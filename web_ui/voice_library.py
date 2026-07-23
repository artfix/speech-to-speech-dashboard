"""Chatterbox voice library for the speech-to-speech dashboard.

This module owns the on-disk voice library that backs the dashboard's
"Voice library" UI. The library is stored as a flat directory of ``.pt`` files
(``<repo>/voices/<name>.pt``) plus a single ``voices.json`` manifest that holds
metadata for each entry. The ``.pt`` files are Chatterbox ``Conditionals`` blobs
produced by ``model.prepare_conditionals(<ref audio>)``; the user supplies the
reference audio at clone time but the audio itself is never persisted on the
server.

The module is intentionally decoupled from FastAPI: it raises plain exceptions
and returns plain dataclasses, and the server translates those into HTTP
responses. ``chatterbox`` is imported lazily so the rest of the dashboard
continues to work when the optional extra isn't installed.

Public API
----------

- ``VOICES_DIR`` -- absolute path to the voices directory.
- ``MANIFEST_PATH`` -- absolute path to ``voices.json``.
- ``class ChatterboxNotInstalled`` -- raised by every public function that
  needs the chatterbox package.
- ``class VoiceEntry`` -- one row in the manifest.
- ``list_voices()`` -- read the manifest, return entries.
- ``save_voice(name, audio_bytes, suffix, model_variant, sample_count, sample_rate)``
  -- write a temp source, run ``prepare_conditionals``, save the Conditionals,
  delete the temp source, add the manifest entry.
- ``delete_voice(name)`` -- remove the ``.pt`` and the manifest entry.
- ``synthesize_test(voice, text, model_variant, **params)`` -- generate a
  short audio preview, return raw ``WAV`` bytes.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
VOICES_DIR = REPO_ROOT / "voices"
MANIFEST_PATH = VOICES_DIR / "voices.json"
INBOX_DIR = VOICES_DIR / "_inbox"


# Default output sample rate for test previews. Matches the pipeline's streamer.
_PREVIEW_SAMPLE_RATE = 16000


class ChatterboxNotInstalled(RuntimeError):
    """Raised when an operation needs the optional ``chatterbox-tts`` package."""


@dataclass
class VoiceEntry:
    """One row in the voice library manifest."""

    name: str
    created_at: float
    model_variant: str
    sample_count: int = 0
    sample_rate: int = 0
    file_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Manifest I/O
# ----------------------------------------------------------------------


def _ensure_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)


def _read_manifest() -> dict[str, dict[str, Any]]:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read voice manifest at %s: %s", MANIFEST_PATH, e)
        return {}
    voices = data.get("voices") if isinstance(data, dict) else None
    if not isinstance(voices, dict):
        return {}
    return voices


def _write_manifest(voices: dict[str, dict[str, Any]]) -> None:
    _ensure_dirs()
    tmp = MANIFEST_PATH.with_suffix(".json.tmp")
    payload = {"version": 1, "voices": voices}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(MANIFEST_PATH)


def list_voices() -> list[VoiceEntry]:
    """Return all voices known to the manifest, sorted by name."""
    return [
        VoiceEntry(**entry)
        for entry in sorted(_read_manifest().values(), key=lambda e: e.get("name", ""))
    ]


def _pt_path(name: str) -> Path:
    return VOICES_DIR / f"{name}.pt"


def _validate_name(name: str) -> None:
    if not name:
        raise ValueError("Voice name must not be empty")
    # Limit to a safe character set so the file is portable and the
    # --chatterbox-voice CLI flag stays shell-friendly.
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
    if any(c not in allowed for c in name):
        raise ValueError(
            f"Voice name {name!r} may only contain letters, digits, '_' and '-'"
        )
    if len(name) > 64:
        raise ValueError(f"Voice name {name!r} is longer than 64 characters")


# ----------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------


def _read_wav_info(path: Path) -> tuple[int, int]:
    """Return (n_samples, sample_rate) for a 16-bit mono PCM WAV. Fallback to (0, 0)."""
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes(), w.getframerate()
    except (wave.Error, EOFError, OSError) as e:
        logger.debug("Could not read WAV header from %s: %s", path, e)
        return 0, 0


def _write_int16_wav(path: Path, audio: Any, sample_rate: int) -> None:
    """Write a float32 numpy array as 16-bit PCM mono WAV."""
    import numpy as np

    audio_int16 = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio_int16.tobytes())


# ----------------------------------------------------------------------
# Lazy chatterbox loader
# ----------------------------------------------------------------------


def _import_chatterbox():
    """Lazy import. Raises ``ChatterboxNotInstalled`` with a helpful message."""
    try:
        import chatterbox  # type: ignore[import-not-found]  # noqa: F401
        import chatterbox.tts  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
        raise ChatterboxNotInstalled(
            "Chatterbox is not installed. Click 'Install chatterbox' in the TTS tab to add it."
        ) from e
    # Patch the s3tokenizer mel-spectrogram dtype bug. The pipeline handler
    # also calls this; doing it here covers the dashboard's voice-clone
    # and test-preview paths which run before the handler is ever loaded.
    try:
        from speech_to_speech.TTS.chatterbox_tts_handler import (  # type: ignore[import-not-found]
            _patch_chatterbox_mel_spectrogram,
        )

        _patch_chatterbox_mel_spectrogram()
    except ImportError:
        # The pipeline module isn't importable (e.g. the dashboard is
        # running standalone). Fall through; the model is still usable
        # in cases where the input happens to land on float32.
        pass
    import chatterbox  # re-import to get the (now patched) module

    return chatterbox


def _resolve_model_class(model_variant: str) -> tuple[type, dict[str, Any], str]:
    """Return (class, from_pretrained_kwargs, tts_module_name) for a variant.

    Mirrors the dispatch logic in ``chatterbox_tts_handler._resolve_model_class``.
    """
    chatterbox = _import_chatterbox()
    if model_variant == "chatterbox":
        return chatterbox.tts.ChatterboxTTS, {"device": "cpu"}, "chatterbox.tts"
    if model_variant == "chatterbox-multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # type: ignore[attr-defined]

        return ChatterboxMultilingualTTS, {"device": "cpu"}, "chatterbox.mtl_tts"
    if model_variant == "chatterbox-turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # type: ignore[attr-defined]

        return ChatterboxTurboTTS, {"device": "cpu"}, "chatterbox.tts_turbo"
    if model_variant == "chatterbox-nano":
        from chatterbox.tts_turbo import ChatterboxTurboTTS  # type: ignore[attr-defined]

        return ChatterboxTurboTTS, {"device": "cpu", "nano": True}, "chatterbox.tts_turbo"
    raise ValueError(
        f"Unknown chatterbox model variant: {model_variant!r}. "
        "Expected one of: chatterbox, chatterbox-turbo, chatterbox-nano, chatterbox-multilingual."
    )


# ----------------------------------------------------------------------
# Public operations
# ----------------------------------------------------------------------


def save_voice(
    name: str,
    audio_bytes: bytes,
    suffix: str,
    model_variant: str,
) -> VoiceEntry:
    """Clone a voice from a reference audio and persist the Conditionals.

    Args:
        name: Display name (also the on-disk filename stem). Must be
            ``[A-Za-z0-9_-]{1,64}``.
        audio_bytes: Raw contents of the reference audio file. WAV is preferred
            (used to read sample_count/sample_rate for the manifest); other
            formats are passed through to ``prepare_conditionals`` which uses
            ``librosa.load`` internally.
        suffix: File extension hint for the temp source (e.g. ``.wav``, ``.mp3``).
            Used to pick the right decoder in ``prepare_conditionals``.
        model_variant: Which chatterbox model class to use for the embedding.
            Stored on the entry so the dashboard knows which variant the
            clone was made with.

    Returns:
        The new :class:`VoiceEntry`.

    Raises:
        ValueError: If the name is invalid or the model variant is unknown.
        ChatterboxNotInstalled: If the ``chatterbox`` package is missing.
    """
    _validate_name(name)
    if _pt_path(name).exists():
        raise FileExistsError(f"Voice {name!r} already exists; delete it first to re-clone")
    _ensure_dirs()

    # Write the source audio to a temp file. ``prepare_conditionals`` expects a
    # path; after it returns, the audio is already encoded into the Conditionals
    # tensor, so we delete the source before the function returns.
    suffix_norm = suffix if suffix.startswith(".") else f".{suffix}"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix_norm, dir=str(INBOX_DIR))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(audio_bytes)
        n_samples, sample_rate = _read_wav_info(tmp_path)

        model_cls, from_kwargs, _ = _resolve_model_class(model_variant)
        logger.info(
            "Cloning voice %r with %s (%.1f KB of audio)",
            name,
            model_cls.__name__,
            len(audio_bytes) / 1024,
        )
        model = model_cls.from_pretrained(**from_kwargs)
        # ``prepare_conditionals`` in chatterbox-tts is a side-effect call:
        # it sets ``model.conds`` but does NOT return it. We read
        # ``model.conds`` after the call to save the resulting
        # Conditionals to disk.
        model.prepare_conditionals(str(tmp_path))
        conds = model.conds
        if conds is None:
            raise RuntimeError(
                f"{model_cls.__name__}.prepare_conditionals did not populate "
                "model.conds. This is unexpected for chatterbox-tts 0.1.7."
            )
        out_path = _pt_path(name)
        conds.save(str(out_path))
    finally:
        # Source audio is intentionally never persisted. The .pt holds the
        # speaker embedding; the original waveform stays on the user's disk.
        try:
            tmp_path.unlink()
        except OSError:
            logger.debug("Could not delete temp source %s", tmp_path)

    # Update the manifest atomically.
    manifest = _read_manifest()
    entry = VoiceEntry(
        name=name,
        created_at=time.time(),
        model_variant=model_variant,
        sample_count=n_samples,
        sample_rate=sample_rate,
        file_size_bytes=out_path.stat().st_size,
    )
    manifest[name] = entry.to_dict()
    _write_manifest(manifest)
    logger.info("Voice %r cloned; %d bytes of Conditionals on disk", name, entry.file_size_bytes)
    return entry


def delete_voice(name: str) -> bool:
    """Remove a voice from disk + manifest. Returns ``True`` if anything was removed."""
    pt = _pt_path(name)
    removed = False
    if pt.exists():
        pt.unlink()
        removed = True
    manifest = _read_manifest()
    if name in manifest:
        del manifest[name]
        _write_manifest(manifest)
        removed = True
    return removed


def _resample_to_16k(audio: Any, native_sr: int) -> Any:
    """Resample a 1-D float numpy array to 16 kHz. No-op if already 16 kHz."""
    if native_sr == _PREVIEW_SAMPLE_RATE:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(native_sr, _PREVIEW_SAMPLE_RATE)
    return resample_poly(audio, up=_PREVIEW_SAMPLE_RATE // g, down=native_sr // g)


def synthesize_test(
    voice: str,
    text: str,
    model_variant: str,
    **params: Any,
) -> bytes:
    """Generate a short audio preview for a voice and return WAV bytes.

    The pipeline runs at 16 kHz; the preview is the same rate so users hear
    exactly what the robot will sound like. The Conditionals are loaded from
    ``<voices_dir>/<voice>.pt`` (matching the pipeline handler's behavior).

    Extra ``params`` map to ``model.generate`` keyword arguments (e.g.
    ``exaggeration``, ``cfg_weight``, ``temperature``, ``top_p``, ``top_k``).
    Multilingual variants require ``language_id`` (default ``"en"``).
    """
    if not text.strip():
        raise ValueError("Test text must not be empty")
    _validate_name(voice)
    pt = _pt_path(voice)
    if not pt.exists():
        raise FileNotFoundError(
            f"Voice {voice!r} not found at {pt}. Clone it first, or pick another voice."
        )

    # Ensure chatterbox is importable before we try to load the model.
    # Raises a clear ChatterboxNotInstalled if the user hasn't installed it.
    _import_chatterbox()
    model_cls, from_kwargs, tts_module = _resolve_model_class(model_variant)
    logger.info("Synthesizing test for voice %r on %s", voice, model_cls.__name__)
    model = model_cls.from_pretrained(**from_kwargs)

    # Load the saved Conditionals and attach them. The pipeline handler does
    # exactly this at runtime; we mirror it so the preview matches the robot.
    if tts_module == "chatterbox.tts":
        from chatterbox.tts import Conditionals  # type: ignore[attr-defined]
    elif tts_module == "chatterbox.mtl_tts":
        from chatterbox.mtl_tts import Conditionals  # type: ignore[attr-defined]
    else:
        from chatterbox.tts_turbo import Conditionals  # type: ignore[attr-defined]
    model.conds = Conditionals.load(str(pt), map_location=from_kwargs["device"]).to(
        from_kwargs["device"]
    )

    is_turbo = model_variant in ("chatterbox-turbo", "chatterbox-nano")
    is_multilingual = model_variant == "chatterbox-multilingual"

    gen_kwargs: dict[str, Any] = {
        "text": text,
        "temperature": float(params.get("temperature", 0.8)),
        "top_p": float(params.get("top_p", 1.0)),
    }
    if is_turbo:
        gen_kwargs["top_k"] = int(params.get("top_k", 1000))
        gen_kwargs["norm_loudness"] = True
        gen_kwargs["exaggeration"] = float(params.get("exaggeration", 0.5))
        gen_kwargs["cfg_weight"] = float(params.get("cfg_weight", 0.5))
    else:
        gen_kwargs["exaggeration"] = float(params.get("exaggeration", 0.5))
        gen_kwargs["cfg_weight"] = float(params.get("cfg_weight", 0.5))
        gen_kwargs["repetition_penalty"] = float(params.get("repetition_penalty", 1.2))
        gen_kwargs["min_p"] = float(params.get("min_p", 0.05))
    if is_multilingual:
        gen_kwargs["language_id"] = str(params.get("language_id") or "en")
    wav = model.generate(**gen_kwargs)

    # ``model.generate`` returns a torch tensor shaped (1, n_samples). Strip
    # the batch dim, resample to 16 kHz, write as int16 WAV.
    import numpy as np

    audio_np = wav.detach().cpu().numpy().squeeze(0).astype(np.float32)
    if audio_np.ndim != 1:
        audio_np = audio_np.reshape(-1)

    # The native sample rate is 24 kHz for every variant.
    native_sr = 24000
    audio_np = _resample_to_16k(audio_np, native_sr)

    # Package as WAV in-memory.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_PREVIEW_SAMPLE_RATE)
        audio_int16 = np.clip(audio_np * 32768, -32768, 32767).astype(np.int16)
        w.writeframes(audio_int16.tobytes())
    return buf.getvalue()
