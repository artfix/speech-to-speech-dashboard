"""Qwen3-TTS voice library for the speech-to-speech dashboard.

Backs the dashboard's qwen3 voice library UI. Unlike Chatterbox
(``voice_library.py``), qwen3 doesn't expose user-cloned voices through
this module: the only "library" of voices the user can pick from is the
9 preset CustomVoice speakers baked into the model weights. The
``synthesize_qwen3_test`` function lets the user preview any of those
speakers (and, where ABI v2 is available, a Base model's ref_audio or a
VoiceDesign instruct) by running a one-shot synthesis through the same
``QwenTTS.from_pretrained().synthesize()`` path the pipeline uses at
runtime.

Why a separate module (instead of adding to ``voice_library.py``):
chatterbox and qwen3-TTS are unrelated backends with different
dependency footprints (chatterbox-tts is an optional extra; qwen3-TTS
ships with the Pascal wheel installer). Coupling them would force
``import chatterbox`` on every qwen3 request and vice versa. Decoupling
also matches the dashboard's per-TTS-backend module convention.

Public API
----------

- ``PRESET_SPEAKERS`` -- ordered list of the 9 CustomVoice speaker
  names with their native-language sample sentence (used by the
  dashboard's voice library UI to render cards + pre-fill test modals).
- ``PRESET_LANGUAGES`` -- mapping from speaker name to the language
  code (``"english"``/``"chinese"``/``"japanese"``/``"korean"``) the
  Qwen3 synthesize API expects. The Qwen3-TTS CustomVoice cards on the
  HuggingFace model card document this; we map per-speaker, not per-
  text, because each speaker is optimized for one native language.
- ``DEFAULT_TEST_TEXT`` -- a one-line English phrase used as the
  initial placeholder in the test-sentence modal.
- ``list_ref_audio_files()`` -- enumerate ``voices/qwen3_refs/``,
  return ``[{name, path, size_bytes, mtime}]`` for the voice-library
  "Reference voices" section.
- ``synthesize_qwen3_test(...)`` -- run a one-shot
  ``QwenTTS.from_pretrained().synthesize(...)`` and return WAV bytes.

The qwen3 binding is imported lazily inside each public function so the
rest of the dashboard keeps working when the wheel isn't installed.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
REF_AUDIO_DIR = REPO_ROOT / "voices" / "qwen3_refs"


# qwen3-TTS CustomVoice ships these 9 preset speakers (verified against
# https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice and the
# matching 0.6B card). Each speaker is optimized for a single native
# language; passing a different ``lang=`` to ``synthesize()`` either
# degrades quality or produces code-switching artifacts, so the
# dashboard keeps each speaker paired with its native language. The
# sample sentence is what the test button pre-fills in the modal so the
# user immediately hears the speaker in its strongest language.
PRESET_SPEAKERS: list[dict[str, str]] = [
    {"name": "Vivian",     "language": "chinese",  "label": "Chinese (Mandarin)",     "sample": "你好，我叫 Vivian，这是我的声音。"},
    {"name": "Serena",     "language": "chinese",  "label": "Chinese (Mandarin)",     "sample": "你好，我叫 Serena，希望你喜欢我的声音。"},
    {"name": "Uncle_Fu",   "language": "chinese",  "label": "Chinese (Mandarin)",     "sample": "你好，我是 Fu 叔叔，听起来怎么样？"},
    {"name": "Dylan",      "language": "chinese",  "label": "Chinese (Beijing)",      "sample": "你好，我叫 Dylan，听我说北京话。"},
    {"name": "Eric",       "language": "chinese",  "label": "Chinese (Sichuan)",      "sample": "你好，我叫 Eric，我讲四川话。"},
    {"name": "Ryan",       "language": "english",  "label": "English",                "sample": "Hi, I'm Ryan, this is how I sound."},
    {"name": "Aiden",      "language": "english",  "label": "English",                "sample": "Hi, I'm Aiden, this is my voice."},
    {"name": "Ono_Anna",   "language": "japanese", "label": "Japanese",               "sample": "こんにちは、Anna です。私の声はこんな感じです。"},
    {"name": "Sohee",      "language": "korean",   "label": "Korean",                 "sample": "안녕하세요, Sohee 입니다. 제 목소리는 이런 느낌이에요."},
]


# Convenience lookup so the UI can build a sample-sentence map without
# re-iterating the list every keystroke.
PRESET_BY_NAME: dict[str, dict[str, str]] = {p["name"]: p for p in PRESET_SPEAKERS}


DEFAULT_TEST_TEXT = "Hello, this is a test of the qwen3 text-to-speech voice."

# Native sample rate for qwen3-TTS output. The CustomVoice GGUF
# decodes at 24 kHz mono; the realtime streamer resamples to 16 kHz
# for the OpenAI Realtime protocol, but the test preview returns the
# raw 24 kHz bytes so the browser plays back exactly what the model
# produced (the Streamer resampling is lossless integer resample).
_NATIVE_SAMPLE_RATE = 24000


@dataclass
class RefAudioEntry:
    """One reference audio file in voices/qwen3_refs/."""

    name: str
    path: str
    size_bytes: int
    mtime: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "mtime": self.mtime,
        }


# ----------------------------------------------------------------------
# Reference audio enumeration (for the Reference voices section)
# ----------------------------------------------------------------------


def list_ref_audio_files() -> list[RefAudioEntry]:
    """List reference audio files uploaded via ``/api/qwen3_ref_audio``.

    Files live under ``<repo>/voices/qwen3_refs/`` and are named
    ``<uuid><ext>`` (uuid generated server-side to prevent collisions).
    We expose the file's basename as ``name`` (stripped of the uuid
    prefix is harder than it's worth — the user sees the basename anyway
    and can rename via the OS if they want a friendlier label).

    Returns an empty list when the directory doesn't exist (the user
    has never uploaded a reference audio file). Skips subdirectories and
    zero-byte files.
    """
    if not REF_AUDIO_DIR.is_dir():
        return []
    out: list[RefAudioEntry] = []
    for p in sorted(REF_AUDIO_DIR.iterdir()):
        if not p.is_file():
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        if stat.st_size <= 0:
            continue
        out.append(
            RefAudioEntry(
                name=p.name,
                path=str(p),
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    return out


# ----------------------------------------------------------------------
# WAV packaging
# ----------------------------------------------------------------------


def _to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """Package a 1-D float32 numpy array as 16-bit PCM mono WAV bytes.

    Inlined here (rather than imported from ``voice_library``) to avoid
    coupling qwen3 with the chatterbox-specific module. Chatterbox and
    qwen3 are unrelated backends; the dashboard already keeps their
    data plumbing separate (chatterbox Conditionals live in
    ``voices/<name>.pt``, qwen3 ref audio lives in ``voices/qwen3_refs/``).
    """
    import numpy as np

    if audio is None:
        raise ValueError("audio is None")
    audio_np = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio_np.ndim != 1:
        raise ValueError(f"audio must be 1-D, got shape {audio_np.shape}")
    audio_int16 = np.clip(audio_np * 32768, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ----------------------------------------------------------------------
# Synthesis (test preview)
# ----------------------------------------------------------------------


class Qwen3NotInstalled(RuntimeError):
    """Raised when an operation needs the qwen3-cpp-python package."""


def _import_qwentts() -> Any:
    """Lazy import of ``qwentts_cpp`` so the rest of the dashboard works when the wheel isn't installed.

    Raises ``Qwen3NotInstalled`` with a clear message.
    """
    try:
        import qwentts_cpp  # type: ignore[import-not-found]  # noqa: F401
        from qwentts_cpp import QwenTTS  # type: ignore[import-not-found]  # noqa: F401

        return QwenTTS
    except ImportError as e:
        raise Qwen3NotInstalled(
            "qwentts-cpp-python is not installed. Pick --tts qwen3 once with "
            "--qwen3-tts-device cuda; the dashboard auto-installs the Pascal "
            "wheel on sm_61 GPUs."
        ) from e


def synthesize_qwen3_test(
    *,
    speaker: Optional[str] = None,
    text: str,
    language: str = "english",
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    instruct: Optional[str] = None,
    seed: int = -1,
    temperature: float = 0.9,
    top_p: float = 1.0,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    device: str = "cuda",
) -> bytes:
    """Synthesize a short preview for a qwen3 voice and return WAV bytes.

    Args:
        speaker: One of the 9 CustomVoice preset names (Vivian, Serena,
            Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee). Required
            for CustomVoice models; ignored for VoiceDesign / Base.
        text: The text to synthesize. Caller must pre-validate non-empty.
        language: Qwen3 language code (``"english"``, ``"chinese"``,
            ``"japanese"``, ``"korean"``, ``"auto"``). Defaults to
            ``"english"``.
        model_id: HF Hub model ID. Defaults to the 1.7B CustomVoice model.
        ref_audio: Path to a reference audio file. Required for Base models.
            Will raise ``QwenTTSError`` if the bundled wheel lacks ABI v2
            (``qt_extract_voice_ref``).
        ref_text: Transcript of the reference audio. Required for Base models.
        instruct: Voice description for VoiceDesign. Required for VoiceDesign.
        seed: RNG seed. ``-1`` (default) lets the model pick.
        temperature: Sampling temperature.
        top_p / top_k / repetition_penalty: Sampling params.
        device: ``"cuda"`` (default) or ``"cpu"``. CPU works but is slow.

    Returns:
        16-bit PCM mono WAV bytes at 24 kHz.

    Raises:
        ValueError: Invalid arguments.
        Qwen3NotInstalled: ``qwentts_cpp`` not importable.
        QwenTTSError: Synthesis failed (model crash, missing ref audio,
            ABI v2 symbol missing, etc.).
    """
    if not text or not text.strip():
        raise ValueError("Test text must not be empty")

    if not speaker and not (ref_audio and ref_text) and not instruct:
        raise ValueError(
            "Must provide either `speaker` (CustomVoice), "
            "`ref_audio`+`ref_text` (Base voice cloning), "
            "or `instruct` (VoiceDesign)."
        )

    QwenTTS = _import_qwentts()

    logger.info(
        "qwen3 test preview: model=%s speaker=%s language=%s text_len=%d ref_audio=%s",
        model_id,
        speaker or "(none)",
        language,
        len(text),
        ref_audio or "(none)",
    )

    # The qwentts_cpp high-level API takes ref_audio_24k as a numpy
    # array; the upstream binding re-samples internally if needed but
    # for the test preview we keep things simple and load via soundfile
    # at native rate, then let the binding handle resampling. The
    # fallback is librosa.load, which is in the venv already for the
    # qwen3_tts_handler.
    ref_audio_24k = None
    if ref_audio:
        try:
            import soundfile as sf  # type: ignore[import-not-found]
            ref_audio_24k, _ = sf.read(ref_audio, dtype="float32", always_2d=False)
            if hasattr(ref_audio_24k, "ndim") and ref_audio_24k.ndim > 1:
                # Downmix to mono if the source is stereo.
                import numpy as np  # noqa: PLC0415
                ref_audio_24k = ref_audio_24k.mean(axis=1).astype(np.float32)
        except ImportError:
            # soundfile may not be available; try scipy.io.wavfile which
            # the venv already has.
            try:
                from scipy.io import wavfile  # type: ignore[import-not-found]
                sr, data = wavfile.read(ref_audio)
                import numpy as np  # noqa: PLC0415
                ref_audio_24k = data.astype(np.float32) / np.iinfo(data.dtype).max
                if hasattr(ref_audio_24k, "ndim") and ref_audio_24k.ndim > 1:
                    ref_audio_24k = ref_audio_24k.mean(axis=1).astype(np.float32)
                if sr != _NATIVE_SAMPLE_RATE:
                    from math import gcd

                    from scipy.signal import resample_poly  # type: ignore[import-not-found]
                    g = gcd(sr, _NATIVE_SAMPLE_RATE)
                    ref_audio_24k = resample_poly(
                        ref_audio_24k,
                        up=_NATIVE_SAMPLE_RATE // g,
                        down=sr // g,
                    ).astype(np.float32)
            except Exception as e:
                raise ValueError(
                    f"Could not load reference audio {ref_audio!r}: {e}. "
                    "Install soundfile or use a 24 kHz mono WAV file."
                ) from e

    t_start = time.perf_counter()
    model = QwenTTS.from_pretrained(model_id)
    t_loaded = time.perf_counter() - t_start

    try:
        audio_np, sample_rate = model.synthesize(
            text=text,
            lang=language,
            speaker=speaker,
            instruct=instruct,
            ref_audio_24k=ref_audio_24k,
            ref_text=ref_text,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    finally:
        # Free the C context eagerly — repeated test invocations
        # otherwise accumulate GPU memory.
        try:
            model.close()
        except Exception:  # noqa: BLE001
            logger.debug("qwen3 model.close() raised; ignoring")

    t_done = time.perf_counter() - t_start
    logger.info(
        "qwen3 test preview done: load=%.2fs synth=%.2fs samples=%d sr=%d",
        t_loaded,
        t_done - t_loaded,
        int(getattr(audio_np, "size", 0) or 0),
        int(sample_rate),
    )

    return _to_wav_bytes(audio_np, int(sample_rate))
