"""Chatterbox TTS handler.

Wraps Resemble AI's Chatterbox TTS family (English, Multilingual, Turbo, Nano) as a
BaseHandler for the speech-to-speech pipeline. Chatterbox is a voice-cloning TTS:
a short reference audio is embedded once via ``model.prepare_conditionals()`` and the
resulting ``Conditionals`` object can be saved to disk and reloaded cheaply for every
subsequent synthesis call.

This module imports the ``chatterbox`` package lazily inside :meth:`setup` so that the
package is only required when the user actually picks ``--tts chatterbox``. The
dashboard's voice library stores one ``.pt`` file per named voice in
``<repo_root>/voices/<name>.pt``; the handler loads the right file at synthesis time.

Unlike the other TTS backends, Chatterbox has no streaming API -- every
``model.generate()`` call returns the full utterance. The pipeline already coalesces
sentences upstream (see ``qwen3_tts_handler._coalesce_pending_tts_input``), so
treating each ``TTSInput`` as one ``generate()`` call is the right granularity.
"""

from __future__ import annotations

import logging
import os
from math import gcd
from threading import Event
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.baseHandler import BaseHandler
from speech_to_speech.pipeline.cancel_scope import CancelScope
from speech_to_speech.pipeline.handler_types import TTSIn, TTSOut
from speech_to_speech.pipeline.messages import AUDIO_RESPONSE_DONE, EndOfResponse
from speech_to_speech.pipeline.speculative_turns import SpeculativeTurnTracker

logger = logging.getLogger(__name__)
console = Console()


# Native output sample rate of all Chatterbox models. Internal S3 tokenizer runs at
# 16 kHz but the mel decoder upsamples to 24 kHz before yielding, so 24 kHz is what
# we get back from ``model.generate()``.
_CHATTERBOX_NATIVE_SR = 24000


def _model_rss_mb() -> float:
    """Best-effort estimate of this Python process's RSS in MB.

    Used by :meth:`ChatterboxTTSHandler.on_unload` to tell the user how much
    RAM the model occupied before it was released. Linux-only path; falls
    back to ``0`` on other platforms.
    """
    try:
        import os

        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _patch_chatterbox_mel_spectrogram() -> None:
    """Idempotently patch chatterbox's mel-spectrogram dtypes.

    chatterbox-tts==0.1.7 ships a few dtype-mismatch bugs that surface
    on CPU when the input audio or magnitude spectrogram is float64
    while the model weights are float32:

    1. ``S3Tokenizer.log_mel_spectrogram`` (in
       ``chatterbox/models/s3tokenizer/s3tokenizer.py``) multiplies a
       float32 mel-filterbank matrix by a magnitude spectrogram that
       can be float64, raising:

           RuntimeError: expected scalar type Double but found Float

    2. ``voice_encoder.melspec.melspectrogram`` produces mels of the
       same dtype as the input wav (float64 if librosa handed it
       float64). The downstream VoiceEncoder LSTM is float32, so the
       forward pass raises:

           ValueError: RNN input dtype (torch.float64) does not match
                       weight dtype (torch.float32)

    The upstream fix is a one-line dtype cast. We patch at import-time
    (idempotent) rather than monkey-patching chatterbox on disk so the
    patch survives ``uv pip install --upgrade``.

    ``voice_encoder.py`` does ``from .melspec import melspectrogram``
    which binds the function into its own module namespace, so we have
    to update that local reference too -- patching only
    ``melspec.melspectrogram`` is not enough.
    """
    if getattr(_patch_chatterbox_mel_spectrogram, "_applied", False):
        return
    try:
        from chatterbox.models.s3tokenizer import s3tokenizer as _s3_mod  # type: ignore[attr-defined]
    except ImportError:
        # chatterbox isn't installed -- nothing to patch. setup() will
        # surface a friendly ImportError when the user actually uses the
        # backend.
        return

    def _wrap_s3_log_mel(original):  # type: ignore[no-untyped-def]
        if getattr(original, "_chatterbox_patched", False):
            return original

        def log_mel_spectrogram(self, wav):  # type: ignore[no-untyped-def]
            import torch

            if wav.dtype != torch.float32:
                wav = wav.to(torch.float32)
            return original(self, wav)

        log_mel_spectrogram._chatterbox_patched = True  # type: ignore[attr-defined]
        return log_mel_spectrogram

    original_s3 = _s3_mod.S3Tokenizer.log_mel_spectrogram
    _s3_mod.S3Tokenizer.log_mel_spectrogram = _wrap_s3_log_mel(original_s3)

    # The voice encoder's melspec returns the natural numpy dtype of the
    # input wav. Wrap it to always return float32. ``voice_encoder.py``
    # imports the symbol with ``from .melspec import melspectrogram``,
    # so we must update its local reference in addition to the source.
    try:
        from chatterbox.models.voice_encoder import melspec as _ve_ms  # type: ignore[attr-defined]
        from chatterbox.models.voice_encoder import voice_encoder as _ve  # type: ignore[attr-defined]

        # Grab the original from the melspec module (before we wrap it)
        # and use it directly inside the wrap. This avoids any recursion
        # with the ``voice_encoder`` module's local reference.
        _original_ve_melspectrogram = _ve_ms.melspectrogram

        def _wrapped_ve_melspectrogram(wav, hp, pad=True):  # type: ignore[no-untyped-def]
            import numpy as np

            if wav.dtype != np.float32:
                wav = wav.astype(np.float32, copy=False)
            out = _original_ve_melspectrogram(wav, hp, pad=pad)
            if out.dtype != np.float32:
                out = out.astype(np.float32, copy=False)
            return out

        # Only wrap once.
        if not getattr(_ve_ms.melspectrogram, "_chatterbox_patched", False):
            _wrapped_ve_melspectrogram._chatterbox_patched = True  # type: ignore[attr-defined]
            _ve_ms.melspectrogram = _wrapped_ve_melspectrogram  # type: ignore[assignment]
            # Update the local name in voice_encoder.py so the existing
            # ``from .melspec import melspectrogram`` call uses our wrap.
            _ve.melspectrogram = _wrapped_ve_melspectrogram  # type: ignore[attr-defined]
    except ImportError:
        # The voice encoder module is only used by the original
        # (English / Multilingual) variants. If the user only installed
        # Turbo we can skip this half of the patch.
        pass

    _patch_chatterbox_mel_spectrogram._applied = True


class ChatterboxTTSHandler(BaseHandler[TTSIn, TTSOut]):
    """TTS handler for the Chatterbox TTS family (Resemble AI).

    Voice cloning is supported via the dashboard's voice library. The handler
    loads a ``Conditionals`` object from ``<voices_dir>/<voice>.pt`` whenever
    ``--chatterbox-voice`` is set; otherwise the model's bundled default voice
    (one ``conds.pt`` shipped with the HF weights) is used.
    """

    def setup(
        self,
        should_listen: Event,
        model_variant: str = "chatterbox-turbo",
        device: str = "auto",
        voice: str = "",
        voices_dir: str = "voices",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        top_k: int = 1000,
        language_id: Optional[str] = None,
        sample_rate: int = 16000,
        blocksize: int = 512,
        split_sentences: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
        cancel_scope: CancelScope | None = None,
        speculative_turns: SpeculativeTurnTracker | None = None,
    ) -> None:
        """Load a Chatterbox model and prepare to synthesize.

        Args:
            should_listen: Event used to release the audio consumer after end-of-response.
            model_variant: One of ``chatterbox``, ``chatterbox-turbo``,
                ``chatterbox-nano``, ``chatterbox-multilingual``.
            device: ``auto`` picks CUDA if available, then MPS, then CPU. Set
                ``cuda``/``mps``/``cpu`` to force.
            voice: Name of a cloned voice (loads ``<voices_dir>/<voice>.pt``). Empty
                uses the bundled default voice.
            voices_dir: Directory containing cloned voice ``.pt`` files, relative
                to the repo root.
            exaggeration/cfg_weight/repetition_penalty/min_p: Sampling knobs
                honored by the English + Multilingual variants only. Ignored on
                Turbo/Nano (the upstream library logs a warning if they're set).
            top_k: Sampling knob honored by Turbo/Nano only.
            language_id: Required for the multilingual variant. Ignored otherwise.
            sample_rate: Output sample rate after resampling. Default 16000 matches
                the pipeline's audio streamer.
            blocksize: Number of int16 samples per yielded chunk.
            split_sentences: When True, split each TTSInput on sentence boundaries
                (NLTK punkt) and synthesize one sentence at a time so the speaker
                starts playing sentence 1 while sentence 2 is still generating.
                Cuts TTFA by ~1-2s on a typical multi-sentence reply. Falls back
                to a single generate() call if NLTK/punkt_tab is unavailable or
                if the input is a single sentence. Default True.
        """
        self.should_listen = should_listen
        self.cancel_scope = cancel_scope
        self.speculative_turns = speculative_turns
        # ``self.device = ...`` triggers the property setter, which resolves
        # ``"auto"`` to cuda/mps/cpu and stores the resolved value in
        # ``self._device``. The setter also handles the case where torch is
        # not installed (falls back to "cpu").
        self.device = device
        self.model_variant = model_variant
        self.voice = voice
        self.voices_dir = voices_dir
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.min_p = min_p
        self.top_p = top_p
        self.top_k = top_k
        self.language_id = language_id
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.split_sentences = split_sentences
        self.gen_kwargs = gen_kwargs or {}

        # Suppress verbose logging from the chatterbox library and the Perth
        # watermarker (mirrors pocket_tts_handler.py:62-65).
        logging.getLogger("chatterbox").setLevel(logging.WARNING)
        logging.getLogger("perth").setLevel(logging.WARNING)

        # Patch the s3tokenizer's log_mel_spectrogram so the input waveform
        # is cast to the mel-filterbank's float32 dtype before the matmul.
        # chatterbox-tts==0.1.7 ships a bug where the filterbank is float32
        # but librosa may hand back a float64 magnitude spectrogram, and the
        # matmul on line ~163 of models/s3tokenizer/s3tokenizer.py raises:
        #     RuntimeError: expected scalar type Double but found Float
        # The fix is a one-line dtype cast inside the function. We patch
        # at import-time (idempotent) rather than monkeypatching chatterbox
        # on disk so the patch survives `uv pip install --upgrade`.
        _patch_chatterbox_mel_spectrogram()

        # The model is loaded lazily so the package is only required when this
        # backend is actually selected. ``importlib.import_module`` lets us
        # surface a clean error if the user hasn't installed the [chatterbox]
        # extra yet.
        self._load_model()
        if self.voice:
            self._load_conditionals(self.voice)
        else:
            logger.info("No --chatterbox-voice set; using the model's bundled default voice")

        # Precompute resampling factors (24 kHz -> target SR). Stays a no-op if the
        # user ever configures sample_rate=24000.
        if self.sample_rate != _CHATTERBOX_NATIVE_SR:
            g = gcd(self.sample_rate, _CHATTERBOX_NATIVE_SR)
            self._resample_up = self.sample_rate // g
            self._resample_down = _CHATTERBOX_NATIVE_SR // g
            self._needs_resampling = True
        else:
            self._resample_up = 1
            self._resample_down = 1
            self._needs_resampling = False

        # Load the voice Conditionals (cloned voice) if one was requested. An
        # empty voice name means "use the bundled default", which the model
        # already carries in its own ``model.conds``.

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        if value == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    value = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    value = "mps"
                else:
                    value = "cpu"
            except ImportError:
                value = "cpu"
        self._device = value

    def _resolve_model_class(self) -> tuple[type, dict[str, Any]]:
        """Map ``model_variant`` to a concrete Chatterbox class + ``from_pretrained`` kwargs.

        Lazy-imports the chatterbox submodules so the import only fails for users
        who actually pick this backend. Raises ``ImportError`` with a clear hint
        that matches the existing pocket/kokoro style at
        ``s2s_pipeline.py:942-961``.
        """
        try:
            if self.model_variant == "chatterbox":
                from chatterbox.tts import ChatterboxTTS

                return ChatterboxTTS, {"device": self._device}
            if self.model_variant == "chatterbox-multilingual":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS

                return ChatterboxMultilingualTTS, {"device": self._device}
            if self.model_variant == "chatterbox-nano":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                return ChatterboxTurboTTS, {"device": self._device, "nano": True}
            if self.model_variant == "chatterbox-turbo":
                from chatterbox.tts_turbo import ChatterboxTurboTTS

                return ChatterboxTurboTTS, {"device": self._device}
        except ImportError as e:
            raise ImportError(
                "Chatterbox is optional. Install it with `pip install "
                '"speech-to-speech[chatterbox]"` (or the dashboard\'s '
                '"Install chatterbox" button in the TTS tab).'
            ) from e
        raise ValueError(
            f"Unknown chatterbox_model_variant: {self.model_variant!r}. "
            "Expected one of: chatterbox, chatterbox-turbo, chatterbox-nano, chatterbox-multilingual."
        )

    def _voice_pt_path(self, voice: str) -> str:
        """Return the absolute path to ``voices/<voice>.pt``.

        Resolved relative to the repo root so the dashboard's voice library
        stays in one well-known location.
        """
        if os.path.isabs(voice):
            return voice
        # The pipeline runs with cwd=<repo_root>, but if the user supplied a
        # relative path under the configured voices_dir, honor that.
        if os.sep in voice or voice.startswith("./"):
            return voice
        return os.path.join(self.voices_dir, voice + ".pt")

    def _load_model(self) -> None:
        """Instantiate the chatterbox model class for the current variant.

        Idempotent: a no-op if ``self.model`` is already populated. Called from
        ``setup()`` and re-called on demand by :meth:`process` if the user
        has issued an UNLOAD_TTS control message (see :meth:`on_unload`).
        """
        if getattr(self, "model", None) is not None:
            return
        model_cls, model_kwargs = self._resolve_model_class()
        logger.info(
            "Loading Chatterbox variant=%s device=%s",
            self.model_variant,
            self.device,
        )
        self.model = model_cls.from_pretrained(**model_kwargs)
        # Move to the requested device. ``from_pretrained`` only honors the
        # ``device`` kwarg if it's a string torch device; some models
        # accept the value as-is, others need a manual move.
        if self.device != "auto":
            try:
                self.model = self.model.to(self.device)
            except Exception:  # noqa: BLE001
                logger.debug("Chatterbox model did not accept .to(%s); leaving on default device", self.device)

    def _ensure_model_loaded(self) -> None:
        """Lazy-reload the model after an UNLOAD_TTS released it.

        Cheap when the model is already in RAM (just an attribute check). When
        the model is missing, replays the same code path that ``setup()`` ran
        (model instantiation, device move) and re-attaches the active voice
        Conditionals.
        """
        if getattr(self, "model", None) is not None:
            return
        logger.info("Chatterbox model not in RAM; reloading (UNLOAD_TTS was issued)")
        self._load_model()
        if self.voice:
            self._load_conditionals(self.voice)

    def on_unload(self) -> None:
        """Drop the chatterbox model from RAM. Reloaded on next TTS request.

        The :class:`BaseHandler.run` loop dispatches the ``UNLOAD_TTS`` control
        message here. We delete the model attribute (forcing the wrapped
        ``nn.Module`` to be garbage-collected) and run ``gc.collect()`` so the
        OS reclaims the pages before the user looks at their memory meter.
        CPU and CUDA caches are both cleared. ``self.voice`` and
        ``self.model_variant`` are preserved so :meth:`_ensure_model_loaded`
        can rebuild on demand.
        """
        if getattr(self, "model", None) is None:
            logger.debug("Chatterbox model already unloaded; nothing to release")
            return
        logger.info("Unloading Chatterbox model from RAM (~%.0f MB free'd)", _model_rss_mb())
        self.model = None
        try:
            import gc

            gc.collect()
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Chatterbox model unloaded; next TTS request will reload (~20s on CPU)")

    def _load_conditionals(self, voice: str) -> None:
        """Load Conditionals from disk and attach them to the model.

        Chatterbox's ``Conditionals`` dataclass exposes a ``.load(path)`` class
        method. The library re-derives the speaker embedding lazily on the next
        ``generate()`` call if the saved Conditionals don't match the current
        model variant, so we trust the file is well-formed.
        """
        from chatterbox.tts import Conditionals  # type: ignore[attr-defined]

        path = self._voice_pt_path(voice)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cloned voice {voice!r} not found at {path!r}. Use the dashboard's voice "
                "library to clone a voice, or clear --chatterbox-voice to use the default."
            )
        logger.info("Loading cloned voice Conditionals from %s", path)
        self.model.conds = Conditionals.load(path, map_location=self._device).to(self._device)

    def _apply_session_voice_override(
        self,
        response: Any = None,
        runtime_config: Any = None,
    ) -> None:
        """Re-load Conditionals if the Realtime response named a different voice.

        Mirrors the same override hook in
        ``qwen3_tts_handler._apply_session_voice_override`` (s2s_pipeline.py:408).
        Reads ``response.audio.output.voice`` first, then
        ``runtime_config.session.audio.output.voice`` as a fallback.
        """
        session_voice: Optional[str] = None
        try:
            if response is not None and getattr(response, "audio", None) is not None:
                output = getattr(response.audio, "output", None)
                if output is not None and getattr(output, "voice", None):
                    session_voice = str(output.voice)
            if not session_voice and runtime_config is not None:
                session = getattr(runtime_config, "session", None)
                audio = getattr(session, "audio", None) if session is not None else None
                output = getattr(audio, "output", None) if audio is not None else None
                if output is not None and getattr(output, "voice", None):
                    session_voice = str(output.voice)
        except Exception:  # noqa: BLE001
            # Defensive: any malformed response object should not break TTS.
            return
        if not session_voice or session_voice == self.voice:
            return
        # The user named a new voice mid-response. Reload Conditionals.
        logger.info("Realtime session voice override: %r -> %r", self.voice, session_voice)
        self.voice = session_voice
        self._load_conditionals(session_voice)

    def process(self, tts_input: TTSIn) -> Iterator[TTSOut]:
        """Synthesize one TTSInput (or signal end-of-response) into int16 audio blocks."""
        # If an UNLOAD_TTS was issued earlier, the model was dropped from
        # RAM. Re-instantiate on demand. No-op when the model is already
        # loaded (the common case).
        if getattr(self, "model", None) is None and not isinstance(tts_input, EndOfResponse):
            self._ensure_model_loaded()
        speculative_turns = getattr(self, "speculative_turns", None)
        if isinstance(tts_input, EndOfResponse):
            if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
                tts_input.turn_id,
                tts_input.turn_revision,
            ):
                return
            yield AUDIO_RESPONSE_DONE
            return

        if speculative_turns and not speculative_turns.is_latest_after_reopen_grace(
            tts_input.turn_id,
            tts_input.turn_revision,
        ):
            logger.debug(
                "Dropping stale TTS input for turn=%s rev=%s",
                tts_input.turn_id,
                tts_input.turn_revision,
            )
            return
        if speculative_turns:
            speculative_turns.commit(tts_input.turn_id, tts_input.turn_revision)

        # Allow a Realtime response to swap the active voice mid-stream.
        self._apply_session_voice_override(
            response=getattr(tts_input, "response", None),
            runtime_config=getattr(tts_input, "runtime_config", None),
        )

        gen = self.cancel_scope.generation if self.cancel_scope else None
        text = tts_input.text
        language_code = tts_input.language_code
        console.print(f"[green]ASSISTANT: {text}")

        logger.debug("Synthesizing Chatterbox (%s) for: %s", self.model_variant, text[:50])

        pipeline_start = perf_counter()
        first_chunk = True

        # Decide whether to split. The flag defaults to True; flipping it off in
        # the dashboard restores the pre-v0.3.8 single-call path used as a
        # safety net when the v0.3.10 split was first introduced.
        sentences: list[str] = self._split_sentences(text) if self.split_sentences else [text]
        if len(sentences) > 1:
            logger.debug(
                "Chatterbox split-sentences ON: %d sentence(s) for: %s",
                len(sentences),
                text[:50],
            )

        for sentence_idx, sentence in enumerate(sentences):
            # Cancellation between sentences: if the user interrupted since the
            # last sentence's chunks finished yielding, don't waste GPU time
            # starting the next one. Checked ONCE per sentence (not per chunk)
            # because the v0.3.8 attempt's per-chunk is_stale() reads interacted
            # badly with the realtime router's discardable/send-loop state when
            # audio flowed in N separate bursts with a small gap between them.
            if gen is not None and self.cancel_scope is not None and self.cancel_scope.is_stale(gen):
                logger.info(
                    "TTS generation cancelled (interruption) between sentences "
                    "(sentence %d/%d)",
                    sentence_idx,
                    len(sentences),
                )
                return

            try:
                wav = self._generate_waveform(sentence, language_code)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Chatterbox generation failed on sentence %d/%d: %s",
                    sentence_idx,
                    len(sentences),
                    e,
                )
                return
            if first_chunk:
                logger.debug("Time to first audio: %.3fs", perf_counter() - pipeline_start)
                first_chunk = False
            elif len(sentences) > 1:
                logger.debug(
                    "Sentence %d/%d ready at %.3fs",
                    sentence_idx + 1,
                    len(sentences),
                    perf_counter() - pipeline_start,
                )

            # Convert to numpy float32 in [-1, 1]. Chatterbox returns a torch tensor
            # shaped (1, n_samples).
            wav_np = wav.detach().cpu().numpy().squeeze(0).astype(np.float32)
            if wav_np.ndim != 1:
                wav_np = wav_np.reshape(-1)

            # Resample 24 kHz -> self.sample_rate using polyphase filtering.
            if self._needs_resampling:
                from scipy.signal import resample_poly

                wav_np = resample_poly(wav_np, up=self._resample_up, down=self._resample_down)

            # Int16 conversion with hard clip to avoid wrap-around.
            audio_int16 = np.clip(wav_np * 32768, -32768, 32767).astype(np.int16)

            # Yield in blocksize chunks, padding the last one with zeros.
            # No per-chunk cancel-scope check: the v0.3.8 attempt had one here
            # and the per-chunk is_stale() reads collided with the realtime
            # router when the audio stream was split into N bursts. The check
            # at the top of the next sentence's loop is sufficient -- it covers
            # mid-utterance interruptions because the router advances the
            # cancel-scope generation on interrupt and the next iteration
            # sees is_stale() == True before doing any further work.
            for i in range(0, len(audio_int16), self.blocksize):
                chunk = audio_int16[i : i + self.blocksize]
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

    def _split_sentences(self, text: str) -> list[str]:
        """Split ``text`` on sentence boundaries using NLTK punkt.

        Used by :meth:`process` (when ``split_sentences=True``) to chunk
        chatterbox synthesis per-sentence so the speaker starts playing
        sentence 1 the moment it's ready, while sentence 2 is still being
        generated on the GPU.

        NLTK punkt_tab is already downloaded at pipeline startup
        (``s2s_pipeline.py:74-80``), so no new dependency when used.

        Falls back to a single-element list on any failure (NLTK import
        error, missing punkt_tab data, empty result, etc.) so the caller
        always behaves like the proven single-call path -- one
        ``_generate_waveform`` call with the full text. This is what
        the v0.3.9 production-safe behavior was, and what the
        ``--chatterbox-split-sentences=False`` flag opts back into.
        """
        if not text or not text.strip():
            return [text] if text else []
        try:
            from nltk import sent_tokenize

            sentences = sent_tokenize(text.strip())
        except Exception:  # noqa: BLE001
            return [text]
        return sentences or [text]

    def _generate_waveform(self, text: str, language_code: Optional[str]) -> Any:
        """Call the variant-appropriate ``model.generate(...)`` and return a torch tensor.

        The parameter set differs by variant:
        - ``chatterbox``: exaggeration, cfg_weight, repetition_penalty, min_p, top_p
        - ``chatterbox-multilingual``: same + language_id (required)
        - ``chatterbox-turbo`` / ``-nano``: top_p, top_k, norm_loudness (exaggeration
          and cfg_weight are accepted but logged as ignored upstream).
        """
        is_turbo_family = self.model_variant in ("chatterbox-turbo", "chatterbox-nano")
        is_multilingual = self.model_variant == "chatterbox-multilingual"

        kwargs: dict[str, Any] = {
            "text": text,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if is_turbo_family:
            kwargs["top_k"] = self.top_k
            kwargs["norm_loudness"] = True
            # Turbo's single-step decoder doesn't use CFG / exaggeration; pass them
            # anyway so the upstream library's "ignored" warnings are visible in
            # the Status & Logs tab and the user can see why their knob didn't help.
            kwargs["exaggeration"] = self.exaggeration
            kwargs["cfg_weight"] = self.cfg_weight
        else:
            kwargs["exaggeration"] = self.exaggeration
            kwargs["cfg_weight"] = self.cfg_weight
            kwargs["repetition_penalty"] = self.repetition_penalty
            kwargs["min_p"] = self.min_p
        if is_multilingual:
            kwargs["language_id"] = self.language_id or language_code or "en"
        # Allow advanced users to inject extras via gen_kwargs (last write wins).
        kwargs.update(self.gen_kwargs)
        return self.model.generate(**kwargs)

    def cleanup(self) -> None:
        """Release the model and clear caches."""
        model = getattr(self, "model", None)
        if model is not None:
            del model
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
