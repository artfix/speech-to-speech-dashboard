"""
Parakeet ONNX Speech-to-Text Handler

Wraps the `onnx-asr` library (istupakov/onnx-asr) for pure ONNX inference
of NVIDIA Parakeet TDT models. CPU/CUDA only — Apple Silicon users should
use `--stt parakeet-tdt` instead (MLX path).

Three variants:

  * ``v2``   — Parakeet TDT 0.6B v2, English-only
  * ``v3``   — Parakeet TDT 0.6B v3, multilingual (25 EU languages, default)
  * ``v3sq`` — Parakeet TDT 0.6B v3, SmoothQuant int8 rebuild

Final-only transcription in v1 — `onnx-asr` does not expose a streaming API
and the ONNX export is monolithic (no separate encoder/decoder/joiner
ONNX files), so we cannot implement live progressive updates like the
nano-parakeet / MLX path. The conversation still works: the LLM receives
the final transcript when the user stops speaking.

See ``docs/PARAKEET_ONNX_PLAN.md`` for rationale.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from threading import Lock
from time import perf_counter
from typing import Any, Iterator, Optional

import numpy as np
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

try:
    from lingua import Language, LanguageDetectorBuilder

    LINGUA_AVAILABLE = True
except ImportError:
    LINGUA_AVAILABLE = False

logger = logging.getLogger(__name__)
console = Console()

# Parakeet TDT v3 supports 25 European languages. Kept in sync with the
# existing ParakeetTDTSTTHandler.
SUPPORTED_LANGUAGES = [
    "en",
    "de",
    "fr",
    "es",
    "it",
    "pt",
    "nl",
    "pl",
    "ru",
    "uk",
    "cs",
    "sk",
    "hu",
    "ro",
    "bg",
    "hr",
    "sl",
    "sr",
    "da",
    "no",
    "sv",
    "fi",
    "et",
    "lv",
    "lt",
]

# Lingua uses "nb" (Bokmål) for Norwegian instead of "no"
_LINGUA_CODE_MAP = {"no": "nb"}

if LINGUA_AVAILABLE:
    _lingua_iso_to_code = {
        lang.iso_code_639_1.name.lower(): lang for lang in Language.all() if lang.iso_code_639_1 is not None
    }
    _lingua_languages = [
        _lingua_iso_to_code[_LINGUA_CODE_MAP.get(code, code)]
        for code in SUPPORTED_LANGUAGES
        if _LINGUA_CODE_MAP.get(code, code) in _lingua_iso_to_code
    ]

    def _build_lingua_detector():
        # Preloading can take multiple seconds on some hardware. Pay that
        # cost at startup instead of on the first user request.
        return LanguageDetectorBuilder.from_languages(*_lingua_languages).with_preloaded_language_models().build()

    _lingua_detector = _build_lingua_detector()


# Variant key → HuggingFace repo id. Mirrors the reachy_conv_app
# stt/parakeet.py pattern. We pass the full repo id rather than the
# short nemo-parakeet-tdt-0.6b-v2 alias to bypass onnx-asr's
# resolver._NEMO_REPOS, which has historically swapped v2 for v3 on disk.
_VARIANTS: dict[str, str] = {
    "v2": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "v3": "istupakov/parakeet-tdt-0.6b-v3-onnx",
    "v3sq": "Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx",
}

# Variant → language hint surfaced in the Transcription. onnx-asr does
# not return a language probability, so we report what the model actually
# supports: "en" for v2 (English-only), "multi" for v3 / v3sq.
_VARIANT_LANG: dict[str, str] = {
    "v2": "en",
    "v3": "multi",
    "v3sq": "multi",
}


class ParakeetOnnxSTTHandler(BaseSTTHandler):
    """
    Handles Speech-to-Text using NVIDIA Parakeet TDT models via onnx-asr.

    CPU/CUDA only. On Apple Silicon (MPS) the setup() method raises
    RuntimeError with a clear message pointing the user at `--stt parakeet-tdt`.

    Final-only transcription in v1 — the handler never yields
    PartialTranscription. The conversation still works: the LLM receives
    the final transcript when the user stops speaking.
    """

    def setup(
        self,
        variant: str = "v3",
        num_threads: int = 0,
        language: Optional[str] = "auto",
        device: str = "auto",
        gen_kwargs: dict[str, Any] = {},
    ) -> None:
        """
        Initialize the Parakeet ONNX model.

        Device resolution (mirrors ``parakeet_tdt_handler.setup``):

          * ``"auto"`` (default) — pick ``CUDAExecutionProvider`` if
            onnxruntime reports it available, else ``CPUExecutionProvider``.
            No try/load needed: onnxruntime's own availability check is
            cheaper and more accurate than a probe-and-fallback.
          * ``"cuda"`` (explicit) — request CUDA. If load fails (broken
            CUDA install, old GPU like a 1080Ti with missing libcudnn, ORT
            GPU wheel missing, …), log a warning and reload on CPU so the
            pipeline always starts. The conversation still works on CPU,
            just slower for the first utterance.
          * ``"cpu"`` (explicit) — CPU only.
          * ``"mps"`` — rejected with a clear pointer at ``--stt parakeet-tdt``
            (onnxruntime's Metal/CoreML providers are not production-stable).

        Args:
            variant: One of "v2", "v3", "v3sq". Default is "v3".
            num_threads: Force onnxruntime thread count. 0 = onnxruntime decides.
            language: Language hint for the model. "auto" = model auto-detects.
                      For v2, the field is ignored (v2 is English-only).
            device: Device string from the global --device flag. "auto" means
                    auto-detect; explicit values request that backend with a
                    CPU fallback for "cuda".
            gen_kwargs: Unused. Accepted for compatibility with the BaseSTTHandler
                        interface.
        """
        if variant not in _VARIANTS:
            raise ValueError(f"unknown parakeet-onnx variant: {variant!r} (want one of {list(_VARIANTS)})")

        # Reject MPS explicitly. onnxruntime's Metal/CoreML providers are
        # not stable enough for production; fail loud so the user falls
        # back to --stt parakeet-tdt (MLX) on Mac.
        if device == "mps":
            raise RuntimeError(
                "parakeet-onnx does not support MPS (Apple Silicon). "
                "Use --stt parakeet-tdt instead — it uses mlx-audio on Mac."
            )

        self.gen_kwargs = gen_kwargs
        self.variant = variant
        self.num_threads = num_threads
        self.language = language
        self.last_language = "en" if variant == "v2" else (language if language and language != "auto" else "en")
        self.compute_lock = Lock()
        self.sample_rate = 16000
        self._model: Any = None
        self._loaded_variant: str = ""

        # Decide which onnxruntime providers to request. _resolve_providers()
        # is pure (no I/O) and unit-testable in isolation.
        self._requested_device = device
        self.providers = self._resolve_providers(device)

        logger.info(
            "Loading Parakeet ONNX model: variant=%s repo=%s num_threads=%s language=%s device=%s providers=%s",
            variant,
            _VARIANTS[variant],
            num_threads,
            language,
            device,
            self.providers,
        )

        # Eager load + warmup. If device="cuda" was requested and load fails
        # (broken CUDA install, missing libcudnn, ORT wheel mismatch, etc.),
        # _ensure_loaded falls back to CPU and re-loads. If even that fails,
        # the original exception re-raises so the dashboard surfaces it.
        self._ensure_loaded()
        self.warmup()

    @staticmethod
    def _resolve_providers(device: str) -> list[str]:
        """
        Map a user-facing ``device`` string to an onnx-asr ``providers`` list.

        Pure function: no I/O, no onnxruntime import, safe to unit-test.

        Returns:
            For ``"auto"``: ``["CUDAExecutionProvider"]`` if onnxruntime
            advertises it, else ``["CPUExecutionProvider"]``. The check
            uses ``onnxruntime.get_available_providers()`` so it reflects
            what's actually loadable on this machine (no false positives
            from torch seeing a different CUDA version than onnxruntime).

            For ``"cuda"``: ``["CUDAExecutionProvider"]`` (explicit request,
            load-time fallback handled in ``_ensure_loaded``).

            For ``"cpu"``: ``["CPUExecutionProvider"]``.

        Raises:
            ValueError: for unsupported device strings (caller should have
                already caught MPS and translated it).
        """
        if device == "auto":
            try:
                import onnxruntime  # local import — the extra may not be installed

                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    return ["CUDAExecutionProvider"]
            except ImportError:
                # onnxruntime not installed → onnx-asr would also fail. Fall
                # through to CPU; the actual ImportError surfaces in
                # _ensure_loaded with the clearer install hint.
                pass
            return ["CPUExecutionProvider"]
        if device == "cuda":
            return ["CUDAExecutionProvider"]
        if device == "cpu":
            return ["CPUExecutionProvider"]
        raise ValueError(
            f"parakeet-onnx: unsupported device {device!r} (want 'auto', 'cuda', or 'cpu')"
        )

    def _ensure_loaded(self) -> Any:
        """Lazily (re)load the model. Used by setup() and on variant swap.

        If ``self.providers`` was set to CUDAExecutionProvider and the load
        fails (broken CUDA install, missing libcudnn, ORT GPU wheel missing,
        etc.), logs a warning and reloads with CPUExecutionProvider so the
        pipeline always starts. If CPU also fails, the original exception
        re-raises and the dashboard surfaces it.
        """
        if self._model is not None and self._loaded_variant == self.variant:
            return self._model
        try:
            import onnx_asr
        except ImportError as e:
            raise ImportError(
                "onnx-asr is required for --stt parakeet-onnx. "
                "Install with: uv pip install 'speech-to-speech[parakeet-onnx]'"
            ) from e

        model_id = _VARIANTS[self.variant]
        providers = self.providers

        try:
            logger.info("Loading onnx-asr model: %s with providers=%s", model_id, providers)
            self._model = onnx_asr.load_model(model_id, providers=providers)
        except Exception as e:
            if providers == ["CUDAExecutionProvider"]:
                logger.warning(
                    "parakeet-onnx: CUDA load failed (%s: %s). Falling back to CPU.",
                    type(e).__name__,
                    e,
                )
                providers = ["CPUExecutionProvider"]
                self.providers = providers
                self._model = onnx_asr.load_model(model_id, providers=providers)
            else:
                raise

        self._loaded_variant = self.variant
        logger.info(
            "parakeet-onnx ready (variant=%s providers=%s)",
            self.variant,
            self.providers,
        )
        return self._model

    @contextmanager
    def _compute_lock_context(self, handler_name: str, timeout: float) -> Iterator[bool]:
        """Thin wrapper around compute_lock — same shape as the existing
        parakeet_tdt_handler so the BaseSTTHandler interface is consistent."""
        lock_start_s = perf_counter()
        acquired = self.compute_lock.acquire(timeout=timeout)
        wait_s = perf_counter() - lock_start_s
        hold_start_s: float | None = None
        if acquired:
            if wait_s >= 0.25:
                logger.info("%s: compute lock acquired after %.2fs", handler_name, wait_s)
            else:
                logger.debug("%s: compute lock acquired after %.3fs", handler_name, wait_s)
            hold_start_s = perf_counter()
        else:
            logger.warning("%s: Failed to acquire compute lock after %.3fs (timeout=%s)", handler_name, wait_s, timeout)
        try:
            yield acquired
        finally:
            if acquired:
                assert hold_start_s is not None
                self.compute_lock.release()
                hold_s = perf_counter() - hold_start_s
                if hold_s >= 0.25:
                    logger.info("%s: compute lock released after holding %.2fs", handler_name, hold_s)
                else:
                    logger.debug("%s: compute lock released after holding %.3fs", handler_name, hold_s)

    def warmup(self) -> None:
        """Warm up the model with a 1-second silence, matching the existing
        parakeet_tdt_handler.warmup() pattern (parakeet_tdt_handler.py:213-234).

        If the warmup itself fails on CUDA (some CUDA installs let the model
        load but blow up on the first inference), fall back to CPU and reload.
        """
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_audio = np.zeros(16000, dtype=np.float32)
        try:
            _ = self._model.recognize(dummy_audio, sample_rate=self.sample_rate)
            logger.info("Model warmed up and ready")
        except Exception as e:
            if self.providers == ["CUDAExecutionProvider"]:
                logger.warning(
                    "parakeet-onnx: CUDA warmup failed (%s: %s). Reloading on CPU.",
                    type(e).__name__,
                    e,
                )
                self._model = None
                self._loaded_variant = ""
                self.providers = ["CPUExecutionProvider"]
                self._ensure_loaded()
                try:
                    _ = self._model.recognize(dummy_audio, sample_rate=self.sample_rate)
                    logger.info("Model warmed up on CPU after CUDA fallback")
                except Exception as e2:
                    logger.warning(f"CPU warmup also failed: {e2}")
            else:
                logger.warning(f"Warmup failed: {e}")

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        """
        Process audio and generate a final transcription.

        Final-only. The handler never yields ``PartialTranscription``. The
        BaseSTTHandler speculative-turn logic still works correctly because
        it gates on ``Transcription`` and ``PartialTranscription`` types.

        Yields:
            :class:`Transcription`
        """
        # Skip progressive chunks: onnx_asr is non-streaming, so every
        # progressive chunk would produce a fresh ``Transcription`` that
        # gets forwarded to the LLM — causing the LLM to respond while
        # the user is still speaking (the "robot cuts me off" bug).
        # The progressive chunks are cumulative prefixes of the final
        # audio, so waiting for the final yields the correct full-text
        # result in one shot. Mirrors parakeet-tdt's progressive/final
        # gating (which is why that handler does not have this bug).
        if vad_audio.mode == "progressive":
            audio_duration_s = len(vad_audio.audio) / self.sample_rate
            logger.debug(
                "Parakeet ONNX: skipping progressive chunk turn=%s rev=%s audio=%.3fs",
                vad_audio.turn_id,
                vad_audio.turn_revision,
                audio_duration_s,
            )
            return

        process_start_s = perf_counter()
        audio_input = vad_audio.audio

        # Ensure audio is float32 numpy array — matches parakeet_tdt_handler
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input, dtype=np.float32)
        else:
            audio_input = audio_input.astype(np.float32)
        audio_duration_s = len(audio_input) / getattr(self, "sample_rate", 16000)
        item_age_s = self._item_age_s(vad_audio)

        logger.info(
            "Parakeet ONNX STT start turn=%s rev=%s audio=%.3fs age=%.3fs variant=%s",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            audio_duration_s,
            item_age_s,
            self.variant,
        )

        inference_s = 0.0
        lock_scope_s = 0.0
        try:
            lock_scope_start_s = perf_counter()
            with self._compute_lock_context(handler_name="ParakeetOnnxSTT", timeout=5.0) as acquired:
                lock_scope_s = perf_counter() - lock_scope_start_s
                if not acquired:
                    logger.error("Failed to acquire compute lock for Parakeet ONNX inference")
                    pred_text = ""
                    language_code = self.last_language
                else:
                    inference_start_s = perf_counter()
                    # onnx-asr wants float32 in [-1, 1] at 16 kHz mono.
                    # Pipeline audio is already float32 in [-1, 1] at this point.
                    # If a caller passes int16 (e.g. CLI smoke test), convert here.
                    if audio_input.dtype == np.int16:
                        pcm = audio_input.astype(np.float32) / 32768.0
                    else:
                        pcm = audio_input.astype(np.float32)
                    pcm = np.ascontiguousarray(pcm)

                    text: str = self._model.recognize(pcm, sample_rate=self.sample_rate)
                    pred_text = (text or "").strip()
                    inference_s = perf_counter() - inference_start_s
                    lock_scope_s = perf_counter() - lock_scope_start_s

            # Language handling
            if self.variant == "v2":
                # v2 is English-only; lock to en regardless of the user's choice
                language_code = "en"
            else:
                if self.language and self.language != "auto":
                    language_code = self.language
                else:
                    detected = self._detect_language_from_text(pred_text)
                    if detected:
                        language_code = detected
                    else:
                        language_code = self.last_language

            if language_code and language_code in SUPPORTED_LANGUAGES:
                self.last_language = language_code
            else:
                language_code = self.last_language

        except Exception as e:
            logger.error(f"Parakeet ONNX inference failed: {e}")
            pred_text = ""
            language_code = self.last_language

        total_s = perf_counter() - process_start_s
        logger.info(
            "Parakeet ONNX STT done turn=%s rev=%s total=%.3fs lock_scope=%.3fs inference=%.3fs chars=%d",
            vad_audio.turn_id,
            vad_audio.turn_revision,
            total_s,
            lock_scope_s,
            inference_s,
            len(pred_text),
        )

        if pred_text.strip():
            console.print(f"[yellow]USER: {pred_text.strip()}")
            if language_code:
                console.print(f"[dim]Language: {language_code}[/dim]")

        yield Transcription(
            text=pred_text,
            language_code=language_code,
            turn_id=vad_audio.turn_id,
            turn_revision=vad_audio.turn_revision,
            speech_stopped_at_s=vad_audio.created_at_s,
        )

    @property
    def timing_log_level(self) -> int:
        return logging.INFO

    def _detect_language_from_text(self, text: str) -> Optional[str]:
        """Detect language from transcribed text using lingua-py.

        Mirrors parakeet_tdt_handler._detect_language_from_text.
        """
        if not LINGUA_AVAILABLE:
            logger.warning("lingua-py not available, cannot detect language from text")
            return None

        if not text or len(text.strip()) < 20:
            return None

        detected = _lingua_detector.detect_language_of(text)
        if detected is None:
            return None

        code = detected.iso_code_639_1.name.lower()
        return {v: k for k, v in _LINGUA_CODE_MAP.items()}.get(code, code)

    def _item_age_s(self, item: object) -> float:
        created_at_s = getattr(item, "created_at_s", None)
        if not isinstance(created_at_s, float):
            return 0.0
        return max(0.0, perf_counter() - created_at_s)

    def cleanup(self) -> None:
        """Clean up model resources."""
        logger.info(f"Cleaning up {self.__class__.__name__}")
        if hasattr(self, "_model") and self._model is not None:
            del self._model
            self._model = None
            self._loaded_variant = ""

    def on_session_end(self) -> None:
        super().on_session_end()
        if self.variant == "v2":
            self.last_language = "en"
        elif self.language and self.language != "auto":
            self.last_language = self.language
        logger.debug("Parakeet ONNX session state reset")
