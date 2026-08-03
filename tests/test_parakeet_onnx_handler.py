"""Tests for the Parakeet ONNX STT handler.

Skip-by-default: every test is gated on `onnx_asr` being importable. If
the extras group `[parakeet-onnx]` isn't installed, the entire module is
skipped. This keeps the existing test suite green for users who don't
use the ONNX backend.

Tests do NOT download any model. They mock `onnx_asr.load_model` and
the model's `recognize` method so the suite is fast and offline.

The handler must auto-detect GPU/CPU and fall back to CPU on CUDA
failure — see Test 12 for the broken-CUDA-installer scenario (e.g.
the user's 1080Ti box, or any machine with a CUDA version that
onnxruntime's GPU wheel doesn't match).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("onnx_asr")

from speech_to_speech.pipeline.messages import Transcription, VADAudio  # noqa: E402
from speech_to_speech.STT.parakeet_onnx_handler import (  # noqa: E402
    _VARIANTS,
    ParakeetOnnxSTTHandler,
)


def _make_handler(**kwargs):
    """Construct a handler without going through the BaseSTTHandler ctor."""
    handler = object.__new__(ParakeetOnnxSTTHandler)
    handler.compute_lock = __import__("threading").Lock()
    handler.sample_rate = 16000
    handler.gen_kwargs = {}
    handler._model = None
    handler._loaded_variant = ""
    handler.last_language = "en"
    handler.super = lambda *a, **kw: None
    return handler


def test_setup_rejects_unknown_variant():
    """Test 1: variant validation — bogus variant raises ValueError."""
    handler = _make_handler()
    with pytest.raises(ValueError, match="unknown parakeet-onnx variant"):
        handler.setup(variant="bogus", device="cpu")


def test_setup_rejects_mps():
    """Test 2: MPS rejection — handler raises RuntimeError with a clear message."""
    handler = _make_handler()
    with pytest.raises(RuntimeError, match="parakeet-onnx does not support MPS"):
        handler.setup(variant="v3", device="mps")


def test_setup_loads_model_eagerly():
    """Test 3: setup() calls onnx_asr.load_model() once with the right providers."""
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = ""

    with patch("onnx_asr.load_model", return_value=fake_model) as mocked_load:
        handler.setup(variant="v3", device="cpu")

    # The right HF repo id is passed, plus providers=[CPUExecutionProvider]
    mocked_load.assert_called_once_with(_VARIANTS["v3"], providers=["CPUExecutionProvider"])
    # The loaded model is cached on the handler
    assert handler._model is fake_model
    assert handler._loaded_variant == "v3"
    assert handler.providers == ["CPUExecutionProvider"]


def test_setup_runs_warmup():
    """Test 4: warmup feeds 1s of silence through model.recognize()."""
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = ""

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v3", device="cpu")

    # Warmup ran — recognize was called at least once with a 1s float32 array
    assert fake_model.recognize.call_count >= 1
    first_call = fake_model.recognize.call_args_list[0]
    audio_arg = first_call[0][0]
    assert audio_arg.shape == (16000,)
    assert audio_arg.dtype == np.float32
    assert first_call[1]["sample_rate"] == 16000


def test_setup_v2_locks_language_to_en():
    """Test 5: v2 is English-only — last_language is locked to 'en' regardless of user input."""
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = ""

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v2", device="cpu", language="fr")

    assert handler.last_language == "en"


def test_resolve_providers_explicit_cpu():
    """Test 6: explicit device='cpu' → CPUExecutionProvider only, no onnxruntime import."""
    providers = ParakeetOnnxSTTHandler._resolve_providers("cpu")
    assert providers == ["CPUExecutionProvider"]


def test_resolve_providers_explicit_cuda():
    """Test 7: explicit device='cuda' → CUDAExecutionProvider (load-time fallback in _ensure_loaded)."""
    providers = ParakeetOnnxSTTHandler._resolve_providers("cuda")
    assert providers == ["CUDAExecutionProvider"]


def test_resolve_providers_auto_with_cuda_available():
    """Test 8: device='auto' → CUDA when onnxruntime reports it available."""
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider", "CUDAExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
        providers = ParakeetOnnxSTTHandler._resolve_providers("auto")
    assert providers == ["CUDAExecutionProvider"]


def test_resolve_providers_auto_without_cuda():
    """Test 9: device='auto' → CPU when onnxruntime has no CUDA provider."""
    fake_ort = MagicMock()
    fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
    with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
        providers = ParakeetOnnxSTTHandler._resolve_providers("auto")
    assert providers == ["CPUExecutionProvider"]


def test_resolve_providers_auto_without_onnxruntime_installed():
    """Test 10: device='auto' → CPU if onnxruntime isn't importable. The real
    ImportError will surface later in _ensure_loaded with a clearer install hint."""
    # Remove onnxruntime from sys.modules if present, then make import fail.
    import sys

    saved = sys.modules.pop("onnxruntime", None)
    import builtins

    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "onnxruntime" or name.startswith("onnxruntime."):
            raise ImportError("onnxruntime not installed (test)")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = _blocked
        providers = ParakeetOnnxSTTHandler._resolve_providers("auto")
    finally:
        builtins.__import__ = real_import
        if saved is not None:
            sys.modules["onnxruntime"] = saved
    assert providers == ["CPUExecutionProvider"]


def test_resolve_providers_rejects_unknown():
    """Test 11: unsupported device string raises ValueError."""
    with pytest.raises(ValueError, match="unsupported device"):
        ParakeetOnnxSTTHandler._resolve_providers("tpu")


def test_setup_cuda_load_falls_back_to_cpu():
    """Test 12: device='cuda' → load fails → reload on CPU with a warning.

    Mirrors the user's box: 1080Ti with broken CUDA/cuDNN, --stt
    parakeet-onnx --device cuda should still start the pipeline on CPU.
    """
    handler = _make_handler()
    fake_cpu_model = MagicMock()
    fake_cpu_model.recognize.return_value = ""

    with patch("onnx_asr.load_model") as mocked_load:
        # First call (CUDA) raises, second call (CPU) succeeds.
        mocked_load.side_effect = [RuntimeError("CUDA provider not available"), fake_cpu_model]
        handler.setup(variant="v3", device="cuda")

    # The handler ended up on CPU
    assert mocked_load.call_count == 2
    assert mocked_load.call_args_list[0].kwargs["providers"] == ["CUDAExecutionProvider"]
    assert mocked_load.call_args_list[1].kwargs["providers"] == ["CPUExecutionProvider"]
    assert handler.providers == ["CPUExecutionProvider"]
    assert handler._model is fake_cpu_model


def test_setup_cpu_load_failure_reraises():
    """Test 13: device='cpu' load failure → no fallback, original exception re-raises.

    We want a loud failure for genuinely broken installs (a quiet swallow
    would leave the user confused why the model never loads)."""
    handler = _make_handler()
    with patch("onnx_asr.load_model", side_effect=RuntimeError("ORT wheel missing")):
        with pytest.raises(RuntimeError, match="ORT wheel missing"):
            handler.setup(variant="v3", device="cpu")


def test_process_returns_final_transcription():
    """Test 14: process() yields a Transcription (never PartialTranscription)."""
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "  hello world  "

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v3", device="cpu")

    # 1 second of float32 audio at 16 kHz
    audio = np.zeros(16000, dtype=np.float32)
    vad_audio = VADAudio(
        audio=audio,
        mode="final",
        turn_id="t1",
        turn_revision=1,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))
    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "hello world"
    assert outputs[0].turn_id == "t1"
    assert outputs[0].turn_revision == 1


def test_process_passes_float32_audio_to_recognize():
    """Test 15: audio is passed to recognize() as a float32 numpy array.

    Mirrors the existing parakeet_tdt_handler.process() pattern: ensure the
    audio is a float32 ndarray before calling recognize(). The pipeline's VAD
    always emits float32 audio in [-1, 1], so this is the realistic check.
    """
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "ok"

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v3", device="cpu")

    # 1 second of float32 audio in [-1, 1] — what the VAD emits
    audio_f32 = np.linspace(-1.0, 1.0, 16000, dtype=np.float32)
    vad_audio = VADAudio(
        audio=audio_f32,
        mode="final",
        turn_id="t1",
        turn_revision=1,
        created_at_s=0.0,
    )

    list(handler.process(vad_audio))

    # Inspect the call to recognize — audio arg should be a float32 ndarray
    recognize_calls = fake_model.recognize.call_args_list
    # First call is warmup (zeros), second is the actual transcribe
    transcribe_call = recognize_calls[1]
    audio_arg = transcribe_call[0][0]
    assert isinstance(audio_arg, np.ndarray)
    assert audio_arg.dtype == np.float32
    assert audio_arg.shape == (16000,)
    assert audio_arg.min() >= -1.0
    assert audio_arg.max() <= 1.0


def test_process_handles_v2_english_only():
    """Test 16: v2 always emits language_code='en' regardless of text."""
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "hello world"

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v2", device="cpu", language="auto")

    audio = np.zeros(16000, dtype=np.float32)
    vad_audio = VADAudio(
        audio=audio,
        mode="final",
        turn_id="t1",
        turn_revision=1,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))
    assert outputs[0].language_code == "en"


def test_cleanup_drops_model():
    """Test 17: cleanup() drops the in-memory model."""
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = ""

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v3", device="cpu")

    assert handler._model is not None
    handler.cleanup()
    assert handler._model is None
    assert handler._loaded_variant == ""


def test_process_skips_progressive_chunks():
    """Test 18: process() yields nothing for progressive chunks.

    onnx_asr is non-streaming, so transcribing every progressive audio chunk
    would yield a fresh ``Transcription`` that gets forwarded to the LLM
    while the user is still speaking — causing the LLM to respond and the
    TTS to start, cutting the user off. The progressive chunks are
    cumulative prefixes of the final audio, so skipping them is correct
    AND faster (saves ~14 wasted recognize() calls per 7s utterance).
    """
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "should not be called"

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v3", device="cpu")

    # setup() warms up with one recognize() call — reset the mock so we can
    # assert that process() makes no additional calls for progressive chunks.
    fake_model.recognize.reset_mock()

    audio = np.zeros(16000, dtype=np.float32)
    vad_audio = VADAudio(
        audio=audio,
        mode="progressive",
        turn_id="t1",
        turn_revision=1,
        created_at_s=0.0,
    )

    outputs = list(handler.process(vad_audio))
    assert outputs == []
    # The model's recognize() must NOT have been called for progressive chunks.
    fake_model.recognize.assert_not_called()


def test_process_progressive_then_final_transcribes_only_final():
    """Test 19: progressive chunks yield nothing; the final chunk yields one Transcription.

    Mirrors the live VAD behavior: VAD yields multiple progressive chunks
    (cumulative prefixes) during speech, then one final chunk after
    speech-end. Only the final one should produce a Transcription.
    """
    handler = _make_handler()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "complete sentence"

    with patch("onnx_asr.load_model", return_value=fake_model):
        handler.setup(variant="v3", device="cpu")

    # setup() warms up with one recognize() call — reset so we measure only
    # the process() calls below.
    fake_model.recognize.reset_mock()

    audio = np.zeros(16000, dtype=np.float32)

    # 3 progressive chunks (cumulative prefixes) — must all be skipped
    for i in range(3):
        progressive = VADAudio(
            audio=audio,
            mode="progressive",
            turn_id="t1",
            turn_revision=1,
            created_at_s=0.0,
        )
        outputs = list(handler.process(progressive))
        assert outputs == [], f"progressive chunk #{i} yielded {outputs!r}"

    # Final chunk — must yield exactly one Transcription
    final = VADAudio(
        audio=audio,
        mode="final",
        turn_id="t1",
        turn_revision=1,
        created_at_s=0.0,
    )
    outputs = list(handler.process(final))
    assert len(outputs) == 1
    assert isinstance(outputs[0], Transcription)
    assert outputs[0].text == "complete sentence"

    # recognize() called exactly once — only for the final chunk
    assert fake_model.recognize.call_count == 1
