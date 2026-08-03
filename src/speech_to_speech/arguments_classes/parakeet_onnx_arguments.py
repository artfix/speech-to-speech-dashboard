from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ParakeetOnnxSTTHandlerArguments:
    """
    Arguments for the Parakeet ONNX Speech-to-Text handler.

    Wraps the `onnx-asr` library (istupakov/onnx-asr) for low-resource,
    pure ONNX inference of NVIDIA Parakeet TDT models. CPU/CUDA only —
    Apple Silicon users should use `--stt parakeet-tdt` instead.

    Three variants are exposed:

      * ``v2``   — Parakeet TDT 0.6B v2, English-only, smallest + fastest
      * ``v3``   — Parakeet TDT 0.6B v3, multilingual (25 EU languages, default)
      * ``v3sq`` — Parakeet TDT 0.6B v3, SmoothQuant int8 rebuild
                   (community quant by Olicorne). Best for long audio.

    See ``docs/PARAKEET_ONNX_PLAN.md`` for the full design rationale.
    """

    parakeet_onnx_variant: Literal["v2", "v3", "v3sq"] = field(
        default="v3",
        metadata={
            "help": "Which Parakeet ONNX model to load. "
            "'v2' = English-only Parakeet TDT 0.6B v2 (istupakov/parakeet-tdt-0.6b-v2-onnx). "
            "'v3' = multilingual Parakeet TDT 0.6B v3, 25 EU languages (istupakov/parakeet-tdt-0.6b-v3-onnx). "
            "'v3sq' = int8 SmoothQuant rebuild of v3, best for long audio "
            "(Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx). Default is 'v3'."
        },
    )
    parakeet_onnx_device: Literal["auto", "cuda", "cpu"] = field(
        default="auto",
        metadata={
            "help": "Which onnxruntime provider to request. "
            "'auto' = CUDAExecutionProvider if onnxruntime reports it available, else CPUExecutionProvider. "
            "'cuda' = request CUDA; falls back to CPU if load or warmup fails. "
            "'cpu' = CPU only. "
            "MPS is not supported (use --stt parakeet-tdt on Mac). Default is 'auto'."
        },
    )
    parakeet_onnx_num_threads: int = field(
        default=0,
        metadata={"help": "Force onnxruntime thread count. 0 = onnxruntime decides automatically. Default is 0."},
    )
    parakeet_onnx_language: Optional[
        Literal[
            "auto",
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
    ] = field(
        default="auto",
        metadata={
            "help": "Language hint passed to onnx-asr. 'auto' = model auto-detects. "
            "For v2 (English-only) this is locked to 'en' in the dashboard. "
            "Default is 'auto'."
        },
    )
