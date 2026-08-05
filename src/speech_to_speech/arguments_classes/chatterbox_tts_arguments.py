from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ChatterboxTTSHandlerArguments:
    chatterbox_model_variant: Literal[
        "chatterbox", "chatterbox-turbo", "chatterbox-nano", "chatterbox-multilingual"
    ] = field(
        default="chatterbox-turbo",
        metadata={
            "help": (
                "Chatterbox model variant to load. One of: 'chatterbox' (English 500M, original, "
                "expressive via exaggeration/cfg_weight), 'chatterbox-turbo' (350M, single-step "
                "decoder, sub-second latency, English), 'chatterbox-nano' (110M, CPU-friendly, "
                "English), 'chatterbox-multilingual' (500M, 23 languages). Default is "
                "'chatterbox-turbo'."
            )
        },
    )
    chatterbox_device: str = field(
        default="auto",
        metadata={
            "help": (
                "Device to run Chatterbox on. 'auto' picks CUDA if available, else MPS, else CPU. "
                "Or set explicitly: 'cuda', 'mps', 'cpu'. Default is 'auto'."
            )
        },
    )
    chatterbox_voice: str = field(
        default="",
        metadata={
            "help": (
                "Name of a cloned voice saved in the dashboard's voice library (lives at "
                "<chatterbox_voices_dir>/<name>.pt). Leave empty to use the model's bundled default "
                "voice. The dashboard manages this field via the 'Set active' button on each voice "
                "card. Default is empty."
            )
        },
    )
    chatterbox_exaggeration: float = field(
        default=0.5,
        metadata={
            "help": (
                "Expressive intensity (0.0 = neutral, 0.7+ = dramatic). Only affects "
                "'chatterbox' and 'chatterbox-multilingual' variants; ignored on "
                "chatterbox-turbo / chatterbox-nano. Default is 0.5."
            )
        },
    )
    chatterbox_cfg_weight: float = field(
        default=0.5,
        metadata={
            "help": (
                "Classifier-free guidance weight. Lower = faster, less accent transfer. Only "
                "affects 'chatterbox' and 'chatterbox-multilingual' variants. Default is 0.5."
            )
        },
    )
    chatterbox_temperature: float = field(
        default=0.8,
        metadata={"help": "Sampling temperature for token generation. Default is 0.8."},
    )
    chatterbox_repetition_penalty: float = field(
        default=1.2,
        metadata={
            "help": (
                "Repetition penalty. Only affects 'chatterbox' and 'chatterbox-multilingual' variants. Default is 1.2."
            )
        },
    )
    chatterbox_min_p: float = field(
        default=0.05,
        metadata={
            "help": (
                "min_p sampling threshold. Only affects 'chatterbox' and 'chatterbox-multilingual' "
                "variants. Default is 0.05."
            )
        },
    )
    chatterbox_top_p: float = field(
        default=1.0,
        metadata={"help": "top_p sampling threshold. Default is 1.0."},
    )
    chatterbox_top_k: int = field(
        default=1000,
        metadata={
            "help": (
                "top_k sampling threshold. Only affects 'chatterbox-turbo' and 'chatterbox-nano' "
                "variants. Default is 1000."
            )
        },
    )
    chatterbox_language_id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Language code for the multilingual variant (e.g. 'en', 'fr', 'zh', 'de', 'es', "
                "'ja', 'ko', ...). Required when chatterbox_model_variant='chatterbox-multilingual'. "
                "Ignored for the English/Turbo/Nano variants. Default is unset."
            )
        },
    )
    chatterbox_voices_dir: str = field(
        default="voices",
        metadata={
            "help": (
                "Directory where the dashboard stores cloned voice Conditionals (.pt files), "
                "relative to the repo root. Default is 'voices'."
            )
        },
    )
    chatterbox_sample_rate: int = field(
        default=16000,
        metadata={
            "help": (
                "Output sample rate in Hz. Chatterbox generates at 24kHz internally and is "
                "resampled to this rate to match the pipeline's audio streamer. Default is 16000."
            )
        },
    )
    chatterbox_blocksize: int = field(
        default=512,
        metadata={"help": "Size of audio blocks to yield for streaming. Default is 512."},
    )
    chatterbox_split_sentences: bool = field(
        default=True,
        metadata={
            "help": (
                "When True (default), each TTSInput is split into sentences (NLTK punkt) and "
                "synthesized one at a time so the speaker starts playing sentence 1 while "
                "sentence 2 is still generating on the GPU. Cuts TTFA by ~1-2s on a typical "
                "3-sentence reply. The per-sentence cost is ~0.5-1s of extra overhead, so on a "
                "single-sentence reply the speedup is zero and the behavior is effectively "
                "identical to a single generate() call. Falls back to a single synthesize() if "
                "NLTK or punkt_tab is unavailable, or if the input is a single sentence. "
                "Set to False to force the pre-v0.3.8 single-call path."
            )
        },
    )
