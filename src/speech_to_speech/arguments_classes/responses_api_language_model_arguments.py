from dataclasses import dataclass, field
from typing import Optional

from speech_to_speech.arguments_classes.language_model_base_arguments import LanguageModelBaseArguments


@dataclass
class ResponsesApiLanguageModelHandlerArguments(LanguageModelBaseArguments):
    model_name: str = field(
        default="gpt-5.4-mini",
        metadata={"help": "The model to use with the OpenAI-compatible API. Default is 'gpt-5.4-mini'."},
    )
    responses_api_api_key: Optional[str] = field(
        default=None,
        metadata={"help": "API key used to authenticate access to the OpenAI-compatible API. Default is None."},
    )
    responses_api_base_url: Optional[str] = field(
        default=None,
        metadata={"help": "Base URL for the OpenAI-compatible API endpoint. Default is None (uses OpenAI)."},
    )
    responses_api_stream: bool = field(
        default=True,
        metadata={
            "help": "The stream parameter typically indicates whether data should be transmitted in a continuous flow rather"
            " than in a single, complete response, often used for handling large or real-time data.Default is True"
        },
    )
    responses_api_disable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Disable provider-side thinking/reasoning when supported by the OpenAI-compatible backend. "
            "For Together Qwen3.5 models this sends chat_template_kwargs.enable_thinking=false."
        },
    )
    responses_api_num_ctx: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional per-request context window (num_ctx) forwarded as extra_body={'options': {'num_ctx': N}}. "
            "Honoured by Ollama (and other providers that accept the Ollama-style options map); ignored by the "
            "official OpenAI server. Use to cap VRAM usage on small Ollama models when the model's default "
            "context (often 128k) is far larger than needed for short chat turns. Default is None (use provider default)."
        },
    )
    responses_api_warmup_timeout_s: float = field(
        default=180.0,
        metadata={
            "help": "Timeout in seconds for the pipeline's startup warmup request to the LLM server. "
            "Cold loads of large Ollama models (e.g. 20B+ at full context) can take 30-60s, so the "
            "default is intentionally larger than the per-request timeout (20s). Set higher if "
            "you're loading a very large model on a slow LAN. Default is 180.0."
        },
    )
    responses_api_warmup_system_prompt: Optional[str] = field(
        default=".",
        metadata={
            "help": "System message sent on the pipeline's startup warmup request. The warmup "
            "exists only to load the model into VRAM before the first real chat turn; its reply "
            "is discarded. Set this to a neutral single character (the default) so the warmup "
            "doesn't prime the model with a system persona that conflicts with the user-facing "
            "personality configured elsewhere. Use a longer string if your model requires a "
            "non-empty system message. Default is '.'."
        },
    )
    responses_api_warmup_user_prompt: Optional[str] = field(
        default=".",
        metadata={
            "help": "User message sent on the pipeline's startup warmup request. Same purpose "
            "as --responses-api-warmup-system-prompt: keep it neutral so the warmup doesn't "
            "shape the model's persona for subsequent real chat turns. Default is '.'."
        },
    )
