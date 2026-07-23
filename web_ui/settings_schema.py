"""Introspect the speech-to-speech argument dataclasses and produce a JSON-friendly schema.

The pipeline exposes all of its configuration as dataclasses under
``src/speech_to_speech/arguments_classes/``. Every field carries a
``metadata={"help": "..."}`` string. This module walks those dataclasses once and
emits a plain-dict schema that the FastAPI server hands to the frontend, so the
dashboard can render every existing CLI flag (and any new ones added to those
dataclasses) without the dashboard itself ever hard-coding field names.

Public API
----------

- ``get_full_schema()`` returns a dict describing every group and every field
  the dashboard should render, with conditional visibility rules for the
  per-backend sub-forms (STT/LLM/TTS).
- ``get_defaults()`` returns a flat dict mapping CLI flag name (e.g.
  ``"--stt_device"``) to its default value, ready to seed an empty
  ``web_ui_settings.json``.
- ``build_argv(settings)`` turns a saved-settings dict into the ``sys.argv``
  list that the existing ``s2s_pipeline.py:main`` expects.

The dashboard never imports or modifies any of the argument classes; it only
reads their public shape (``dataclasses.fields``) and instantiates them via
``cls()`` to read defaults. That keeps the contract with the pipeline one-way
and survives future additions cleanly.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Union, get_args, get_origin

# Argument dataclasses from the pipeline. We import them by name so a missing
# optional extra (e.g. faster-whisper not installed) only fails if the user
# actually picks that backend, not at dashboard import time.
from speech_to_speech.arguments_classes.chat_completions_language_model_arguments import (
    ChatCompletionsLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.chat_tts_arguments import ChatTTSHandlerArguments
from speech_to_speech.arguments_classes.facebookmms_tts_arguments import FacebookMMSTTSHandlerArguments
from speech_to_speech.arguments_classes.faster_whisper_stt_arguments import (
    FasterWhisperSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.kokoro_tts_arguments import KokoroTTSHandlerArguments
from speech_to_speech.arguments_classes.language_model_arguments import (
    LanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.mlx_audio_whisper_arguments import (
    MLXAudioWhisperSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.module_arguments import ModuleArguments
from speech_to_speech.arguments_classes.paraformer_stt_arguments import (
    ParaformerSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.parakeet_tdt_arguments import (
    ParakeetTDTSTTHandlerArguments,
)
from speech_to_speech.arguments_classes.pocket_tts_arguments import PocketTTSHandlerArguments
from speech_to_speech.arguments_classes.qwen3_tts_arguments import Qwen3TTSHandlerArguments
from speech_to_speech.arguments_classes.responses_api_language_model_arguments import (
    ResponsesApiLanguageModelHandlerArguments,
)
from speech_to_speech.arguments_classes.socket_receiver_arguments import (
    SocketReceiverArguments,
)
from speech_to_speech.arguments_classes.socket_sender_arguments import (
    SocketSenderArguments,
)
from speech_to_speech.arguments_classes.vad_arguments import VADHandlerArguments
from speech_to_speech.arguments_classes.websocket_streamer_arguments import (
    WebSocketStreamerArguments,
)
from speech_to_speech.arguments_classes.whisper_stt_arguments import WhisperSTTHandlerArguments

# ----------------------------------------------------------------------
# Field-name <-> CLI-flag conversion
# ----------------------------------------------------------------------


def field_to_flag(name: str) -> str:
    """Convert a Python attribute name into the CLI flag the pipeline uses.

    ``qwen3_tts_model_name`` -> ``--qwen3-tts-model-name``. Identical mapping to
    what Hugging Face ``HfArgumentParser`` does, which is what the pipeline
    uses internally.
    """
    return "--" + name.replace("_", "-")


def flag_to_field(flag: str) -> str:
    """Inverse of :func:`field_to_flag`."""
    return flag.lstrip("-").replace("-", "_")


# ----------------------------------------------------------------------
# Type introspection
# ----------------------------------------------------------------------

# A few special field names that take a long string (system prompt, ref text).
# We render these as textareas in the UI rather than single-line inputs.
_TEXTAREA_FIELDS = {
    "init_chat_prompt",
    "qwen3_tts_ref_text",
    "qwen3_tts_instruct",
}


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    """Return (inner_type, is_optional) for a type that may be Optional[X]."""
    if get_origin(tp) is Union:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(tp):
            return args[0], True
    return tp, False


def _detect_type(tp: Any) -> str:
    """Map a typing annotation to one of our UI type names.

    UI types: ``string``, ``int``, ``float``, ``bool``, ``enum`` (Literal),
    ``optional_string`` (Optional[str] / Optional[Literal[...]]).
    """
    inner, is_optional = _unwrap_optional(tp)
    if get_origin(inner) is Literal:
        return "enum"
    if inner is bool:
        return "bool"
    if inner is int:
        return "int"
    if inner is float:
        return "float"
    if inner is str:
        return "string"
    if is_optional and inner is str:
        return "optional_string"
    if is_optional and get_origin(inner) is Literal:
        return "enum"
    # Fallback: best-effort string
    return "string"


def _field_choices(tp: Any) -> list[Any] | None:
    """For Literal / Optional[Literal] fields, return the allowed values."""
    inner, _ = _unwrap_optional(tp)
    if get_origin(inner) is Literal:
        return list(get_args(inner))
    return None


def _field_default(name: str, default: Any) -> Any:
    """Convert non-JSON-friendly defaults (like ``float('inf')``) into strings."""
    if default is None:
        return None
    if isinstance(default, float) and default == float("inf"):
        return "inf"
    if isinstance(default, bool):
        return default  # bool is a subclass of int; keep it as bool for clarity
    return default


# ----------------------------------------------------------------------
# Schema generation
# ----------------------------------------------------------------------


def _schema_for_class(
    cls: type,
    *,
    group: str,
    title: str,
    visible_when: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Produce a list of field specs for a dataclass, in declaration order."""
    instance = cls()
    out: list[dict[str, Any]] = []
    for f in dataclasses.fields(cls):
        tp = _detect_type(f.type)
        out.append(
            {
                "name": f.name,
                "flag": field_to_flag(f.name),
                "type": tp,
                "default": _field_default(f.name, getattr(instance, f.name)),
                "help": (f.metadata.get("help") or "").strip(),
                "choices": _field_choices(f.type),
                "ui": "textarea" if f.name in _TEXTAREA_FIELDS else _ui_for_type(tp),
                "group": group,
                "title": title,
            }
        )
    if visible_when is not None:
        out[0]["visible_when"] = visible_when
    return out


def _ui_for_type(tp: str) -> str:
    """Map a UI type to an HTML control kind."""
    return {
        "bool": "checkbox",
        "int": "number",
        "float": "number",
        "string": "text",
        "enum": "select",
        "optional_string": "text",
    }.get(tp, "text")


# Per-backend groups. Each tuple is (backend_value, dataclass, display_title).
# The key ``visible_when`` carries the rule that the frontend uses to know when
# to render this sub-form (e.g. only show qwen3-tts fields when ``--tts`` is
# ``qwen3``).
STT_BACKENDS: list[tuple[str, type, str, dict[str, Any]]] = [
    ("whisper", WhisperSTTHandlerArguments, "Whisper (Transformers)", {"field": "stt", "equals": "whisper"}),
    (
        "whisper-mlx",
        WhisperSTTHandlerArguments,
        "Whisper (Lightning MLX, Apple Silicon)",
        {"field": "stt", "equals": "whisper-mlx"},
    ),
    (
        "mlx-audio-whisper",
        MLXAudioWhisperSTTHandlerArguments,
        "Whisper (MLX Audio, Apple Silicon)",
        {"field": "stt", "equals": "mlx-audio-whisper"},
    ),
    (
        "faster-whisper",
        FasterWhisperSTTHandlerArguments,
        "Faster-Whisper (CTranslate2)",
        {"field": "stt", "equals": "faster-whisper"},
    ),
    (
        "parakeet-tdt",
        ParakeetTDTSTTHandlerArguments,
        "Parakeet TDT (default)",
        {"field": "stt", "equals": "parakeet-tdt"},
    ),
    (
        "paraformer",
        ParaformerSTTHandlerArguments,
        "Paraformer (FunASR, Chinese-oriented)",
        {"field": "stt", "equals": "paraformer"},
    ),
]

LLM_BACKENDS: list[tuple[str, type, str, dict[str, Any]]] = [
    (
        "responses-api",
        ResponsesApiLanguageModelHandlerArguments,
        "OpenAI-compatible Responses API (/v1/responses)",
        {"field": "llm_backend", "equals": "responses-api"},
    ),
    (
        "chat-completions",
        ChatCompletionsLanguageModelHandlerArguments,
        "OpenAI-compatible Chat Completions (/v1/chat/completions)",
        {"field": "llm_backend", "equals": "chat-completions"},
    ),
    (
        "transformers",
        LanguageModelHandlerArguments,
        "Local Transformers (CUDA/CPU)",
        {"field": "llm_backend", "equals": "transformers"},
    ),
    (
        "mlx-lm",
        LanguageModelHandlerArguments,
        "Local MLX-LM (Apple Silicon)",
        {"field": "llm_backend", "equals": "mlx-lm"},
    ),
]

TTS_BACKENDS: list[tuple[str, type, str, dict[str, Any]]] = [
    (
        "qwen3",
        Qwen3TTSHandlerArguments,
        "Qwen3-TTS (default, multilingual)",
        {"field": "tts", "equals": "qwen3"},
    ),
    (
        "kokoro",
        KokoroTTSHandlerArguments,
        "Kokoro-82M (English-focused, lightweight)",
        {"field": "tts", "equals": "kokoro"},
    ),
    (
        "pocket",
        PocketTTSHandlerArguments,
        "Pocket TTS (CPU, voice cloning)",
        {"field": "tts", "equals": "pocket"},
    ),
    (
        "chatTTS",
        ChatTTSHandlerArguments,
        "ChatTTS (English/Chinese, expressive)",
        {"field": "tts", "equals": "chatTTS"},
    ),
    (
        "facebookMMS",
        FacebookMMSTTSHandlerArguments,
        "Facebook MMS TTS (multilingual)",
        {"field": "tts", "equals": "facebookMMS"},
    ),
]


def get_full_schema() -> dict[str, Any]:
    """Return the full dashboard schema, organized by tab / group."""
    groups: list[dict[str, Any]] = []

    # Mode + VAD + IO. These are always visible.
    groups.append(
        {
            "id": "mode",
            "title": "Mode",
            "description": "Pick how the pipeline receives audio and delivers responses.",
            "fields": _schema_for_class(ModuleArguments, group="mode", title="Mode"),
        }
    )
    groups.append(
        {
            "id": "vad",
            "title": "VAD (Voice Activity Detection)",
            "description": "Silero VAD v5 detects when the user is speaking. Tunable below.",
            "fields": _schema_for_class(VADHandlerArguments, group="vad", title="VAD"),
        }
    )
    groups.append(
        {
            "id": "websocket",
            "title": "WebSocket / Realtime server",
            "description": "Used by `--mode websocket` and `--mode realtime`.",
            "fields": _schema_for_class(
                WebSocketStreamerArguments, group="websocket", title="WebSocket"
            ),
        }
    )
    groups.append(
        {
            "id": "socket",
            "title": "TCP Socket (--mode socket)",
            "description": "Used by `--mode socket`. Raw PCM in, raw PCM out.",
            "fields": [
                *_schema_for_class(SocketReceiverArguments, group="socket", title="Socket receiver"),
                *_schema_for_class(SocketSenderArguments, group="socket", title="Socket sender"),
            ],
        }
    )

    # STT: a parent group with the backend dropdown (from ModuleArguments --stt)
    # plus a list of sub-groups, one per backend, that the UI shows/hides.
    stt_parent_fields = [
        f
        for f in _schema_for_class(ModuleArguments, group="stt", title="STT")
        if f["name"] == "stt"
    ]
    stt_subgroups = []
    for value, cls, title, visible in STT_BACKENDS:
        stt_subgroups.append(
            {
                "id": f"stt_{value}",
                "title": title,
                "visible_when": visible,
                "fields": _schema_for_class(cls, group="stt", title=title),
            }
        )
    groups.append(
        {
            "id": "stt",
            "title": "STT (Speech to Text)",
            "description": "Pick the speech recognition backend. Each backend has its own settings below.",
            "fields": stt_parent_fields,
            "subgroups": stt_subgroups,
        }
    )

    # LLM: same pattern.
    llm_parent_fields = [
        f
        for f in _schema_for_class(ModuleArguments, group="llm", title="LLM")
        if f["name"] == "llm_backend"
    ]
    llm_subgroups = []
    for value, cls, title, visible in LLM_BACKENDS:
        llm_subgroups.append(
            {
                "id": f"llm_{value}",
                "title": title,
                "visible_when": visible,
                "fields": _schema_for_class(cls, group="llm", title=title),
            }
        )
    groups.append(
        {
            "id": "llm",
            "title": "LLM (Language Model)",
            "description": "Pick the LLM backend. Either a remote OpenAI-compatible API or a local model.",
            "fields": llm_parent_fields,
            "subgroups": llm_subgroups,
        }
    )

    # TTS: same pattern.
    tts_parent_fields = [
        f
        for f in _schema_for_class(ModuleArguments, group="tts", title="TTS")
        if f["name"] == "tts"
    ]
    tts_subgroups = []
    for value, cls, title, visible in TTS_BACKENDS:
        tts_subgroups.append(
            {
                "id": f"tts_{value}",
                "title": title,
                "visible_when": visible,
                "fields": _schema_for_class(cls, group="tts", title=title),
            }
        )
    groups.append(
        {
            "id": "tts",
            "title": "TTS (Text to Speech)",
            "description": "Pick the speech synthesis backend. Each backend has its own settings below.",
            "fields": tts_parent_fields,
            "subgroups": tts_subgroups,
        }
    )

    # The remaining ModuleArguments fields (log_level, live_transcription, ...)
    # go on an Advanced tab. We don't duplicate the ones we already showed.
    shown_in_mode = {"stt", "llm_backend", "tts", "mode", "device", "local_mac_optimal_settings"}
    advanced_fields = [
        f
        for f in _schema_for_class(ModuleArguments, group="advanced", title="Advanced")
        if f["name"] not in shown_in_mode
    ]
    groups.append(
        {
            "id": "advanced",
            "title": "Advanced",
            "description": "Log level, live transcription, and other pipeline-wide options.",
            "fields": advanced_fields,
        }
    )

    return {"groups": groups, "backend_meta": _backend_meta()}


def _backend_meta() -> dict[str, Any]:
    """Lightweight metadata about backends for the Guide / overview."""
    return {
        "stt": [
            {"value": v, "title": title}
            for v, _, title, _ in STT_BACKENDS
        ],
        "llm": [{"value": v, "title": title} for v, _, title, _ in LLM_BACKENDS],
        "tts": [{"value": v, "title": title} for v, _, title, _ in TTS_BACKENDS],
    }


def get_defaults() -> dict[str, Any]:
    """Flat ``{flag: default_value}`` map covering every dashboard field.

    Includes a ``theme`` entry for the UI itself and an ``env`` list for
    additional environment variables. These two are not introspected from
    argument classes -- they live in the settings file.
    """
    out: dict[str, Any] = {"theme": "cyberpunk-neon", "env": []}
    for group in get_full_schema()["groups"]:
        for f in group["fields"]:
            if f["default"] is not None or f["type"] in ("string", "optional_string", "enum"):
                out[f["flag"]] = f["default"]
        for sub in group.get("subgroups", []):
            for f in sub["fields"]:
                if f["default"] is not None or f["type"] in ("string", "optional_string", "enum"):
                    out[f["flag"]] = f["default"]
    return out


# ----------------------------------------------------------------------
# Settings -> CLI argv
# --------------------------------------------------------------------


# Fields the user should never forward to the pipeline as CLI args.
# ``theme`` controls the UI; ``env`` is passed via the subprocess environment.
_NON_FORWARDED = {"theme", "env"}


def build_argv(settings: dict[str, Any]) -> list[str]:
    """Turn a settings dict (from ``web_ui_settings.json``) into ``sys.argv``.

    Booleans that are True become ``--flag``. False / None are dropped (the
    dataclass default is the right behavior for "off"). Strings, ints, and
    floats are emitted as ``--flag value``. Empty strings are dropped (the
    pipeline treats ``""`` and "not set" the same in most places; some
    fields like ``--responses_api_api_key`` do accept ``""`` and we preserve
    that by checking ``"value" in settings``).
    """
    argv: list[str] = ["-m", "speech_to_speech.s2s_pipeline"]
    for flag, value in settings.items():
        if flag in _NON_FORWARDED:
            continue
        if not flag.startswith("--"):
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            continue
        if isinstance(value, str) and value == "" and flag == "--responses-api-api-key":
            # Empty string IS a meaningful value here (Ollama ignores the key).
            argv.extend([flag, value])
            continue
        if isinstance(value, str) and value == "":
            # Otherwise treat "" as "not set" and skip.
            continue
        argv.extend([flag, str(value)])
    return argv


def settings_from_argv(flags: list[str]) -> dict[str, str]:
    """Inverse helper for the test smoke checks: turn argv back into a flag map.

    Only used by tests; the production path always round-trips through the
    dashboard form.
    """
    out: dict[str, str] = {}
    i = 0
    while i < len(flags):
        tok = flags[i]
        if tok.startswith("--") and i + 1 < len(flags) and not flags[i + 1].startswith("--"):
            out[tok] = flags[i + 1]
            i += 2
        else:
            out[tok] = "true"
            i += 1
    return out
