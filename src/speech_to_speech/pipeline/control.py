from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControlKind(str, Enum):
    """Strongly-typed kinds for :class:`PipelineControlMessage`."""

    SESSION_END = "session_end"
    UNLOAD_TTS = "unload_tts"


@dataclass(frozen=True)
class PipelineControlMessage:
    kind: ControlKind
    # Session that enqueued the message, when known. Lets the pooled realtime
    # send loop ignore a SESSION_END from a force-released session so it can't
    # satisfy the drain wait of the session that claimed the unit afterwards.
    session_id: str | None = None


SESSION_END = PipelineControlMessage(ControlKind.SESSION_END)
# UNLOAD_TTS is broadcast (not session-scoped): the user clicks "Unload TTS
# model" in the dashboard, and every pipeline unit drops its TTS model from
# RAM. The model reloads on the next TTS request.
UNLOAD_TTS = PipelineControlMessage(ControlKind.UNLOAD_TTS)


def is_control_message(message: object, kind: ControlKind | None = None) -> bool:
    return isinstance(message, PipelineControlMessage) and (kind is None or message.kind == kind)
