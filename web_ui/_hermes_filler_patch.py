"""Runtime patch that injects filler audio while Hermes is thinking.

Loaded inside the speech-to-speech pipeline process via the dashboard's
``sitecustomize.py`` snippet (see :mod:`web_ui.process_manager`). The patch is
a strict no-op unless ``DASHBOARD_HERMES_FILLER_ENABLED=1`` is set before the
interpreter starts.

What it does
============

The patch monkey-patches ``s2s_pipeline._build_realtime_pipeline_unit()`` so
that every realtime pipeline unit gets a small watchdog thread. The watchdog:

1. Arms a timer when a response starts (``response.create`` / implicit VAD
   path). The timer delay comes from ``DASHBOARD_HERMES_FILLER_DELAY_MS``.
2. Cancels the timer as soon as the LLM emits its first text chunk that
   actually reaches the TTS stage.
3. If the timer fires first, it picks a random phrase from
   ``DASHBOARD_HERMES_FILLER_PHRASES`` and injects a :class:`TTSInput` into
   the pipeline's ``lm_processed_queue``. The existing TTS handler speaks it
   in the user's configured voice.

Why this is safe
================

- Filler is injected **after** the LLM stage, so Hermes / the upstream LLM
  never sees the filler text in the conversation history.
- It uses only existing pipeline message types and queues; no source edits.
- It only activates in realtime mode, and only when the dashboard sets the
  env var.
- The injected audio carries the current ``cancel_generation``, so user
  interruption drops it through the existing CancelScope machinery.
- The patch is removed from the venv when the pipeline stops.
"""

from __future__ import annotations

import logging
import os
import random
import threading
from typing import Any

logger = logging.getLogger("_dash_hermes_filler_patch")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    return os.environ.get("DASHBOARD_HERMES_FILLER_ENABLED") == "1"


def _read_phrases() -> list[str]:
    raw = os.environ.get("DASHBOARD_HERMES_FILLER_PHRASES", "")
    return [p.strip() for p in raw.split("|") if p.strip()]


def _read_delay_s() -> float:
    raw = os.environ.get("DASHBOARD_HERMES_FILLER_DELAY_MS", "1500")
    try:
        ms = int(raw)
    except ValueError:
        logger.warning("Invalid DASHBOARD_HERMES_FILLER_DELAY_MS=%r; using 1500", raw)
        ms = 1500
    return max(0, ms) / 1000.0


# ---------------------------------------------------------------------------
# Watchdog
# ---------------------------------------------------------------------------


class FillerWatchdog:
    """One watchdog per realtime pipeline unit."""

    def __init__(self, unit: Any) -> None:
        self._unit = unit
        self._enabled = _is_enabled()
        self._delay_s = _read_delay_s()
        self._phrases = _read_phrases()
        if not self._phrases:
            self._phrases = ["one moment"]

        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._injected = False
        self._current_ctx: dict[str, Any] | None = None

    # -- public hooks -------------------------------------------------------

    def response_started(self, conn_id: str) -> None:
        """Call when a response is created for the unit's connection."""
        if not self._enabled:
            return
        with self._lock:
            self._cancel_timer()
            self._injected = False
            self._current_ctx = self._build_ctx(conn_id)
            if self._delay_s <= 0 or not self._phrases:
                return
            self._timer = threading.Timer(self._delay_s, self._fire)
            self._timer.start()

    def response_ended(self, conn_id: str) -> None:
        """Call when the current response closes (completed/cancelled/failed)."""
        with self._lock:
            self._cancel_timer()
            self._injected = False
            self._current_ctx = None

    def first_text_seen(self) -> None:
        """Call when the LM output processor emits real text for TTS."""
        with self._lock:
            self._cancel_timer()

    # -- internals ----------------------------------------------------------

    def _build_ctx(self, conn_id: str) -> dict[str, Any] | None:
        try:
            st = self._unit.service._state(conn_id)
        except Exception:
            logger.debug("No connection state for %s", conn_id)
            return None
        return {
            "conn_id": conn_id,
            "turn_id": st.speculative_user_turn_id,
            "turn_revision": st.speculative_user_turn_revision,
            "speech_stopped_at_s": st.speculative_user_speech_stopped_at_s,
            "runtime_config": st.runtime_config,
            "response": st.current_response_params,
            "language_code": None,
            "cancel_generation": self._unit.cancel_scope.generation,
        }

    def _cancel_timer(self) -> None:
        t = self._timer
        self._timer = None
        if t is not None:
            t.cancel()

    def _fire(self) -> None:
        with self._lock:
            if self._injected:
                return
            self._injected = True
            ctx = self._current_ctx
            if ctx is None or not self._phrases:
                return

        try:
            text = random.choice(self._phrases)
            from speech_to_speech.pipeline.messages import TTSInput

            msg = TTSInput(
                text=text,
                language_code=ctx.get("language_code"),
                runtime_config=ctx["runtime_config"],
                response=ctx["response"],
                turn_id=ctx["turn_id"],
                turn_revision=ctx["turn_revision"],
                speech_stopped_at_s=ctx["speech_stopped_at_s"],
                cancel_generation=ctx["cancel_generation"],
            )
            self._unit.lm_processed_queue.put(msg)
            logger.info("Injected hermes filler audio: %r", text)
        except Exception:
            logger.exception("Failed to inject hermes filler audio")


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


def _find_lm_processor(handlers: list[Any]) -> Any | None:
    from speech_to_speech.LLM.lm_output_processor import LMOutputProcessor

    for h in handlers:
        if isinstance(h, LMOutputProcessor):
            return h
    return None


def _is_text_chunk(item: Any) -> bool:
    from speech_to_speech.pipeline.messages import LLMResponseChunk

    return isinstance(item, LLMResponseChunk) and bool(item.text)


def _attach_watchdog(unit: Any) -> None:
    """Attach a FillerWatchdog to one PipelineUnit."""
    watchdog = FillerWatchdog(unit)
    response_handler = unit.service.response
    lm_processor = _find_lm_processor(unit.handlers)
    if lm_processor is None:
        logger.warning("LMOutputProcessor not found; hermes filler disabled for unit %s", unit.index)
        return

    # Hook response start.
    original_response_create = response_handler.handle_response_create

    def _patched_response_create(conn_id: str, event: Any) -> Any:
        # Let the original create the response and queue the LLM request,
        # then arm our timer so we capture the fresh current_response_params.
        result = original_response_create(conn_id, event)
        watchdog.response_started(conn_id)
        return result

    response_handler.handle_response_create = _patched_response_create

    # Hook response end (completed/cancelled/failed all go through _end_response).
    original_end_response = response_handler._end_response

    def _patched_end_response(conn_id: str, status: Any = "completed") -> None:
        watchdog.response_ended(conn_id)
        return original_end_response(conn_id, status)

    response_handler._end_response = _patched_end_response

    # Hook LM output so we cancel the timer when real assistant text appears.
    original_process = lm_processor.process

    def _patched_process(*args: Any, **kwargs: Any) -> Any:
        first = True
        for item in original_process(*args, **kwargs):
            if first:
                first = False
                if _is_text_chunk(item):
                    watchdog.first_text_seen()
                # Tool-only chunks keep the timer running: filler is meant to
                # cover dead time during tool calls.
            yield item

    lm_processor.process = _patched_process


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


def _install() -> None:
    import speech_to_speech.s2s_pipeline as s2s_pipeline

    original = s2s_pipeline._build_realtime_pipeline_unit

    def _patched_build_realtime_pipeline_unit(*args: Any, **kwargs: Any) -> Any:
        unit = original(*args, **kwargs)
        try:
            _attach_watchdog(unit)
        except Exception:
            logger.exception("Failed to attach hermes filler watchdog")
        return unit

    s2s_pipeline._build_realtime_pipeline_unit = _patched_build_realtime_pipeline_unit


if _is_enabled():
    try:
        _install()
        logger.info("Hermes filler patch installed")
    except Exception:
        logger.exception("Hermes filler patch failed to install")
