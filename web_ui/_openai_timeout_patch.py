"""OpenAI-SDK monkey-patch for the Hermes LLM read-timeout knob.

What this does
==============

The pipeline's ``BaseOpenAICompatibleHandler.setup()`` hardcodes a
20-second read timeout for every chat / response call. When the user
asks Hermes to read a long passage, or a large model takes longer
than 20 s to first byte, the httpx client times out, the pipeline
hits its ``except httpx.ReadTimeout:`` branch, and the canned
fallback ``"Wow I'm a bit slow today, could you repeat that?"`` is
spoken by the TTS. The user sees Hermes as "stuck in a loop".

The pipeline's ``src/speech_to_speech/`` is treated as a black box
(see CLAUDE.md) so we can't add a CLI flag for the request timeout.
This module is a dashboard-side monkey-patch that intercepts the
``openai.resources.chat.completions.Completions.create`` and
``openai.resources.responses.Responses.create`` methods to inject
the dashboard-configured timeout. If no timeout is configured, the
patch is a strict no-op (it returns whatever the original method
returns).

Loading
=======

The dashboard writes a copy of this file to the venv's
``site-packages`` directory as
``_dash_openai_timeout_patch.py`` and a sibling ``sitecustomize.py``
that imports it at Python interpreter startup. Both are removed on
pipeline shutdown. Auto-loading via ``sitecustomize.py`` is the
documented PEP-370 startup-hook mechanism -- it fires before
``src/speech_to_speech/`` is imported, so the patch is in place
before the pipeline's ``OpenAI(...)`` constructor ever runs.

Scope
=====

The patch only replaces the timeout when the call's ``timeout`` kwarg
matches the pipeline's hardcoded 20 s shape. If the user (or the
SDK default) set a different timeout, the patch leaves the call
alone -- we don't want to clobber a deliberate value.

Env var contract
================

``DASHBOARD_LLM_READ_TIMEOUT_S`` -- integer, set by the dashboard
before every pipeline start, regardless of LLM backend. The value is
the user's per-backend pick from the LLM Settings tab (defaults:
hermes = 0, ollama/vllm/llama.cpp/mlx-lm/transformers = 120,
responses-api = 20).

* ``0`` means *no timeout* -- ``httpx.Timeout(None)``. Wait forever
  for the LLM to close the stream.
* ``>0`` -- read timeout in seconds, ``httpx.Timeout(N)``.

If the env var is unset, invalid, or set to a non-integer, the patch
is a no-op and the pipeline's 20 s default stands.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# Exact shape the pipeline builds:
#   self.request_timeout = httpx.Timeout(self.request_timeout_s, connect=min(10.0, self.request_timeout_s))
# i.e. httpx.Timeout(20.0, connect=10.0). We match on it so we don't
# clobber a deliberate value (e.g. someone hand-passes
# httpx.Timeout(None) for a long-running call).
_PIPELINE_REQUEST_TIMEOUT_READ_S = 20.0
_PIPELINE_REQUEST_TIMEOUT_CONNECT_S = 10.0


def _read_dashboard_timeout_s() -> int | None:
    """Read ``DASHBOARD_LLM_READ_TIMEOUT_S`` from the env. ``None`` = no-op."""
    raw = os.environ.get("DASHBOARD_LLM_READ_TIMEOUT_S")
    if raw is None or raw == "":
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "DASHBOARD_LLM_READ_TIMEOUT_S=%r is not an integer; timeout patch is a no-op",
            raw,
        )
        return None
    if v < 0:
        logger.warning(
            "DASHBOARD_LLM_READ_TIMEOUT_S=%d is negative; timeout patch is a no-op",
            v,
        )
        return None
    return v


def _is_pipeline_hardcoded_20s(timeout: Any) -> bool:
    """True if ``timeout`` is exactly ``httpx.Timeout(20.0, connect=10.0)``.

    The pipeline constructs that object once per LLM handler and
    reuses it for every call; we identify it on the shape (read and
    connect fields) rather than on object identity so we still
    replace it even if ``httpx.Timeout`` rebuilds the same shape.
    """
    import httpx  # imported lazily so this patch module never breaks an import chain

    if not isinstance(timeout, httpx.Timeout):
        return False
    return (
        timeout.read == _PIPELINE_REQUEST_TIMEOUT_READ_S
        and timeout.connect == _PIPELINE_REQUEST_TIMEOUT_CONNECT_S
    )


def _build_replacement_timeout(seconds: int) -> Any:
    """Build the ``httpx.Timeout`` the dashboard configured.

    ``seconds == 0`` -> ``httpx.Timeout(None)`` (no timeout at all).
    ``seconds > 0``  -> ``httpx.Timeout(seconds)`` (all four phases).
    """
    import httpx

    if seconds == 0:
        return httpx.Timeout(None)
    return httpx.Timeout(float(seconds))


def _patch_create_method(method_name: str, klass: Any, configured_seconds: int) -> None:
    """Wrap ``klass.<method_name>`` to inject the configured timeout."""
    original = getattr(klass, method_name, None)
    if original is None:
        # The SDK doesn't expose this method on this class; nothing to patch.
        return
    # Idempotency: don't wrap twice (e.g. if sitecustomize.py is
    # imported more than once via -c, site reload, etc.).
    if getattr(original, "_dash_openai_timeout_patch_wrapped", False):
        return

    replacement = _build_replacement_timeout(configured_seconds)
    logger.info(
        "openai timeout patch: wrapping %s.%s -- injecting httpx.Timeout(%s)",
        klass.__name__,
        method_name,
        "None (no timeout)" if configured_seconds == 0 else f"{configured_seconds}s",
    )

    def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Replace the pipeline's hardcoded 20 s only; leave every
        # other timeout shape untouched so we never clobber a
        # deliberate value the user has chosen.
        if _is_pipeline_hardcoded_20s(kwargs.get("timeout")):
            kwargs["timeout"] = replacement
        return original(self, *args, **kwargs)

    _wrapped._dash_openai_timeout_patch_wrapped = True  # type: ignore[attr-defined]
    setattr(klass, method_name, _wrapped)


def apply_patch() -> bool:
    """Apply the openai-SDK monkey-patch. Returns True if the patch
    is active, False if the env var is unset / invalid (no-op)."""
    seconds = _read_dashboard_timeout_s()
    if seconds is None:
        return False

    try:
        import httpx  # noqa: F401  -- imported so monkey-patched code can rely on it being available
        from openai.resources.chat import completions as _chat_completions_mod
        from openai.resources.responses import responses as _responses_mod
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "openai timeout patch: could not import openai / httpx (%s); patch is a no-op",
            exc,
        )
        return False

    _patch_create_method("create", _chat_completions_mod.Completions, seconds)
    _patch_create_method("create", _responses_mod.Responses, seconds)
    return True


# Auto-apply on import. The sitecustomize.py the dashboard drops
# into site-packages does ``import _dash_openai_timeout_patch`` at
# interpreter startup, which triggers this.
try:
    _ACTIVE = apply_patch()
except Exception:  # noqa: BLE001
    logger.exception("openai timeout patch: apply_patch crashed; patch is a no-op")
    _ACTIVE = False


def is_active() -> bool:
    """Whether the patch is currently replacing the 20 s timeout."""
    return _ACTIVE