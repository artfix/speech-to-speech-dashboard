"""Dashboard-side reverse proxy for Hermes Agent.

This module mounts a small set of routes on the dashboard's existing
uvicorn server (port 8050) under the path prefix ``/hermes-proxy/``.
The routes transparently forward requests to the hermes subprocess
(running on ``127.0.0.1:8642`` by default) while injecting two
headers the pipeline never sends on its own:

- ``Authorization: Bearer <API_SERVER_KEY>`` — hermes-agent enforces
  this on every endpoint other than ``/health`` (verified upstream
  in ``hermes_cli/security_audit_startup.py`` for 0.19+).
- ``X-Hermes-Session-Id: <uuid>`` — without this, hermes treats each
  request as stateless: no memory, no evolving persona, no
  conversation continuity, no skill state. With it, the same voice
  conversation is one continuous hermes session.

Why this exists
===============

The pipeline (the speech-to-speech package under
``src/speech_to_speech/``) builds its OpenAI client with
``OpenAI(api_key=..., base_url=...)`` and no custom headers. We
can't edit the pipeline (per the project constraints in CLAUDE.md), so
we can't make it send ``X-Hermes-Session-Id`` directly. The proxy
sits in the middle, on the dashboard's own port, and the pipeline
points at it instead of at hermes directly. End result: hermes sees
the session header, pipeline is unchanged.

Resource cost
=============

This proxy runs **inside the dashboard's existing uvicorn process**.
No new port, no new process, no model loaded, no state held. Per
request, it does a lock + dict lookup + httpx stream passthrough.
CPU at idle is zero. Memory footprint is the size of one in-flight
SSE response (a few KB).

The proxy is mounted at ``/hermes-proxy/v1/*`` on the existing 8050
port, so the pipeline's ``--responses-api-base-url`` is something
like ``http://127.0.0.1:8050/hermes-proxy/v1``.

Latency notes
=============

The proxy is on the hot path of every LLM token chunk. To keep it as
cheap as possible:

- The hermes config block (host, port, api_key) is cached in memory
  and only refreshed when the dashboard writes it (see
  :func:`refresh_hermes_config`). We never re-read the settings file
  on the request path.
- The session id is a module-level uuid created lazily on first use
  and held in a slot — no allocation per request.
- Hop-by-hop headers are stripped once, upstream, instead of per chunk.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)


# Where the proxy is mounted on the dashboard's own port. The pipeline
# is configured to point at this base URL when "Hermes Agent" is
# selected in the LLM dropdown.
PROXY_MOUNT_PREFIX = "/hermes-proxy"


# ----------------------------------------------------------------------
# Cached config + session id
# ----------------------------------------------------------------------
# The proxy reads these on every request, so they live in module-level
# slots guarded by a lock. The dashboard's write paths (start, stop,
# filler) call :func:`refresh_hermes_config` to keep them in sync.
#
# We deliberately don't read the settings file on the request path:
# every LLM token chunk would otherwise trigger a ``stat()`` + read,
# which is exactly the kind of avoidable latency a voice pipeline
# should not eat.

_CONFIG_LOCK = threading.Lock()
_CACHED_BASE_URL: str = ""  # empty = "hermes not configured yet"
_CACHED_API_KEY: str = ""

# Session id is owned by the *pipeline* lifecycle, not the dashboard
# process lifetime. The dashboard calls reset_session_id() at:
#   - /api/process/start      (fresh pipeline -> fresh hermes context)
#   - /api/process/restart    (same)
#   - /api/hermes/kill        (killed hermes can't serve the old id)
# Polite stop and cancel do NOT reset — those preserve hermes context
# per the plan §6 ("Cancellation flow").
_SESSION_LOCK = threading.Lock()
_SESSION_ID: Optional[str] = None


def get_or_create_session_id() -> str:
    """Return the current hermes session id, creating one on first use.

    One uuid per pipeline lifetime. Reset only on pipeline start /
    restart or on hermes kill (via :func:`reset_session_id`). The id
    is a uuid4 hex string — hermes accepts any opaque value.
    """
    global _SESSION_ID
    with _SESSION_LOCK:
        if _SESSION_ID is None:
            _SESSION_ID = uuid.uuid4().hex
        return _SESSION_ID


def reset_session_id() -> None:
    """Force a fresh session id on the next request.

    Called by the dashboard when the pipeline is started/restarted
    or when hermes is hard-killed (a killed hermes loses its context,
    so the old id is stale and would return 404 from hermes's
    api_server).
    """
    global _SESSION_ID
    with _SESSION_LOCK:
        _SESSION_ID = None


def refresh_hermes_config(cfg: Optional[dict[str, Any]]) -> None:
    """Update the cached hermes config block.

    Called by the dashboard's /api/hermes/* write paths (start, stop,
    filler updates, etc.) so the proxy always uses the latest
    settings without disk I/O on the request path. Safe to call from
    any thread; no-op when ``cfg`` is None or missing required keys.
    """
    global _CACHED_BASE_URL, _CACHED_API_KEY
    with _CONFIG_LOCK:
        if not isinstance(cfg, dict):
            return
        host = cfg.get("host") or "127.0.0.1"
        if not isinstance(host, str) or not host:
            host = "127.0.0.1"
        try:
            port = int(cfg.get("port") or 8642)
        except (TypeError, ValueError):
            port = 8642
        api_key = cfg.get("api_key") or ""
        if not isinstance(api_key, str):
            api_key = ""
        _CACHED_BASE_URL = f"http://{host}:{port}"
        _CACHED_API_KEY = api_key


def _target() -> tuple[str, str]:
    """Return ``(base_url, api_key)`` from the in-memory cache.

    Empty strings mean "hermes not running / not configured". Callers
    respond with a 502 in that case so the pipeline sees a meaningful
    failure rather than a hung request.
    """
    with _CONFIG_LOCK:
        return _CACHED_BASE_URL, _CACHED_API_KEY


# The router is mounted in :mod:`web_ui.server` at app construction
# time. We export a single router here so the server module can do
# ``app.include_router(hermes_proxy.router, prefix=PROXY_MOUNT_PREFIX)``.
router = APIRouter()


async def _proxy_passthrough(
    request: Request,
    method: str,
    path: str,
) -> Response:
    """Forward an HTTP request to the hermes subprocess and stream the response.

    Handles every method, every header (except ``host`` and
    ``content-length`` which httpx manages), and streams the body
    both directions. SSE responses (``text/event-stream``) are
    forwarded chunk-by-chunk so /v1/chat/completions streams
    token-by-token just like a direct call to hermes would.

    If hermes is not running or the proxy has no config yet, we
    return 502 with a clear message rather than letting the request
    hang — the pipeline's own retry logic will surface a useful
    error to the user.
    """
    base_url, api_key = _target()
    if not base_url or not api_key:
        return JSONResponse(
            {
                "error": "hermes_not_running",
                "message": ("Hermes Agent is not running. Start it from the Hermes tab in the dashboard."),
            },
            status_code=502,
        )
    target_url = f"{base_url}{path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"

    # Build the upstream headers. We always set Authorization (hermes
    # rejects requests without a key, even on loopback) and the
    # session id (so hermes keeps the conversation continuous).
    # Everything else from the inbound request is forwarded
    # verbatim — including ``accept``, ``content-type``, custom user
    # headers, etc.
    inbound_headers = dict(request.headers)
    # Drop headers that don't make sense to forward.
    inbound_headers.pop("host", None)
    inbound_headers.pop("content-length", None)
    inbound_headers["Authorization"] = f"Bearer {api_key}"
    inbound_headers["X-Hermes-Session-Id"] = get_or_create_session_id()

    body_bytes = await request.body()
    # The httpx client and the upstream response MUST outlive the
    # StreamingResponse — Starlette reads the body async via the
    # generator, and once we return the client + upstream stay in scope
    # only if we keep a reference to them. We capture both in the
    # closure that the generator and the BackgroundTask close over.
    # This is the standard Starlette + httpx reverse-proxy pattern;
    # without it the upstream stream is closed before Starlette can
    # read the first chunk and the client sees an empty body.
    client = httpx.AsyncClient(timeout=None)
    try:
        upstream = await client.send(
            client.build_request(
                method,
                target_url,
                headers=inbound_headers,
                content=body_bytes if body_bytes else None,
            ),
            stream=True,
        )
    except httpx.ConnectError as e:
        await client.aclose()
        return JSONResponse(
            {
                "error": "hermes_unreachable",
                "message": f"Could not reach hermes at {target_url}: {e}",
            },
            status_code=502,
        )
    except Exception as e:  # noqa: BLE001
        await client.aclose()
        logger.exception("hermes proxy error")
        return JSONResponse(
            {
                "error": "proxy_error",
                "message": f"{type(e).__name__}: {e}",
            },
            status_code=502,
        )

    # Copy response headers but strip hop-by-hop ones AND any header
    # uvicorn sets itself (``date`` / ``server``) — forwarding them
    # produces duplicate headers that strict HTTP clients reject with
    # an empty body / JSONDecodeError.
    resp_headers = dict(upstream.headers)
    for h in (
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "date",
        "server",
    ):
        resp_headers.pop(h, None)
    if (resp_headers.get("content-type") or "").startswith("text/event-stream"):
        resp_headers["X-Accel-Buffering"] = "no"
        resp_headers["Cache-Control"] = "no-cache"

    status_code = upstream.status_code

    async def _body_iter() -> Any:
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    yield chunk
        except httpx.StreamClosed:
            # Pipeline disconnected mid-stream. End gracefully — the
            # BackgroundTask will still close the client cleanly.
            return
        except Exception:
            logger.exception("hermes proxy: stream read error")
            return

    async def _cleanup() -> None:
        try:
            await upstream.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass

    return StreamingResponse(
        content=_body_iter(),
        status_code=status_code,
        headers=resp_headers,
        background=BackgroundTask(_cleanup),
    )


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
# We catch-all on ``/v1/{path:path}`` so the proxy can serve every
# hermes endpoint transparently: /v1/chat/completions, /v1/models,
# /v1/responses, /v1/capabilities, /v1/runs/... and so on. The path
# the pipeline uses is the only one we *must* support, but routing
# them all is free and means the console / model-picker / etc. can
# all work without bespoke handlers.


@router.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def proxy_v1(path: str, request: Request) -> Response:
    return await _proxy_passthrough(
        request=request,
        method=request.method,
        path=f"/v1/{path}",
    )


@router.get("/health")
async def proxy_health() -> dict[str, Any]:
    """Health endpoint for the proxy itself (NOT a passthrough).

    The dashboard's status panel uses this to surface "proxy
    mounted?" without a forward to hermes. We DO also try to ping
    hermes's /health so the response includes whether hermes is
    actually up — that way one call answers both questions.
    """
    base_url, _api_key = _target()
    hermes_reachable = False
    hermes_error: Optional[str] = None
    if base_url:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{base_url}/health")
            hermes_reachable = 200 <= r.status_code < 300
            if not hermes_reachable:
                hermes_error = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            hermes_reachable = False
            hermes_error = f"{type(e).__name__}: {e}"
    return {
        "proxy": "ok",
        "hermes_reachable": hermes_reachable,
        "hermes_error": hermes_error,
        "session_id": get_or_create_session_id(),
    }
