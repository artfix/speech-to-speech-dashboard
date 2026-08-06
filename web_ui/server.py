"""FastAPI server for the speech-to-speech web dashboard.

This module wires together:

- :mod:`web_ui.settings_schema` -- produces the JSON form schema from the
  pipeline's argument classes.
- :mod:`web_ui.process_manager` -- spawns and supervises the pipeline subprocess.
- A small set of REST + websocket endpoints for the single-page frontend.
- A static-file mount for ``web_ui/static`` (HTML / JS / CSS) and
  ``web_ui/themes`` (one CSS file per theme).

The server listens on ``0.0.0.0:8050`` by default. Port can be overridden with
``SPEECH_TO_SPEECH_WEB_PORT``. Host with ``SPEECH_TO_SPEECH_WEB_HOST``.

Two settings files are involved:

- ``web_ui_settings.json`` in the repo root -- the user's saved settings.
  Created on first Save, deleted on Reset.
- This module's runtime state lives only in memory and in the subprocess.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from web_ui import __version__ as _DASHBOARD_VERSION
from web_ui.gpu import (
    check_gpu,
    install_wheel,
    resolution_for_device,
)
from web_ui.hermes_manager import HermesProcess
from web_ui.hermes_proxy import (
    PROXY_MOUNT_PREFIX as _HERMES_PROXY_PREFIX,
)
from web_ui.hermes_proxy import (
    refresh_hermes_config as _hermes_proxy_refresh_config,
)
from web_ui.hermes_proxy import (
    reset_session_id as _hermes_proxy_reset_session,
)
from web_ui.hermes_proxy import (
    router as _hermes_proxy_router,
)
from web_ui.llm_keepalive import KeepaliveStatus, LLMKeepAliver
from web_ui.process_manager import LogLine, PipelineProcess
from web_ui.qwentts_voice_library import (
    PRESET_SPEAKERS,
    Qwen3NotInstalled,
    list_ref_audio_files,
    synthesize_qwen3_test,
)
from web_ui.settings_schema import get_defaults, get_full_schema
from web_ui.voice_library import (
    ChatterboxNotInstalled,
    delete_voice,
    list_voices,
    save_voice,
    synthesize_test,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "web_ui_settings.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"
THEMES_DIR = Path(__file__).resolve().parent / "themes"
GUIDE_PATH = Path(__file__).resolve().parent / "Guide.md"

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8050


# ----------------------------------------------------------------------
# Settings file I/O
# ----------------------------------------------------------------------


def _read_settings() -> Optional[dict[str, Any]]:
    if not SETTINGS_PATH.exists():
        return None
    try:
        settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read %s: %s", SETTINGS_PATH, e)
        return None

    # One-time migration: upstream v0.2.12 renamed the raw PCM mode from
    # "websocket" to "raw-websocket" and removed the compatibility alias.
    # Old web_ui_settings.json files saved with the previous default would
    # otherwise fail to start the pipeline after the rename. We rewrite the
    # value on load so users don't have to recreate their settings.
    if settings.get("mode") == "websocket":
        settings["mode"] = "raw-websocket"
        _write_settings(settings)
        logger.info("Migrated saved mode 'websocket' -> 'raw-websocket'")
    return settings


def _write_settings(settings: dict[str, Any]) -> None:
    """Atomically write settings to disk (write .tmp, then rename)."""
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)


def _delete_settings() -> None:
    if SETTINGS_PATH.exists():
        SETTINGS_PATH.unlink()


# ----------------------------------------------------------------------
# App + state
# ----------------------------------------------------------------------


class DashboardState:
    """Single FastAPI-process state container.

    Holds the :class:`PipelineProcess` and a list of connected websocket
    subscribers.
    """

    def __init__(self) -> None:
        self.process = PipelineProcess(repo_root=REPO_ROOT)
        self.hermes = HermesProcess(repo_root=REPO_ROOT)
        self.websocket_clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def broadcast(self, line: LogLine) -> None:
        """Push a log line to all connected websocket clients.

        Called from the process manager's fanout thread; we hop to the asyncio
        loop to actually send. Each call is a no-op if there are no clients.

        Source-tagging is now done by the per-subprocess subscribers
        (``_pipeline_log_cb`` / ``_hermes_log_cb``), not here, so the
        websocket can distinguish which subprocess produced each line.
        """
        if not self.websocket_clients:
            return
        loop = self._loop
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_all(line.to_dict()), loop)

    async def _send_to_all(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        text = json.dumps(payload)
        for ws in list(self.websocket_clients):
            try:
                await ws.send_text(text)
            except Exception:  # noqa: BLE001
                dead.append(ws)
        for ws in dead:
            self.websocket_clients.discard(ws)


state = DashboardState()


# Background pinger that refreshes Ollama's ``keep_alive`` timer when the
# user has set ``--llm-keepalive`` to anything other than ``"0"``. The
# pipeline's CLI flags don't expose ``keep_alive`` (CLAUDE.md forbids
# editing ``src/speech_to_speech/``), so we refresh it from the dashboard
# side instead. See ``web_ui/llm_keepalive.py`` for the pinger itself.
llm_keepaliver = LLMKeepAliver()


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------


def _make_source_stamping_subscriber(source: str) -> "callable":
    """Build a subscriber that stamps a ``source`` field on each log line.

    The pipeline and hermes both fan out through the same websocket,
    so the frontend needs to know which subprocess a line came from.
    Wrapping the broadcast call in a per-source adapter keeps the
    log_line's dataclass clean and avoids mutating shared state from
    multiple threads.
    """

    def _cb(line: LogLine) -> None:
        # Build a payload with the source tag. We do this here (not in
        # ``broadcast``) because ``broadcast`` is the shared sink and
        # already does the asyncio hop — the per-source tagging belongs
        # at the producer side, before the line gets handed off.
        d = line.to_dict()
        d["source"] = source
        if not state.websocket_clients or state._loop is None:
            return
        asyncio.run_coroutine_threadsafe(state._send_to_all(d), state._loop)

    return _cb


# Two source-stamping callbacks, one per subprocess. They share the
# same broadcast path but tag the lines so the frontend can split
# them. ``unsubscribed=False`` because we manually re-attach the same
# callback on the subscribe side; we keep the same function reference
# for clean unsubscribe in ``lifespan``.
_pipeline_log_cb = _make_source_stamping_subscriber("pipeline")
_hermes_log_cb = _make_source_stamping_subscriber("hermes")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state._loop = asyncio.get_running_loop()
    # Subscribe the source-stamping callbacks to each subprocess's log
    # fanout. The callbacks hand the stamped line to the shared
    # broadcast path so the single ``/ws/logs`` websocket serves both
    # pipelines' logs.
    state.process.subscribe(_pipeline_log_cb)
    state.hermes.subscribe(_hermes_log_cb)
    try:
        yield
    finally:
        try:
            state.process.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping pipeline on shutdown")
        # Skip hermes stop if the dashboard is restarting for a wheel
        # install (the suppression flag is set in _schedule_dashboard_restart).
        # The user's hermes session keeps running in its own process group
        # and the new dashboard re-uses it. The explicit /api/hermes/kill
        # endpoint still works for forced shutdown.
        if not _SUPPRESS_HERMES_STOP_ON_SHUTDOWN:
            try:
                state.hermes.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Error stopping hermes on shutdown")
        state.process.unsubscribe(_pipeline_log_cb)
        state.hermes.unsubscribe(_hermes_log_cb)
        # Tear down the keepalive pinger so the dashboard exits cleanly.
        try:
            llm_keepaliver.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping llm keepalive pinger on shutdown")


app = FastAPI(title="speech-to-speech dashboard", lifespan=lifespan)


# Reverse proxy for Hermes Agent. The pipeline's LLM slot is pointed
# at ``http://<dashboard-host>:<dashboard-port>/hermes-proxy/v1`` when
# the user picks Hermes Agent in the LLM dropdown. The proxy mounts
# on the dashboard's own uvicorn (same process, same port) and
# transparently forwards requests to the hermes subprocess while
# injecting the ``Authorization`` and ``X-Hermes-Session-Id`` headers
# the OpenAI client used by the pipeline doesn't know how to set.
# Resource cost is negligible: the proxy is a few FastAPI routes, no
# extra process, no extra port. See web_ui/hermes_proxy.py for the
# full design rationale.
app.include_router(_hermes_proxy_router, prefix=_HERMES_PROXY_PREFIX)


# No-cache middleware for all static assets. Without this, browsers
# happily serve a stale ``app.js`` across dashboard restarts, which
# means a user iterating on dashboard code can spend a long time
# wondering why their changes don't appear. ETag + Last-Modified
# headers alone aren't enough — Chromium often ignores them on
# same-origin GETs after a soft refresh.
@app.middleware("http")
async def _no_cache_static(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/") or path.startswith("/static-repo/"):
        # ``no-cache`` lets the browser cache, but requires revalidation
        # every time. ``no-store`` would force a full re-download every
        # page load, which is wasteful for our 80 KB app.js — revalidate
        # is the right balance.
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# ---- Settings --------------------------------------------------------------


@app.get("/api/schema")
def api_schema() -> dict[str, Any]:
    return get_full_schema()


@app.get("/api/settings")
def api_get_settings() -> dict[str, Any]:
    saved = _read_settings()
    defaults = get_defaults()
    if saved is None:
        return {"saved": False, "settings": defaults, "path": str(SETTINGS_PATH)}
    # Merge: saved wins, defaults fill in anything missing.
    merged = {**defaults, **saved}
    return {"saved": True, "settings": merged, "path": str(SETTINGS_PATH)}


@app.post("/api/settings")
def api_post_settings(body: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    # The frontend sends the full settings object. We only persist it.
    _write_settings(body)
    # Keep the Ollama keepalive pinger in sync with the new value. We
    # call this AFTER the write so a transient Ollama outage doesn't lose
    # the user's saved setting — the pinger reads from the settings dict
    # the user just confirmed, not from disk.
    # 0.3.2+: the pinger has been replaced by the one-shot warmup
    # (see _one_shot_keepalive_async below). The call is kept as a
    # no-op for rollback parity.
    try:
        llm_keepaliver.update_from_settings(body)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update llm keepalive from settings save")
    # Fire a one-shot Ollama warmup so the selected model is loaded
    # into VRAM and the unload timer is set in a single request.
    # Background-thread, never blocks the Save response.
    _one_shot_keepalive_async(body)
    return {"ok": True, "path": str(SETTINGS_PATH)}


@app.post("/api/settings/patch")
def api_patch_settings(body: dict[str, Any]) -> dict[str, Any]:
    """Merge `body` into the saved settings file (creates it if missing).

    Used for per-field auto-saves like the theme picker -- the frontend
    doesn't need to know the rest of the file's contents. Only the keys in
    `body` are touched; everything else on disk is preserved.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    current = _read_settings() or {}
    current.update(body)
    _write_settings(current)
    # Re-evaluate the keepalive pinger after a per-field patch too — this
    # catches the (uncommon) case where someone POSTs `--llm-keepalive`
    # via the patch endpoint directly.
    try:
        llm_keepaliver.update_from_settings(current)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update llm keepalive from settings patch")
    # 0.3.2+: also fire a one-shot warmup so per-field patches (e.g.
    # theme-style auto-saves from the model dropdown) load the model
    # the moment the user picks it, no Save click required.
    _one_shot_keepalive_async(current)
    return {"ok": True, "path": str(SETTINGS_PATH), "settings": current}


@app.post("/api/reset")
def api_reset() -> dict[str, Any]:
    _delete_settings()
    return {"ok": True, "settings": get_defaults()}


# ---- Voice library + Chatterbox install --------------------------------


def _chatterbox_installed() -> bool:
    """Best-effort probe for the optional chatterbox-tts package.

    The top-level ``import chatterbox`` is not enough on its own: chatterbox
    pulls in ``onnx`` at import time, and onnx in turn pulls in
    ``ml_dtypes`` (a transitive dep). If the user installed chatterbox
    with ``--no-deps`` and didn't separately install ml_dtypes, the
    top-level import succeeds but actually loading the model crashes
    later. Importing ``chatterbox.tts.ChatterboxTTS`` exercises the full
    import chain and surfaces that case here.
    """
    try:
        import chatterbox  # type: ignore[import-not-found]  # noqa: F401
        import chatterbox.tts  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def _platform_label() -> str:
    if sys.platform.startswith("win"):
        return "Windows"
    return "Linux"


def _chatterbox_install_command() -> str:
    """Build the platform-aware install command shown in the install modal.

    ``chatterbox-tts==0.1.7`` pins ``transformers==5.2.0`` and
    ``torchaudio==2.6.0`` strictly. The dashboard's auto-install tosses
    the chatterbox package in with ``--no-deps`` and then adds the
    chatterbox-specific runtime deps (s3tokenizer, conformer, diffusers,
    etc.) that the package expects but pip cannot resolve alongside the
    project's other pins. ``transformers==5.2.0`` is added separately so
    it gets installed with its own deps (the bundled LlamaModel is what
    chatterbox's TTS model uses internally — the pipeline's LLM slot
    still talks to Ollama/responses-api independently).
    """
    return (
        "uv pip install --no-deps chatterbox-tts==0.1.7 && "
        "uv pip install transformers==5.2.0 "
        "s3tokenizer conformer==0.3.2 resemble-perth "
        "diffusers omegaconf pykakasi pyloudnorm onnx"
    )


def _chatterbox_install_needed(settings: dict[str, Any]) -> Optional[str]:
    """Return an install command string if the user picked chatterbox TTS but the
    package isn't available, else ``None``.
    """
    if settings.get("--tts") != "chatterbox":
        return None
    if _chatterbox_installed():
        return None
    return _chatterbox_install_command()


@app.get("/api/voices")
def api_list_voices() -> dict[str, Any]:
    return {"voices": [v.to_dict() for v in list_voices()]}


@app.post("/api/voices/clone")
async def api_clone_voice(
    name: str = Form(...),
    audio: UploadFile = File(...),
    model_variant: str = Form("chatterbox-turbo"),
) -> dict[str, Any]:
    """Clone a voice from a reference audio uploaded by the browser.

    Multipart fields: ``name`` (string), ``model_variant`` (string, optional,
    defaults to ``chatterbox-turbo``), and ``audio`` (file, any audio format
    that ``prepare_conditionals`` understands -- WAV is best).
    """
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(status_code=400, detail="Field 'name' is required")
    raw = await audio.read()
    original = audio.filename or "reference.wav"
    suffix = os.path.splitext(original)[1] or ".wav"
    try:
        entry = save_voice(
            name=name,
            audio_bytes=raw,
            suffix=suffix,
            model_variant=model_variant,
        )
    except ChatterboxNotInstalled as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chatterbox_not_installed",
                "install_command": _chatterbox_install_command(),
                "platform": _platform_label(),
            },
        ) from e
    except (ValueError, FileExistsError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "voice": entry.to_dict()}


@app.post("/api/qwen3_ref_audio")
async def api_qwen3_ref_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    """Save a reference audio file for Qwen3-TTS voice cloning.

    The file is written under ``voices/qwen3_refs/`` with a UUID-prefixed
    filename so concurrent uploads don't collide. Returns the absolute path
    that should be passed as ``--qwen3-tts-ref-audio``.

    The qwen3-TTS handler accepts any audio format its underlying decoder
    supports; we keep the original extension (defaulting to .wav) so the
    handler's audio loader can find a matching decoder.
    """
    if not audio or not audio.filename:
        raise HTTPException(status_code=400, detail="Missing audio file")
    raw = await audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio file")
    import uuid as _uuid

    suffix = os.path.splitext(audio.filename)[1].lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aac"}:
        # Be permissive but default to .wav if the extension is something
        # exotic so the file still has a sensible name on disk.
        suffix = ".wav"
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(repo_root, "voices", "qwen3_refs")
    os.makedirs(target_dir, exist_ok=True)
    fname = f"{_uuid.uuid4().hex}{suffix}"
    target = os.path.join(target_dir, fname)
    with open(target, "wb") as f:
        f.write(raw)
    return {"ok": True, "path": target, "size_bytes": len(raw)}


# ---- Qwen3-TTS voice library -----------------------------------------------
#
# The CustomVoice model variant exposes 9 preset speakers (Vivian, Serena,
# Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee). The dashboard's
# voice library UI lets the user browse + preview those speakers, and
# (when a Base model is selected) the reference audio files they've
# uploaded under ``voices/qwen3_refs/``. Voice cloning itself runs
# through the model at runtime -- the dashboard's only job here is to
# surface speakers + run one-shot test synthesis. Synthesis happens
# through ``web_ui.qwentts_voice_library.synthesize_qwen3_test``, which
# uses the same qwentts_cpp high-level API the pipeline calls.


@app.get("/api/qwen3/voices")
def api_qwen3_list_voices() -> dict[str, Any]:
    """Return the 9 CustomVoice preset speakers + any uploaded ref-audio files.

    The frontend renders this as the "Voice library" panel inside the
    TTS tab when ``--tts qwen3`` is selected. The response is intentionally
    minimal: the client already knows the speaker metadata (sample
    sentences, language codes) from ``PRESET_SPEAKERS`` baked into
    ``qwen3_voice_library_ui.js``; we just send ref-audio file metadata
    + the active speaker flag so the active card gets the highlight.
    """
    current = _read_settings() or {}
    active_speaker = current.get("--qwen3-tts-speaker") or ""
    ref_files = [e.to_dict() for e in list_ref_audio_files()]
    return {
        "presets": PRESET_SPEAKERS,
        "active_speaker": active_speaker,
        "ref_audio_files": ref_files,
    }


@app.post("/api/qwen3/voice/{speaker}/set-active")
def api_qwen3_set_active_voice(speaker: str) -> dict[str, Any]:
    """Write ``speaker`` into ``--qwen3-tts-speaker`` in the settings file.

    Mirrors the chatterbox ``/api/voices/{name}/set-active`` endpoint so
    the voice library UI can use the same one-click pattern. Also flips
    ``--tts`` to ``qwen3`` if it isn't already so the dashboard and the
    pipeline agree on the backend -- otherwise the saved flag would be
    silently dropped by the form schema (which only persists flags
    whose parent backend is selected).
    """
    current = _read_settings() or {}
    # Reject anything outside the 9 verified speaker names. We do NOT
    # accept arbitrary strings here even though the form field allows
    # them, because the voice library UI specifically says "Set active"
    # for a known preset. Custom values can still be typed directly into
    # the ``--qwen3-tts-speaker`` dropdown.
    preset_names = {p["name"] for p in PRESET_SPEAKERS}
    if speaker not in preset_names:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Speaker {speaker!r} is not one of the 9 CustomVoice presets: "
                f"{sorted(preset_names)}. Use the dropdown's 'Custom' option to "
                "set a non-preset value."
            ),
        )
    current["--qwen3-tts-speaker"] = speaker
    if current.get("--tts") != "qwen3":
        current["--tts"] = "qwen3"
    _write_settings(current)
    return {"ok": True, "speaker": speaker, "settings": current}


# ----------------------------------------------------------------------
# Ollama model discovery + one-shot keep-alive (0.3.2+)
# ----------------------------------------------------------------------
# Replaces the previous background pinger in ``web_ui/llm_keepalive.py``
# (now deprecated). The flow is:
#
# 1. The frontend asks ``GET /api/ollama/models?base_url=...`` to populate
#    the ``--model-name`` dropdown on the LLM tab.
# 2. When the user clicks **Save Settings** or **Start Pipeline**, the
#    dashboard fires ``POST /api/ollama/keepalive`` in a background
#    thread, which sends ONE ``/api/generate`` request to Ollama with
#    ``keep_alive=<user's value>`` — that single call both loads the
#    model into VRAM and sets the unload deadline. Ollama itself
#    enforces the deadline; no pings are needed.
#
# Both endpoints are best-effort: any failure is logged + surfaced to
# the in-app Logs tab, but never blocks the Save/Start response.
def _looks_like_ollama_url(url: str) -> bool:
    """Heuristic: does this base URL look like a local/remote Ollama server?

    Used by the frontend to decide whether to show the model dropdown.
    Defaults to allowing the dropdown on any URL containing ``:11434``
    (Ollama's default port) or the substring ``/ollama``. Real OpenAI
    URLs (``api.openai.com``) fail the check and fall back to free-text.
    """
    s = (url or "").strip().lower()
    if not s:
        return False
    return ":11434" in s or "/ollama" in s


def _normalize_ollama_keep_alive(raw: str | int | float | None) -> Optional[str]:
    """Translate the dashboard's keep_alive string to an Ollama-accepted duration.

    The dashboard stores ``--llm-keepalive`` as a plain string the user
    picks from a curated list (``"0"``, ``"5m"``, ``"30m"``, ``"1h"``,
    ``"2h"``, ``"12h"``, ``"Forever (-1)"``). Ollama's native
    ``/api/generate`` only accepts duration strings with a unit suffix
    (``"5m"``, ``"-1s"`` for never-unload, ``"0s"`` for unload-after),
    or the bare integer ``0``. The "Forever" sentinel ``-1`` is a
    dashboard convention that Ollama rejects with HTTP 400
    ("time: missing unit in duration \"-1\"") — so we translate it
    here.

    Rules:
    - empty / None / ``"0"`` → returns ``None`` (caller skips the call).
    - already has a unit suffix (m/h/s/d) → returned as-is.
    - bare integer ``-1`` → ``"-1s"`` (never unload).
    - bare integer ``>= 1`` → ``"Ns"`` (interpreted as seconds; matches
      the dashboard's keepalive picker where bare numbers are seconds).
    - anything else → returned as-is and let Ollama reject it (so the
      error surfaces for the user to debug).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s == "0":
        return None
    # Already has a unit suffix — pass through.
    if s[-1] in ("m", "h", "s", "d", "w"):
        return s
    # Bare integer: try to parse.
    try:
        n = int(s)
    except (TypeError, ValueError):
        return s  # let Ollama reject it for the user to see
    if n == -1:
        return "-1s"  # Ollama's "never unload" sentinel
    if n > 0:
        return f"{n}s"  # bare positive int → seconds
    return s  # weird value, pass through


@app.get("/api/ollama/models")
def api_ollama_list_models(
    base_url: str = Query(...),
    api_key: Optional[str] = Query(None),
) -> dict[str, Any]:
    """List models installed on the Ollama server at ``base_url``.

    Calls ``{base_url}/models`` (OpenAI-compat) and returns
    ``{"models": [{"id": ...}, ...], "error": str|None}``. The frontend
    silently falls back to a free-text input on any error — we never
    raise HTTPException here.
    """
    if not base_url.strip():
        return {"models": [], "error": "base_url is required"}
    key = (api_key or "ollama").strip() or "ollama"
    url = base_url.rstrip("/") + "/models"
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json",
                },
            )
        if not (200 <= resp.status_code < 300):
            # Truncate the body so a 50 KB error page doesn't bloat the
            # response — first 200 chars is enough to debug.
            err_body = (resp.text or "")[:200]
            return {
                "models": [],
                "error": f"HTTP {resp.status_code}: {err_body}",
            }
        data = resp.json()
        # OpenAI shape: {"object": "list", "data": [{"id": "...", ...}, ...]}.
        # Some Ollama versions / proxies may also return {"models": [...]}.
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            items = data.get("models") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {"models": [], "error": "unexpected response shape"}
        models = []
        for it in items:
            if not isinstance(it, dict):
                continue
            mid = it.get("id") or it.get("name")
            if isinstance(mid, str) and mid:
                models.append({"id": mid})
        return {"models": models, "error": None}
    except httpx.HTTPError as e:
        return {"models": [], "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"models": [], "error": f"{type(e).__name__}: {e}"}


@app.post("/api/ollama/keepalive")
def api_ollama_keepalive(body: dict[str, Any]) -> dict[str, Any]:
    """Send ONE ``/api/generate`` to Ollama to load the model + set keep_alive.

    Body:
    - ``base_url`` (required): the OpenAI-compat URL the user configured.
    - ``api_key`` (optional): defaults to ``"ollama"``.
    - ``model`` (required): the Ollama model identifier.
    - ``keep_alive`` (required): an Ollama duration string (``"5m"``,
      ``"-1"`` for forever, ``"0"`` to skip). Empty / ``"0"`` short-
      circuits with ``{"ok": true, "skipped": "..."}`` — the user has
      chosen to disable the warmup.
    - ``timeout_s`` (optional): integer seconds to wait for Ollama to
      load the model. Defaults to ``60`` (matches the dashboard's
      ``--ollama-load-timeout-seconds`` default). Must be in
      ``[5, 600]`` — anything outside the range is clamped.
    - ``num_ctx`` (optional): integer context window forwarded as
      ``options.num_ctx``. When set, the warmup loads the model at the
      chosen context size so the very first chat request doesn't have
      to pay a second cold-load for the smaller window. Matches what the
      pipeline sends on every subsequent chat-completions call.

    Returns ``{"ok": bool, "skipped": str|None, "message": str|None}``.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    base_url = (body.get("base_url") or "").strip()
    model = (body.get("model") or "").strip()
    keep_alive_raw = (body.get("keep_alive") or "").strip()
    keep_alive = _normalize_ollama_keep_alive(keep_alive_raw)
    api_key = (body.get("api_key") or "ollama").strip() or "ollama"
    # Parse + clamp the timeout so a misconfigured value can't hang the
    # background thread for hours. Anything under 5s is useless for a
    # cold load; anything over 10 min is almost certainly a typo.
    try:
        timeout_s = int(body.get("timeout_s") or 60)
    except (TypeError, ValueError):
        timeout_s = 60
    timeout_s = max(5, min(600, timeout_s))
    # Optional context-window cap. Forwarded as ``options.num_ctx`` on the
    # warmup request so the model is loaded at the user's chosen size from
    # the start. Invalid (non-positive, non-numeric) values are ignored
    # silently — Ollama's own model-default is a sane fallback.
    num_ctx_raw = body.get("num_ctx")
    num_ctx: Optional[int] = None
    if num_ctx_raw is not None and num_ctx_raw != "":
        try:
            n = int(num_ctx_raw)
            if n > 0:
                num_ctx = n
        except (TypeError, ValueError):
            pass
    if not base_url:
        return {"ok": False, "skipped": None, "message": "base_url is required"}
    if not model:
        return {"ok": False, "skipped": None, "message": "model is required"}
    if keep_alive is None:
        return {"ok": True, "skipped": "keep_alive is 0/empty", "message": None}
    # We call Ollama's native ``/api/generate`` rather than the
    # OpenAI-compat ``/v1/chat/completions`` because (a) it accepts
    # ``keep_alive`` directly in the body without any hack, and (b) with
    # an empty ``prompt`` and ``stream=false`` it returns ~immediately
    # once the model is loaded. 10-second timeout because a cold load
    # can legitimately take 10-30s.
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]
    url = origin + "/api/generate"
    payload = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": keep_alive,
    }
    if num_ctx is not None:
        # Ollama accepts ``options`` as a free-form map of inference knobs;
        # ``num_ctx`` caps the context window so a 128k model loaded at 8k
        # doesn't pay the full memory cost. Ignored silently on Ollama
        # builds that don't recognise the field.
        payload["options"] = {"num_ctx": num_ctx}
    import httpx

    try:
        # ``timeout_s`` is the user-controlled max wait for a cold load
        # (clamped to [5, 600] above). Default 60 s; users on a slow
        # LAN loading a 70 B model can bump it from the dashboard. The
        # call runs in a background thread so blocking here is fine.
        with httpx.Client(timeout=float(timeout_s)) as client:
            resp = client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
        if 200 <= resp.status_code < 300:
            return {"ok": True, "skipped": None, "message": None}
        snippet = (resp.text or "")[:200]
        return {
            "ok": False,
            "skipped": None,
            "message": f"HTTP {resp.status_code}: {snippet}",
        }
    except httpx.HTTPError as e:
        return {
            "ok": False,
            "skipped": None,
            "message": f"{type(e).__name__}: {e}",
        }
    except Exception as e:  # noqa: BLE001
        return {
            "ok": False,
            "skipped": None,
            "message": f"{type(e).__name__}: {e}",
        }


# ------------------------------------------------------------------
# /api/ollama/ps + /api/ollama/reload (0.4.3+ Ollama lifecycle section)
# ------------------------------------------------------------------
# Ollama's `/api/ps` returns the list of currently-loaded models with
# their current context window, VRAM footprint, and unload deadline.
# The LLM tab's "Ollama model lifecycle" section polls this every 2s
# to render the live context badge ("Context: 16000"). The reload
# endpoint is the manual button: it sends `keep_alive=0` to unload the
# model, then fires the warmup endpoint with `num_ctx=<current value>`
# so the next load sticks at the user's chosen context.
#
# Both endpoints short-circuit with `{"ok": false, "message": ...}`
# when the URL doesn't look like Ollama — same heuristic the keepalive
# path uses. The frontend hides the whole section when the URL isn't
# Ollama; these endpoints are a second line of defense.


@app.get("/api/ollama/ps")
def api_ollama_ps(
    base_url: str = Query(...),
    api_key: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Return the list of models currently loaded in Ollama's VRAM.

    Calls ``{base_url}/api/ps`` (Ollama's native, NOT OpenAI-compat,
    endpoint). Parses each entry's ``context_length`` into a flat
    ``context`` integer so the frontend can show "Context: 16000"
    without re-parsing Ollama's nested shape.

    Returns ``{"ok": bool, "models": [...], "error": str|None}``.
    ``ok`` is false when Ollama is unreachable or the URL doesn't
    look like Ollama.
    """
    if not _looks_like_ollama_url(base_url):
        return {"ok": False, "models": [], "error": "not an ollama url"}
    base_url = base_url.strip()
    if not base_url:
        return {"ok": False, "models": [], "error": "base_url is required"}
    key = (api_key or "ollama").strip() or "ollama"
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]
    url = origin + "/api/ps"
    import httpx

    try:
        # 5s is plenty for localhost / LAN. The endpoint is cheap.
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Accept": "application/json",
                },
            )
        if not (200 <= resp.status_code < 300):
            snippet = (resp.text or "")[:200]
            return {
                "ok": False,
                "models": [],
                "error": f"HTTP {resp.status_code}: {snippet}",
            }
        data = resp.json()
        # Ollama shape: {"models": [{"name", "size_vram", "context_length",
        # "expires_at", ...}]}. Older builds may omit context_length; we
        # treat that as "unknown" rather than dropping the entry.
        items = data.get("models") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {"ok": True, "models": [], "error": "unexpected response shape"}
        models = []
        for it in items:
            if not isinstance(it, dict):
                continue
            name = it.get("name")
            if not isinstance(name, str) or not name:
                continue
            ctx_raw = it.get("context_length")
            try:
                ctx_int: Optional[int] = int(ctx_raw) if ctx_raw is not None else None
            except (TypeError, ValueError):
                ctx_int = None
            # expires_at is an ISO timestamp; we surface it as a
            # human-readable "until" string for the badge.
            until = it.get("expires_at")
            models.append(
                {
                    "name": name,
                    "context": ctx_int,
                    "size_vram": it.get("size_vram"),
                    "until": until if isinstance(until, str) else None,
                }
            )
        return {"ok": True, "models": models, "error": None}
    except httpx.HTTPError as e:
        return {"ok": False, "models": [], "error": f"{type(e).__name__}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "models": [], "error": f"{type(e).__name__}: {e}"}


@app.post("/api/ollama/reload")
def api_ollama_reload(body: dict[str, Any]) -> dict[str, Any]:
    """Unload the Ollama model, then warm it back up at the current num_ctx.

    Two-step dance because Ollama ignores ``num_ctx`` on requests for a
    model already resident in its KV cache at a smaller context. The
    only way to apply a new context is to fully unload first.

    Body (all optional except ``base_url``):
    - ``base_url`` (required): Ollama base URL.
    - ``api_key`` (optional): defaults to ``"ollama"``.
    - ``model`` (required): Ollama model identifier.
    - ``num_ctx`` (optional): integer context. Forwarded on the warmup.
    - ``keep_alive`` (optional): the post-warmup keep-alive duration.
      Defaults to ``"-1"`` (forever) so the freshly-loaded model sticks.
      Honored by the same `_looks_like_ollama_url` heuristic as the rest
      of the Ollama endpoints.
    - ``unload_timeout_s`` (optional): seconds to wait for the unload
      step. Defaults to 10 (the unload is essentially instant when the
      model is loaded; we use a real timeout only as a hang guard).
    - ``warmup_timeout_s`` (optional): seconds to wait for the warmup
      step. Defaults to 60. Same clamping as the keepalive endpoint.

    Returns ``{"ok": bool, "message": str|None, "duration_ms": int,
    "context": int|None}``. ``context`` is the value the model is
    expected to be loaded at (caller can verify with the next
    /api/ollama/ps poll).
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    base_url = (body.get("base_url") or "").strip()
    if not _looks_like_ollama_url(base_url):
        return {"ok": False, "message": "not an ollama url", "duration_ms": 0, "context": None}
    model = (body.get("model") or "").strip()
    api_key = (body.get("api_key") or "ollama").strip() or "ollama"
    keep_alive = _normalize_ollama_keep_alive(body.get("keep_alive")) or "-1s"
    num_ctx_raw = body.get("num_ctx")
    num_ctx: Optional[int] = None
    if num_ctx_raw is not None and num_ctx_raw != "":
        try:
            n = int(num_ctx_raw)
            if n > 0:
                num_ctx = n
        except (TypeError, ValueError):
            pass
    try:
        unload_timeout_s = int(body.get("unload_timeout_s") or 10)
    except (TypeError, ValueError):
        unload_timeout_s = 10
    unload_timeout_s = max(2, min(60, unload_timeout_s))
    try:
        warmup_timeout_s = int(body.get("warmup_timeout_s") or 60)
    except (TypeError, ValueError):
        warmup_timeout_s = 60
    warmup_timeout_s = max(5, min(600, warmup_timeout_s))
    if not model:
        return {"ok": False, "message": "model is required", "duration_ms": 0, "context": num_ctx}
    import time as _time

    import httpx

    started = _time.monotonic()
    origin = base_url.rstrip("/")
    if origin.endswith("/v1"):
        origin = origin[: -len("/v1")]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Step 1: unload. ``keep_alive=0`` tells Ollama to drop the model
    # after serving this request. We send an empty /api/generate so
    # the unload happens immediately. We don't fail the whole call if
    # the model wasn't loaded (Ollama returns 404 — that's fine, we
    # wanted it unloaded anyway).
    unload_url = origin + "/api/generate"
    unload_payload = {"model": model, "prompt": "", "stream": False, "keep_alive": 0}
    try:
        with httpx.Client(timeout=float(unload_timeout_s)) as client:
            unload_resp = client.post(unload_url, json=unload_payload, headers=headers)
        # 200 = unloaded, 404 = wasn't loaded (fine). Anything else = real error.
        if unload_resp.status_code not in (200, 404):
            snippet = (unload_resp.text or "")[:200]
            return {
                "ok": False,
                "message": f"unload HTTP {unload_resp.status_code}: {snippet}",
                "duration_ms": int((_time.monotonic() - started) * 1000),
                "context": num_ctx,
            }
    except httpx.HTTPError as e:
        return {
            "ok": False,
            "message": f"unload failed: {type(e).__name__}: {e}",
            "duration_ms": int((_time.monotonic() - started) * 1000),
            "context": num_ctx,
        }
    # Step 2: warmup with the user's num_ctx. Reuses the keepalive
    # endpoint so the success/failure shape is identical to what the
    # dashboard already handles in _one_shot_keepalive_async.
    warmup_url = origin + "/api/generate"
    warmup_payload: dict[str, Any] = {
        "model": model,
        "prompt": "",
        "stream": False,
        "keep_alive": keep_alive,
    }
    if num_ctx is not None:
        warmup_payload["options"] = {"num_ctx": num_ctx}
    try:
        with httpx.Client(timeout=float(warmup_timeout_s)) as client:
            warmup_resp = client.post(warmup_url, json=warmup_payload, headers=headers)
        duration_ms = int((_time.monotonic() - started) * 1000)
        if 200 <= warmup_resp.status_code < 300:
            ctx_msg = f" at {num_ctx} ctx" if num_ctx is not None else ""
            return {
                "ok": True,
                "message": f"reloaded{ctx_msg}",
                "duration_ms": duration_ms,
                "context": num_ctx,
            }
        snippet = (warmup_resp.text or "")[:200]
        return {
            "ok": False,
            "message": f"warmup HTTP {warmup_resp.status_code}: {snippet}",
            "duration_ms": duration_ms,
            "context": num_ctx,
        }
    except httpx.HTTPError as e:
        return {
            "ok": False,
            "message": f"warmup failed: {type(e).__name__}: {e}",
            "duration_ms": int((_time.monotonic() - started) * 1000),
            "context": num_ctx,
        }


def _one_shot_keepalive_async(settings: dict[str, Any]) -> None:
    """Fire-and-forget Ollama warmup for the user's current settings.

    Background-thread entry point. No-ops if the LLM backend is not
    OpenAI-compat, the keep_alive is empty/0, or the URL doesn't look
    like Ollama. Surfaces the result in the in-app Logs tab.
    """
    backend = (settings.get("--llm-backend") or "").strip()
    if backend not in ("chat-completions", "responses-api"):
        return
    base_url = (settings.get("--responses-api-base-url") or "").strip()
    if not _looks_like_ollama_url(base_url):
        return
    model = (settings.get("--model-name") or "").strip()
    if not model:
        return
    keep_alive = (settings.get("--llm-keepalive") or "").strip()
    if not keep_alive or keep_alive == "0":
        return
    api_key = (settings.get("--responses-api-api-key") or "ollama").strip() or "ollama"
    # The user-tunable max wait for a cold load. Defaults to 60 s; the
    # dashboard exposes this as ``--ollama-load-timeout-seconds`` on the
    # LLM tab next to ``--llm-keepalive``. The endpoint clamps to
    # ``[5, 600]``; we just pass it through.
    try:
        timeout_s = int(settings.get("--ollama-load-timeout-seconds") or 60)
    except (TypeError, ValueError):
        timeout_s = 60
    # Optional context-window cap. Forwarded as options.num_ctx on the
    # warmup call so the model is loaded at the chosen size from the
    # start. Invalid / non-positive values are dropped at the endpoint.
    num_ctx_raw = settings.get("--responses-api-num-ctx")
    num_ctx: Optional[int] = None
    if num_ctx_raw is not None and num_ctx_raw != "":
        try:
            n = int(num_ctx_raw)
            if n > 0:
                num_ctx = n
        except (TypeError, ValueError):
            pass

    def _runner() -> None:
        import httpx

        try:
            # 10 s is plenty for a localhost-to-localhost HTTP call. The
            # *real* timeout (potentially 60+ s for a cold load) is the
            # ``timeout_s`` we forward to the keepalive endpoint, which
            # applies to the Ollama call itself, not this one.
            with httpx.Client(timeout=10.0) as client:
                payload: dict[str, Any] = {
                    "base_url": base_url,
                    "api_key": api_key,
                    "model": model,
                    "keep_alive": keep_alive,
                    "timeout_s": timeout_s,
                }
                if num_ctx is not None:
                    payload["num_ctx"] = num_ctx
                resp = client.post(
                    "http://127.0.0.1:8050/api/ollama/keepalive",
                    json=payload,
                )
            if 200 <= resp.status_code < 300:
                data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
                if data.get("skipped"):
                    _publish_log(
                        f"[ollama] warmup skipped: {data['skipped']}",
                        level="info",
                    )
                elif data.get("ok"):
                    _publish_log(
                        f"[ollama] warmup ok model='{model}' keep_alive='{keep_alive}'",
                        level="info",
                    )
                else:
                    _publish_log(
                        f"[ollama] warmup failed: {data.get('message')}",
                        level="warning",
                    )
            else:
                _publish_log(
                    f"[ollama] warmup HTTP {resp.status_code}: {resp.text[:200]}",
                    level="warning",
                )
        except Exception as e:  # noqa: BLE001
            _publish_log(
                f"[ollama] warmup error: {type(e).__name__}: {e}",
                level="warning",
            )

    t = threading.Thread(target=_runner, name="ollama-one-shot-keepalive", daemon=True)
    t.start()


# Tracks the warmup-trigger callback so Stop / the next Start can detach
# it before a fresh one is armed. A single global slot is enough — only
# one pipeline runs at a time.
_WARMUP_TRIGGER_CB = {"cb": None}


def _arm_warmup_trigger(settings: dict[str, Any]) -> None:
    """Fire ``_one_shot_keepalive_async`` as soon as the pipeline logs warmup done.

    The pipeline's LLM handler logs ``<Handler>:  warmed up! time: ...``
    on stdout the moment its startup ``/v1/chat/completions`` request
    returns. We subscribe to the log stream and trigger our
    ``/api/generate`` warmup on that exact line — no polling, no fixed
    sleep, no assumptions about model load time. Since Ollama serialises
    requests against a single model, our request will be the next one
    Ollama handles, and ``options.num_ctx`` lands deterministically.

    The callback unsubscribes itself after firing once. If the pipeline
    never logs a warmup line (e.g. non-Ollama backend, or warmup errored)
    the callback stays subscribed harmlessly — it just never matches.
    """
    cb_state = _WARMUP_TRIGGER_CB
    # Detach any previous trigger (Stop / new Start) so we don't double-fire.
    prev = cb_state.get("cb")
    if prev is not None:
        try:
            state.process.unsubscribe(prev)
        except Exception:  # noqa: BLE001
            pass
        cb_state["cb"] = None

    def _on_log(entry) -> None:
        try:
            text = getattr(entry, "text", "") or ""
        except Exception:  # noqa: BLE001
            return
        # TEMP DEBUG: log every invocation to the dashboard log so we can
        # see whether the callback is firing at all.
        logger.info("[warmup-trigger] saw line: %s", text[:120])
        if "warmed up" in text or "ChatCompletions" in text:
            _publish_log(f"[ollama] warmup trigger saw line: {text[:120]}", level="info")
        # Match the pipeline's warmup-done marker. The pipeline emits this
        # line in ``src/speech_to_speech/LLM/*.py``'s ``warmup()`` methods.
        if "warmed up" not in text:
            return
        # Detach ourselves so subsequent log lines don't re-trigger.
        try:
            state.process.unsubscribe(_on_log)
        except Exception:  # noqa: BLE001
            pass
        cb_state["cb"] = None
        _publish_log("[ollama] pipeline warmup detected — pinning num_ctx + keep_alive", level="info")
        _one_shot_keepalive_async(settings)

    try:
        state.process.subscribe(_on_log)
        cb_state["cb"] = _on_log
    except Exception:  # noqa: BLE001
        logger.exception("Failed to arm warmup trigger")
        # Fall back to the immediate fire so we don't lose the keepalive
        # entirely if the subscription path is broken for any reason.
        _one_shot_keepalive_async(settings)


@app.post("/api/qwen3/voice/test")
def api_qwen3_voice_test(body: dict[str, Any]) -> Response:
    """Synthesize a short preview for a qwen3 voice and return WAV bytes.

    Body shape (all optional except ``text``):

    - ``text``: string to synthesize.
    - ``speaker``: one of the 9 CustomVoice presets (CustomVoice).
    - ``language``: qwen3 language code (``"english"``, ``"chinese"``,
      ``"japanese"``, ``"korean"``, ``"auto"``). Defaults to the
      speaker's native language when ``speaker`` is set, else
      ``"english"``.
    - ``model_id``: HF Hub model ID. Defaults to
      ``Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice``.
    - ``ref_audio``: path to a reference audio file (Base voice cloning).
    - ``ref_text``: transcript of the reference audio.
    - ``instruct``: voice description (VoiceDesign).
    - ``seed``/``temperature``/``top_p``/``top_k``/``repetition_penalty``:
      sampling params forwarded straight to ``QwenTTS.synthesize``.
    - ``device``: ``"cuda"`` (default) or ``"cpu"``.

    Returns ``audio/wav`` (24 kHz mono int16 PCM).

    The frontend uses this for both the CustomVoice preset Test button
    AND the Reference-voice Test button. For Reference voices on this
    Pascal wheel the synthesis will fail with
    ``QwenTTSError: qt_extract_voice_ref is unavailable; voice reference
    extraction requires qwentts.cpp ABI v2`` -- we surface that as a 400
    with the exact qwentts error message in the JSON detail so the UI
    can show the ABI v2 explanation banner.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' is required")

    # Translate the speaker name -> language code when the caller
    # omitted ``language``. The CustomVoice speakers are optimized for
    # their native language, so using the speaker's native lang for the
    # test gives the user the truest preview.
    speaker = body.get("speaker")
    language = body.get("language")
    if not language and isinstance(speaker, str) and speaker:
        from web_ui.qwentts_voice_library import PRESET_BY_NAME  # noqa: PLC0415

        match = PRESET_BY_NAME.get(speaker)
        if match:
            language = match["language"]

    try:
        wav_bytes = synthesize_qwen3_test(
            speaker=speaker if isinstance(speaker, str) and speaker else None,
            text=text,
            language=language or "english",
            model_id=body.get("model_id") or "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            ref_audio=body.get("ref_audio") if isinstance(body.get("ref_audio"), str) else None,
            ref_text=body.get("ref_text") if isinstance(body.get("ref_text"), str) else None,
            instruct=body.get("instruct") if isinstance(body.get("instruct"), str) else None,
            seed=int(body.get("seed", -1)),
            temperature=float(body.get("temperature", 0.9)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k", 50)),
            repetition_penalty=float(body.get("repetition_penalty", 1.05)),
            device=body.get("device") or "cuda",
        )
    except Qwen3NotInstalled as e:
        raise HTTPException(
            status_code=409,
            detail={"error": "qwen3_not_installed", "message": str(e)},
        ) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": "bad_request", "message": str(e)}) from e
    except Exception as e:  # noqa: BLE001
        # Catch QwenTTSError (and anything else the binding raises) and
        # surface as a 400 with the original message. The frontend shows
        # this verbatim in a toast -- if the message mentions ABI v2
        # the UI already knows to show the "needs ABI v2" banner.
        msg = str(e) or e.__class__.__name__
        raise HTTPException(
            status_code=400,
            detail={"error": "synthesis_failed", "message": msg},
        ) from e
    return Response(content=wav_bytes, media_type="audio/wav")


@app.delete("/api/qwen3_ref_audio/{name}")
def api_qwen3_delete_ref_audio(name: str) -> dict[str, Any]:
    """Remove an uploaded reference audio file from ``voices/qwen3_refs/``.

    Used by the voice library UI's Reference voices section so the user
    can delete a stale upload. We don't validate that the file is in
    the ref-audio directory (only its basename is matched) to keep the
    endpoint simple; the matcher below restricts to that directory by
    construction.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid file name")
    from web_ui.qwentts_voice_library import REF_AUDIO_DIR  # noqa: PLC0415

    target = REF_AUDIO_DIR / name
    if not target.is_file() or REF_AUDIO_DIR not in target.resolve().parents:
        raise HTTPException(status_code=404, detail=f"Ref audio {name!r} not found")
    try:
        target.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Could not delete {name}: {e}") from e
    # If the deleted file was the currently configured ref_audio, clear it.
    current = _read_settings() or {}
    if current.get("--qwen3-tts-ref-audio") == str(target):
        current["--qwen3-tts-ref-audio"] = ""
        _write_settings(current)
    return {"ok": True, "name": name}


@app.delete("/api/voices/{name}")
def api_delete_voice(name: str) -> dict[str, Any]:
    removed = delete_voice(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Voice {name!r} not found")
    return {"ok": True, "name": name}


@app.post("/api/voices/{name}/set-active")
def api_set_active_voice(name: str) -> dict[str, Any]:
    """Write the voice name into ``--chatterbox-voice`` in the settings file.

    This is a thin convenience: the frontend can otherwise just edit the form
    field and click Save, but a dedicated endpoint makes the
    "Set active" button on a voice card a one-click action. We also flip
    ``--tts`` to ``chatterbox`` if it isn't already, so the form and the
    pipeline agree on which backend to use — without this, a user who clones
    a voice while running qwen3 would have to remember to switch the dropdown
    before saving or the chatterbox-voice flag would be silently dropped.
    """
    current = _read_settings() or {}
    current["--chatterbox-voice"] = name
    if current.get("--tts") != "chatterbox":
        current["--tts"] = "chatterbox"
    _write_settings(current)
    return {"ok": True, "name": name, "settings": current}


@app.post("/api/voices/test")
def api_voice_test(body: dict[str, Any]) -> Response:
    """Synthesize a short preview for a voice and return raw WAV bytes.

    The body shape matches the chatterbox settings (the preview uses the
    current form values so the user hears exactly the voice their robot will
    produce). Returns ``audio/wav``.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    voice = body.get("voice")
    text = body.get("text") or "Hello, this is a test of my cloned voice."
    model_variant = body.get("model_variant") or "chatterbox-turbo"
    if not isinstance(voice, str) or not voice:
        raise HTTPException(status_code=400, detail="Field 'voice' is required")
    # Pull the rest of the chatterbox params straight from the body so the
    # preview matches whatever the user has set on the form.
    param_keys = (
        "exaggeration",
        "cfg_weight",
        "temperature",
        "repetition_penalty",
        "min_p",
        "top_p",
        "top_k",
        "language_id",
    )
    params = {k: body[k] for k in param_keys if k in body and body[k] is not None}
    try:
        wav_bytes = synthesize_test(
            voice=voice,
            text=text,
            model_variant=model_variant,
            **params,
        )
    except ChatterboxNotInstalled as e:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chatterbox_not_installed",
                "install_command": _chatterbox_install_command(),
                "platform": _platform_label(),
            },
        ) from e
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Response(content=wav_bytes, media_type="audio/wav")


# Background-thread install. The frontend kicks this off and watches
# ``/ws/logs`` for the streamed output. The thread pushes each pip output
# line into the dashboard's log fanout (mirroring what ``process_manager``
# does for the pipeline subprocess).
_INSTALL_LOCK = threading.Lock()
_INSTALL_RUNNING = {"value": False}


@app.post("/api/install/chatterbox")
def api_install_chatterbox() -> dict[str, Any]:
    """Run the chatterbox install command in a background thread.

    Returns ``{"ok": True, "started": True}`` immediately. Install logs are
    pushed into the same websocket stream the user watches for the pipeline.
    The frontend watches the log for a "Successfully installed chatterbox-tts"
    marker (or the failure equivalent) to know when to close the install
    modal.
    """
    with _INSTALL_LOCK:
        if _INSTALL_RUNNING["value"]:
            raise HTTPException(status_code=409, detail="Install already in progress")
        _INSTALL_RUNNING["value"] = True

    def _publish(line: str, level: str = "info") -> None:
        """Push a log line to the same fanout the pipeline uses."""
        from web_ui.process_manager import LogLine

        loop = state._loop
        if loop is None:
            return
        try:
            log = LogLine(text=line, level=level, index=-1, timestamp=time.time())
            asyncio.run_coroutine_threadsafe(state._send_to_all(log.to_dict()), loop)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to publish install log line", exc_info=True)

    def _runner() -> None:
        try:
            cmd = _chatterbox_install_command()
            _publish("[install] $ " + cmd)
            # Stream stdout+stderr line-by-line. ``text=True`` so we don't
            # have to decode bytes; ``bufsize=1`` so the lines come in
            # promptly.
            import subprocess

            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _publish("[install] " + line.rstrip())
            rc = proc.wait()
            if rc == 0:
                _publish(
                    "[install] Done. chatterbox-tts is now installed (v0.1.7).",
                    level="info",
                )
            else:
                _publish(
                    f"[install] Failed with exit code {rc}. See the log above for details.",
                    level="error",
                )
        except Exception as e:  # noqa: BLE001
            _publish(f"[install] Crashed: {e}", level="error")
            logger.exception("Chatterbox install runner crashed")
        finally:
            with _INSTALL_LOCK:
                _INSTALL_RUNNING["value"] = False

    threading.Thread(target=_runner, daemon=True, name="chatterbox-install").start()
    return {"ok": True, "started": True, "command": _chatterbox_install_command()}


# ---- Auto-install onnx_asr for the parakeet-onnx STT backend ----------------
#
# Mirrors the chatterbox installer pattern. The parakeet-onnx backend pulls
# ``onnx_asr`` at runtime (inside the pipeline subprocess), so unlike the
# chatterbox TTS handler it's safe to install on demand without restarting the
# dashboard — the dashboard never imports onnx_asr itself.
#
# ``onnx-asr`` is pure-Python and depends on ``onnxruntime`` (which has a
# pre-built wheel for cu126 + cpu). We install with ``uv pip install`` (NOT
# ``uv sync``) so the user's locked torch + nvidia-cudnn-cu12 stack is never
# touched. Once-only: subsequent Starts see ``onnx_asr`` importable and skip.
def _onnx_asr_installed() -> bool:
    """Best-effort probe for the optional ``onnx_asr`` package.

    onnx_asr's top-level import is enough — it lazily loads its model classes.
    If the user installed a different STT backend in the meantime, we still
    return False here (caller checks the --stt flag).
    """
    try:
        import onnx_asr  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def _onnx_asr_install_command() -> str:
    """Return the shell command used to install onnx-asr.

    Intentionally minimal — no version pins, no extras. The package's own
    metadata picks the right onnxruntime wheel for the user's platform.
    """
    return "uv pip install onnx-asr"


def _parakeet_onnx_install_needed(settings: dict[str, Any]) -> bool:
    """True when the user selected --stt parakeet-onnx but onnx_asr isn't importable."""
    if settings.get("--stt") != "parakeet-onnx":
        return False
    return not _onnx_asr_installed()


# Same lock + running flag as chatterbox — keeps a manual /api/install/* call
# and the auto-install from racing each other (the chatterbox install runs
# `uv pip install transformers==5.2.0` separately, so simultaneous installs
# could clobber each other; safer to serialize).
_PARAKEET_ONNX_INSTALL_DONE: threading.Event = threading.Event()


def _ensure_parakeet_onnx_installed_async() -> None:
    """Kick off the onnx-asr install in a background thread if not already
    running. Safe to call from any context; idempotent.
    """
    with _INSTALL_LOCK:
        if _INSTALL_RUNNING["value"]:
            return
        _INSTALL_RUNNING["value"] = True
    _PARAKEET_ONNX_INSTALL_DONE.clear()

    def _publish(line: str, level: str = "info") -> None:
        from web_ui.process_manager import LogLine

        loop = state._loop
        if loop is None:
            return
        try:
            log = LogLine(text=line, level=level, index=-1, timestamp=time.time())
            asyncio.run_coroutine_threadsafe(state._send_to_all(log.to_dict()), loop)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to publish onnx-asr install log line", exc_info=True)

    def _runner() -> None:
        try:
            cmd = _onnx_asr_install_command()
            _publish(f"[install] $ {cmd}")
            import subprocess

            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _publish("[install] " + line.rstrip())
            rc = proc.wait()
            if rc == 0:
                _publish("[install] Done. onnx-asr is now installed.", level="info")
            else:
                _publish(f"[install] onnx-asr install failed (exit {rc}).", level="error")
        except Exception as e:  # noqa: BLE001
            _publish(f"[install] onnx-asr install crashed: {e}", level="error")
        finally:
            with _INSTALL_LOCK:
                _INSTALL_RUNNING["value"] = False
            _PARAKEET_ONNX_INSTALL_DONE.set()

    threading.Thread(target=_runner, daemon=True, name="parakeet-onnx-install-auto").start()


# Tracking for the chatterbox install started from `api_process_start`.
# The auto-install kicks it off in a background thread and `api_process_start`
# waits on the event before launching the pipeline. The existing
# `/api/install/chatterbox` endpoint uses the same lock so a manual install
# and the auto-install don't fight each other.
_CHATTERBOX_INSTALL_DONE: tuple[threading.Event, str] = (threading.Event(), "chatterbox")


def _ensure_chatterbox_installed_async() -> None:
    """Kick off the chatterbox install in a background thread if not already
    running. Safe to call from any context; idempotent.
    """
    with _INSTALL_LOCK:
        if _INSTALL_RUNNING["value"]:
            return
        _INSTALL_RUNNING["value"] = True
    _CHATTERBOX_INSTALL_DONE[0].clear()

    def _publish(line: str, level: str = "info") -> None:
        from web_ui.process_manager import LogLine

        loop = state._loop
        if loop is None:
            return
        try:
            log = LogLine(text=line, level=level, index=-1, timestamp=time.time())
            asyncio.run_coroutine_threadsafe(state._send_to_all(log.to_dict()), loop)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to publish install log line", exc_info=True)

    def _runner() -> None:
        try:
            cmd = _chatterbox_install_command()
            _publish("[install] $ " + cmd)
            import subprocess

            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                _publish("[install] " + line.rstrip())
            rc = proc.wait()
            if rc == 0:
                _publish("[install] Done. chatterbox-tts is now installed.", level="info")
            else:
                _publish(f"[install] Failed with exit code {rc}.", level="error")
        except Exception as e:  # noqa: BLE001
            _publish(f"[install] Crashed: {e}", level="error")
        finally:
            with _INSTALL_LOCK:
                _INSTALL_RUNNING["value"] = False
            _CHATTERBOX_INSTALL_DONE[0].set()

    threading.Thread(target=_runner, daemon=True, name="chatterbox-install-auto").start()


# ---- GPU detection + auto-install ---------------------------------------


def _schedule_dashboard_restart(delay: float = 1.5) -> None:
    """Spawn a child process that kills the current uvicorn and respawns
    a fresh one. Used after a torch wheel swap so the new torch is loaded.

    The child uses the same ``uv run --no-sync`` invocation the user
    started with, so the wheel that the user just installed survives the
    restart (otherwise ``uv sync`` would clobber it back to the lock-pinned
    version).
    """

    def _do_restart() -> None:
        time.sleep(delay)
        # Use the env var to find the same uv run the user used
        import os
        import subprocess

        cmd = os.environ.get("SPEECH_TO_SPEECH_DASHBOARD_RESTART_CMD")
        if not cmd:
            # Fall back: try to find the original command line via /proc
            try:
                with open(f"/proc/{os.getppid()}/cmdline", "rb") as f:
                    raw = f.read().decode("utf-8", "replace")
                # cmdline is null-separated; reconstruct.
                parts = raw.split("\x00")
                cmd = " ".join(p for p in parts if p)
            except Exception:  # noqa: BLE001
                cmd = "uv run --no-sync speech-to-speech-web"
        # Spawn detached: kill the parent (current dashboard) and exec the new one.
        logger.info("Restarting dashboard: %s", cmd)
        # Tell the lifespan exit handler to leave hermes alone. The
        # dashboard is restarting for a wheel install, not shutting
        # down for a reason that should take hermes with it. The user
        # still has the explicit Kill button to force hermes down.
        global _SUPPRESS_HERMES_STOP_ON_SHUTDOWN
        _SUPPRESS_HERMES_STOP_ON_SHUTDOWN = True
        try:
            subprocess.Popen(
                cmd,
                shell=True,
                cwd=REPO_ROOT,
                start_new_session=True,
                stdout=open("/tmp/dashboard.log", "ab"),
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to spawn new dashboard process")
        # Stop the current process so uvicorn exits and the new one binds.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=_do_restart, daemon=True, name="dashboard-restart").start()


# Locks + state for the background GPU install. The dashboard auto-runs
# this whenever the user picks a GPU device but the installed torch
# doesn't list the GPU's compute capability — the user just sees the
# pipeline start on GPU a moment later, no manual intervention.
_GPU_FIX_LOCK = threading.Lock()
_GPU_FIX_RUNNING = {"value": False}

# When the dashboard is restarting itself (e.g. after a torch wheel
# install), set this flag so the lifespan exit handler knows to leave
# the hermes subprocess alone. Without this, every torch restart
# also killed hermes and forced the user to re-start it. The only way
# to actually kill hermes is the explicit /api/hermes/kill endpoint
# (the red Kill button) — a graceful dashboard shutdown for a wheel
# swap is not a reason to take down the user's hermes session.
_SUPPRESS_HERMES_STOP_ON_SHUTDOWN = False


@app.get("/api/gpu/check")
def api_gpu_check() -> dict[str, Any]:
    """Return the GPU/compute-capability vs installed-torch compatibility."""
    return check_gpu().to_dict()


@app.get("/api/qwentts/pascal_wheel")
def api_qwentts_pascal_wheel() -> dict[str, Any]:
    """Status of the Pascal (sm_61) qwentts-cpp-python wheel bundle.

    Reports whether a Pascal-compatible wheel is bundled, whether the
    installed ``qwentts-cpp-python`` matches it, and the result of the
    last install attempt (cached in ``~/.cache/speech-to-speech-dashboard/``).
    Useful for the UI to surface "Pascal wheel installed" / "wheel bundled
    but not installed" badges next to the qwen3 TTS dropdown.
    """
    from web_ui.qwentts_installer import (
        _bundled_wheel_dir,
        _detect_compute_capability,
        _matching_wheel,
        _read_cache,
    )

    cc = _detect_compute_capability()
    wheel_dir = _bundled_wheel_dir()
    bundled = _matching_wheel()
    installed_version = None
    try:
        from importlib.metadata import version

        installed_version = version("qwentts-cpp-python")
    except Exception:
        pass
    return {
        "compute_capability": cc,
        "is_pascal": cc in {"6.1", "6.2"},
        "wheel_dir": str(wheel_dir),
        "bundled_wheel": str(bundled) if bundled else None,
        "installed_version": installed_version,
        "last_install": _read_cache(),
    }


@app.post("/api/gpu/fix")
def api_gpu_fix() -> dict[str, Any]:
    """Probe the GPU, install the matching torch wheel in the background.

    Returns immediately. Logs stream to the same ``/ws/logs`` websocket the
    pipeline uses. The frontend shows a "preparing GPU..." status while
    the install runs.
    """
    with _GPU_FIX_LOCK:
        if _GPU_FIX_RUNNING["value"]:
            return {"ok": True, "started": False, "already_running": True}
        report = check_gpu()
        if not report.has_gpu:
            return {"ok": False, "error": "no GPU detected"}
        if report.supported:
            return {"ok": True, "started": False, "already_supported": True}
        if report.suggested_wheel is None:
            return {"ok": False, "error": "no wheel available for this GPU"}
        wheel = report.suggested_wheel
        _GPU_FIX_RUNNING["value"] = True

    def _publish(line: str, level: str = "info") -> None:
        from web_ui.process_manager import LogLine

        loop = state._loop
        if loop is None:
            return
        try:
            log = LogLine(text=line, level=level, index=-1, timestamp=time.time())
            asyncio.run_coroutine_threadsafe(state._send_to_all(log.to_dict()), loop)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to publish gpu-fix log line", exc_info=True)

    def _runner() -> None:
        try:
            install_wheel(wheel, _publish)
            # Re-check after install — if supported now, great; if not, log it.
            new_report = check_gpu()
            if new_report.supported:
                _publish(
                    f"[gpu-fix] GPU now compatible: torch {new_report.torch_version} supports CC {new_report.gpu_cc}.",
                    level="info",
                )
            elif new_report.recommend_cpu:
                _publish(
                    "[gpu-fix] Could not find a torch wheel for this GPU. The pipeline will run on CPU.",
                    level="error",
                )
            else:
                _publish(
                    f"[gpu-fix] Still not compatible after install: "
                    f"CC {new_report.gpu_cc}, archs {new_report.torch_archs}. "
                    "The pipeline will run on CPU.",
                    level="error",
                )
        except Exception as e:  # noqa: BLE001
            _publish(f"[gpu-fix] Crashed: {e}", level="error")
            logger.exception("GPU fix runner crashed")
        finally:
            with _GPU_FIX_LOCK:
                _GPU_FIX_RUNNING["value"] = False

    threading.Thread(target=_runner, daemon=True, name="gpu-fix").start()
    return {
        "ok": True,
        "started": True,
        "command": wheel.install_command(),
        "torch_version": wheel.torch_version,
        "cuda_tag": wheel.cuda_tag,
    }


# ---- Process control -------------------------------------------------------


# Device flags the user might point at the GPU. When any of these resolves
# to "use the GPU", the dashboard checks that the installed torch supports
# the GPU's compute capability. If not, the matching wheel is installed
# in the background before the pipeline starts.
_GPU_DEVICE_FLAGS = (
    "--device",
    "--stt-device",
    "--faster-whisper-stt-device",
    "--parakeet-tdt-device",
    "--paraformer-stt-device",
    "--llm-device",
    "--qwen3-tts-device",
    "--kokoro-device",
    "--pocket-tts-device",
    "--chat-tts-device",
    "--facebook-mms-device",
    "--chatterbox-device",
)

_GPU_DEVICE_VALUES = {"cuda", "auto", "gpu"}


def _settings_request_gpu(settings: dict[str, Any]) -> bool:
    """True if any device flag in ``settings`` is set to a GPU value."""
    for flag in _GPU_DEVICE_FLAGS:
        v = settings.get(flag)
        if isinstance(v, str) and v.lower() in _GPU_DEVICE_VALUES:
            return True
    return False


def _publish_log(line: str, level: str = "info") -> None:
    """Push a log line to the dashboard's log fanout."""
    from web_ui.process_manager import LogLine

    loop = state._loop
    if loop is None:
        return
    try:
        log = LogLine(text=line, level=level, index=-1, timestamp=time.time())
        asyncio.run_coroutine_threadsafe(state._send_to_all(log.to_dict()), loop)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to publish log line", exc_info=True)


def _ensure_gpu_ready(settings: dict[str, Any], timeout_s: float = 600.0) -> tuple[dict[str, Any], str]:
    """If the user's settings point at a GPU but installed torch can't
    run on it, install the matching wheel before starting the pipeline.

    Returns ``(patched_settings, message)``. ``patched_settings`` is the
    same dict the caller passed in, possibly with device flags demoted
    from ``cuda``/``auto`` to ``cpu`` if no GPU wheel exists for this
    hardware. ``message`` is a human-readable summary the UI can show.
    """
    if not _settings_request_gpu(settings):
        return settings, "no GPU device selected"

    report = check_gpu()
    if not report.has_gpu:
        # No GPU on this box — demote any cuda/auto flags to cpu.
        for flag in _GPU_DEVICE_FLAGS:
            v = settings.get(flag)
            if isinstance(v, str) and v.lower() in _GPU_DEVICE_VALUES:
                settings[flag] = "cpu"
        return settings, "no GPU detected — running on CPU"

    # Helper: resolve each GPU flag against the report.
    def _resolve_one(flag: str) -> Optional[str]:
        v = settings.get(flag)
        if not (isinstance(v, str) and v.lower() in _GPU_DEVICE_VALUES):
            return None
        effective, wheel = resolution_for_device(v.lower(), report)
        if wheel is not None and effective == "cuda":
            # Will install.
            return "cuda"
        if effective == "cpu":
            return "cpu"
        return None

    # Check if any flag actually needs the install.
    any_needs_install = False
    for flag in _GPU_DEVICE_FLAGS:
        v = settings.get(flag)
        if not (isinstance(v, str) and v.lower() in _GPU_DEVICE_VALUES):
            continue
        eff, wheel = resolution_for_device(v.lower(), report)
        if wheel is not None and eff == "cuda":
            any_needs_install = True
            break

    if not any_needs_install:
        # Either already supported, or no wheel available — fall back to cpu
        # silently if a flag was set to cuda/auto on unsupported hardware.
        if not report.supported:
            for flag in _GPU_DEVICE_FLAGS:
                v = settings.get(flag)
                if isinstance(v, str) and v.lower() in _GPU_DEVICE_VALUES:
                    settings[flag] = "cpu"
            return settings, (
                f"GPU {report.gpu_name} (CC {report.gpu_cc}) is not supported "
                f"by any available torch wheel — running on CPU"
            )
        return settings, f"GPU ready: {report.gpu_name} (CC {report.gpu_cc})"

    # Install the matching wheel. Block until done (this is a foreground
    # operation; the UI is watching the log).
    wheel = report.suggested_wheel
    assert wheel is not None  # any_needs_install proves this
    _publish_log(
        f"[gpu-fix] Installed torch {report.torch_version} does not support "
        f"{report.gpu_name} (CC {report.gpu_cc}). Installing "
        f"torch=={wheel.torch_version} ({wheel.cuda_tag})...",
        level="info",
    )

    done = threading.Event()
    install_failed = {"value": False}

    def _publish(line: str, level: str = "info") -> None:
        _publish_log(line, level=level)

    def _runner() -> None:
        try:
            install_wheel(wheel, _publish)
        finally:
            done.set()

    with _GPU_FIX_LOCK:
        if _GPU_FIX_RUNNING["value"]:
            # Another install is already running; wait for it.
            pass
        else:
            _GPU_FIX_RUNNING["value"] = True
            threading.Thread(target=_runner, daemon=True, name="gpu-fix-on-start").start()

    if not done.wait(timeout=timeout_s):
        install_failed["value"] = True
        _publish_log(
            f"[gpu-fix] Install timed out after {timeout_s:.0f}s — falling back to CPU",
            level="error",
        )

    # The dashboard's in-process torch is now stale (the .so files on
    # disk have been replaced). The cleanest fix is to restart the
    # dashboard process so the new torch is loaded. We do this on a
    # background thread so the current request can return a useful
    # response. The next request from the UI will hit the fresh server.
    _publish_log(
        "[gpu-fix] Restarting dashboard to load the new torch...",
        level="info",
    )
    _schedule_dashboard_restart()
    # Tell the UI to reload the page. The settings are saved so the
    # user's choices are preserved.
    return (
        settings,
        f"Installed {wheel.torch_version}+{wheel.cuda_tag} for {report.gpu_name}; dashboard is restarting...",
    )


@app.post("/api/process/start")
def api_process_start(body: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = (body or {}).get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Body must include 'settings' object")
    # Auto-install onnx-asr in the background if the user picked
    # --stt parakeet-onnx but the package isn't importable yet. onnx-asr is
    # pure-Python (no native build, no torch/cudnn touch) and the dashboard
    # never imports it itself, so this doesn't require a dashboard restart
    # the way the torch-wheel swap does.
    if _parakeet_onnx_install_needed(settings):
        _ensure_parakeet_onnx_installed_async()
        if not _PARAKEET_ONNX_INSTALL_DONE.wait(timeout=600):
            _publish_log(
                "[install] onnx-asr install timed out — continuing anyway",
                level="error",
            )
        # If the install failed and the package is still missing, demote
        # back to parakeet-tdt so the pipeline can still start instead of
        # crashing with ImportError.
        if not _onnx_asr_installed():
            _publish_log(
                "[install] onnx-asr install didn't complete; falling back to parakeet-tdt.",
                level="error",
            )
            settings["--stt"] = "parakeet-tdt"
    # Auto-install chatterbox in the background if the user picked it but
    # the package isn't there yet. We don't return 409 — the user pressed
    # Start, the dashboard handles the rest. The pipeline starts as soon
    # as the install completes.
    chatterbox_needed = _chatterbox_install_needed(settings)
    if chatterbox_needed:
        _ensure_chatterbox_installed_async()
        # Wait for chatterbox to finish before the GPU check (which
        # might also install) — otherwise the two installs race and
        # one can clobber the other.
        if not _CHATTERBOX_INSTALL_DONE[0].wait(timeout=600):
            _publish_log(
                "[install] Chatterbox install timed out — continuing anyway",
                level="error",
            )
    # Auto-install the matching torch wheel if the user picked GPU.
    # If a torch install is needed, this also schedules a dashboard
    # restart so the new torch is loaded — the pipeline then starts
    # on the next user click.
    settings, gpu_msg = _ensure_gpu_ready(settings)
    if "restarting" in gpu_msg.lower():
        # The dashboard is about to restart. Tell the UI to retry.
        return {
            "ok": False,
            "restarting": True,
            "message": gpu_msg,
            "retry_after_ms": 4000,
        }
    # If the chatterbox install still didn't take, demote to pocket (or
    # whatever the default TTS is) so the pipeline can start.
    if chatterbox_needed and _chatterbox_install_needed(settings):
        _publish_log(
            "[install] Chatterbox install didn't complete; starting with the default TTS instead.",
            level="error",
        )
        settings["--tts"] = "pocket"
    try:
        state.process.start(settings)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    # Fresh pipeline = fresh hermes session (plan §5: one session per
    # pipeline lifetime). Reset BEFORE the keepalive arm so the proxy
    # module's next /v1/chat/completions call gets a new uuid.
    _hermes_proxy_reset_session()
    # Push the latest hermes config into the proxy's in-memory cache
    # so the very first LLM request doesn't pay a disk read. The
    # settings file is the source of truth; we re-push on every write.
    _hermes_proxy_refresh_config(_read_settings().get("hermes"))
    # Spin up the Ollama keepalive pinger using the freshly-saved settings.
    # We do this AFTER the pipeline subprocess has been started so a slow
    # pinger startup can't delay the pipeline's first LLM call. The pinger
    # itself is a no-op if the user picked a non-Ollama backend or left
    # ``--llm-keepalive`` at its default "0".
    # 0.3.2+: the pinger call is a no-op; the actual warmup happens via
    # the one-shot request below. Both run after Start so a slow Ollama
    # load doesn't block the pipeline.
    try:
        llm_keepaliver.update_from_settings(settings)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update llm keepalive from process start")
    # Arm the warmup trigger: fire the dashboard's Ollama keepalive the
    # instant the pipeline logs ``warmed up!`` (which is the deterministic
    # signal that the pipeline's own startup warmup has just been
    # processed by Ollama). Our subsequent ``/api/generate`` will be the
    # next request Ollama handles, so ``options.num_ctx`` lands reliably.
    _arm_warmup_trigger(settings)
    status = state.process.get_status()
    status["gpu_message"] = gpu_msg
    return {"ok": True, "status": status}


@app.post("/api/process/stop")
def api_process_stop() -> dict[str, Any]:
    state.process.stop()
    # Polite stop: do NOT reset the hermes session id. Per plan §6,
    # polite stop preserves hermes context so the user can resume
    # their conversation on the next Start with continuity.
    # Stop the Ollama keepalive pinger too — there's no pipeline to keep
    # the model warm for, and the user's next Start may pick a different
    # backend / model.
    try:
        llm_keepaliver.stop()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to stop llm keepalive on process stop")
    return {"ok": True, "status": state.process.get_status()}


@app.post("/api/process/restart")
def api_process_restart(body: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = (body or {}).get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Body must include 'settings' object")
    state.process.restart(settings)
    # Fresh pipeline = fresh hermes session (same as Start above).
    _hermes_proxy_reset_session()
    _hermes_proxy_refresh_config(_read_settings().get("hermes"))
    # Same keepalive refresh as Start — the user may have edited
    # ``--llm-keepalive`` since the last start.
    try:
        llm_keepaliver.update_from_settings(settings)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update llm keepalive on restart")
    # Arm the warmup trigger: same as Start — fire the dashboard's Ollama
    # keepalive the instant the pipeline logs ``warmed up!``. Deterministic
    # ordering with the pipeline's startup warmup, no polling, no fixed
    # wait.
    _arm_warmup_trigger(settings)
    return {"ok": True, "status": state.process.get_status()}


@app.get("/api/process/status")
def api_process_status() -> dict[str, Any]:
    return state.process.get_status()


# ---- Hermes Agent ---------------------------------------------------------
#
# The Hermes tab surfaces one ``hermes gateway run --accept-hooks``
# subprocess that exposes hermes-agent as an OpenAI-compatible HTTP
# server (default port 8642). The dashboard owns the subprocess
# lifecycle here; the pipeline then points its LLM slot at this same
# URL via the "Hermes Agent" toggle in the LLM tab.
#
# All hermes-related routes live in this section. Anything that would
# affect a different tab (e.g. injecting X-Hermes-Session-Id into the
# pipeline env) lives in the corresponding tab's section instead.


@app.get("/api/hermes/status")
def api_hermes_status() -> dict[str, Any]:
    """Snapshot of the hermes subprocess for the Hermes tab status panel.

    Always fast — the heavy lifting (the ``/health`` probe) runs on a
    background thread inside :class:`HermesProcess`.
    """
    settings = _read_settings() or {}
    return state.hermes.get_status(settings)


@app.post("/api/hermes/start")
def api_hermes_start(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start the hermes-agent subprocess.

    Persists the (possibly just-generated) ``api_key`` to
    ``web_ui_settings.json`` so subsequent restarts reuse the same key
    and the pipeline's LLM slot can pick it up.

    Body shape (all optional):
        - ``port`` (int): override the api_server port. Default 8642.
        - ``host`` (str): bind host. Default 127.0.0.1.
        - ``model_name`` (str): the model name hermes should advertise
          on ``/v1/models``. The user picks this inside hermes; the
          dashboard just shows whatever is configured.
        - ``log_level`` (str): one of ``default``, ``verbose``, ``debug``.
          Controls the ``-v`` / ``-vv`` flags passed to ``hermes gateway run``.

    Returns ``{"ok": True, "status": <status dict>}`` on success, or
    409 if hermes is already running.
    """
    current = _read_settings() or {}
    # Apply any inline overrides from the body. Body wins over the
    # persisted file so the UI can edit and Start without a Save click.
    overrides: dict[str, Any] = {}
    if isinstance(body, dict):
        for k in ("port", "host", "model_name", "log_level"):
            v = body.get(k)
            if v is not None:
                overrides[k] = v
    if overrides:
        hermes_cfg = current.setdefault("hermes", {})
        hermes_cfg.update(overrides)

    # Generate a key if missing. The ``ensure_api_key`` helper mutates
    # the dict in place and returns the value; we write the file right
    # after so the key survives a process restart.
    HermesProcess.ensure_api_key(current)
    _write_settings(current)

    try:
        state.hermes.start(current)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except FileNotFoundError as e:
        # The hermes CLI is not on PATH. Surface a clean 400 with the
        # same message the user would see in the dashboard log so the
        # UI can render it verbatim.
        raise HTTPException(status_code=400, detail=str(e)) from e
    # Push the latest hermes config (host/port/api_key) into the
    # proxy's in-memory cache so the next LLM request doesn't pay a
    # disk read. We do this AFTER start so the subprocess is up
    # before the proxy gets its first call.
    _hermes_proxy_refresh_config(_read_settings().get("hermes"))
    return {"ok": True, "status": state.hermes.get_status(current)}


@app.post("/api/hermes/stop")
def api_hermes_stop() -> dict[str, Any]:
    """Politely stop the hermes subprocess (SIGTERM, escalating to SIGKILL).

    No-op if not running. The hermes session context is preserved on
    the hermes side (the api_server keeps the session in memory across
    a stop+start only if the user picks the same session id on both
    sides; the dashboard does that via the pipeline's
    ``X-Hermes-Session-Id`` header).
    """
    state.hermes.stop()
    return {"ok": True, "status": state.hermes.get_status(_read_settings() or {})}


@app.post("/api/hermes/cancel")
def api_hermes_cancel() -> dict[str, Any]:
    """Send a cancel signal to the running hermes agent session.

    Best-effort: if no run is in flight, hermes returns 404 and we
    treat that as a no-op success. This is what the amber "Cancel
    hermes" button maps to.
    """
    state.hermes.cancel()
    return {"ok": True, "status": state.hermes.get_status(_read_settings() or {})}


@app.post("/api/hermes/kill")
def api_hermes_kill(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Hard-kill the hermes subprocess (SIGKILL, immediate).

    Requires ``confirm: true`` in the body to prevent accidental
    clicks. Herme context is lost; the next start gives hermes a
    fresh process. The session id (if any) the pipeline was using
    becomes stale — the next /v1/chat/completions call with that id
    will return a 404 until the pipeline sends a fresh id.
    """
    confirm = bool((body or {}).get("confirm"))
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Set confirm=true in the body to kill hermes. This is a destructive action.",
        )
    state.hermes.kill()
    # Hard kill: the old session id is orphaned on the hermes side
    # (a fresh hermes process has no memory of it). Force a new id
    # on the next request so the pipeline doesn't get spurious 404s
    # until the user manually restarts.
    _hermes_proxy_reset_session()
    return {"ok": True, "status": state.hermes.get_status(_read_settings() or {})}


@app.post("/api/hermes/reset_session")
def api_hermes_reset_session() -> dict[str, Any]:
    """Drop the dashboard's current hermes session id and mint a new one.

    This is the practical "compress context" lever for v1: hermes-agent
    does not currently expose a documented ``compress_context`` tool
    (verified via search of NousResearch/hermes-agent docs, 2026-07),
    so we can't auto-summarize the conversation. Rotating the session
    id has the same effect from the user's perspective: the next LLM
    call lands on a fresh hermes conversation slot, the robot starts
    talking with no memory of the previous exchange.

    Plan §5 reserved this slot for an auto-scheduler; for v1 the
    rotation is manual via the Hermes tab's "Reset session" button.
    When upstream ships a compress tool, swap this endpoint to call
    it instead of (or in addition to) the id rotation.
    """
    _hermes_proxy_reset_session()
    return {"ok": True, "session_id": _get_or_create_proxy_session_id()}


def _get_or_create_proxy_session_id() -> str:
    """Read the proxy's current session id without triggering a reset."""
    from web_ui.hermes_proxy import get_or_create_session_id as _g

    return _g()


@app.get("/api/hermes/logs")
def api_hermes_logs(since: int = Query(default=0, ge=0)) -> dict[str, Any]:
    """Recent log lines from the hermes subprocess.

    Polled by the Hermes tab's logs panel. The client passes the
    last-seen ``index`` so reconnect-after-page-refresh catches up
    cleanly without re-fetching the whole buffer.
    """
    return {"lines": state.hermes.get_log_buffer(since=since)}


@app.put("/api/hermes/filler")
def api_hermes_filler(body: dict[str, Any]) -> dict[str, Any]:
    """Update the filler audio configuration.

    Body shape:
        - ``enabled`` (bool, optional): turn filler audio on/off.
        - ``phrases`` (list[str], optional): one phrase per line. Up
          to 10 phrases are kept; anything beyond that is dropped
          silently. Empty strings are filtered.

    Persists into ``web_ui_settings.json["hermes"]``. The pipeline
    consumer (filler audio scheduler, task #9) reads this on every
    turn.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    current = _read_settings() or {}
    hermes_cfg = current.setdefault("hermes", {})
    if "enabled" in body:
        hermes_cfg["filler_enabled"] = bool(body["enabled"])
    if "phrases" in body:
        raw = body["phrases"]
        if not isinstance(raw, list):
            raise HTTPException(status_code=400, detail="phrases must be a list of strings")
        # Cap at 20, strip whitespace, drop empties. The UI shows one
        # box per phrase and limits additions to 20 total.
        phrases = [str(p).strip() for p in raw if isinstance(p, str) and str(p).strip()]
        hermes_cfg["filler_phrases"] = phrases[:20]
    if "compress_context_every_n_turns" in body:
        try:
            n = int(body["compress_context_every_n_turns"])
            # 0 disables the scheduler entirely; negative is invalid.
            # We clamp to [0, 1000] so a typo can't turn the
            # scheduler into a tight loop.
            n = max(0, min(1000, n))
            hermes_cfg["compress_context_every_n_turns"] = n
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="compress_context_every_n_turns must be a non-negative integer",
            ) from None
    _write_settings(current)
    return {
        "ok": True,
        "filler": {
            "enabled": hermes_cfg.get("filler_enabled", True),
            "phrases": hermes_cfg.get("filler_phrases", []),
            "compress_context_every_n_turns": hermes_cfg.get("compress_context_every_n_turns", 20),
        },
    }


@app.get("/api/hermes/filler")
def api_hermes_get_filler() -> dict[str, Any]:
    """Read the current filler audio configuration.

    Used by the Hermes tab's filler section to render the textarea
    + toggle on first load. Mirrors :func:`api_hermes_filler` so the
    two endpoints can be polled independently.
    """
    current = _read_settings() or {}
    hermes_cfg = current.get("hermes") or {}
    return {
        "enabled": hermes_cfg.get("filler_enabled", True),
        "phrases": hermes_cfg.get("filler_phrases", []),
        "compress_context_every_n_turns": hermes_cfg.get("compress_context_every_n_turns", 20),
    }


@app.get("/api/hermes/models")
def api_hermes_models() -> dict[str, Any]:
    """Proxy hermes's ``/v1/models`` so the UI can show the active model.

    We can't just trust the user-typed ``model_name`` — hermes may
    reject it (typo, model not loaded, etc.) and the api_server
    echoes the resolved model on its own endpoint. The Hermes tab's
    status panel shows whatever hermes actually reports.
    """
    import httpx

    settings = _read_settings() or {}
    cfg = HermesProcess.resolve_config(settings)
    if not state.hermes.is_running():
        return {"models": [], "error": "hermes not running"}
    url = f"http://{cfg['host']}:{cfg['port']}/v1/models"
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(
                url,
                headers={"Authorization": f"Bearer {cfg['api_key']}"},
            )
        if not (200 <= r.status_code < 300):
            return {
                "models": [],
                "error": f"HTTP {r.status_code}: {r.text[:200]}",
            }
        data = r.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {"models": [], "error": "unexpected response shape"}
        models = []
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("id"), str):
                models.append({"id": it["id"]})
        return {"models": models, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"models": [], "error": f"{type(e).__name__}: {e}"}


@app.post("/api/hermes/chat")
def api_hermes_chat(body: dict[str, Any]) -> Response:
    """Stream a chat turn to hermes (SSE) for the Hermes tab's console.

    Body shape:
        - ``message`` (str, required): the user's text.
        - ``session_id`` (str, optional): ``X-Hermes-Session-Id`` to
          send with the request. Defaults to the dashboard-managed
          session id (one per pipeline lifetime, see task #8).
        - ``model`` (str, optional): model name to put in the
          request body. Defaults to whatever the user has configured
          in ``web_ui_settings.json["hermes"]["model_name"]``.

    Returns a Server-Sent Events stream: each line is the literal
    hermes SSE chunk (``data: {...}\\n\\n``), with the dashboard
    adding ``event: chunk`` headers so the browser's ``EventSource``
    can route them. The console panel renders each ``choices[0].delta.content``
    in order to produce the streaming text effect.

    The pipeline never uses this endpoint — it's the Hermes tab's
    text-only console, separate from the voice path.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be an object")
    message = body.get("message")
    if not isinstance(message, str) or not message.strip():
        raise HTTPException(status_code=400, detail="Field 'message' is required")
    settings = _read_settings() or {}
    cfg = HermesProcess.resolve_config(settings)
    if not state.hermes.is_running():
        raise HTTPException(status_code=409, detail="Hermes is not running")
    session_id = body.get("session_id") or cfg.get("api_key", "") + "-console"
    if not isinstance(session_id, str) or not session_id:
        session_id = "dashboard-console"
    model = body.get("model") or cfg.get("model_name") or "hermes-agent"

    import httpx

    url = f"http://{cfg['host']}:{cfg['port']}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": True,
    }

    def _event_stream():
        # Stream hermes's SSE chunks verbatim. If hermes returns a
        # non-2xx (e.g. session expired), we surface the first 200
        # chars of the body as an SSE error event so the UI can show
        # a useful message instead of just hanging.
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {cfg['api_key']}",
                        "Content-Type": "application/json",
                        "X-Hermes-Session-Id": session_id,
                        "Accept": "text/event-stream",
                    },
                ) as r:
                    if not (200 <= r.status_code < 300):
                        body_snippet = r.read().decode("utf-8", "replace")[:200]
                        yield (
                            f'event: error\ndata: {{"status": {r.status_code}, "body": {json.dumps(body_snippet)}}}\n\n'
                        )
                        return
                    for line in r.iter_lines():
                        if not line:
                            # Blank line — SSE event boundary. We
                            # forward as-is so the browser's
                            # EventSource picks it up.
                            yield "\n"
                            continue
                        # hermes SSE lines look like:
                        #   data: {"id": "...", "choices": [...]}
                        #   data: [DONE]
                        # Forward verbatim. The UI parses ``data:``
                        # lines to extract ``choices[0].delta.content``.
                        yield line + "\n"
                        # SSE event boundary (one blank line after the
                        # ``data:`` line). EventSource consumes this
                        # implicitly when our writes are well-formed.
        except Exception as e:  # noqa: BLE001
            yield (f'event: error\ndata: {{"error": {json.dumps(f"{type(e).__name__}: {e}")}}}\n\n')

    # ``Response`` calls ``.render(content)`` which tries to ``.encode()`` the
    # content; passing a generator crashes with ``'generator' object has no
    # attribute 'encode'``. ``StreamingResponse`` is the right tool here --
    # it forwards each yielded chunk to the client as-is, which is exactly
    # what an SSE passthrough needs. (Same fix we applied to the proxy in
    # ``hermes_proxy.py``.)
    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering
        },
    )


# ---- LLM keepalive (Ollama) ----------------------------------------------
#
# When the user picks a remote OpenAI-compatible LLM backend
# (``chat-completions`` or ``responses-api``, the slots that point at
# Ollama / vLLM / llama.cpp) and sets ``--llm-keepalive`` to anything
# other than "0", we run a dashboard-side pinger that posts a tiny
# /v1/chat/completions request to that endpoint with
# ``keep_alive: <value>`` in the body. This resets Ollama's idle timer
# so a long conversation gap doesn't pay a ~20s reload penalty before
# the next reply.
#
# The pinger itself lives in ``web_ui.llm_keepalive.py``; the surface
# here is intentionally tiny -- three endpoints to inspect + control
# it manually if the UI ever needs that. The ``/api/settings`` and
# process-start/stop/restart hooks already drive it from the main
# settings flow, so most users will never hit these directly.


def _keepalive_status_to_dict(s: KeepaliveStatus) -> dict[str, Any]:
    """Convert a KeepaliveStatus dataclass into a JSON-safe dict.

    The ``config`` field carries another dataclass; flatten it here so
    FastAPI's JSON encoder can serialize it without complaint.
    """
    out = {
        "running": s.running,
        "last_ping_at": s.last_ping_at,
        "last_ping_ok": s.last_ping_ok,
        "last_ping_error": s.last_ping_error,
        "next_ping_at": s.next_ping_at,
        "ping_count": s.ping_count,
        "ping_failures": s.ping_failures,
        "started_at": s.started_at,
        "thread_id": s.thread_id,
        "config": None,
    }
    if s.config is not None:
        c = s.config
        out["config"] = {
            "base_url": c.base_url,
            "api_key": "***" if c.api_key else "",
            "model_name": c.model_name,
            "keepalive": c.keepalive,
            "ping_interval_s": c.ping_interval_s,
        }
    return out


@app.get("/api/llm-keepalive/status")
def api_llm_keepalive_status() -> dict[str, Any]:
    """Snapshot of the Ollama keepalive pinger for the LLM-tab badge.

    Returns the full status (running, ping cadence, last error, etc.)
    plus the active config (with the API key masked). Safe to poll.
    """
    return _keepalive_status_to_dict(llm_keepaliver.status())


@app.post("/api/llm-keepalive/restart")
def api_llm_keepalive_restart(body: dict[str, Any] | None = None) -> dict[str, Any]:
    """Restart the keepalive pinger with the supplied config.

    Useful for "I just changed --llm-keepalive in the form but haven't
    pressed Save yet" — the UI can POST a partial config to apply it
    immediately without rewriting the settings file. The actual settings
    persistence happens via the regular ``/api/settings`` flow.
    """
    settings = (body or {}).get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Body must include 'settings' object")
    try:
        llm_keepaliver.update_from_settings(settings)
    except Exception as e:  # noqa: BLE001
        logger.exception("Failed to restart llm keepalive pinger")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"ok": True, "status": _keepalive_status_to_dict(llm_keepaliver.status())}


@app.post("/api/llm-keepalive/stop")
def api_llm_keepalive_stop() -> dict[str, Any]:
    """Stop the keepalive pinger immediately (no-op if not running)."""
    llm_keepaliver.stop()
    return {"ok": True, "status": _keepalive_status_to_dict(llm_keepaliver.status())}


@app.post("/api/process/unload_tts")
async def api_process_unload_tts() -> dict[str, Any]:
    """Drop the TTS model from RAM on every pipeline unit.

    Proxies the request to the pipeline's ``POST /v1/admin/unload_tts``,
    which broadcasts an ``UNLOAD_TTS`` control message into each unit's
    input queue. The TTS handler's ``on_unload`` releases the model; the
    model reloads transparently on the next TTS request.
    """
    import httpx

    status = state.process.get_status()
    if not status["running"]:
        raise HTTPException(status_code=409, detail="Pipeline is not running")

    argv = status["argv"]
    ws_port = 8765  # default
    if "--ws-port" in argv:
        i = argv.index("--ws-port")
        if i + 1 < len(argv):
            try:
                ws_port = int(argv[i + 1])
            except ValueError:
                pass

    url = f"http://127.0.0.1:{ws_port}/v1/admin/unload_tts"
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(url)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not reach pipeline at {url}: {e}") from e

    if r.status_code == 200:
        return r.json()
    raise HTTPException(status_code=r.status_code, detail=r.text)


# ---- Logs ------------------------------------------------------------------


@app.get("/api/logs")
def api_logs(since: int = Query(default=0, ge=0)) -> dict[str, Any]:
    return {"lines": state.process.get_log_buffer(since=since)}


@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    await websocket.accept()
    state.websocket_clients.add(websocket)
    try:
        # Send the existing buffer immediately so the client is caught up.
        for line in state.process.get_log_buffer(0):
            await websocket.send_text(json.dumps(line))
        while True:
            # The client doesn't need to send us anything; we just keep the
            # socket alive. Receiving is a heartbeat.
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send a ping-style no-op to keep the connection healthy.
                await websocket.send_text(json.dumps({"__ping__": True}))
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        logger.exception("Websocket error")
    finally:
        state.websocket_clients.discard(websocket)


# ---- System + Pool + Themes ------------------------------------------------


@app.get("/api/system")
def api_system() -> dict[str, Any]:
    """Lightweight system info for the status bar. Best-effort."""
    import psutil  # type: ignore[import-not-found]  # optional dep, may be missing

    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(REPO_ROOT))

    gpu = _read_gpu_usage()

    return {
        "cpu_percent": cpu,
        "ram_percent": mem.percent,
        "ram_used_gb": round(mem.used / 1024**3, 2),
        "ram_total_gb": round(mem.total / 1024**3, 2),
        "disk_percent": disk.percent,
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "gpu": gpu,
    }


def _read_gpu_usage() -> Optional[dict[str, Any]]:
    """Best-effort NVIDIA GPU usage via nvidia-smi. Returns None if unavailable."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
        ).decode("utf-8", "replace")
    except (subprocess.SubprocessError, OSError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            util, mem_used, mem_total = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        gpus.append(
            {
                "name": parts[3],
                "util_percent": util,
                "mem_used_mb": mem_used,
                "mem_total_mb": mem_total,
            }
        )
    return {"gpus": gpus} if gpus else None


@app.get("/api/pool")
async def api_pool() -> dict[str, Any]:
    """Proxy the pipeline's existing /v1/pool status endpoint.

    Returns ``{"available": false}`` if the pipeline isn't running in
    realtime mode or the pool endpoint isn't reachable.
    """
    import httpx

    status = state.process.get_status()
    if not status["running"]:
        return {"available": False, "reason": "pipeline not running"}

    # The pipeline exposes its port on whatever was passed via --ws-port.
    argv = status["argv"]
    ws_port = 8765  # default
    if "--ws-port" in argv:
        i = argv.index("--ws-port")
        if i + 1 < len(argv):
            try:
                ws_port = int(argv[i + 1])
            except ValueError:
                pass

    url = f"http://127.0.0.1:{ws_port}/v1/pool"
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            r = await client.get(url)
            if r.status_code == 200:
                return {"available": True, "data": r.json(), "url": url}
            return {"available": False, "reason": f"HTTP {r.status_code}", "url": url}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": str(e), "url": url}


@app.get("/api/themes")
def api_themes() -> dict[str, Any]:
    if not THEMES_DIR.exists():
        return {"themes": []}
    themes = sorted(p.stem for p in THEMES_DIR.glob("*.css"))
    return {"themes": themes}


# ---- Version ---------------------------------------------------------------

# Single source of truth for the dashboard version. The frontend fetches
# this on load and renders it in the title bar, so bumping the number in
# web_ui/__init__.py is enough to make the new version visible to users.


@app.get("/api/version")
def api_version() -> dict[str, str]:
    return {"version": _DASHBOARD_VERSION}


# ---- Guide -----------------------------------------------------------------


@app.get("/api/guide")
def api_guide() -> Response:
    if not GUIDE_PATH.exists():
        return Response(content="Guide not found", media_type="text/plain", status_code=404)
    return FileResponse(GUIDE_PATH, media_type="text/markdown")


# ---- Shutdown --------------------------------------------------------------


@app.post("/api/shutdown")
def api_shutdown() -> dict[str, Any]:
    """Stop the subprocess and ask the uvicorn server to exit."""
    try:
        state.process.stop()
    except Exception:  # noqa: BLE001
        logger.exception("Error stopping pipeline during shutdown")

    # Schedule server exit on a background thread so the response gets sent
    # before uvicorn tears down.
    def _do_shutdown() -> None:
        import time

        time.sleep(0.2)
        # SIGTERM uvicorn (works on POSIX). On Windows uvicorn also handles
        # KeyboardInterrupt cleanly. We use os.kill so we don't depend on
        # the current process being in the main thread.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to signal shutdown")
            sys.exit(0)

    threading.Thread(target=_do_shutdown, daemon=True).start()
    return {"ok": True, "shutting_down": True}


# ---- Static + root ---------------------------------------------------------


# Themes are exposed as static files so the frontend can swap the active
# stylesheet without a page reload.
if THEMES_DIR.exists():
    app.mount("/static/themes", StaticFiles(directory=str(THEMES_DIR)), name="themes")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# The example settings file lives at the repo root (so users who clone
# the repo find it next to ``web_ui_settings.json`` and can `cp` it into
# place). Mounting the repo root under ``/static-repo/`` lets the
# dashboard's "Load Example" button fetch it without an extra API hop.
# Path is namespaced to avoid clashing with the dashboard's own
# ``/static/`` mount.
if REPO_ROOT.exists():
    app.mount(
        "/static-repo",
        StaticFiles(directory=str(REPO_ROOT), check_dir=False),
        name="repo-root",
    )


@app.get("/")
def root_index() -> Response:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "index.html not found"}, status_code=500)
    return FileResponse(index)


# ---- Entrypoint ------------------------------------------------------------


def main() -> None:
    """Console-script entry point: ``speech-to-speech-web``."""
    logging.basicConfig(
        level=os.environ.get("SPEECH_TO_SPEECH_WEB_LOG_LEVEL", "info").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    host = os.environ.get("SPEECH_TO_SPEECH_WEB_HOST", DEFAULT_HOST)
    port = int(os.environ.get("SPEECH_TO_SPEECH_WEB_PORT", DEFAULT_PORT))

    logger.info("Starting speech-to-speech dashboard on http://%s:%d", host, port)
    logger.info("Settings file: %s", SETTINGS_PATH)
    logger.info("Repo root: %s", REPO_ROOT)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
