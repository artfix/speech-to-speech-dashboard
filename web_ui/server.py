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
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from web_ui import __version__ as _DASHBOARD_VERSION
from web_ui.process_manager import LogLine, PipelineProcess
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
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read %s: %s", SETTINGS_PATH, e)
        return None


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
        self.websocket_clients: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def broadcast(self, line: LogLine) -> None:
        """Push a log line to all connected websocket clients.

        Called from the process manager's fanout thread; we hop to the asyncio
        loop to actually send. Each call is a no-op if there are no clients.
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


# ----------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state._loop = asyncio.get_running_loop()
    # Subscribe the broadcast callback to log fanout.
    state.process.subscribe(state.broadcast)
    try:
        yield
    finally:
        try:
            state.process.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Error stopping pipeline on shutdown")
        state.process.unsubscribe(state.broadcast)


app = FastAPI(title="speech-to-speech dashboard", lifespan=lifespan)


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
    ``torchaudio==2.6.0`` strictly, which would downgrade the rest of this
    project on most platforms. So we install ``chatterbox-tts`` with
    ``--no-deps`` and then re-add the chatterbox-specific runtime deps
    (s3tokenizer, conformer, diffusers, etc.) that the package expects but
    pip cannot resolve alongside the project's other pins. The
    ``transformers==5.2.0`` and ``torchaudio==2.6.0`` lines are intentionally
    NOT included -- the project's existing pins (which work with the
    chatterbox code in practice) take precedence.

    The companion packages (s3tokenizer, conformer, etc.) are installed
    WITHOUT ``--no-deps`` so they pull in their normal transitive deps
    (ml_dtypes for onnx, etc.). Without this, onnx imports fail at
    runtime with ``ModuleNotFoundError: No module named 'ml_dtypes'``,
    which makes the dashboard's "_chatterbox_installed" probe fail
    even after a successful install.
    """
    return (
        "uv pip install --no-deps chatterbox-tts==0.1.7 && "
        "uv pip install s3tokenizer conformer==0.3.2 resemble-perth "
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
            log = LogLine(text=line, level=level, ts=time.time())
            asyncio.run_coroutine_threadsafe(
                state._send_to_all(log.to_dict()), loop
            )
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


# ---- Process control -------------------------------------------------------


@app.post("/api/process/start")
def api_process_start(body: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = (body or {}).get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Body must include 'settings' object")
    install_cmd = _chatterbox_install_needed(settings)
    if install_cmd:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "chatterbox_not_installed",
                "install_command": install_cmd,
                "platform": _platform_label(),
            },
        )
    try:
        state.process.start(settings)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"ok": True, "status": state.process.get_status()}


@app.post("/api/process/stop")
def api_process_stop() -> dict[str, Any]:
    state.process.stop()
    return {"ok": True, "status": state.process.get_status()}


@app.post("/api/process/restart")
def api_process_restart(body: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = (body or {}).get("settings")
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Body must include 'settings' object")
    state.process.restart(settings)
    return {"ok": True, "status": state.process.get_status()}


@app.get("/api/process/status")
def api_process_status() -> dict[str, Any]:
    return state.process.get_status()


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
