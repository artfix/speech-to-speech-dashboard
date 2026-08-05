"""Subprocess lifecycle for the speech-to-speech pipeline.

The dashboard never imports the pipeline. Instead, it spawns
``python -m speech_to_speech.s2s_pipeline <args>`` as a subprocess, captures its
combined stdout+stderr, and serves those lines to the browser over a websocket.

Responsibilities of this module:

- Build a ``sys.argv`` from a settings dict (delegated to
  :mod:`web_ui.settings_schema`).
- Spawn the subprocess with a list argv (no shell, cross-platform).
- Read its output line by line in a background thread.
- Classify each line by log level and keep a ring buffer of the last
  ``BUFFER_SIZE`` lines for the UI to fetch on reconnect.
- Fan new lines out to subscribers (the websocket layer).
- Stop / restart cleanly across Windows and POSIX.

The subprocess inherits the parent's environment, with the dashboard's
``env`` setting (a list of ``KEY=VALUE`` strings) merged on top.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TextIO

from web_ui.settings_schema import build_argv

logger = logging.getLogger(__name__)


BUFFER_SIZE = 5000
STOP_TIMEOUT_S = 5.0

# Directory under the repo root where each pipeline run's captured stdout/stderr
# is written. Persisted logs survive dashboard crashes and restarts.
LOG_DIR_NAME = "logs"


def _default_ld_library_path() -> str:
    """Collect the nvidia-*/lib/ paths from this interpreter's site-packages.

    The torch 2.6.0+cu126 wheel we install for Pascal (sm_61) ships its
    CUDA / cuDNN / cuSPARSE / NCCL / nvshmem runtime libs under
    ``site-packages/nvidia/<name>/lib/`` -- five of them: ``cu13``,
    ``cudnn``, ``cusparselt``, ``nccl``, ``nvshmem``. ``import torch``
    resolves those via dlopen, which only consults ``LD_LIBRARY_PATH``
    and the system ldconfig cache -- it does not look inside
    site-packages. On the user's box none of those paths are otherwise
    on the linker search path, so the first ``import torch`` inside
    the pipeline subprocess dies with::

        ImportError: libcusparseLt.so.0: cannot open shared object file:
        No such file or directory

    even though the file is sitting right there. We assemble those
    paths here and prepend them to ``LD_LIBRARY_PATH`` for the
    subprocess (and for the dashboard process itself, so the qwentts
    probe API can ``import qwentts_cpp``).

    Cross-platform: on macOS the nvidia dir never exists, so this
    returns ``""`` and we leave the env alone.
    """
    try:
        import site  # noqa: PLC0415

        sp_paths = [Path(p) for p in site.getsitepackages()]
    except Exception:  # noqa: BLE001
        sp_paths = [Path(p) for p in sys.path if "site-packages" in p]

    out: list[str] = []
    for sp in sp_paths:
        nvidia = sp / "nvidia"
        if not nvidia.is_dir():
            continue
        for sub in sorted(nvidia.iterdir()):
            lib = sub / "lib"
            if lib.is_dir():
                out.append(str(lib))
    return ":".join(out)


# Set LD_LIBRARY_PATH in our own process env the first time this module
# is imported, IF the user hasn't already set it. Idempotent: re-imports
# are no-ops because we only set when the key is missing. The dashboard
# doesn't import torch at startup, so this only matters for handlers
# that probe GPU state at request time (e.g. /api/qwentts/pascal_wheel).
#
# Critical: ``os.environ`` is a Python dict that mirrors the C runtime's
# env table at process startup. Mutating it does NOT change the C
# runtime's view -- and dlopen (which torch's __init__.py calls into) is
# a C-level API that consults the C env, not Python's. So we have to
# call libc ``setenv(3)`` ourselves via ctypes. Otherwise the env looks
# fine in Python but in-process handlers still fail with ``ImportError:
# libcusparseLt.so.0``. (For the SUBprocess the dashboard spawns, the
# fix lives in ``_build_env`` below -- that path copies ``os.environ``
# into a fresh env dict that ``Popen`` hands to the child, and the
# child's C runtime picks up LD_LIBRARY_PATH via execve.)
_DEFAULT_LD = _default_ld_library_path()
if _DEFAULT_LD and "LD_LIBRARY_PATH" not in os.environ:
    os.environ["LD_LIBRARY_PATH"] = _DEFAULT_LD
    try:
        import ctypes  # noqa: PLC0415

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        # setenv(name, value, overwrite=1). Explicit argtypes/rtype --
        # the ctypes default would truncate the c_char_p return of
        # subsequent getenv() and silently mis-cast pointers.
        libc.setenv.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        libc.setenv.restype = ctypes.c_int
        libc.setenv(b"LD_LIBRARY_PATH", _DEFAULT_LD.encode("utf-8"), 1)
        logger.info(
            "Set LD_LIBRARY_PATH in C runtime (in-process): %s",
            _DEFAULT_LD[:120] + ("..." if len(_DEFAULT_LD) > 120 else ""),
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "Could not propagate LD_LIBRARY_PATH to the C runtime; "
            "GPU-using endpoints may fail with libcusparseLt.so.0 not found."
        )


# Matches the pipeline's own logging format:
#   "2026-07-23 14:33:12,345 - speech_to_speech.s2s_pipeline - INFO - hello"
# We make the timestamp and name optional so that other writes (warnings during
# model load, library banners, tracebacks) still get a sensible level.
_LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d+)?)?\s*"
    r"(?:-?\s*(?P<name>[\w.]+)\s*-\s*)?"
    r"(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*-\s*"
    r"(?P<msg>.*)$"
)

_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50, "OTHER": 0}


@dataclass
class LogLine:
    """A single captured line from the subprocess."""

    index: int
    timestamp: float
    level: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "level": self.level,
            "text": self.text,
        }


@dataclass
class _State:
    """Mutable state guarded by ``_lock``."""

    process: Optional[subprocess.Popen] = None
    started_at: Optional[float] = None
    last_argv: list[str] = field(default_factory=list)
    last_env: dict[str, str] = field(default_factory=dict)
    # Monotonic counter assigned to each LogLine. Used by the client to
    # request ``/api/logs?since=<n>`` on reconnect.
    log_index: int = 0
    # Buffer is created in __post_init__ because deque(maxlen=...) has
    # mutable semantics that the dataclass machinery doesn't like as a default.
    buffer: deque = field(default_factory=lambda: deque(maxlen=BUFFER_SIZE))
    subscribers: list[Callable[[LogLine], None]] = field(default_factory=list)
    reader_thread: Optional[threading.Thread] = None
    subscriber_queue: "queue.Queue[LogLine]" = field(default_factory=queue.Queue)


class PipelineProcess:
    """Owns one speech-to-speech subprocess at a time.

    Instantiate once per dashboard. Thread-safe.
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = Path(repo_root)
        self._lock = threading.Lock()
        self._state = _State()
        self._fanout_thread: Optional[threading.Thread] = None
        self._fanout_stop = threading.Event()
        self._log_dir = self._repo_root / LOG_DIR_NAME
        self._log_file: Optional[TextIO] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, settings: dict[str, Any]) -> None:
        """Start (or restart) the pipeline with the given settings."""
        with self._lock:
            if self._state.process is not None and self._state.process.poll() is None:
                raise RuntimeError("Pipeline is already running; stop it first.")
            # Best-effort Pascal wheel install for qwen3-TTS on sm_61 GPUs.
            # No-op on Volta+ or when qwen3 isn't selected. Raises are swallowed.
            try:
                from web_ui.qwentts_installer import install_pascal_wheel_if_needed

                install_result = install_pascal_wheel_if_needed(settings)
                if install_result.get("status") not in (None, "skipped", "already-installed"):
                    logger.info("qwentts Pascal wheel installer: %s", install_result)
            except Exception:  # noqa: BLE001
                logger.exception("qwentts Pascal wheel installer crashed; continuing")
            argv = build_argv(settings)
            env = self._build_env(settings.get("env") or [])
            self._maybe_inject_openai_api_key(settings, env)
            self._maybe_install_openai_timeout_patch(settings, env)
            self._state.last_argv = argv
            self._state.last_env = env
            self._state.started_at = time.time()
            self._state.log_index = 0
            self._state.buffer.clear()
            self._ensure_log_file()

            logger.info("Starting pipeline: %s", " ".join(argv))
            self._state.process = self._spawn(argv, env)
            self._state.reader_thread = threading.Thread(
                target=self._reader_loop, args=(self._state.process,), daemon=True
            )
            self._state.reader_thread.start()

        if self._fanout_thread is None or not self._fanout_thread.is_alive():
            self._fanout_stop.clear()
            self._fanout_thread = threading.Thread(target=self._fanout_loop, daemon=True)
            self._fanout_thread.start()

    def stop(self) -> None:
        """Stop the pipeline (no-op if not running)."""
        with self._lock:
            proc = self._state.process
            if proc is None or proc.poll() is not None:
                self._state.process = None
                self._remove_openai_timeout_patch()
                self._close_log_file()
                return
            logger.info("Stopping pipeline (pid=%s)", proc.pid)
            self._terminate(proc)
        # Wait outside the lock so the reader thread can finish flushing.
        try:
            proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("Pipeline did not exit within %ss; killing", STOP_TIMEOUT_S)
            with self._lock:
                p = self._state.process
            if p is not None and p.poll() is None:
                p.kill()
                p.wait()
        with self._lock:
            self._state.process = None
            self._state.started_at = None
            self._remove_openai_timeout_patch()
            self._close_log_file()

    def restart(self, settings: dict[str, Any]) -> None:
        self.stop()
        self.start(settings)

    def is_running(self) -> bool:
        with self._lock:
            return self._state.process is not None and self._state.process.poll() is None

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            proc = self._state.process
            running = proc is not None and proc.poll() is None
            return {
                "running": running,
                "pid": proc.pid if proc is not None else None,
                "exit_code": proc.returncode if proc is not None else None,
                "started_at": self._state.started_at,
                "uptime_s": (time.time() - self._state.started_at) if (running and self._state.started_at) else 0.0,
                "argv": self._state.last_argv,
                "command_line": " ".join(self._state.last_argv),
            }

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_log_buffer(self, since: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return [line.to_dict() for line in self._state.buffer if line.index >= since]

    def subscribe(self, callback: Callable[[LogLine], None]) -> None:
        with self._lock:
            self._state.subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[LogLine], None]) -> None:
        with self._lock:
            try:
                self._state.subscribers.remove(callback)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _spawn(self, argv: list[str], env: dict[str, str]) -> subprocess.Popen:
        """Launch the subprocess in a cross-platform way.

        List argv (no shell) and explicit cwd ensure the same behavior on
        Linux, macOS, and Windows.
        """
        kwargs: dict[str, Any] = {
            "args": [sys.executable, *argv],
            "cwd": str(self._repo_root),
            "env": env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "bufsize": 1,  # line buffered
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            # Detach so Ctrl+C / closing the dashboard terminal doesn't propagate
            # down. ``subprocess.CREATE_NEW_PROCESS_GROUP`` is the right knob on
            # Windows; we then call ``proc.send_signal(signal.CTRL_BREAK_EVENT)``
            # to ask it to shut down cleanly.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return subprocess.Popen(**kwargs)

    def _terminate(self, proc: subprocess.Popen) -> None:
        """Politely ask the subprocess to stop, escalating to kill if needed."""
        if os.name == "nt":
            try:
                proc.send_signal(subprocess.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                return
            except (OSError, ValueError):
                pass
        try:
            proc.terminate()
        except (OSError, ProcessLookupError):
            pass

    def _reader_loop(self, proc: subprocess.Popen) -> None:
        """Read stdout until EOF and append each line to the buffer / queue."""
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                self._append(line)
        except Exception:  # noqa: BLE001
            logger.exception("Reader loop crashed")
        finally:
            self._append(f"[dashboard] subprocess exited with code {proc.returncode}")

    def _append(self, text: str) -> None:
        level = "OTHER"
        m = _LOG_LINE_RE.match(text)
        if m:
            level = m.group("level") or "OTHER"
        with self._lock:
            self._state.log_index += 1
            entry = LogLine(
                index=self._state.log_index,
                timestamp=time.time(),
                level=level,
                text=text,
            )
            self._state.buffer.append(entry)
            self._state.subscriber_queue.put(entry)
            self._write_log_line(entry)

    def _ensure_log_file(self) -> None:
        """Open a new timestamped log file for the upcoming pipeline run."""
        self._close_log_file()
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self._log_dir / f"pipeline_{timestamp}.log"
            self._log_file = open(path, "a", encoding="utf-8", errors="replace")
            self._write_log_line(
                LogLine(
                    index=self._state.log_index + 1,
                    timestamp=time.time(),
                    level="INFO",
                    text=f"[dashboard] Pipeline log file: {path}",
                )
            )
            self._state.log_index += 1
        except Exception:  # noqa: BLE001
            logger.exception("Failed to open pipeline log file; logs will stay in memory only")
            self._log_file = None

    def _close_log_file(self) -> None:
        """Flush and close the current pipeline log file."""
        fh = self._log_file
        self._log_file = None
        if fh is not None:
            try:
                fh.flush()
                fh.close()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to close pipeline log file")

    def _write_log_line(self, entry: LogLine) -> None:
        """Append one captured line to the persistent log file, if open."""
        fh = self._log_file
        if fh is None:
            return
        ts = datetime.fromtimestamp(entry.timestamp).isoformat(timespec="milliseconds")
        try:
            fh.write(f"{ts} [{entry.level}] {entry.text}\n")
        except Exception:  # noqa: BLE001
            # Logging must never crash the dashboard.
            logger.exception("Failed to write pipeline log line")

    def _fanout_loop(self) -> None:
        """Single consumer of the log queue; calls each subscriber without blocking."""
        while not self._fanout_stop.is_set():
            try:
                entry = self._state.subscriber_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                subs = list(self._state.subscribers)
            for cb in subs:
                try:
                    cb(entry)
                except Exception:  # noqa: BLE001
                    logger.exception("Log subscriber raised")

    def _build_env(self, env_list: Iterable[str]) -> dict[str, str]:
        """Merge ``KEY=VALUE`` strings onto the current process env.

        Items that don't contain ``=`` are ignored (with a warning). The
        existing process environment is preserved so things like
        ``PATH``, ``HOME``, and platform defaults still work.

        We also prepend the nvidia-* runtime library paths discovered
        by :func:`_default_ld_library_path` to ``LD_LIBRARY_PATH`` --
        ``import torch`` in the subprocess needs them on Pascal (sm_61)
        systems where torch ships cu126 nvidia wheels but the system
        ldconfig cache has no sm_61-compatible libcusparseLt.
        """
        out = dict(os.environ)
        for raw in env_list:
            if not isinstance(raw, str) or "=" not in raw:
                logger.warning("Ignoring malformed env entry: %r", raw)
                continue
            k, _, v = raw.partition("=")
            out[k.strip()] = v
        # Make sure the bundled nvidia-*/lib/ paths are reachable by
        # the subprocess. The user's env editor can still override
        # LD_LIBRARY_PATH entirely (we just merge onto whatever is
        # there).
        if _DEFAULT_LD:
            existing = out.get("LD_LIBRARY_PATH", "")
            out["LD_LIBRARY_PATH"] = _DEFAULT_LD + (":" + existing if existing else "")
        # Force unbuffered output from the child so logs stream in real time.
        out["PYTHONUNBUFFERED"] = "1"
        return out

    @staticmethod
    def _maybe_inject_openai_api_key(settings: dict[str, Any], env: dict[str, str]) -> None:
        """Make the pipeline happy when pointing at non-OpenAI endpoints.

        The OpenAI client rejects ``api_key=""`` at construction time, but
        Ollama / vLLM / llama.cpp servers accept any placeholder string and
        ignore the key. If the user picked a non-OpenAI base URL and left
        the API key blank, inject ``OPENAI_API_KEY=ollama`` so the pipeline
        can construct the client. The user can still override via the
        ``env`` editor in the Settings tab.
        """
        if "OPENAI_API_KEY" in env and env["OPENAI_API_KEY"]:
            return  # user already set it; respect that
        llm_backend = settings.get("--llm-backend")
        if llm_backend not in ("chat-completions", "responses-api"):
            return
        base_url = (settings.get("--responses-api-base-url") or "").rstrip("/")
        if base_url and "api.openai.com" in base_url:
            return  # official OpenAI: keep whatever the user provided
        api_key = settings.get("--responses-api-api-key")
        if api_key:  # user explicitly set a non-empty key; respect it
            return
        env["OPENAI_API_KEY"] = "ollama"

    # ------------------------------------------------------------------
    # OpenAI SDK monkey-patch (LLM request timeout knob, 0.4.2+)
    # ------------------------------------------------------------------
    # The pipeline hardcodes a 20 s read timeout on every OpenAI-
    # compatible LLM call. When the LLM is slow (Ollama cold load,
    # large local model, slow vLLM), the call times out and the user
    # hears the canned "Wow I'm a bit slow today, could you repeat
    # that?" fallback -- which they perceive as the model being
    # permanently broken. ``DASHBOARD_LLM_READ_TIMEOUT_S`` overrides
    # that hardcoded 20 s via a tiny monkey-patch installed in the
    # venv's site-packages and auto-imported through a guarded
    # ``sitecustomize.py`` snippet. The patch is a strict no-op
    # when the env var is unset; setting it to ``0`` means
    # ``httpx.Timeout(None)`` (wait forever); positive integers are
    # seconds. We remove the patch files on pipeline stop so the venv
    # stays clean.
    #
    # 0.4.2 change: previously gated on ``--llm-backend-type ==
    # 'hermes'`` only. Now generalized -- every LLM backend honors
    # the per-backend ``llm_request_timeout_s`` value the user
    # picked in the LLM Settings tab. ``hermes`` keeps its 0
    # default; ``ollama`` and other local backends default to 120 s.
    # See ``get_llm_request_timeout_s`` in settings_schema.py for
    # the resolution logic.

    _OPENAI_PATCH_FILENAME = "_dash_openai_timeout_patch.py"
    _SITECUSTOMIZE_FILENAME = "sitecustomize.py"
    _SITECUSTOMIZE_PROBE_MARKER = "_dash_openai_timeout_patch"

    @staticmethod
    def _resolve_llm_request_timeout_s(settings: dict[str, Any]) -> int:
        """Resolve the LLM read timeout for the currently selected backend.

        Thin wrapper around ``settings_schema.get_llm_request_timeout_s``
        so this module stays self-contained about where the value
        comes from. Returns seconds (0 = infinite).
        """
        # Local import keeps the module-load-time cost of settings_schema
        # off the import path for tests that don't touch the pipeline.
        from web_ui.settings_schema import get_llm_request_timeout_s

        return get_llm_request_timeout_s(settings)

    @classmethod
    def _site_packages_dir(cls) -> Optional[Path]:
        """Find the venv's site-packages directory.

        Returns None if it can't be found -- the patch is then a
        no-op and the pipeline keeps its 20 s default. The user
        will see the canned fallback loop, but the dashboard itself
        will still work.
        """
        try:
            import site as _site  # noqa: PLC0415

            candidates = [Path(p) for p in _site.getsitepackages()]
        except Exception:  # noqa: BLE001
            candidates = []
        # Prefer the one that actually contains ``site.py`` -- a
        # virtualenv's site-packages is always a parent of ``site``.
        for sp in candidates:
            if (sp / "site.py").exists():
                return sp
        if candidates:
            return candidates[0]
        return None

    @classmethod
    def _maybe_install_openai_timeout_patch(cls, settings: dict[str, Any], env: dict[str, str]) -> None:
        """Install the openai monkey-patch + set the per-backend timeout env var.

        As of 0.4.2 the patch is no longer gated on ``--llm-backend-type
        == 'hermes'`` -- every LLM backend honors the per-backend
        ``llm_request_timeout_s`` value. The patch itself is a strict
        no-op when ``DASHBOARD_LLM_READ_TIMEOUT_S`` is unset, so it's
        safe to leave installed between pipeline starts; we still drop
        the files on pipeline stop for venv cleanliness.
        """
        timeout_s = cls._resolve_llm_request_timeout_s(settings)
        env["DASHBOARD_LLM_READ_TIMEOUT_S"] = str(timeout_s)

        sp = cls._site_packages_dir()
        if sp is None:
            logger.warning(
                "Could not locate venv site-packages; Hermes timeout knob will "
                "not take effect (pipeline keeps the 20 s default)."
            )
            return

        try:
            repo_root = Path(__file__).resolve().parent.parent
            patch_src = repo_root / "web_ui" / "_openai_timeout_patch.py"
            patch_dst = sp / cls._OPENAI_PATCH_FILENAME
            sitecustomize_path = sp / cls._SITECUSTOMIZE_FILENAME

            if not patch_src.exists():
                logger.warning(
                    "OpenAI timeout patch source missing at %s; knob disabled",
                    patch_src,
                )
                return

            # Copy the patch into site-packages so the
            # sitecustomize.py importer can find it as a top-level
            # module (without the ``web_ui.`` package prefix).
            patch_dst.write_text(patch_src.read_text(encoding="utf-8"), encoding="utf-8")

            # If a sitecustomize.py already exists, append a guarded
            # import so we don't trample whatever the user has there.
            # The probe marker keeps the append idempotent across
            # repeated pipeline (re)starts.
            existing = sitecustomize_path.read_text(encoding="utf-8") if sitecustomize_path.exists() else ""
            if cls._SITECUSTOMIZE_PROBE_MARKER not in existing:
                snippet = (
                    "\n\n# Auto-installed by speech-to-speech-dashboard when the "
                    "LLM request timeout knob is active. Safe to delete; "
                    "it is removed on pipeline stop.\n"
                    "try:\n"
                    "    import _dash_openai_timeout_patch  # noqa: F401\n"
                    "except Exception:  # noqa: BLE001\n"
                    "    pass\n"
                )
                sitecustomize_path.write_text(existing + snippet, encoding="utf-8")

            logger.info(
                "Installed openai timeout patch (timeout_s=%s for backend=%s) at %s",
                timeout_s,
                settings.get("--llm-backend"),
                sp,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to install openai timeout patch; pipeline will use the "
                "20 s default. The LLM request timeout knob will not take effect."
            )

    @classmethod
    def _remove_openai_timeout_patch(cls) -> None:
        """Best-effort removal of the patch + its sitecustomize marker.

        Called on pipeline stop so the venv doesn't keep a
        sitecustomize.py fragment around between runs. The patch is
        re-installed on the next start, so removing it is purely a
        cleanliness measure.
        """
        sp = cls._site_packages_dir()
        if sp is None:
            return
        try:
            patch_path = sp / cls._OPENAI_PATCH_FILENAME
            if patch_path.exists():
                patch_path.unlink()
            sitecustomize_path = sp / cls._SITECUSTOMIZE_FILENAME
            if sitecustomize_path.exists():
                txt = sitecustomize_path.read_text(encoding="utf-8")
                if cls._SITECUSTOMIZE_PROBE_MARKER in txt:
                    # Strip our appended block but leave any other
                    # sitecustomize content the user (or another
                    # tool) put there.
                    marker = "# Auto-installed by speech-to-speech-dashboard"
                    idx = txt.find(marker)
                    if idx != -1:
                        # Walk back to the start of the preceding blank-line block
                        cut = txt[:idx].rstrip() + "\n"
                        if cut.strip():
                            sitecustomize_path.write_text(cut, encoding="utf-8")
                        else:
                            sitecustomize_path.unlink()
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to remove openai timeout patch from %s; harmless but worth a manual cleanup",
                sp,
            )
