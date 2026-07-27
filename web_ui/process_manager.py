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
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from web_ui.settings_schema import build_argv

logger = logging.getLogger(__name__)


BUFFER_SIZE = 5000
STOP_TIMEOUT_S = 5.0


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
            self._state.last_argv = argv
            self._state.last_env = env
            self._state.started_at = time.time()
            self._state.log_index = 0
            self._state.buffer.clear()

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
            out["LD_LIBRARY_PATH"] = (
                _DEFAULT_LD + (":" + existing if existing else "")
            )
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
