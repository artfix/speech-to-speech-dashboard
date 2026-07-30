"""Background pinger that keeps an Ollama model loaded between user turns.

Ollama unloads a model from VRAM after ``keep_alive`` of inactivity (default
5 minutes). For a low-latency voice-agent pipeline this is fatal: the next
user utterance after a 5-minute pause would pay the full ~20s reload cost
before the LLM responds. The dashboard's pipeline wrapper can't pass
``keep_alive`` to Ollama through the upstream OpenAI client (the
``_extra_body`` is hardcoded in ``src/speech_to_speech/``, off-limits per
CLAUDE.md), so we refresh the keepalive timer from this side instead.

A dedicated daemon thread posts a tiny ``/v1/chat/completions`` request to
Ollama every ``interval / 2`` minutes (refresh well before the unload
deadline). The request body includes ``keep_alive: <interval>`` so each
ping resets Ollama's idle timer to the user's chosen value. ``max_tokens=1``
keeps the ping cheap — we discard the response.

This pinger is complementary to the one-shot warmup that
``web_ui/server.py`` fires at Start / Restart time (see
``POST /api/ollama/keepalive``): the warmup loads the model at the user's
``num_ctx`` and arms the initial ``keep_alive`` timer; this pinger then
keeps that timer fresh until the next Start / Restart re-arms it.

CLI flags reach this code via the dashboard's ``web_ui_settings.json``
through the settings save / pipeline start hooks in ``web_ui/server.py``.
This module is intentionally self-contained: no FastAPI imports, no
references to ``state``, so it can be unit-tested in isolation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# Default ping cadence as a fraction of the keepalive window. We refresh at
# half the interval so a single missed ping (network blip, transient 5xx)
# still leaves us well under the unload deadline.
_PING_FRACTION = 0.5

# Lower bound on the interval so a too-eager user doesn't hammer Ollama.
# 30 s matches Ollama's own default load-unload granularity.
_MIN_INTERVAL_S = 30.0

# HTTP timeout for one ping. Pings should be near-instant; if they take
# longer than this the Ollama server is sick and we should let the next
# attempt recover rather than block the thread.
_PING_TIMEOUT_S = 10.0


@dataclass
class KeepaliveConfig:
    """Resolved keepalive parameters.

    ``base_url`` is the Ollama ``/v1`` endpoint, e.g. ``http://1.2.3.4:11434/v1``.
    ``api_key`` is the OpenAI-compat key (Ollama accepts any non-empty
    placeholder, the dashboard injects ``"ollama"``).
    ``model_name`` is the Ollama model identifier (matches ``--model-name``).
    ``keepalive`` is the user-chosen value. Three shapes are supported:

    - ``"5m"``, ``"30m"``, ``"2h"`` — pass through verbatim. Ollama parses
      these natively.
    - ``"-1"`` — Ollama-specific "keep loaded forever".
    - ``"0"`` — Ollama's default ("unload immediately after request"). We
      treat this as "feature off" and skip the pinger entirely.
    """

    base_url: str
    api_key: str
    model_name: str
    keepalive: str
    # Optional override for the ping cadence. Mostly useful for tests.
    ping_interval_s: Optional[float] = None
    # Optional context-window cap (``options.num_ctx``). When set, every
    # ping includes it so the model's loaded KV-cache stays pinned to the
    # user's chosen size (Ollama otherwise falls back to the model's
    # native default, e.g. 131072 for gemma4). Mirrors the parsing in
    # ``web_ui/server.py``'s ``_one_shot_keepalive_async`` so the two
    # keepalive paths agree on the value.
    num_ctx: Optional[int] = None


@dataclass
class KeepaliveStatus:
    """Snapshot of the pinger for the ``/api/llm-keepalive/status`` endpoint."""

    running: bool = False
    config: Optional[KeepaliveConfig] = None
    last_ping_at: Optional[float] = None
    last_ping_ok: Optional[bool] = None
    last_ping_error: Optional[str] = None
    next_ping_at: Optional[float] = None
    ping_count: int = 0
    ping_failures: int = 0
    started_at: Optional[float] = None
    thread_id: Optional[int] = field(default=None)


def _parse_keepalive_to_seconds(value: str) -> Optional[float]:
    """Parse an Ollama ``keep_alive`` string into seconds.

    Returns ``None`` for ``"-1"`` (infinite — caller treats as "keep
    loaded forever"). Returns ``None`` for ``""`` / ``"0"`` (feature off).
    Raises ``ValueError`` for unparseable input — caught by the form
    validator upstream so the user sees a clean error.
    """
    s = (value or "").strip()
    if not s or s == "0":
        return None
    if s == "-1":
        return None  # sentinel — handled by caller
    if len(s) < 2:
        raise ValueError(f"keepalive {value!r} too short")
    unit = s[-1]
    try:
        n = float(s[:-1])
    except ValueError as e:
        raise ValueError(f"keepalive {value!r} is not a number+unit") from e
    if unit == "s":
        return n
    if unit == "m":
        return n * 60.0
    if unit == "h":
        return n * 3600.0
    raise ValueError(f"keepalive unit {unit!r} not in (s, m, h)")


def _is_keepalive_enabled(value: str) -> bool:
    """True iff the user picked anything other than ``""`` / ``"0"``.

    A value of ``"-1"`` (forever) is enabled. ``"5m"`` etc. are enabled.
    Empty string and ``"0"`` are the dashboard's "feature off" sentinels.
    """
    s = (value or "").strip()
    return bool(s) and s != "0"


class LLMKeepAliver:
    """Owns the keepalive pinger thread lifecycle.

    Lifecycle:

    - ``start(config)`` is idempotent — re-calling with a new config
      replaces the running pinger (the old thread is signaled to stop and
      joined before the new one spawns).
    - ``stop()`` is idempotent and safe to call when no thread is running.
    - ``status()`` returns a thread-safe snapshot for the dashboard's
      status endpoint and the LLM-tab badge.

    Concurrency: the pinger reads a single ``threading.Event`` to detect
    shutdown and uses a ``threading.Lock`` to guard config swap. No other
    shared mutable state, so the lock is mostly a formality.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event: Optional[threading.Event] = None
        self._config: Optional[KeepaliveConfig] = None
        self._status = KeepaliveStatus()
        self._thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────--

    def start(self, config: KeepaliveConfig) -> None:
        """Start (or replace) the pinger with ``config``.

        If the keepalive value is the "feature off" sentinel, calls
        ``stop()`` instead. Otherwise joins any existing thread cleanly
        before spawning a fresh one.
        """
        if not _is_keepalive_enabled(config.keepalive):
            logger.info(
                "llm_keepalive: keepalive=%r is disabled — not starting pinger",
                config.keepalive,
            )
            self.stop()
            return

        with self._lock:
            self._stop_old_thread_locked()
            stop_event = threading.Event()
            self._stop_event = stop_event
            self._config = config
            self._status = KeepaliveStatus(
                running=True,
                config=config,
                started_at=time.time(),
            )
            thread = threading.Thread(
                target=self._runner,
                args=(config, stop_event),
                name="llm-keepalive-pinger",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            self._status.thread_id = thread.ident
        logger.info(
            "llm_keepalive: started pinger for model=%r keepalive=%r base_url=%r",
            config.model_name,
            config.keepalive,
            config.base_url,
        )

    def stop(self) -> None:
        """Stop the pinger (no-op if not running)."""
        with self._lock:
            self._stop_old_thread_locked()
            self._status.running = False

    def status(self) -> KeepaliveStatus:
        """Return a copy of the current status (safe to expose via JSON)."""
        with self._lock:
            return KeepaliveStatus(
                running=self._status.running,
                config=self._config,
                last_ping_at=self._status.last_ping_at,
                last_ping_ok=self._status.last_ping_ok,
                last_ping_error=self._status.last_ping_error,
                next_ping_at=self._status.next_ping_at,
                ping_count=self._status.ping_count,
                ping_failures=self._status.ping_failures,
                started_at=self._status.started_at,
                thread_id=self._status.thread_id,
            )

    def update_from_settings(self, settings: dict) -> None:
        """Restart the pinger with config derived from a settings dict.

        Reads:

        - ``--model-name`` for the Ollama model identifier.
        - ``--responses-api-base-url`` for the Ollama endpoint.
        - ``--llm-keepalive`` for the keepalive string (dashboard-only key,
          see ``web_ui.settings_schema._NON_FORWARDED``).
        - ``--responses-api-num-ctx`` for the optional context-window cap.
          Mirrors the parsing in ``web_ui.server._one_shot_keepalive_async``
          so the pinger and the one-shot warmup agree on the value.

        Only acts if the keepalive is set AND ``--llm-backend`` is one of
        the OpenAI-compat backends (``chat-completions`` /
        ``responses-api``). Local in-process transformers models don't
        talk to Ollama, so the pinger would be a no-op there.
        """
        backend = settings.get("--llm-backend") or ""
        if backend not in ("chat-completions", "responses-api"):
            self.stop()
            return
        keepalive = (settings.get("--llm-keepalive") or "").strip()
        if not _is_keepalive_enabled(keepalive):
            self.stop()
            return
        base_url = (settings.get("--responses-api-base-url") or "").strip()
        if not base_url:
            logger.debug("llm_keepalive: --responses-api-base-url is empty, skipping")
            self.stop()
            return
        api_key = (settings.get("--responses-api-api-key") or "ollama").strip() or "ollama"
        model_name = (settings.get("--model-name") or "").strip()
        if not model_name:
            logger.debug("llm_keepalive: --model-name is empty, skipping")
            self.stop()
            return
        # Same ``--responses-api-num-ctx`` parsing as
        # ``web_ui.server._one_shot_keepalive_async``: missing / empty /
        # non-int / non-positive all collapse to ``None`` so the pinger
        # omits the ``options`` block entirely when the user hasn't picked
        # a value.
        num_ctx_raw = settings.get("--responses-api-num-ctx")
        num_ctx: Optional[int] = None
        if num_ctx_raw is not None and num_ctx_raw != "":
            try:
                n = int(num_ctx_raw)
                if n > 0:
                    num_ctx = n
            except (TypeError, ValueError):
                pass
        self.start(
            KeepaliveConfig(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                keepalive=keepalive,
                num_ctx=num_ctx,
            )
        )

    # ── internal ───────────────────────────────────────────────────────────

    def _stop_old_thread_locked(self) -> None:
        """Signal the running thread to exit and join it. Caller holds lock."""
        if self._stop_event is not None:
            self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # Bound the join so a hung thread can't block forever. The
            # thread is a daemon, so even if the join times out the
            # process can still exit cleanly.
            thread.join(timeout=2.0)
        self._stop_event = None
        self._thread = None

    def _runner(self, config: KeepaliveConfig, stop_event: threading.Event) -> None:
        """Ping Ollama on a fixed cadence until ``stop_event`` is set.

        On any error (HTTP failure, timeout, network unreachable) we log
        and keep trying — a transient Ollama restart should not kill the
        pinger. ``next_ping_at`` is updated after every ping (or skip)
        so the dashboard's status badge can show "next ping in N s".
        """
        # Resolve ping cadence once at thread entry. If the user picks a
        # tiny interval we clamp so we don't hammer Ollama.
        keepalive_s = _parse_keepalive_to_seconds(config.keepalive)
        if keepalive_s is None:
            # "-1" (forever) — there's no actual unload deadline, so we
            # still want to send a keep_alive=-1 ping every so often in
            # case Ollama was restarted and lost the previous "keep
            # loaded" state. Use 5 minutes as a reasonable refresh.
            ping_interval_s = 300.0
        else:
            ping_interval_s = max(_MIN_INTERVAL_S, keepalive_s * _PING_FRACTION)
        if config.ping_interval_s is not None:
            # Tests / power-user override.
            ping_interval_s = config.ping_interval_s

        # Cap our ping interval at Ollama's own default keep_alive. The
        # pipeline's per-turn /v1/chat/completions requests don't carry
        # keep_alive in their extra_body, so until our pinger arms the
        # user's value Ollama applies its 5-minute default and unloads
        # the model. Without this cap a user picking --llm-keepalive
        # 30m would set our ping to 15 min — but Ollama's 5-min default
        # would still trip in between, dropping the model silently.
        # Clamping to 5 min keeps us strictly ahead of any unload the
        # server might decide on its own. The user's chosen keep_alive
        # is still what gets sent to Ollama in each ping body — the
        # cap only affects how often we refresh.
        ping_interval_s = min(ping_interval_s, 300.0)

        logger.info(
            "llm_keepalive: runner loop started, ping_interval=%.1fs",
            ping_interval_s,
        )

        # Fire the first ping immediately on start. Without this, the
        # model sits at Ollama's default keep_alive (5 min) from
        # pipeline-start until our first scheduled ping — which for
        # --llm-keepalive 30m is 15 min away. Arming the user's value
        # right away is what makes the chosen keepalive actually take
        # effect.
        if not stop_event.is_set():
            self._do_ping(config)
            with self._lock:
                self._status.next_ping_at = time.time() + ping_interval_s

        # Sleep in small slices so stop() takes effect promptly. The
        # 1-second slice is fine for a sub-minute ping cadence and still
        # responsive for a 30-minute cadence.
        sleep_slice_s = min(1.0, ping_interval_s / 2.0)
        elapsed = 0.0
        while not stop_event.is_set():
            if elapsed >= ping_interval_s:
                self._do_ping(config)
                elapsed = 0.0
                with self._lock:
                    self._status.next_ping_at = time.time() + ping_interval_s
            time.sleep(sleep_slice_s)
            elapsed += sleep_slice_s

        logger.info("llm_keepalive: runner loop exiting (stop signal)")

    def _do_ping(self, config: KeepaliveConfig) -> None:
        """Send one keepalive ping to Ollama. Updates status under the lock.

        We use the OpenAI-compat ``/v1/chat/completions`` endpoint with
        ``keep_alive`` in the body. Ollama honours ``keep_alive`` on this
        path (unlike ``options.num_ctx`` which it silently drops), so each
        ping resets the model's unload timer to the user's chosen value.
        ``max_tokens=1`` keeps the request cheap — Ollama returns ~empty
        content once the model is loaded and we discard the response.

        The user-supplied ``num_ctx`` is logged here for diagnostics but
        intentionally NOT included in the ping body: sending it would
        require the Ollama-native ``/api/chat`` endpoint, which takes
        26 s on a cold load (timeout exceeded) and pollutes conversation
        state with an empty user message. The ``num_ctx`` value is pinned
        by the one-shot warmup at Start / Restart time (``api_ollama_
        keepalive`` in ``web_ui/server.py``), so by the time the pinger
        fires the model is already loaded at the right context.
        """
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": ""}],
            "max_tokens": 1,
            "stream": False,
            "keep_alive": config.keepalive,
        }
        url = config.base_url.rstrip("/") + "/chat/completions"
        try:
            with httpx.Client(timeout=_PING_TIMEOUT_S) as client:
                resp = client.post(url, json=body, headers=headers)
            ok = 200 <= resp.status_code < 300
            err: Optional[str] = None
            if not ok:
                # Truncate the body so a 50 KB error page doesn't blow up the
                # status JSON. The first 200 chars are enough to debug.
                err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except httpx.HTTPError as e:
            ok = False
            err = f"{type(e).__name__}: {e}"

        with self._lock:
            self._status.last_ping_at = time.time()
            self._status.last_ping_ok = ok
            self._status.last_ping_error = err
            self._status.ping_count += 1
            if not ok:
                self._status.ping_failures += 1
        if ok:
            logger.debug(
                "llm_keepalive: ping ok model=%r keep_alive=%r (num_ctx=%r pinned at warmup)",
                config.model_name,
                config.keepalive,
                config.num_ctx,
            )
        else:
            logger.warning("llm_keepalive: ping failed: %s", err)
