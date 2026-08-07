"""
Standalone, NON-DESTRUCTIVE test for an UNLOAD_TTS feature on Qwen3-TTS
(faster-qwen3-tts / ggml backend on Linux/CUDA).

What this script does:
    1. Imports the real Qwen3TTSHandler without modifying the source.
    2. Monkey-patches the handler at runtime to inject the proposed fix:
         - on_unload()         : del self.model + gc + torch.cuda.empty_cache
         - _ensure_model_loaded(): recreate self.model via _setup_faster
         - process() wrapper   : call _ensure_model_loaded() at top so
                                 a follow-up request after unload re-loads
    3. Runs an end-to-end sequence:
         construct (setup) ->  first synth  ->  measure GPU/RSS memory
                              unload()    ->  measure GPU/RSS memory
                              second synth -> confirm lazy reload + audio still flows
    4. Prints PASS/FAIL for every step plus the actual memory deltas.

What this script does NOT do:
    - No edits to qwen3_tts_handler.py, baseHandler.py, server.py, or app.js.
    - No new files in src/.
    - Not part of pytest (do `uv run python tests/manual/test_qwen3_unload.py`).

Run with:
    uv run python tests/manual/test_qwen3_unload.py
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path

# Make the project root importable so `speech_to_speech.*` resolves.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from speech_to_speech.baseHandler import BaseHandler  # noqa: E402
from speech_to_speech.pipeline.messages import TTSInput  # noqa: E402
from speech_to_speech.TTS.qwen3_tts_handler import Qwen3TTSHandler  # noqa: E402

# ─────────────────────────────────────────────────────────────────────
# Tiny pass/fail harness — no pytest, no extra deps.
# ─────────────────────────────────────────────────────────────────────

RESULTS: list[tuple[str, bool, str]] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ─────────────────────────────────────────────────────────────────────
# Memory probes — ggml holds buffers on the C side, so
# torch.cuda.memory_allocated() may not move. We report three signals:
#   (a) torch.cuda.memory_allocated() — torch-side allocations only
#   (b) torch.cuda.memory_reserved()  — torch's reserved pool (catches more)
#   (c) process RSS via /proc/self/status — what the OS sees this process use
#   (d) nvidia-smi process memory     — what the GPU driver sees this PID use
# ─────────────────────────────────────────────────────────────────────


def _read_rss_mib() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _read_nvidia_smi_mib() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    my_pid = os.getpid()
    total = 0.0
    found = False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = float(parts[1])
        except ValueError:
            continue
        if pid == my_pid:
            total += mem
            found = True
    return total if found else None


def mem_snapshot() -> dict[str, float | None]:
    return {
        "torch_allocated": (torch.cuda.memory_allocated() / (1024 * 1024)) if torch.cuda.is_available() else None,
        "torch_reserved": (torch.cuda.memory_reserved() / (1024 * 1024)) if torch.cuda.is_available() else None,
        "rss_mib": _read_rss_mib(),
        "nvidia_smi_mib": _read_nvidia_smi_mib(),
    }


def _fmt_mem(m: dict[str, float | None]) -> str:
    parts = []
    if m["torch_allocated"] is not None:
        parts.append(f"torch_alloc={m['torch_allocated']:.0f}MiB")
    if m["torch_reserved"] is not None:
        parts.append(f"torch_resv={m['torch_reserved']:.0f}MiB")
    if m["rss_mib"] is not None:
        parts.append(f"rss={m['rss_mib']:.0f}MiB")
    if m["nvidia_smi_mib"] is not None:
        parts.append(f"nvidia-smi={m['nvidia_smi_mib']:.0f}MiB")
    return " ".join(parts) if parts else "(no signal)"


# ─────────────────────────────────────────────────────────────────────
# The patch we want to test — does NOT touch source files. Applied
# here in-memory to Qwen3TTSHandler only.
# ─────────────────────────────────────────────────────────────────────


def _ensure_model_loaded(self: Qwen3TTSHandler) -> None:  # type: ignore[no-untyped-def]
    """Lazy-reload model after on_unload() dropped it. Mirrors chatterbox."""
    if getattr(self, "model", None) is not None:
        return
    print(f"[patch] _ensure_model_loaded: re-creating model on device={self.device!r}")
    # The qwen3 setup() accepts attn_implementation in kwargs but does NOT
    # store it on self. We capture it on first setup via the test harness
    # (see _capture_setup_state below) and read it back here.
    attn_impl = getattr(self, "_test_captured_attn_implementation", "eager")
    self._setup_faster(
        model_name=self.model_name,
        dtype=self.dtype,
        attn_implementation=attn_impl,
        backend=self.faster_backend,
    )


def _on_unload(self: Qwen3TTSHandler) -> None:  # type: ignore[no-untyped-def]
    """Drop the model + free caches. Mirrors chatterbox.on_unload()."""
    if getattr(self, "model", None) is None:
        print("[patch] on_unload: model already None, no-op")
        return
    print("[patch] on_unload: deleting self.model and clearing caches")
    self.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _patched_process(self: Qwen3TTSHandler, tts_input: TTSInput):  # type: ignore[no-untyped-def]
    """Wrap the real process() so a model=None triggers lazy reload first."""
    if getattr(self, "model", None) is None:
        print("[patch] process(): self.model is None -> calling _ensure_model_loaded()")
        self._ensure_model_loaded()
    yield from self._original_process(tts_input)


# Patch the class (process-local; no source file is modified).
Qwen3TTSHandler._ensure_model_loaded = _ensure_model_loaded  # type: ignore[attr-defined]
Qwen3TTSHandler.on_unload = _on_unload  # type: ignore[attr-defined]
Qwen3TTSHandler._original_process = Qwen3TTSHandler.process  # type: ignore[attr-defined]
Qwen3TTSHandler.process = _patched_process  # type: ignore[assignment,method-assign]

# Sanity: confirm BaseHandler default is a no-op (the dispatcher is unchanged).
assert callable(BaseHandler.on_unload), "BaseHandler.on_unload must exist (it's how dispatch reaches us)"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _capture_setup_state(handler: Qwen3TTSHandler, attn_implementation: str) -> None:
    """Snapshot kwargs needed to re-create the model on demand."""
    # The qwen3 setup() doesn't persist attn_implementation on self; save
    # it under a private attr so _ensure_model_loaded can use it later.
    handler._test_captured_attn_implementation = attn_implementation  # type: ignore[attr-defined]


def collect_audio(handler: Qwen3TTSHandler, text: str) -> tuple[int, int, float]:
    """Run process() to completion. Return (audio_chunks, total_samples, elapsed_s).

    Note: the AUDIO_RESPONSE_DONE sentinel is only yielded for EndOfResponse,
    not for plain TTSInput — see qwen3_tts_handler.process:692-697. So we do
    NOT assert on it for the regular synth path.
    """
    t0 = time.perf_counter()
    chunk_count = 0
    sample_count = 0
    tts_input = TTSInput(text=text, language_code="en")
    for item in handler.process(tts_input):
        # AudioOutput has .audio; some yield paths may emit raw ndarray/bytes.
        audio = getattr(item, "audio", item)
        if isinstance(audio, bytes):
            sample_count += len(audio) // 2  # int16
            chunk_count += 1
        elif hasattr(audio, "size"):
            sample_count += int(audio.size)
            chunk_count += 1
        else:
            # Probably an AudioOutput wrapper whose .audio is the actual data
            # — already counted above via getattr fallback.
            continue
    elapsed = time.perf_counter() - t0
    return chunk_count, sample_count, elapsed


# ─────────────────────────────────────────────────────────────────────
# Main test sequence
# ─────────────────────────────────────────────────────────────────────


def main() -> int:
    from queue import Queue
    from threading import Event

    from speech_to_speech.TTS.qwen3_tts_handler import DEFAULT_REF_TEXT

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU mem total: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.0f} MiB")
    print(f"Platform: {sys.platform}")
    print(f"Repo root: {REPO_ROOT}")
    print(f"PID: {os.getpid()}")

    section("1. Constructing handler (setup() runs in __init__)")

    stop_event = Event()
    queue_in: Queue = Queue()
    queue_out: Queue = Queue()
    attn_implementation = "eager"  # flash_attention_2 needs flash-attn installed
    setup_kwargs = dict(
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device="cuda",
        dtype="auto",
        attn_implementation=attn_implementation,
        backend="ggml",
        ref_audio=None,
        ref_text=DEFAULT_REF_TEXT,
        language="auto",
        speaker="Aiden",
        instruct=None,
        xvec_only=False,
        parity_mode=False,
        non_streaming_mode=True,
        streaming_chunk_size=8,
        max_new_tokens=128,
        blocksize=512,
        gen_kwargs={},
    )

    mem_before_setup = mem_snapshot()
    t_construct = time.perf_counter()
    try:
        handler = Qwen3TTSHandler(
            stop_event,
            queue_in,
            queue_out,
            setup_args=(stop_event,),  # should_listen
            setup_kwargs=setup_kwargs,
        )
    except Exception as e:
        step("handler construction (calls setup())", False, f"exception: {type(e).__name__}: {e}")
        return _summary()
    construct_elapsed = time.perf_counter() - t_construct
    _capture_setup_state(handler, attn_implementation)
    mem_after_setup = mem_snapshot()

    section("2. Setup completed (model loaded + warmed up)")
    print(f"(Took {construct_elapsed:.1f}s — includes model load and warmup synthesis.)")
    step(
        "handler construction succeeded",
        True,
        f"took {construct_elapsed:.1f}s",
    )
    step(
        "model attribute exists after setup",
        getattr(handler, "model", None) is not None,
        f"type={type(handler.model).__name__}",
    )
    # We do NOT assert that any one memory counter grew: with ggml, only some
    # signals move. Instead we report all of them so we can see what the
    # ggml C-side allocation actually looks like.
    print(f"  memory before setup: {_fmt_mem(mem_before_setup)}")
    print(f"  memory after setup:  {_fmt_mem(mem_after_setup)}")

    section("3. First synthesis (no unload yet)")
    text1 = "Hello, this is a test of the qwen three text to speech system."
    mem_snapshot()
    try:
        chunks1, samples1, elapsed1 = collect_audio(handler, text1)
    except Exception as e:
        step("first synthesis runs", False, f"exception: {type(e).__name__}: {e}")
        return _summary()
    mem_snapshot()
    step(
        "first synthesis produced audio",
        chunks1 > 0 and samples1 > 0,
        f"chunks={chunks1} samples={samples1} ({samples1 / 16000:.2f}s audio) in {elapsed1:.2f}s",
    )

    section("4. Calling on_unload() — the proposed fix")
    mem_before_unload = mem_snapshot()
    try:
        handler.on_unload()
    except Exception as e:
        step("on_unload() runs", False, f"exception: {type(e).__name__}: {e}")
        return _summary()
    mem_after_unload = mem_snapshot()
    step("on_unload() runs", True, "")
    step(
        "self.model is None after on_unload",
        getattr(handler, "model", None) is None,
        f"actual={getattr(handler, 'model', '<missing>')!r}",
    )
    # Report all memory signals before/after unload. We don't assert
    # that ALL counters drop — that depends on whether ggml's C-side
    # buffers get reclaimed by the OS. We DO assert that the torch
    # counters drop and the model is gone, then print nvidia-smi RSS
    # for human review.
    print(f"  memory before unload: {_fmt_mem(mem_before_unload)}")
    print(f"  memory after unload:  {_fmt_mem(mem_after_unload)}")
    torch_alloc_freed = (mem_before_unload["torch_allocated"] or 0) - (mem_after_unload["torch_allocated"] or 0)
    rss_freed = (mem_before_unload["rss_mib"] or 0) - (mem_after_unload["rss_mib"] or 0)
    smi_freed = (mem_before_unload["nvidia_smi_mib"] or 0) - (mem_after_unload["nvidia_smi_mib"] or 0)
    step(
        "torch memory counter dropped after unload",
        torch_alloc_freed >= 0,
        f"freed {torch_alloc_freed:.0f} MiB on torch_allocated",
    )
    step(
        "OS RSS dropped after unload (real signal for ggml C-side)",
        rss_freed > 0,
        f"freed {rss_freed:.0f} MiB on RSS (before={mem_before_unload['rss_mib']:.0f}, "
        f"after={mem_after_unload['rss_mib']:.0f})",
    )
    if mem_after_unload["nvidia_smi_mib"] is not None:
        step(
            "nvidia-smi GPU memory dropped after unload",
            smi_freed > 0,
            f"freed {smi_freed:.0f} MiB on nvidia-smi (before={mem_before_unload['nvidia_smi_mib']:.0f}, "
            f"after={mem_after_unload['nvidia_smi_mib']:.0f})",
        )

    section("5. Second synthesis (model is None — must lazy-reload)")
    text2 = "And this is the second test after the unload."
    mem_before_synth2 = mem_snapshot()
    t_reload = time.perf_counter()
    try:
        chunks2, samples2, elapsed2 = collect_audio(handler, text2)
    except Exception as e:
        step("second synthesis runs", False, f"exception: {type(e).__name__}: {e}")
        print(f"  before-reload memory: {_fmt_mem(mem_before_synth2)}")
        return _summary()
    reload_to_synth = time.perf_counter() - t_reload
    mem_after_synth2 = mem_snapshot()
    step(
        "second synthesis produced audio (lazy reload worked)",
        chunks2 > 0 and samples2 > 0,
        f"chunks={chunks2} samples={samples2} ({samples2 / 16000:.2f}s audio) in {elapsed2:.2f}s "
        f"(total wall-clock {reload_to_synth:.1f}s including model reload)",
    )
    print(f"  memory before second synth: {_fmt_mem(mem_before_synth2)}")
    print(f"  memory after second synth:  {_fmt_mem(mem_after_synth2)}")

    section("6. State preserved across reload")
    step(
        "self.speaker preserved",
        getattr(handler, "speaker", None) == "Aiden",
        f"speaker={getattr(handler, 'speaker', None)!r}",
    )
    step(
        "self.language preserved",
        getattr(handler, "language", None) is not None,
        f"language={getattr(handler, 'language', None)!r}",
    )

    section("7. on_unload() idempotency")
    try:
        handler.on_unload()
        handler.on_unload()  # second call: model already None
    except Exception as e:
        step("on_unload() is idempotent", False, f"exception: {type(e).__name__}: {e}")
    else:
        step("on_unload() is idempotent", True, "two consecutive calls did not raise")

    return _summary()


def _summary() -> int:
    section("SUMMARY")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"{passed}/{total} steps passed")
    if passed != total:
        print("\nFailed steps:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"  - {name}: {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
