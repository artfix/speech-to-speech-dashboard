# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`speech-to-speech` is a low-latency, fully modular voice-agent pipeline. The default install
runs **VAD → STT → LLM → TTS** as a cascade of threads connected by queues, exposed through an
**OpenAI Realtime-compatible WebSocket API** at `/v1/realtime` (and optionally over WebRTC). Every
component is swappable via CLI flags. In production it backs the conversation server for thousands
of [Reachy Mini](https://huggingface.co/blog/reachy-mini) robots.

## Web dashboard (this fork)

This fork adds a **self-hosted web dashboard** at `web_ui/` that wraps the pipeline as a
subprocess. **It does not modify any code in `src/speech_to_speech/`.** The dashboard
introspects the pipeline's argument dataclasses at runtime and renders a web form for every
CLI flag.

### How to run the dashboard

```bash
./start_web_ui.sh          # Linux / macOS
start_web_ui.bat           # Windows
```

→ opens http://localhost:8050 (configurable via `SPEECH_TO_SPEECH_WEB_HOST` /
`SPEECH_TO_SPEECH_WEB_PORT`).

### Dashboard architecture (one-liner)

`web_ui/server.py` (FastAPI) spawns `python -m speech_to_speech.s2s_pipeline <args>` via
`web_ui/process_manager.py` (subprocess with list argv, no shell, cross-platform).
`web_ui/settings_schema.py` introspects the pipeline's `*Arguments` dataclasses
(`ModuleArguments` + every per-backend `STT/LLM/TTS` class) and produces the JSON form
schema the JS frontend uses. The frontend (`web_ui/static/app.js` + `index.html` + `base.css`
+ 8 theme files in `web_ui/themes/`) is vanilla JS, no build step.

The dashboard ships a **user-facing Guide** at `web_ui/Guide.md` (rendered into the
in-app Guide tab) covering Quickstart, the Ollama-on-another-machine recipe, the
OpenAI / HF Inference / vLLM / llama.cpp provider table, per-backend overview, and
common issues.

### Hard constraints (do not violate when extending the dashboard)

1. **No edits to `src/speech_to_speech/`.** The pipeline is treated as a black box.
   All behavior must come from existing CLI flags, the existing `/v1/realtime`
   WebSocket API, and the existing `/v1/pool` status endpoint.
2. **Auto-introspect, never hardcode.** Adding a new CLI flag to the pipeline's
   argument classes must show up in the UI on next dashboard reload. Do not add
   per-flag special cases in `app.js`.
3. **Settings file location:** `web_ui_settings.json` in the repo root, gitignored.
   Atomic write (`.tmp` + rename) so a crash mid-save can't corrupt the file.
4. **Cross-platform:** launcher scripts handle both POSIX and Windows;
   `process_manager.py` uses `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK_EVENT` on
   Windows, `SIGTERM` + `SIGKILL` on POSIX.

### Critical env-injection gotcha

The `openai` Python client rejects `api_key=""` at construction time, but Ollama /
vLLM / llama.cpp servers accept any placeholder string. When the user picks a
non-OpenAI base URL and leaves the API key blank, the dashboard auto-injects
`OPENAI_API_KEY=ollama` into the subprocess environment via
`PipelineProcess._maybe_inject_openai_api_key`. Users can still override via
the Settings → Environment Variables editor.

### How to bump the dashboard version

Single source of truth: `web_ui/__init__.py` → `__version__`. The frontend
fetches `/api/version` on load and renders the result as a badge in the
title bar and in the page `<title>`. Bump the string in `__init__.py`, and
the new version appears in the UI on next reload. The pipeline's own
version (`src/speech_to_speech/__init__.py` and `pyproject.toml`) is
upstream's number and is **not** touched by dashboard changes.

### Known TTS backends on CPU-only boxes

| Backend | CPU-only status |
|---|---|
| `pocket` | ✅ works (pure-Python, no system deps) |
| `qwen3` | ❌ the installed `qwentts-cpp-python` wheel links `libcudart.so.12` even on the `+cpu` variant — known upstream packaging issue |
| `kokoro` | ❌ broken — `spacy`/`thinc` prebuilt wheels don't match numpy 2.x, source build fails to compile |
| `chatTTS` | untested |
| `facebookMMS` | untested |

### Git remotes in this fork

- `origin` → `huggingface.co/spaces/ArtFix0/speech-to-speech-dashboard` (this fork's HF Space)
- `upstream` → `github.com/huggingface/speech-to-speech` (the original repo, preserved for `git fetch upstream`)

To push dashboard changes: `git push origin main`. To sync from upstream: `git fetch upstream && git merge upstream/main`.

### Current running state (as of last session)

A pipeline subprocess is running on this box, started by the dashboard with these
settings (saved to `web_ui_settings.json`):

- `--mode realtime`
- `--llm-backend chat-completions` pointing at local Ollama (`http://127.0.0.1:11434/v1`)
- `--model-name minimax-m3:cloud`
- `--stt parakeet-tdt`, `--tts pocket` (voice `jean`)
- Dashboard on `0.0.0.0:8050`, pipeline WebSocket on `0.0.0.0:8765`

To check it: `curl -s http://127.0.0.1:8050/api/process/status` (requires the dashboard process to be alive; PID is whatever the last `nohup uv run speech-to-speech-web` produced).

## Common commands

Setup is `uv`-based (see `pyproject.toml`):

```bash
uv sync                       # install package + dev deps in editable mode
uv run speech-to-speech -h    # full CLI reference
```

Run modes (set via `--mode`):

- `realtime` (default) — OpenAI Realtime WebSocket at `ws://<host>:8765/v1/realtime`
- `local` — talks through the local mic / speakers
- `websocket` — raw PCM over WebSocket (no Realtime protocol)
- `socket` — raw PCM over TCP

Useful invocations:

```bash
# Local chat (macOS): picks STT=parakeet-tdt, LLM=mlx-lm, TTS=qwen3, device=mps
uv run speech-to-speech --local_mac_optimal_settings

# Point the OpenAI-compatible LLM slot at a local llama.cpp server
uv run speech-to-speech \
    --llm_backend responses-api \
    --model_name "ggml-org/gemma-4-E4B-it-GGUF" \
    --responses_api_base_url "http://127.0.0.1:8080/v1" \
    --responses_api_api_key ""

# Run with a pool of N independent pipelines (realtime mode only)
uv run speech-to-speech --num_pipelines 2

# Connect a second terminal as a Realtime client (mic + speaker)
uv run python scripts/listen_and_play_realtime.py --host 127.0.0.1 --port 8765
```

Backends are installed as extras, e.g. `pip install "speech-to-speech[kokoro]"` or
`"speech-to-speech[pocket]"`; see the table in `README.md`.

### Tests, lint, type-check

The CI workflow (`.github/workflows/ci.yml`) runs four jobs; these are the local equivalents:

```bash
# Install dev deps
uv sync --group dev

# Ruff (lint + format check) over src/ and tests/
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Mypy (src/ only, ignore_missing_imports on)
uv run mypy src/

# Pytest — NLTK data needed for sent_tokenize
uv run python -c "import nltk; nltk.download('punkt_tab')"
uv run pytest tests/ -x -q

# Build the wheel + sdist and run the install smoke test (no model loading, no API calls)
uv build
uvx twine check --strict dist/*
```

`pytest-asyncio` is in `asyncio_mode = "auto"` (see `pyproject.toml`), so async tests run
without per-test markers.

A few tests `cd` into the repo to run; for an individual file:

```bash
uv run pytest tests/test_chat.py -x -q
uv run pytest tests/test_chat.py::TestClass::test_method -x -q
```

## High-level architecture

### Pipeline as a thread-and-queue cascade

`src/speech_to_speech/s2s_pipeline.py` is the single entry point — the console script
`speech-to-speech` is `speech_to_speech.s2s_pipeline:main`. The flow:

1. `parse_arguments()` registers every per-handler dataclass-based `HfArgumentParser` argument
   class (`ModuleArguments` plus one per STT/LLM/TTS backend in `arguments_classes/`). The
   `chat-completions` LLM backend reuses the `responses-api` connection fields by subclassing,
   so the parser pre-resolves `--llm_backend` to register only the right class and avoid
   duplicate field errors.
2. `prepare_all_args()` applies `optimal_mac_settings` (when `--local_mac_optimal_settings`),
   `--device` overrides, and Mac platform checks. It also calls `rename_args()` to strip
   prefixes (e.g. `--stt_*` → backend kwargs, `--llm_gen_*` → `gen_kwargs["..."]`).
3. `build_pipeline()` switches on `--mode` to wire the IO layer and the handler chain:
   - `local` / `socket` / `websocket` modes own the queues and the handler chain directly
   - `realtime` mode builds a pool of `_build_realtime_pipeline_unit()` instances, then
     `RealtimeServer` runs a single uvicorn/FastAPI app and a websocket route that claims
     a free `PipelineUnit` per connection (`--num_pipelines` sets the pool size; max
     concurrent sessions = pool size; further connections are rejected).
4. `_build_pipeline_handlers()` returns the canonical six-stage chain:
   **VAD → STT → TranscriptionNotifier → LM → LMOutputProcessor → TTS** — instantiated via
   `get_stt_handler()`, `get_llm_handler()`, `get_tts_handler()` which dispatch on the
   `--stt` / `--llm_backend` / `--tts` enum values.
5. `ThreadManager.start()` spawns one thread per handler and `wait()` joins them; SIGINT
   sets the shared `stop_event` so handlers drain and put `PIPELINE_END` on their output
   queue to unblock the next stage.

### `BaseHandler` and the inter-stage contract

`src/speech_to_speech/baseHandler.py` defines `BaseHandler[InT, OutT]`. Every pipeline stage
subclasses it with `setup()`, `process(input) -> Iterator[output]`, and optional
`should_process_input()` / `should_emit_output()` / `cleanup()` / `on_session_end()` hooks.
The `run()` loop:

- Polls `queue_in` with a 100 ms timeout (so `stop_event` is checked promptly)
- Unwraps `PipelineControlMessage(kind=SESSION_END)` → calls `on_session_end()` and forwards
  the message (used to reset per-session state without stopping the thread)
- Terminates when it sees the `PIPELINE_END` bytes sentinel (placed in the input queue to
  prevent deadlock when a downstream stage has stopped)
- On exit, places `PIPELINE_END` on the output queue
- Tags produced audio chunks via `output_for_queue()` so the send loop can drop them on cancel

### Typed messages and queue types

`src/speech_to_speech/pipeline/` is the single source of truth for what flows between stages:

- `messages.py` — Pydantic `PipelineMessage` subclasses: `VADAudio`, `PartialTranscription`,
  `Transcription`, `LLMResponseChunk`, `TTSInput`, `AudioOutput`, `GenerateResponseRequest`,
  `TokenUsage`, `EndOfResponse`. Most carry `turn_id`/`turn_revision` for speculative-turn
  bookkeeping, and `cancel_generation` for the CancelScope. Binary sentinels:
  `PIPELINE_END = b"END"` and `AUDIO_RESPONSE_DONE = b"__RESPONSE_DONE__"`.
- `events.py` — `PipelineEvent` subclasses that ride the side-channel `text_output_queue`
  (consumed by the realtime router): `SpeechStartedEvent`, `SpeechStoppedEvent`,
  `PartialTranscriptionEvent`, `TranscriptionCompletedEvent`, `AssistantTextEvent`,
  `TokenUsageEvent`, `ResponseFailedEvent`.
- `queue_types.py` — `TypeAlias` unions for each queue (e.g. `LMOutItem = LLMOut | PipelineInternalItem`).
  Treat these as the canonical contract when adding a new handler.
- `handler_types.py` — `TypeAlias` for the per-stage input/output types so handler generics
  stay readable (`BaseHandler[STTIn, STTOut]`).
- `control.py` — `PipelineControlMessage(kind: ControlKind)` dataclass and the `SESSION_END`
  singleton. Adding a new control kind means extending the enum and teaching `BaseHandler.run()`.

### Cancellation and interruption handling

`src/speech_to_speech/pipeline/cancel_scope.py` defines `CancelScope`, a single object
shared between the asyncio router and pipeline threads. It uses a generation counter so LLM
and TTS handlers can detect supersession without timing games (capture `gen = cancel_scope.generation`
at the start of a response, abort when `is_stale(gen)` flips true), plus a `discarding` flag
the async send loop reads to drop in-flight output between `cancel()` and `__RESPONSE_DONE__`.
The full state machine — including `response_done()`, `new_response()`, `reset()`, and the
discard-generation sentinel matching — is described in the
[Realtime Engine README](src/speech_to_speech/api/openai_realtime/README.md) under
"Interruption Handling".

`src/speech_to_speech/pipeline/speculative_turns.py` (`SpeculativeTurnTracker`) is a separate
thread-safe revision tracker keyed by `turn_id`/revision that lets late partial STT work
discover it has been superseded by a reopen without losing data — see the
"CancelScope design" and "speculative turns" notes in the realtime README.

### Realtime API server (WebSocket + WebRTC)

`src/speech_to_speech/api/openai_realtime/` is the OpenAI Realtime-compatible server. Key files:

- `pipeline_unit.py` — `PipelineUnit` (queues, events, handlers, optional `SessionState`) and
  `SessionState` (transport, session_id, drain Event). A unit is "claimed" by a session by
  setting `unit.session = SessionState(...)` and "released" by setting it back to `None`.
- `server.py` — `RealtimeServer` owns the pool and runs one uvicorn server; signals `should_exit`
  when `stop_event` is set.
- `websocket_router.py` — FastAPI app, the `/v1/realtime` WebSocket endpoint, the `/v1/pool`
  introspection endpoint, the `/v1/realtime/calls` WebRTC endpoint, the per-unit async send
  loop (`_send_loop`), and the SESSION_END drain + quarantine logic
  (`SESSION_END_DRAIN_TIMEOUT_S`, `SESSION_END_QUARANTINE_TIMEOUT_S`).
- `service.py` — `RealtimeService` translates Realtime client events to pipeline work and back,
  holds the per-session conversation chat, and owns `PIPELINE_SAMPLE_RATE = 16000` /
  `CHUNK_SAMPLES = 512` (inbound audio is resampled to 16 kHz and split into 512-sample
  chunks for the VAD).
- `runtime_config.py` — `RuntimeConfig`, a Pydantic model holding `chat: Chat` and
  `session: RealtimeSessionCreateRequest`. Written by `RealtimeService` on `session.update`
  (with `_apply_update` deep-merging into nested BaseModels so partial nested updates
  don't wipe unset fields), read by VAD (turn-detection thresholds), LLM (instructions,
  tools), and TTS (voice). Python's GIL makes primitive reads/writes atomic — no explicit
  locking needed.
- `handlers/` — one file per inbound event type: `audio.py` (audio append + speech
  start/stop), `session.py`, `conversation.py`, `response.py`. `handlers/base.py` is the
  shared base. Each handler returns the list of server events to emit.
- `webrtc_session.py` + `transports.py` — WebRTC GA handshake (`POST /v1/realtime/calls` with
  SDP), aiortc `RTCPeerConnection` plumbing, Opus/48 kHz RTP with a stateful resampler to
  the 16 kHz pipeline. ICE servers come from the `SPEECH_TO_SPEECH_ICE_SERVERS` env var
  (JSON list). WebRTC is optional — the import is wrapped so the server still starts without
  the `webrtc` extra.

The full Realtime protocol event reference, supported event set, tool-calling design, and
WebRTC specifics live in `src/speech_to_speech/api/openai_realtime/README.md`.

### Per-component modules

Each stage has a directory of swappable backends selected via CLI flags.

- `VAD/vad_handler.py` + `VAD/vad_iterator.py` — Silero VAD v5 wrapper with the speech /
  silence thresholds documented in `VADHandlerArguments` (`--thresh`, `--min_speech_ms`,
  `--min_speech_continuation_ms`, `--min_silence_ms`, `--short_segment_merge_ms`,
  `--unanswered_reopen_ms`).
- `STT/` — `parakeet_tdt_handler.py` (default; MLX on Apple Silicon, nano-parakeet on
  CUDA/CPU; supports live transcription), `whisper_stt_handler.py`, `faster_whisper_handler.py`
  (extra), `lightning_whisper_mlx_handler.py` (extra), `mlx_audio_whisper_handler.py`,
  `paraformer_handler.py` (extra, Chinese-oriented). All inherit from `STT/base_stt_handler.py`
  and `STT/transcription_notifier.py` taps partial transcripts for the realtime
  `transcription.delta` events.
- `LLM/` — three backends wired into the parser: `LanguageModelHandler` (transformers +
  mlx-lm, also handles VLMs), `ResponsesApiModelHandler` (OpenAI-compatible `/v1/responses`),
  `ChatCompletionsApiModelHandler` (OpenAI-compatible `/v1/chat/completions`; subclass of
  ResponsesApiModelHandler reusing the same `--responses_api_*` connection flags). Local
  models render tools as `<code>...</code>` blocks in a system prompt (`LLM/tool_call/`)
  and parse them with a regex; the OpenAI API path gets tool calls natively.
- `TTS/` — `qwen3_tts_handler.py` (default; `faster-qwen3-tts` GGML on Linux,
  `mlx-audio` on Apple Silicon), `kokoro_handler.py` (extra on non-macOS), `pocket_tts_handler.py`
  (extra; voice cloning with preset voices `alba`, `marius`, `javert`, `jean`, `fantine`,
  `cosette`, `eponine`, `azelma`), `chatTTS_handler.py` (extra), `facebookmms_handler.py`
  (extra).
- `connections/` — IO adapters: `local_audio_streamer.py` (mic/speakers), `socket_receiver.py`
  + `socket_sender.py` (TCP), `websocket_streamer.py` (raw PCM WebSocket).

Deprecated implementations, including MeloTTS, live in `archive/` and are intentionally not
wired into `s2s_pipeline.py`.

### CLI argument structure

CLI flags come from `arguments_classes/`. Convention:

- Each backend has its own dataclass; field name is the flag (`--stt_device`, etc.)
- `rename_args(args, prefix)` (in `s2s_pipeline.py`) strips a per-handler prefix and folds
  `*_gen_*` keys into `gen_kwargs[...]` so e.g. `--llm_gen_max_new_tokens 128` becomes
  `gen_kwargs={"max_new_tokens": 128}` for the local LLM
- LLM model selection (`--model_name`, `--chat_size`, `--init_chat_prompt`,
  `--enable_lang_prompt`) is shared across all LLM backends
- For the OpenAI-compatible backends the connection flags use the `responses_api_` prefix
  (`--responses_api_base_url`, `--responses_api_api_key`, `--responses_api_stream`,
  `--responses_api_reasoning_effort`)

### Scripts

`scripts/` contains standalone helpers, each runnable with `python scripts/<name>.py --help`:

- `listen_and_play.py` — TCP socket client (mic + speaker)
- `listen_and_play_realtime.py` — OpenAI Realtime client (mic + speaker)
- `synthetic_conversation_realtime_client.py` — drives the server with synthetic audio for
  end-to-end tests
- `benchmark_stt.py` / `benchmark_tts.py` — comparison harnesses; the TTS one accepts
  `--qwen3_mlx_quantizations bf16 4bit 6bit 8bit` to compare MLX quantizations

## Repository conventions

- **Branch / PR naming:** never include `codex` in branch names or PR titles
  (per `AGENTS.md`).
- **Releases:** `AGENTS.md` documents the flow — bump `version` in `pyproject.toml` and
  `__version__` in `src/speech_to_speech/__init__.py`, then tag `vX.Y.Z`; PyPI publishing
  is handled by `.github/workflows/publish.yml` and should not be done manually unless the
  workflow is unavailable.
- **No committed build artifacts** — `dist/`, `build/`, wheels, and sdists stay local.
- **Code style:** ruff with `select = ["E", "F", "I", "W"]`, `ignore = ["E501"]`, line
  length 120 (see `pyproject.toml`). Format-check is part of CI. Many files begin with
  `# ruff: noqa: I001` because Pydantic / OpenAI / numpy typing import ordering clashes
  with isort defaults.
- **Mypy:** `pyproject.toml` has `[tool.mypy]` with `ignore_missing_imports = true` and
  `check_untyped_defs = false`; also see `mypy.ini`. The CI job runs `mypy src/`.
- **Tests:** `pytest-asyncio` is in `auto` mode. `tests/install_smoke.py` is the only test
  CI runs against the built wheel (with `OPENAI_API_KEY` unset, to verify the CLI imports
  without loading models or calling OpenAI).

## Key reference docs

- `README.md` — user-facing overview, quickstart, backend matrix, run modes, LLM provider
  table, CLI reference
- `AGENTS.md` — release process and PyPI publishing rules
- `src/speech_to_speech/api/openai_realtime/README.md` — Realtime protocol events,
  architecture diagrams, CancelScope + interruption design, WebRTC transport, tool calling
- `src/speech_to_speech/STT/README.md` — per-handler language coverage
- `src/speech_to_speech/LLM/README.md` — LLM backends, args prefixes, sample invocations
- `src/speech_to_speech/TTS/README.md` — TTS backends and the Qwen3-TTS CUDA wheel wheelhouse
