# Changelog

All notable changes to this fork of `speech-to-speech` are documented here. The
format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and this project does not follow SemVer — versions reflect the dashboard's
own release cadence.

## [Unreleased] — Web dashboard

### 0.5.10 — 2026-08-10

- Removed the **Settings** sidebar tab (9 → 8 tabs). Its content moved into a
  **Configuration** section at the top of the Status & Logs tab, right under
  the Controls cards (settings file status, Environment Variables editor,
  Save / Reset / Export / Import / Load Example). Behaviour-neutral: the same
  standalone functions and the same `web_ui_settings.json` file back it. A
  stale `#settings` URL hash now falls back to Mode.
- Mode / VAD / STT / LLM / TTS tabs restyled to group their fields into panel
  cards matching the other tabs. CSS-only — no field ids, classes, handlers,
  or DOM structure changed, so disabled-when, the Hermes backend toggle,
  Ollama keepalive, the model dropdown, and the voice libraries are untouched.
- Refreshed the HF Space landing page (`index.html`) to v0.5.10 and removed
  the unreferenced duplicate `app_index.html`. Fixed stale references to the
  deleted repo-root `web_ui_settings.example.json` (the live example is
  `web_ui/settings.example.json`, loaded by the Load Example button).

### 0.5.9 — 2026-08-10

- **Hermes tab redesigned into a card dashboard**: a Status row of metric
  cards (State, Model, Endpoint, Uptime, PID, Port), a Controls row of
  action cards (Start, Polite Stop, Cancel, Reset session, Open Hermes
  Dashboard, Kill), and Endpoint / Tuning / Filler / Console / Logs panels.
- The Hermes **Model card shows the actually-loaded model** (read from
  `hermes config get`), fetched on paint + start/stop + a 30 s drift timer —
  not the stale dashboard `--model-name`.
- `--responses-api-num-ctx` hidden in the LLM tab when the Hermes backend is
  selected (it is Ollama-native and ignored by the Hermes proxy).
- Hermes Logs panel respects scroll position (auto-follow only at the
  bottom).
- Guide tab redesigned into sub-tabs (one per section) laid out as cards
  with GFM tables and callout blocks.

### 0.5.8 — 2026-08-09

- Dashboard version bump.

### 0.5.7 — 2026-08-07

- **30 themes** total (10 new color themes + 2 to round up). The theme
  picker auto-discovers `.css` files in `web_ui/themes/`.
- Per-flag "Use" checkboxes; default profile switched to llama.cpp + Qwen3;
  Qwen3 dashboard-side unload.
- Restored the original upstream `demo/` folder.
- Removed the unused repo-root `web_ui_settings.example.json` (the live
  example now lives at `web_ui/settings.example.json`).

### 0.5.6 — 2026-08-07

- Qwen3-TTS unload button support (dashboard-side).
- Disabled the Hermes filler patch loader and Ollama lifecycle poll by
  default.

### 0.5.5 — 2026-08-06

- Hermes filler audio runtime patch with configurable delay.

### 0.5.4 — 2026-08-06

- Hermes log tailing, stderr verbosity control, per-phrase filler UI, and
  tab persistence.

### 0.5.2 — 2026-08-05

- Reasoning-strip parser (strips LLM reasoning tokens from TTS input) +
  `--responses-api-max-tokens` + LLM tab cleanup.

### 0.5.1 — 2026-08-03

- **Parakeet ONNX STT backend** (`--stt parakeet-onnx`): three NVIDIA Parakeet
  TDT variants via `onnx-asr` (pure ONNX, no PyTorch for STT) — `v2`
  (English), `v3` (25 EU langs, default), `v3sq` (int8 SmoothQuant). Auto-
  install, GPU auto-detect, progressive/final gating. CPU / CUDA only;
  Apple Silicon keeps `parakeet-tdt`.

### 0.4.3 — 2026-08-03

- **Per-backend LLM request timeout (s)** dropdown on the LLM tab —
  overrides the pipeline's hardcoded 20 s openai-SDK read timeout.
- **Ollama model lifecycle section** at the bottom of the LLM tab: live
  `ollama ps` status + an Unload & reload button (fixes "num_ctx doesn't
  stick"). Auto-unloads when `--responses-api-num-ctx` changes.
- Keepalive fix.

### 0.4.2 — 2026-08-03

- Per-backend LLM request timeout plumbed through; the `sitecustomize.py`
  OpenAI-SDK timeout patch is no longer gated to Hermes-only mode.

### 0.4.1 — 2026-07-30

- **Hermes-only LLM read-timeout knob** (`hermes.read_timeout_s`) via a
  zero-touch `sitecustomize.py` monkey-patch loaded into the venv on
  pipeline start, removed on stop.
- Read-only `--model-name` label when the Hermes backend is selected (the
  model is chosen via `hermes model` in the terminal).

### 0.4.0 — 2026-07-30

- **Hermes Agent integration**: a new Hermes tab (start/stop/cancel/kill,
  text console, filler-phrase config) plus an LLM-tab "Backend type"
  dropdown that auto-fills the reverse-proxy URL and API key. The reverse
  proxy lives inside the dashboard's own uvicorn — no extra port, process,
  or dependency. One session id per pipeline lifetime.
- **Pascal GPU (sm_61) restart-loop fix**: `gpu.py` pins torch to the
  bundled CPU wheel on Pascal + a runtime CPU fallback for any other CUDA
  init failure; `process_manager.py` tracks the restart count; the qwen3
  installer defers its heavy CUDA work until the first TTS request.
- **Ollama LLM keepalive** (`web_ui/llm_keepalive.py`) pings the endpoint
  so the model stays loaded in VRAM between turns.

### 0.3.10 — 2026-07-28

- Chatterbox per-sentence split with safer cancel-scope placement.

### 0.3.9 — 2026-07-28

- Reverted per-sentence split in chatterbox TTS to restore realtime audio.

### 0.3.8 — 2026-07-28

- Chatterbox sentence-splitting + split voice library mounts.

### 0.3.7 — 2026-07-28

- Reverted warmup defaults to the original persona-priming strings.

### 0.3.6 — 2026-07-28

- Render warmup prompts as textareas in the dashboard.

### 0.3.5 — 2026-07-28

- Expose warmup system / user prompts in the dashboard.

### 0.3.4 — 2026-07-28

- Fix pipeline warmup timeout; re-enable `num_ctx` on warmup. Curated
  dropdown for `--responses-api-reasoning-effort`.

### 0.3.3 — 2026-07-28

- Cap the Ollama context window via `--responses-api-num-ctx`.

### 0.3.2 — 2026-07-28

- Ollama model dropdown + one-shot keepalive on Save / Start.

### 0.3.0 — 2026-07-27

#### Added

- **qwen3-TTS voice library in the dashboard.** Card grid of the 9
  CustomVoice preset speakers (Vivian, Serena, Uncle_Fu, Dylan, Eric,
  Ryan, Aiden, Ono_Anna, Sohee) with native-language badges, sample
  sentences, "Set active" and "Test" buttons. Mirrors the Chatterbox
  voice library UX. New files: `web_ui/static/qwen3_voice_library_ui.js`,
  `web_ui/qwentts_voice_library.py`.
- **qwen3-TTS dropdowns for speaker and language.** `--qwen3-tts-speaker`
  is now a `<select>` with the 9 presets; `--qwen3-tts-language` lists
  `auto` + 10 languages (english, chinese, japanese, korean, german,
  french, russian, portuguese, spanish, italian). Pinning the language
  per-request fixes the cross-lingual artefact (e.g. French text
  sent to `Ono_Anna` emitting French-across-Japanese-voice).
  Device / dtype / attn-implementation also get curated dropdowns.
  Values live in `web_ui/static/field_choices.json`.
- **All 5 qwen3-TTS models in the picker.** `qwen3_models.json` lists
  CustomVoice 1.7B/0.6B, Base 1.7B/0.6B, VoiceDesign 1.7B with variant
  metadata and `needs_abi_v2` flags. The dashboard shows a yellow
  info banner when a Base or VoiceDesign model is selected.
- **Reference voices section + upload widget.** Lists uploaded files
  under `voices/qwen3_refs/` with size + mtime, "Use as
  `--qwen3-tts-ref-audio`" and "Test" actions; header reads "require
  ABI v2 — synthesis will fail" in red.
- **Voice Library Test modal language selector.** Pre-filled with the
  speaker's native language (or the dashboard's pinned
  `--qwen3-tts-language` if set to a non-`auto` value).
- **Test-while-running guard.** When the pipeline is running, the qwen3
  voice Test button pops a modal explaining the OOM and offering
  one-click "Stop pipeline & test" (force-refreshes status, waits
  800 ms for CUDA context release, retries).
- **Bundled Pascal wheel** `qwentts_cpp_python-0.3.1-py3-none-linux_x86_64.whl`
  (126 MB) in `web_ui/wheels/`. Re-included in `.gitignore` per the
  `!web_ui/wheels/` rule.
- **Venv shim patcher** in `web_ui/qwentts_installer.py`
  (`_patch_faster_qwen3_tts_if_needed`). The Pascal wheel is ABI v1
  only — the upstream `get_supported_speakers()` calls into the
  missing ABI v2 `runtime.speaker_names()` symbol, raising
  `QwenTTSError: Speaker enumeration requires ABI v2` on every OpenAI
  Realtime `session.update`. The shim rewrites the method body to
  `return []` (idempotent, marker comment), causing the upstream
  handler's existing "ignore client voice, use configured" branch
  to fire — the configured `--qwen3-tts-speaker` wins. Reapplied
  automatically on every `uv sync` via the installer.
- **Five new endpoints** in `web_ui/server.py`: `GET /api/qwen3/voices`,
  `POST /api/qwen3/voice/{speaker}/set-active`,
  `POST /api/qwen3/voice/test`,
  `DELETE /api/qwen3_ref_audio/{name}`,
  `GET /api/qwen3_ref_audio/list`.
- **Example settings file** `web_ui_settings.example.json` — a real
  working configuration (Ollama + qwen3-TTS + parakeet STT, realtime
  mode). The Settings tab's Import button can load it as a starting
  point. The `192.168.1.2` IP and `Ono_Anna` voice pick are the
  author's actual working config — no secrets exposed.
- **Three new Guide.md troubleshooting entries**: qwen3 voice selection
  in Realtime clients, ABI v2 limitation, model picker guidance.

#### Changed

- `web_ui/static/app.js` — qwen3 voice library mount, ABI v2 banner,
  ref-audio upload widget.
- `web_ui/static/index.html` — loads `qwen3_voice_library_ui.js` before
  `app.js`.
- `web_ui/static/qwen3_models.json` — version 1 → 2.
- `web_ui/static/field_choices.json` — new file with curated dropdown
  values for plain-`str` fields.
- `web_ui/settings_schema.py` — 3-line patch teaching schema
  introspection about the new curated fields.
- `web_ui/process_manager.py` — small refactor plus a new
  `_install_qwentts_pascal_wheel` step that runs once before the
  pipeline subprocess starts (idempotent).

#### Hard-constraint preservation

- **Zero changes to `src/speech_to_speech/`.** Confirmed by
  `git diff --stat main^..main -- src/` returning empty.

### 0.2.2 — 2026-07-27

- Fix inline `?` help buttons on tabs with duplicate flag IDs.

### 0.2.1 — 2026-07-27

- Auto-detect GPU + install the matching torch; auto-fallback for cudnn on
  Pascal.

### 0.2.0 — 2026-07-23

- **Chatterbox TTS backend** (the sixth TTS backend) with an on-disk voice
  cloning library (`voices/<name>.pt`): list / clone / delete / test from
  the dashboard. Three variants: `chatterbox` (English 500M),
  `chatterbox-turbo` (350M), `chatterbox-nano` (110M). A `UNLOAD_TTS`
  control message walks the handler chain so only the TTS handler drops
  the model. One-click install streams the `uv pip install` log into the
  Status tab.

### 0.1.3 — 2026-07-23

#### Added

- **10 new themes** (18 total now in the dropdown, sorted alphabetically).
  All WCAG-checked: body text on background passes AAA (≥ 7:1) on every
  theme; the muted `--text-dim` used for labels passes AA on most, AA-large
  on the intentionally low-contrast `matrix` and `hacker-terminal` (which
  are designed to look like a phosphor terminal — dim on dim is the point).
  - `dracula` — soft purples on dark indigo
  - `nord` — arctic blue-grey, very low eye-strain
  - `solarized-dark` — classic Ethan Schoonover palette
  - `gruvbox` — warm retro amber + olive on near-black
  - `tokyo-night` — soft blue with pink and cyan accents
  - `one-dark` — Atom editor's signature palette
  - `monokai` — iconic Sublime Text palette
  - `catppuccin` — soft pastels on warm dark
  - `paperwhite` — light theme, dark text on warm off-white
  - `high-contrast` — black + white + bright yellow, max readability

#### Fixed

- **Log console hardcoded to pure black** in `base.css` (`.log-console
  { background: #000 }`) — replaced with `var(--bg)` so the console
  now picks up the active theme's background.
- **Log line colors hardcoded** (`#cccccc`, `#888888`) — replaced
  with `var(--text)` and `var(--text-dim)` so INFO/DEBUG/OTHER lines
  follow the active theme. WARNING/ERROR/CRITICAL already used
  theme tokens.

### 0.1.2 — 2026-07-23

#### Fixed

- **Theme switcher did nothing.** `index.html` loaded the theme stylesheet
  BEFORE `base.css`, so `base.css`'s `:root` color variables always won the
  cascade regardless of which theme was selected. Swapped the `<link>` order
  and split `base.css`'s `:root` block in two — structural tokens
  (`--font`, `--radius`, `--transition`) declared first, color fallbacks
  declared after. The active theme's `:root` rules (loaded last) now win
  and the dropdown actually repaints the UI in the chosen palette.

### 0.1.1 — 2026-07-23

First release of the `web_ui/` package: a self-hosted cyberpunk-themed
dashboard that wraps the `speech-to-speech` CLI without modifying any
code under `src/speech_to_speech/`.

### Added

- **`web_ui/` package** — a FastAPI server that introspects the
  pipeline's argument dataclasses at runtime and renders a web form for
  every CLI flag. No fields are hardcoded; adding a new flag to the
  pipeline's argument classes shows up in the UI on next load.
  - `web_ui/settings_schema.py` — dataclass → JSON schema with
    conditional visibility rules for the per-backend STT / LLM / TTS
    sub-forms.
  - `web_ui/process_manager.py` — cross-platform subprocess lifecycle
    (POSIX SIGTERM, Windows `CREATE_NEW_PROCESS_GROUP` + `CTRL_BREAK`),
    background reader thread, in-memory ring buffer (5 000 lines),
    WebSocket fanout to subscribed browsers.
  - `web_ui/server.py` — FastAPI app: REST endpoints (`/api/schema`,
    `/api/settings`, `/api/process/*`, `/api/system`, `/api/pool`,
    `/api/themes`, `/api/guide`, `/api/shutdown`), WebSocket
    `/ws/logs`, static file mount.
  - `web_ui/static/index.html` + `app.js` + `base.css` — single-page
    vanilla-JS UI, no build step.
  - `web_ui/themes/*.css` — 8 cyberpunk theme presets (cyberpunk-neon,
    matrix, synthwave, blade-runner, tron, vaporwave, hacker-terminal,
    dark-glass). Each is just CSS custom properties on `:root`; adding
    a new `.css` file shows up in the theme dropdown automatically.
  - `web_ui/Guide.md` — in-app walkthrough covering Quickstart, the
    Ollama-on-another-machine recipe, the OpenAI / HF Inference /
    vLLM / llama.cpp provider table, per-backend overview, and
    common issues (CUDA wheel mismatch for Qwen3-TTS, missing
    `espeak-ng` for Kokoro, etc.).

- **Launcher scripts** — `start_web_ui.sh` (Linux / macOS) and
  `start_web_ui.bat` (Windows) install via `uv sync`, attempt to open
  the browser, then run `speech-to-speech-web`.

- **10 dashboard tabs** — Mode, VAD, STT, LLM, TTS, Advanced,
  Status & Logs, Guide, Settings, Control. Each STT / LLM / TTS tab
  shows a backend dropdown plus the per-backend sub-form for the
  selected backend.

- **Status & Logs** — live pipeline state, system resource usage
  (CPU / RAM / GPU via `psutil` + `nvidia-smi`), realtime `/v1/pool`
  proxy, level filter (ALL / INFO+ / WARNING+ / ERROR-only), Toggle
  Verbose restarts the pipeline with `--log-level debug` after a
  confirmation modal.

- **Settings persistence** — `web_ui_settings.json` in the repo root
  (gitignored). Atomic write (`.tmp` + rename) so a crash mid-save
  can't corrupt the file. Import / Export buttons in the Settings tab.

- **Shutdown button** — single click stops the pipeline and the
  dashboard web server.

- **Env-injection helper** (`web_ui/process_manager.py` →
  `_maybe_inject_openai_api_key`) — when the user points the
  OpenAI-compatible LLM slot at a non-OpenAI host (Ollama, vLLM,
  llama.cpp) and leaves the API key blank, the dashboard
  auto-injects `OPENAI_API_KEY=ollama` into the subprocess
  environment so the `openai` Python client can construct itself.
  Users can still override the key in the Settings → Environment
  Variables editor.

- **Version endpoint** (`/api/version`) — single source of truth is
  `web_ui/__version__`. The frontend fetches it on load and renders
  it as a badge in the title bar and in the page `<title>`. Bumping
  `web_ui/__init__.py` is enough to make the new version visible
  to users — no JS or CSS changes required.

### Changed

- **`pyproject.toml`** — added the `speech-to-speech-web` console
  script entry point, declared `web_ui*` as a discoverable package,
  listed `web_ui/static/`, `web_ui/themes/`, `web_ui/Guide.md`, and
  `web_ui/README.md` as package data, added `psutil>=5.9.0` to
  dependencies (used by the dashboard's `/api/system` endpoint for
  CPU / RAM metrics).
- **`README.md`** — added a "Option A: Web dashboard" section at the
  top with a one-liner to launch the dashboard, plus a "From Source"
  mention.
- **`.gitignore`** — added `web_ui_settings.json` so each user's
  saved dashboard config stays local.

### Notes for upstream maintainers

- **Zero changes to `src/`.** The dashboard is an additive wrapper.
  All behaviour comes from existing CLI flags, the existing
  `/v1/realtime` WebSocket API, and the existing `/v1/pool` status
  endpoint.
- **Optional dependency:** `psutil` is the only new dep. The
  pipeline itself is unchanged.
- **Tested stack on a CPU-only Linux box** (no GPU, no CUDA):
  - VAD: Silero v5 (bundled)
  - STT: Parakeet TDT (default)
  - LLM: Ollama (local) via `--llm-backend chat-completions` and
    `--responses-api-base-url http://127.0.0.1:11434/v1`
  - TTS: Pocket TTS (pure-Python CPU backend)
  - Confirmed end-to-end: pipeline → Ollama → response → pool
    reports `available: true`, `size: 1`, `in_use: 0`.

### Known issues / next steps

- **Kokoro TTS** is on the back-end dependency tree, but its
  `spacy` + `thinc` wheels are built against numpy <2 and this box
  has numpy 2.4.3. Forcing a source rebuild of `thinc` against
  numpy 2.x fails (Cython compile error). Downgrading numpy would
  force rebuilding torch + scipy as well, which is too risky for a
  v1. **Recommendation:** ship Pocket as the default TTS on
  CPU-only installs; document the numpy constraint for Kokoro.
- **Qwen3-TTS** — the `qwentts-cpp-python` wheel in this environment
  links to `libcudart.so.12` even when downloaded from the
  `whl/cpu/` path on the HF wheelhouse. The CPU variant appears to
  ship the same CUDA backend. Users on CUDA-less boxes must pick
  a different TTS.
- **Authentication** — the dashboard binds to `0.0.0.0:8050` by
  default. For LAN exposure, recommend binding to `127.0.0.1` only
  (set `SPEECH_TO_SPEECH_WEB_HOST=127.0.0.1` before launching).
