# Changelog

All notable changes to this fork of `speech-to-speech` are documented here. The
format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and this project does not follow SemVer — versions reflect the dashboard's
own release cadence.

## [Unreleased] — Web dashboard

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
