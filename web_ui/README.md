# Speech-to-Speech Web Dashboard

A cyberpunk-styled browser GUI that wraps the `speech-to-speech` CLI. The dashboard introspects the pipeline's argument classes at runtime, so every CLI flag the pipeline exposes shows up in the UI automatically — no hardcoded field list to keep in sync.

The dashboard **does not modify any code in `src/speech_to_speech/`**. It spawns the pipeline as a subprocess with the user's chosen settings and streams its logs back to the browser over a WebSocket.

## Install + run

```bash
git clone https://github.com/huggingface/speech-to-speech.git
cd speech-to-speech
uv sync
./start_web_ui.sh          # Linux / macOS
start_web_ui.bat           # Windows
```

The browser opens to `http://localhost:8050`. If it doesn't (headless box, no `xdg-open` etc.), open it manually.

## What you get

- **Tabs:** Mode, VAD, STT, LLM, TTS, Advanced, Status & Logs, Guide, Settings, Control.
- **Auto-generated forms.** Every CLI flag the pipeline knows about shows up in the relevant tab. Help text is shown next to every field via a `?` button.
- **Conditional sub-forms.** When you change `--tts` from `qwen3` to `pocket`, the TTS fields swap to Pocket TTS's settings. Same for STT and LLM.
- **8 cyberpunk themes.** Cyberpunk Neon (default), Matrix, Synthwave, Blade Runner, Tron, Vaporwave, Hacker Terminal, Dark Glass. Switch from the top-right dropdown.
- **Save / Reset / Import / Export** the settings JSON. The file lives at `web_ui_settings.json` in the repo root.
- **Start / Stop / Restart** buttons for the pipeline. The pipeline is spawned as a subprocess with the current settings.
- **Live logs** in the Status & Logs tab. Filter by level (ALL / INFO / WARNING / ERROR). Toggle Verbose restarts the pipeline with `--log-level debug` (with a confirmation).
- **Realtime pool status.** When the pipeline is running in `realtime` mode, the existing `/v1/pool` endpoint is polled and shown.
- **Shutdown button.** Closes the pipeline AND the dashboard web server in one click.
- **Settings start empty.** First run, the form shows the pipeline's dataclass defaults. Nothing is saved until you click **Save Settings**. **Reset** wipes back to defaults.

## Architecture

```
┌─────────────────┐     subprocess      ┌──────────────────────────┐
│  FastAPI server │ ─────────────────── │  python -m               │
│  (uvicorn:8050) │ ◀──── stdout+stderr │  speech_to_speech        │
└────────┬────────┘     (log thread)    │  .s2s_pipeline <args>    │
         │                              └──────────────────────────┘
         │ WebSocket /ws/logs                    ▲
         ▼                                       │ arguments
   ┌──────────┐                                  │
   │ Browser  │   settings (JSON)                │
   │  (app.js)│ ───────────────────────────────►─┘
   └──────────┘
```

- `web_ui/settings_schema.py` — introspects the argument dataclasses and produces the form schema the JS uses.
- `web_ui/process_manager.py` — owns the subprocess lifecycle (cross-platform), tails stdout+stderr, fans log lines out to websocket subscribers.
- `web_ui/server.py` — FastAPI app: REST endpoints, WebSocket, static files, shutdown.
- `web_ui/static/index.html` + `app.js` — single-page UI, vanilla JS, no build step.
- `web_ui/themes/*.css` — 8 theme files, each just defines CSS custom properties.
- `web_ui/Guide.md` — the in-app walkthrough (Quickstart, Ollama recipe, common issues, …).
- `web_ui_settings.json` — saved user settings. Gitignored.

## Configuration via environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SPEECH_TO_SPEECH_WEB_HOST` | `0.0.0.0` | Interface the dashboard binds to. |
| `SPEECH_TO_SPEECH_WEB_PORT` | `8050` | Port the dashboard listens on. |
| `SPEECH_TO_SPEECH_WEB_LOG_LEVEL` | `info` | Python log level for the dashboard server. |

## Custom themes

Drop a new `.css` file in `web_ui/themes/`. It should only override CSS custom properties on `:root`:

```css
:root {
    --bg: #...;          /* page background */
    --bg-elev: #...;     /* panels */
    --border: #...;      /* panel borders */
    --text: #...;        /* body text */
    --text-dim: #...;    /* secondary text */
    --accent: #...;      /* primary highlight */
    --accent-2: #...;    /* secondary highlight */
    --success: #...;
    --warning: #...;
    --error: #...;
    --info: #...;
    --debug: #...;
    --font: '...', ...;      /* UI font */
    --font-mono: '...', ...; /* monospace font */
}
```

The theme name (filename without `.css`) appears in the top-right dropdown automatically.

## What the dashboard does NOT do

- It does not modify any code in `src/speech_to_speech/`. The pipeline package is treated as a black box.
- It does not auto-restart the pipeline on crash. Click Restart.
- It does not provide authentication. Bind to localhost (`127.0.0.1`) if you don't want others on the LAN to access it.
- It does not support multiple saved profiles. One settings file, one config.
- It does not pull models for you. Run `ollama pull` / etc. on the relevant machine before starting the pipeline.

## Troubleshooting

- **Port 8050 in use:** set `SPEECH_TO_SPEECH_WEB_PORT`.
- **Browser doesn't auto-open:** headless box, no `xdg-open`. Just open `http://localhost:8050` manually.
- **Settings not persisting:** check the **Settings** tab — it shows the file path and whether it's saved.
- **Pipeline won't start:** open **Status & Logs** and click **Toggle Verbose** to see the full error.
- **Theming broken:** check the browser console. Most likely a CSS syntax error in a custom theme.

## License

Apache 2.0 (same as the parent project).
