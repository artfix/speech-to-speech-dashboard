# Hermes Agent tab — design plan (v1, locked 2026-07-30)

This document captures the locked v1 design for adding **Hermes Agent**
integration to the speech-to-speech dashboard. It is the source of truth —
code, changelog entries, and Guide updates must align with this plan.

The decision was made **after a prior conversation** in which we weighed
six candidate architectures (build a shim, add a new pipeline backend,
wrap hermes in MCP, etc.) and found that Hermes Agent already ships a
built-in OpenAI-compatible HTTP server on port 8642
(`NousResearch/hermes-agent/gateway/platforms/api_server.py`), so the
integration reduces to a thin dashboard subprocess manager.

Prior art surveyed before locking the plan:

- 4 Reachy Mini + Hermes projects: `The-Focus-AI/hermes-body`,
  `ai-ag2026/reachy-hermes-stack`, `mrjonathanm/hermes-sts-server`,
  and the seven HF Spaces community apps under
  `pollen-robotics-reachy-mini.hf.space/apps`.
- ~20 voice + Hermes projects of which the most relevant are
  `eadmin2/jarvis_ai` (FastAPI, ★108), `bielcarpi/hermes-live-voice`
  (TypeScript, ★16), `VivaldiCode/voice-gateway` (Electron + aiohttp
  bridge), `baladithyab/hermes-s2s` (latency matrix), and
  `Anquietas86/hermes-realtime-bridge` (Realtime adapters).

## 1. Goals (in the user's own words)

> "When I talk to the robot, the robot uses everything hermes-agent can do
> — its skills, its HA control, its evolving personality/memory. Hermes
> is the brain, the model (gpt-oss-20b) is just the engine inside
> hermes. The model is picked inside hermes, not in the dashboard. But I
> want to keep the speech-to-speech pipeline because users can swap
> stt/tts/vad/llm there."

What this means concretely:

- **hermes-agent is the brain.** Skills, memory, MCP servers, HA control,
  evolving behavior all live in hermes-agent. The dashboard does not
  reimplement any of this.
- **The dashboard is hermes-agent's voice + console.** Voice via the
  existing realtime pipeline, text via a chat panel in the Hermes tab.
- **The model lives inside hermes-agent.** Model is picked via hermes's
  own model picker (e.g. `ollama launch hermes`). The dashboard shows
  *which* model hermes is currently using but does not pick it.
- **The pipeline keeps its job.** Users still configure stt/tts/vad in
  the dashboard. Hermes does not replace the pipeline; it sits *behind*
  it as the LLM endpoint.

## 2. Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ dashboard (FastAPI + vanilla JS)                               │
│                                                                │
│   ┌──────────────────┐         ┌────────────────────────────┐  │
│   │  existing tabs   │         │  NEW: Hermes tab           │  │
│   │  Mode / VAD /    │         │  start / stop / cancel /   │  │
│   │  STT / LLM / TTS │         │  kill / filler phrases /   │  │
│   │  / Settings /    │         │  console / logs            │  │
│   │  Guide / Control │         └──────────────┬─────────────┘  │
│   └────────┬─────────┘                        │                │
│            │ spawns                           │ spawns         │
│            ▼                                  ▼                │
│   ┌──────────────────┐         ┌────────────────────────────┐  │
│   │  pipeline proc   │  http   │  hermes-agent proc         │  │
│   │  (speech-to-     │ ◀─────▶ │  (gateway run on :8642,    │  │
│   │   speech)        │         │   /v1/chat/completions     │  │
│   │  WebSocket       │         │   OpenAI-compatible SSE)   │  │
│   │  on :8765        │         └──────────────┬─────────────┘  │
│   └──────────────────┘                        │                │
└────────────────────────────────────────────────┼───────────────┘
                                                 │ http
                                                 ▼
                                    ┌────────────────────────┐
                                    │  ollama / vllm / etc.  │
                                    │  (model server)        │
                                    └────────────────────────┘
```

**Two subprocesses owned by the dashboard.** No edits to
`src/speech_to_speech/`. The pipeline talks to hermes-agent the same way
it would talk to ollama — base URL + (optional) bearer key + a session
id header.

## 3. The Hermes tab (new)

All hermes-agent functionality lives in this tab. No hermes controls leak
to other tabs. This is the locked placement rule.

### 3.1 Layout (top to bottom)

```
┌─────────────────────────────────────────────────────────────┐
│ HERMES AGENT                                  ● running :8642│
├─────────────────────────────────────────────────────────────┤
│  [ Start ]  [ Polite Stop ]                                 │
│                                                              │
│  Model: gpt-oss-20b (via ollama)        API key: ●●●●●●●●●●  │
│                                                              │
│  ── Filler audio ──────────────────────────────────────────  │
│  [●] Enable filler phrases                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ one moment                                           │    │
│  │ let me check                                         │    │
│  │ thinking...                                          │    │
│  │ (one phrase per line, up to 10)                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ── Console (text chat with hermes) ───────────────────────  │
│  [type here.................................] [ Send ]       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ hermes: hi, what can I help with?                    │    │
│  │ you:   what's the weather?                           │    │
│  │ hermes: I'll check via my HA skill...                │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ── Logs ──────────────────────────────────────────────────  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ [INFO] hermes-agent started on :8642                 │    │
│  │ [INFO] loaded skill: home_assistant                  │    │
│  │ [INFO] loaded model: gpt-oss-20b                     │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  ── Emergency ─────────────────────────────────────────────  │
│  [ Cancel hermes ]    [ KILL HERMES ]    (red, with confirm) │
│                                                              │
│  [ Open hermes dashboard ↗ ]                                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Buttons

| Button           | Color  | What it does                                                                 | Context preserved? |
|------------------|--------|------------------------------------------------------------------------------|--------------------|
| **Start**        | green  | Spawns hermes-agent subprocess on port 8642 with auto-generated `API_SERVER_KEY`. | n/a                |
| **Polite Stop**  | gray   | Sends shutdown signal; hermes finishes current turn, then exits.             | yes                |
| **Cancel**       | amber  | Sends hermes-agent cancel signal mid-turn. Clean stop.                       | yes                |
| **Kill Hermes**  | red    | `SIGKILL` on the hermes subprocess. Hard kill. Confirmation prompt required. | no (fresh start)   |
| **Open hermes dashboard** | link | Opens hermes-agent's own web UI in a new tab (if running).             | n/a                |

**Kill is intentionally not bound to voice or keyboard shortcuts.** It
requires the button + confirmation. Cancel is the safe mid-turn stop;
kill is the last-resort safety net for "hermes is about to do something
destructive in HA and won't stop."

### 3.3 Filler audio

- Fires when hermes has been mid-tool-call (>1.5s without producing
  text) and the user is waiting for an answer.
- Uses the **pipeline's existing tts** (no new dependency, no new voice
  config). The user already configured their preferred tts.
- Picks a random phrase from the 10-line box. Default phrases:
  `one moment`, `let me check`, `thinking...`, `give me a second`.
- On/off toggle (default on).

## 4. LLM settings — "use hermes" toggle

A new dropdown at the top of the LLM settings tab:

```
Backend type:  ( • ) Direct backend       ( ) Hermes Agent
              ┌──────────────────────────────────────────┐
              │ ● green = hermes tab running             │
              │ ● gray  = hermes tab not running         │
              │ Auto-fills when Hermes is selected:      │
              │   base_url = http://127.0.0.1:8642/v1    │
              │   api_key  = <from Hermes tab>           │
              │   header   = X-Hermes-Session-Id: <uuid> │
              └──────────────────────────────────────────┘
```

- Picking **Direct backend** restores today's behavior (URL + model
  name + api key fields work as before).
- Picking **Hermes Agent** auto-fills the LLM settings so the pipeline
  points at hermes-agent instead of ollama directly.
- The session id is regenerated on every pipeline restart (intentional —
  fresh conversation chapter).

## 5. Session handling (decision A)

**One hermes session per pipeline lifetime.**

- Dashboard generates a uuid at pipeline start.
- Stored in `web_ui_settings.json`.
- Every LLM request sends `X-Hermes-Session-Id: <uuid>`. Hermes keeps
  the conversation in memory for that session id.
- Pipeline restart → new uuid → fresh hermes context (intentional).
- `compress_context` is invoked periodically (every ~20 turns) via
  hermes-agent's built-in tool to keep the session bounded. Configurable
  in the Hermes tab.

## 6. Cancellation flow

Three layers, all opt-in:

1. **Pipeline barge-in** (already works). Sends `response.cancel` on
   the SSE stream. Hermes-agent honors it (verified by
   `mrjonathanm/hermes-sts-server`'s `ws_cancel_smoke.py`).
2. **Cancel button** (manual, amber). Same signal as barge-in but
   user-initiated.
3. **Kill button** (manual, red). Subprocess kill. Last resort.

## 7. Settings schema

New top-level section in `web_ui_settings.json`:

```json
{
  "hermes": {
    "enabled": false,
    "port": 8642,
    "api_key": "<auto-generated hex>",
    "session_id": "<auto-generated uuid, refreshed on pipeline restart>",
    "filler_enabled": true,
    "filler_phrases": [
      "one moment",
      "let me check",
      "thinking...",
      "give me a second"
    ],
    "compress_context_every_n_turns": 20
  }
}
```

`api_key` is generated on first start (32 random bytes via
`secrets.token_hex(32)`) and persisted. It never leaves the machine.

## 8. Server endpoints (new)

Mirroring the existing `/api/process/*` pattern:

| Method | Path                            | Purpose                                    |
|--------|---------------------------------|--------------------------------------------|
| POST   | `/api/hermes/start`             | Start hermes-agent subprocess              |
| POST   | `/api/hermes/stop`              | Polite stop                                |
| POST   | `/api/hermes/cancel`            | Send cancel signal                         |
| POST   | `/api/hermes/kill`              | Kill subprocess (requires confirm token)   |
| GET    | `/api/hermes/status`            | running? port? model loaded?               |
| POST   | `/api/hermes/chat`              | Stream chat via SSE for the console        |
| GET    | `/api/hermes/logs?tail=N`       | Recent hermes stdout/stderr lines          |
| PUT    | `/api/hermes/filler`            | Update filler phrases / toggle             |
| PUT    | `/api/hermes/compress_interval` | Update compress_context cadence            |

## 9. Constraints respected

- ✅ No edits to `src/speech_to_speech/`. Pipeline is unchanged.
- ✅ Auto-introspect (no edits): the LLM dropdown change reuses existing
  fields with new defaults; no per-flag special cases in `app.js`.
- ✅ Settings file: `web_ui_settings.json` in repo root, gitignored.
- ✅ Cross-platform subprocess management (POSIX `SIGTERM`/`SIGKILL`,
  Windows `CTRL_BREAK_EVENT` + taskkill).

## 10. Out of scope for v1

The following were considered and **deferred**:

- **Voice-activated cancel.** Discussed and dropped by user decision.
  May return in a future version once voice activity detection and
  phrase matching can be done without editing `src/speech_to_speech/`.
- **Hermes-as-realtime-frontend** (the `gpt-realtime` + hermes-as-tool
  pattern from `The-Focus-AI/hermes-body`). Different product; current
  v1 is hermes as direct LLM endpoint, matching the user's mental model.
- **MCP integration from the dashboard side.** Hermes-agent already
  speaks MCP; the dashboard doesn't need to.

## 11. Implementation order

Tasks are tracked in TaskList. Dependency order:

1. `docs/HERMES_TAB_PLAN.md` (this file) — capture decisions.
2. `web_ui/hermes_manager.py` — subprocess lifecycle.
3. `web_ui/server.py` — `/api/hermes/*` endpoints.
4. `web_ui/settings_schema.py` — schema additions.
5. `web_ui/static/app.js` + `index.html` + `base.css` — Hermes tab UI.
6. Session id injection into the pipeline env.
7. `compress_context` background scheduler.
8. `web_ui/Guide.md` — "Connecting to Hermes Agent" section.
9. Version bump + CHANGELOG entry.

## 12. Why no shim

Hermes-agent already ships an OpenAI-compatible HTTP API server
(`gateway/platforms/api_server.py`, aiohttp) on port 8642 with:

- `POST /v1/chat/completions` (streaming SSE + non-streaming)
- `POST /v1/responses` (stateful via `previous_response_id`)
- `GET /v1/models` (model discovery)
- Session headers via `X-Hermes-Session-Id`

This is exactly the protocol the speech-to-speech pipeline already
speaks to ollama. Writing a shim would be reinventing what already
exists upstream. The dashboard's job is to manage hermes-agent's
subprocess and surface its controls — nothing more.