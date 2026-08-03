# Ollama num_ctx reload — locked plan

## Context

`--responses-api-num-ctx` is the Ollama context window knob. The dashboard
saves it, the pipeline forwards it on every chat-completions call as
`extra_body={"options":{"num_ctx": N}}`, and the dashboard's own warmup
fires it on pipeline startup. **All the plumbing is correct.**

The bug: Ollama ignores `num_ctx` on requests for a model that is
**already loaded** in its KV cache at a smaller context. Once a model
is resident at, say, 4096 ctx, every subsequent request with
`num_ctx=16000` is silently dropped — `ollama ps` keeps showing 4096
until the model is fully unloaded.

Repro on this box: user set `--responses-api-num-ctx=16000`, pipeline
received `--responses-api-num-ctx 16000` in argv, warmup fired, but
`ollama ps` still showed `CONTEXT = 4096` because the model was
already resident from a prior session.

## Fix

Two coordinated changes so the user's saved `num_ctx` actually sticks:

1. **Auto-unload on `num_ctx` change.** When the user edits the field
   and the value differs from the value Ollama currently holds, the
   dashboard silently sends `keep_alive=0` to unload the model. The
   *next* request (chat completion or warmup) loads it fresh at the
   new context.
2. **Manual "Reload" button.** For the case where the auto didn't
   catch it (stale state from a prior session, or user just wants to
   test a different context). Visible in the LLM tab.

## UI — new "Ollama model lifecycle" section in the LLM tab

Visible only when the URL looks like Ollama (same `_looks_like_ollama_url`
heuristic the existing keepalive uses). Hidden for OpenAI / HF / vLLM /
llama.cpp / non-Ollama OpenAI-compat.

Layout, top-to-bottom:

```
┌─ Ollama model lifecycle ──────────────────────────────────┐
│                                                           │
│  Model:    lfm2.5:8b        (read-only label)             │
│  Endpoint: http://192.168.1.3:11434                       │
│  Context:  4096  (refreshed from /api/ps every 2s)        │
│                                                           │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  🔄 Unload & reload Ollama model with current ctx    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                           │
│  Last action: ok — reloaded at 16000 ctx (3s ago)         │
│                or: skipped — model not loaded             │
│                or: failed — connection refused (after 5s) │
└───────────────────────────────────────────────────────────┘
```

## New server endpoints

```
GET  /api/ollama/ps
   → { ok: true, models: [{ name, size_vram, context, until }] }
   → returns ok: false if Ollama unreachable
   → no-op when URL doesn't look like Ollama

POST /api/ollama/reload
   body: { num_ctx?: int, timeout_s?: int }
   → { ok: true, message, duration_ms }
   → { ok: false, message }
```

## Files

- `web_ui/server.py` — new `/api/ollama/ps` + `/api/ollama/reload` endpoints.
- `web_ui/static/app.js` — `renderOllamaLifecycleField()` in the LLM
  tab. Wired to a 2s poller (matches Hermes tab cadence) that calls
  `/api/ollama/ps` while the LLM tab is visible. Auto-unload hook on
  `--responses-api-num-ctx` change (only fires when Ollama is reachable
  and the model is currently loaded).
- `web_ui/Guide.md` — new section after "LLM request timeout":
  "Ollama model lifecycle" — explains the stale-model gotcha, the
  button, the auto-unload.
- `web_ui/__init__.py` — `__version__` 0.4.2 → 0.4.3.

## Hard constraints

- **No edits to `src/speech_to_speech/`.** All logic in the dashboard.
- **Section hidden when URL isn't Ollama.** No button, no auto-unload,
  no polling for non-Ollama backends.
- **Dashboard-only state.** Reload endpoint reads
  `settings["--responses-api-num-ctx"]` from `web_ui_settings.json`,
  doesn't mutate it.

## Edge cases

1. `num_ctx` null/unset → skip auto-unload; button uses Ollama default.
2. Pipeline running → button still works (chat request to Ollama
   directly reloads at the new context; running pipeline's next chat
   lands on the fresh model). No pipeline restart needed.
3. `--llm-keepalive=-1` → no conflict; keepalive controls unload
   timing *after* warmup, not the unload itself.
4. Button disabled when Ollama unreachable; spinner while reloading.
5. Auto-unload hook fires only when (a) URL is Ollama, (b) Ollama is
   reachable, (c) the model is currently loaded at a different context
   than the user's setting. Silent (no toast) — the user will see the
   result on the next pipeline turn or via the status badge refresh.

## Verification (user-perspective)

1. Restart dashboard on 0.4.3, open LLM tab.
2. Confirm "Ollama model lifecycle" section visible (URL is Ollama).
3. Confirm "Context: 4096" badge matches `ollama ps`.
4. Edit `--responses-api-num-ctx` to 16000. Save Settings. Confirm
   auto-unload fires (check logs / status badge updates).
5. Click Reload button. Confirm new context loads. `ollama ps` should
   show `CONTEXT = 16000`.
6. Switch `--llm-backend-type` to Hermes. Confirm section hides.

## How to apply

Read this file before touching any num_ctx / Ollama-reload code. If
the dashboard already has a 0.4.3 with this feature, skip the plan and
just verify it works.

## Status

Approved by user. Implementation pending — do not start until user
says "go" (we have a pattern of user interrupting plan-mode for
implementation).
