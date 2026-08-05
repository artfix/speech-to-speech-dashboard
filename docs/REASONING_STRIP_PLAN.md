# Reasoning-strip plan — locked plan

## Context

Reasoning-capable LLMs (Qwen3, gpt-oss, DeepSeek-R1, OpenAI o-series, etc.)
emit chain-of-thought tokens **before** their final answer. When such a
model is served by `llama-server` and called through either `/v1/responses`
or `/v1/chat/completions`, those reasoning tokens come back to the
pipeline as **first-class stream events** that the current parsers don't
handle.

The two broken paths:

| Backend | Reasoning arrives as | Current parser |
|---|---|---|
| `responses-api` (`/v1/responses`) | `ResponseReasoningTextDeltaEvent` SSE events + a `ResponseOutputItemDoneEvent` whose `item.type == "reasoning"` | `_iter_stream_events` in `src/speech_to_speech/LLM/responses_api_language_model.py:117` only handles `ResponseTextDeltaEvent`, `ResponseOutputItemDoneEvent`, `ResponseOutputMessage`, and `ResponseCompletedEvent`. Reasoning events fall through silently — and on some reasoning models the missing event classes break the stream-consumed loop. |
| `chat-completions` (`/v1/chat/completions`) | `delta.reasoning_content` field (separate from `delta.content`) on each `ChatCompletionChunk` | `_iter_stream_events` in `src/speech_to_speech/LLM/chat_completions_language_model.py:206` only reads `delta.content` / `delta.refusal` / `delta.tool_calls`. `reasoning_content` is silently dropped. **Less fatal** — reasoning tokens are already separated from the answer in the protocol — but the dashboard has no visibility into them. |

### Repro on this box

`llama-server` running `Qwen3.5-4B-Q8_0.gguf + mmproj`, then a curl
replay of the pipeline's warmup body to `/v1/responses` with
`stream=true`:

```
event: response.output_item.added
data: {"type":"response.output_item.added","item":{"id":"rs_...","type":"reasoning", ...}}

event: response.reasoning_text.delta
data: {"type":"response.reasoning_text.delta","delta":"Thinking Process:\n\n1. **Analyze the Request:** ..."}
```

The pipeline's `responses-api` warmup then hangs forever (no
`response.completed` arrives cleanly because the parser doesn't know
about reasoning items), and the dashboard's pipeline subprocess is
terminated externally.

### Goal

1. **The robot never speaks the LLM's chain-of-thought.** Hard
   invariant, regardless of model, server, or button state.
2. **The pipeline works with any model the user loads** — reasoning
   or non-reasoning, served by llama.cpp / Ollama / vLLM / OpenAI.
3. **One single source of UX truth** for "do you want thinking":
   the dashboard `--responses-api-disable-thinking` checkbox.
4. **Three-layer defense:**
   - **Wire flag** — best-effort, the existing `chat_template_kwargs.enable_thinking=false` mechanism stays. Models that honor it don't reason; models that ignore it still surface reasoning that we strip.
   - **Parser strip** — guaranteed. Reasoning tokens never reach `lm_output_processor` / TTS, no matter what the server emitted.
   - **Log visibility** — opt-in. When the user enables a separate "show reasoning in logs" toggle, reasoning tokens stream into the Status & Logs panel for debugging. Default off.

### Why a separate "show reasoning in logs" toggle (not the same button)

`--responses-api-disable-thinking` semantics in the dashboard today:
unchecked = "let it think", checked = "try to make it stop thinking".
That's a *server-side* preference.

Reasoning-in-logs is a *dashboard-side* preference for visibility. They
should be independent. A user may want the server to think (button
unchecked) but **not** want the verbose chain-of-thought in their logs.
Or the user may want to know *why* a reasoning model is misbehaving
even with thinking disabled at the wire. Decoupling them is the
honest design.

## Fix (three coordinated changes)

### 1. Parser strips reasoning tokens — pipeline side

**Files:**
- `src/speech_to_speech/LLM/responses_api_language_model.py`
- `src/speech_to_speech/LLM/chat_completions_language_model.py`
- `src/speech_to_speech/LLM/base_openai_compatible_language_model.py` (base class)
- `src/speech_to_speech/pipeline/events.py` (new event type)

**New ProviderEvent variant** in `base_openai_compatible_language_model.py`:

```python
class ReasoningDelta(BaseModel):
    """Incremental reasoning text from a thinking/reasoning model.
    Never forwarded to TTS. Optionally surfaced to the dashboard logs."""

    text: str

# Add to the union:
ProviderEvent = TextDelta | AssistantMessage | ToolCall | Usage | ReasoningDelta
```

**`responses-api` parser** (`responses_api_language_model.py:117`): add
two branches before the existing `ResponseTextDeltaEvent` branch:

```python
elif isinstance(raw_event, ResponseReasoningTextDeltaEvent):
    yield ReasoningDelta(text=raw_event.delta)
elif isinstance(raw_event, ResponseOutputItemDoneEvent):
    item = raw_event.item
    if isinstance(item, ResponseReasoningItem):
        # final reasoning summary, also stripped from TTS
        yield ReasoningDelta(text=item.encrypted_content or "")
    elif isinstance(item, ResponseFunctionToolCall):
        ...
```

**`chat-completions` parser** (`chat_completions_language_model.py:234`):
read `delta.reasoning_content` and yield a `ReasoningDelta`:

```python
reasoning_piece = getattr(delta, "reasoning_content", None)
if reasoning_piece:
    yield ReasoningDelta(text=reasoning_piece)
text_piece = delta.content or getattr(delta, "refusal", None)
if text_piece:
    raw_text += text_piece
    yield TextDelta(text=text_piece)
```

**Critical:** ReasoningDelta is consumed only by the new
`LMOutputProcessor` reasoning-tap (next section). The base class's
`_consume_streaming` ignores events it doesn't know about, so we don't
need to thread the toggle through every consumer. Reasoning tokens
**silently drop** unless the dashboard has wired the log sink.

### 2. Optional log sink — pipeline + dashboard

**New pipeline event** in `pipeline/events.py`:

```python
class ReasoningTextEvent(PipelineEvent):
    type: Literal["reasoning_text"] = "reasoning_text"
    text: str
    turn_id: str | None = None
    turn_revision: int | None = None
```

**New module** `src/speech_to_speech/LLM/reasoning_sink.py`:

```python
class ReasoningSink:
    """Forwards ReasoningDelta to the text_output_queue as ReasoningTextEvent
    only when reasoning-log visibility is enabled."""

    def __init__(self, text_output_queue, enabled: Callable[[], bool]):
        self.queue = text_output_queue
        self._enabled = enabled

    def push(self, delta: ReasoningDelta, turn_id, turn_revision) -> None:
        if not self._enabled():
            return
        self.queue.put_nowait(
            ReasoningTextEvent(text=delta.text, turn_id=turn_id, turn_revision=turn_revision)
        )
```

**`LMOutputProcessor`** (in `src/speech_to_speech/LLM/lm_output_processor.py`):
when consuming `ReasoningDelta` from the LLM stream, route to the sink
instead of forwarding to `LMResponseChunk`. Always strip from TTS.

**Dashboard wiring** (`web_ui/server.py` + `web_ui/static/app.js`):

- New endpoint: `GET /api/settings/reasoning_logs_enabled` (bool, default `false`).
- New endpoint: `POST /api/settings/reasoning_logs_enabled` (body: `{enabled: bool}`).
- The setting is passed to the pipeline subprocess via env var
  `SPEECH_TO_SPEECH_SHOW_REASONING_LOGS=1` (similar to the existing
  `--responses-api-num-ctx` flow).
- `PipelineProcess.__init__` reads it from env and threads it into
  `ReasoningSink._enabled`.
- WebSocket send loop in `src/speech_to_speech/api/openai_realtime/websocket_router.py`
  forwards `ReasoningTextEvent` as a new server event `reasoning.delta`
  to the client (parallel to `response.audio.delta`).
- Frontend `app.js` Status & Logs panel renders `reasoning.delta` in a
  collapsible "🧠 Reasoning" sub-panel below the standard log stream.

### 3. UX guarantees (the locked semantics)

| Dashboard button `--responses-api-disable-thinking` | Wire payload | What happens server-side | Spoken by robot | Visible in logs (if reasoning-logs ON) |
|---|---|---|---|---|
| **unchecked** (default) | no `chat_template_kwargs` | model uses its default (likely reasoning on for Qwen, off for LFM2.5) | never | yes (when reasoning emitted) |
| **checked** | `chat_template_kwargs={"enable_thinking": false}` | best-effort — Qwen3 ignores, LFM2.5 already off | never | yes (when reasoning emitted — for diagnostics) |

**The user never hears reasoning, regardless of model or button state.**
That's the invariant. The button controls intent; the parser enforces
the invariant; the log sink is opt-in visibility.

## Why this is a pipeline change, not a dashboard-only change

Hard constraint #1 in `CLAUDE.md` says "no edits to
`src/speech_to_speech/`." That constraint exists to keep the dashboard
fork's diffs upstreamable. This fix is **genuinely a pipeline bug** —
the parser was written before reasoning models were common. The right
place to fix it is the parser. We are making an explicit exception
because:

1. The bug is reproducible against *any* reasoning model served via
   `/v1/responses`, including OpenAI's own o-series. This is upstream-
   worthy; a clean PR is plausible.
2. A dashboard-only monkeypatch would replace the streaming method
   dynamically, which is fragile (breaks on signature changes, module
   refactors) and lives forever as tech debt.
3. Reasoning-strip is a **safety property** for the user. It belongs
   in the pipeline, not in a downstream consumer that could forget to
   apply it.

If the upstream PR is rejected, the patch stays in this fork; it's a
small, contained diff.

## Phase 1 — pipeline parser change (ship-ready independently)

Scope: change 1 only, no dashboard toggle, no log sink. Reasoning
tokens are silently dropped from the spoken output. Existing dashboard
behaviour is unchanged.

**Files touched:**
- `src/speech_to_speech/LLM/responses_api_language_model.py` (+12 lines)
- `src/speech_to_speech/LLM/chat_completions_language_model.py` (+5 lines)
- `src/speech_to_speech/LLM/base_openai_compatible_language_model.py` (+5 lines)

**Verification on this box:**
1. Restart `llama-server` with `Qwen3.5-4B-Q8_0.gguf`.
2. Save `--llm-backend=responses-api`, `--model-name=Qwen3.5-4B-Q8_0.gguf`,
   `--responses-api-base-url=http://127.0.0.1:8080/v1` in the dashboard.
3. Click Start. Pipeline should reach "running" state. Send a turn.
4. Confirm: the robot speaks the final answer only, no chain-of-thought.
5. Status & Logs should show normal pipeline activity; no `reasoning_text`
   events (they're not surfaced yet, but they're being correctly dropped).

**Out of scope for Phase 1:**
- Reasoning log visibility toggle.
- Dashboard "🧠 Reasoning" sub-panel in Status & Logs.

## Phase 2 — reasoning-log visibility (independent ship)

Scope: change 2 only (log sink + dashboard toggle). Independent of
Phase 1.

**New CLI flag:** `--show-reasoning-logs` (default `false`).
**New env var:** `SPEECH_TO_SPEECH_SHOW_REASONING_LOGS=1`.
**New dashboard checkbox:** "Show LLM reasoning in logs" in the LLM
tab, sibling of `--responses-api-disable-thinking`.
**New server endpoint:** `/api/realtime` sends a new event
`reasoning.delta` to subscribed WebSocket clients when the toggle is on.
**Frontend:** collapsible "🧠 Reasoning" sub-panel in Status & Logs.

**Verification on this box:**
1. With the working LFM2.5 setup, toggle "Show reasoning in logs" on.
2. Send a turn. Status & Logs shows the reasoning sub-panel (empty,
   since LFM2.5 doesn't reason).
3. Switch to Qwen3.5-4B, toggle reasoning-logs on.
4. Send a turn. Status & Logs shows the reasoning sub-panel populated
   with Qwen's chain-of-thought in real time.
5. Toggle reasoning-logs off. Sub-panel disappears; reasoning is
   still dropped from speech.

## Why no version bump is required

Phase 1 is a bug fix to the parser. The wire protocol and CLI flags
are unchanged. Existing `web_ui_settings.json` continues to work.

Phase 2 adds one CLI flag, one env var, one dashboard checkbox. All
opt-in, all defaulted off. Existing settings are unchanged.

If we ever ship the manager subprocess (the llama-server-managed
plan from `docs/MANAGED_LLAMA_PLAN.md`-to-be), version bumps can be
co-located.

## Open questions (locked answers)

1. **Should the dashboard button be inverted in the UI?**
   No. The label "Disable thinking" + unchecked-default semantics are
   consistent with the existing pipeline flag and the screenshot the
   user already approved. Don't churn.

2. **Should the parser strip happen for ALL models, or only when the
   wire flag is on?**
   **All models, always.** The invariant is "robot never speaks
   thinking." Wire flag is best-effort; strip is the guarantee.

3. **Should the existing Ollama `--keep-alive` field have any effect
   on this?**
   No. Unrelated. Ollama keep-alive is about model residency in VRAM,
   not reasoning tokens.

4. **What about the old `gpt-oss-20b` model that's already in
   `web_ui_settings.json`?**
   It's a reasoning model. With Phase 1 applied, it works correctly
   out of the box. Pre-Phase 1, it likely hung at warmup for the same
   reason Qwen3.5 does.

## Definition of done (Phase 1)

- [ ] `responses-api` parser handles `ResponseReasoningTextDeltaEvent` +
      `ResponseReasoningItem`, yields `ReasoningDelta`.
- [ ] `chat-completions` parser handles `delta.reasoning_content`, yields
      `ReasoningDelta`.
- [ ] `ReasoningDelta` is consumed by the LMOutputProcessor and dropped
      before reaching TTS (no `LLMResponseChunk` is emitted for it).
- [ ] No regression for non-reasoning models (LFM2.5).
- [ ] Qwen3.5-4B reasoning model loads, warms up, responds to a turn,
      and the robot speaks the final answer only.
- [ ] No new CLI flag, no new env var, no dashboard UI change.

## Definition of done (Phase 2)

- [ ] `ReasoningSink` module exists and is wired into
      `LMOutputProcessor`.
- [ ] `--show-reasoning-logs` CLI flag added to
      `ModuleArguments` / language model args.
- [ ] Dashboard checkbox renders, persists to `web_ui_settings.json`,
      env-injects `SPEECH_TO_SPEECH_SHOW_REASONING_LOGS=1`.
- [ ] `reasoning.delta` server event emitted over `/v1/realtime`
      WebSocket.
- [ ] Frontend renders reasoning sub-panel in Status & Logs.
- [ ] Round-trip verified on LFM2.5 (no content) + Qwen3.5 (content).
