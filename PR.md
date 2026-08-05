# Fix: OpenAI-compatible `/v1/chat/completions` ignores `keep_alive` and `options.num_ctx`

## Problem

Since Ollama 0.32, the OpenAI-compatible endpoint at `/v1/chat/completions` silently
ignores two important request fields:

1. **`keep_alive`** — Ollama always uses its server-side default (5m), regardless of
   what the client sends. This means models unload after 5 minutes of silence even
   when the client explicitly asks for `-1` (forever), `"30m"`, etc.

2. **`options.num_ctx`** — Ollama always loads/keeps the model at its default context
   (typically 4096), regardless of what the client sends. The native `/api/chat`
   endpoint honors `options.num_ctx`; the OpenAI-compat endpoint does not.

This is a regression from pre-0.32 where both fields were forwarded. It breaks every
OpenAI-compatible client that needs Ollama to keep models loaded between sessions or
that wants to pin a specific context window (e.g. low-VRAM machines wanting to cap a
128k-native model at 8k).

### Reproduction

```bash
# Force unload
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4:e4b","prompt":"","keep_alive":0}'

# Call /v1/chat/completions with keep_alive=30m — Ollama should keep model
# loaded for 30 min, instead drops to default 5m.
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"gemma4:e4b",
    "messages":[{"role":"user","content":"hi"}],
    "keep_alive":"30m"
  }'

# Check state — `expires_at` is ~5 minutes from now, not 30.
curl http://localhost:11434/api/ps
```

Same reproduction with `options.num_ctx` — model stays at Ollama's default context
instead of the requested value.

## Root cause

In `openai/openai.go`, `FromChatRequest` builds an `api.ChatRequest` but only copies
a hardcoded subset of fields into the `Options` map (`stop`, `num_predict`,
`temperature`, `seed`, `frequency_penalty`, `presence_penalty`, `top_p`). It never
reads `keep_alive` from the OpenAI request body, and never reads `options.num_ctx`
either. Both fields are dropped at the HTTP boundary.

## Fix

Two changes in `openai/openai.go`:

1. Add `KeepAlive` and pass-through for `Options["num_ctx"]` to `ChatCompletionRequest`.
2. Wire them into the returned `api.ChatRequest`.

### Patch 1: add fields to `ChatCompletionRequest`

```go
type ChatCompletionRequest struct {
	Model string `json:"model"`
	Messages []Message `json:"messages"`
	Stream bool `json:"stream"`
	StreamOptions *StreamOptions `json:"stream_options"`
	MaxTokens *int `json:"max_tokens"`
	Seed *int `json:"seed"`
	Stop any `json:"stop"`
	Temperature *float64 `json:"temperature"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
	PresencePenalty *float64 `json:"presence_penalty"`
	TopP *float64 `json:"top_p"`
	ResponseFormat *ResponseFormat `json:"response_format"`
	Tools []api.Tool `json:"tools"`
	Reasoning *Reasoning `json:"reasoning,omitempty"`
	ReasoningEffort *string `json:"reasoning_effort,omitempty"`
	Logprobs *bool `json:"logprobs"`
	TopLogprobs int `json:"top_logprobs"`
	DebugRenderOnly bool `json:"_debug_render_only"`

	// Fix #1: forward Ollama-specific fields that the OpenAI-compat layer
	// was silently dropping since v0.32.
	KeepAlive *api.Duration `json:"keep_alive,omitempty"`
	NumCtx    *int          `json:"num_ctx,omitempty"`
}
```

### Patch 2: forward them in `FromChatRequest`

In the `Options` map construction, add:

```go
// Forward the explicit num_ctx from the request body (when present) so
// OpenAI-compat clients can pin the context window the same way the
// native /api/chat endpoint has always allowed. Without this, Ollama
// silently falls back to the model's default context.
if r.NumCtx != nil {
	options["num_ctx"] = *r.NumCtx
}
```

And in the returned `api.ChatRequest`:

```go
return &api.ChatRequest{
	Model: r.Model,
	Messages: messages,
	Format: format,
	Options: options,
	Stream: &r.Stream,
	Tools: r.Tools,
	Think: think,
	Logprobs: r.Logprobs != nil && *r.Logprobs,
	TopLogprobs: r.TopLogprobs,
	DebugRenderOnly: r.DebugRenderOnly,

	// Fix #1 (cont'd): carry the OpenAI-compat keep_alive into the
	// internal request so the model's unload timer is reset to the
	// caller's chosen value, not Ollama's 5-minute default.
	KeepAlive: r.KeepAlive,
}, nil
```

### Same fix for `FromCompleteRequest` (text completion endpoint)

`FromCompleteRequest` has the same Options-only construction. Apply the identical
two-step change (add `NumCtx` field, copy into Options, return same shape).

## Test plan

1. Unload model: `POST /api/generate` with `keep_alive:0`.
2. Call `/v1/chat/completions` with `{"keep_alive":"30m", "num_ctx":32000, "messages":[…]}`.
3. Verify `GET /api/ps` shows `context_length: 32000` and `expires_at` ≈ now + 30 min.
4. Repeat for `/v1/completions`.
5. Run Ollama's existing OpenAI-compat test suite — should still pass; `NumCtx` and
   `KeepAlive` are pointers with `omitempty`, so unset requests still produce the
   same `api.ChatRequest` shape they do today.

## Why this is safe

- Both new fields are pointer types with `omitempty` JSON tags, so existing clients
  that don't set them serialize identically to before.
- `KeepAlive *api.Duration` is the same type the native `/api/chat` endpoint already
  uses internally — no new parsing code, just a field pass-through.
- `NumCtx` is an `*int` matching the existing `Options["num_ctx"]` consumer (the
  inference engine reads it from `Options` already on the native path).

## Impact

Fixes every OpenAI-compat client that needs persistent model loading (voice
pipelines, IDE assistants, long-running agents) and every client that needs to cap
VRAM usage by pinning context. Backwards-compatible — no existing client breaks.

## References

- Reported in [ollama/ollama#issues](https://github.com/ollama/ollama/issues) —
  search for "keep_alive v1 chat completions regression 0.32" to find the matching
  issue thread.
- Native `/api/chat` (`openai/openai.go` was forked from `server/routes.go`'s
  chat handler) has always honored both fields; this PR brings the OpenAI-compat
  endpoint to parity.