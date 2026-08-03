# Quickstart

This guide walks you through getting a working voice conversation in a few minutes.

## 1. First-time setup

If you just cloned the repo:

```bash
uv sync
```

That's it. `uv` installs the package, all Python deps, and registers the `speech-to-speech` and `speech-to-speech-web` console scripts.

## 2. Start the dashboard

Linux / macOS:
```bash
./start_web_ui.sh
```

Windows:
```bat
start_web_ui.bat
```

Your browser opens to `http://localhost:8050`. You'll see a sidebar with tabs: **Mode**, **VAD**, **STT**, **LLM**, **TTS**, **Advanced**, **Status & Logs**, **Guide**, **Settings**, **Control**.

## 3. Your first conversation (local microphone + local LLM)

The defaults are a working configuration. If your machine has a CUDA GPU and you have `OPENAI_API_KEY` exported, the defaults will Just Work:

1. Open the **Control** tab.
2. Click **▶ Start Pipeline**.
3. Click **Status & Logs** to see it come up.
4. Talk to your microphone (the pipeline's default mode is `realtime` — it waits for an OpenAI Realtime client to connect via WebSocket).

To use your machine's microphone directly without writing a client, set the mode to `local`:

1. Open the **Mode** tab.
2. Change `--mode` to `local`.
3. Click **Save Settings** then **Control → Restart Pipeline**.

---

# Pointing at Ollama on another machine

This is the most common setup for a network with a beefy Windows/Linux box running models and a lighter Linux/Mac box running the voice pipeline.

## On the machine running Ollama

1. Pull the model you want:
   ```bash
   ollama pull gemma3:4b
   ```

2. Make Ollama listen on the network (default is `127.0.0.1:11434` only). On Windows, set the system env var `OLLAMA_HOST=0.0.0.0:11434`, then restart Ollama. On Linux/macOS, run:
   ```bash
   OLLAMA_HOST=0.0.0.0:11434 ollama serve
   ```

3. Allow inbound TCP 11434 through the firewall. On Windows:
   ```powershell
   New-NetFirewallRule -DisplayName "Ollama" -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
   ```

4. Find the machine's LAN IP (`ipconfig` on Windows, `ip addr` on Linux). It'll be something like `192.168.1.42`.

5. From the **other** machine, verify connectivity:
   ```bash
   curl http://192.168.1.42:11434/v1/models
   ```
   You should get JSON listing the models Ollama has pulled.

## In the dashboard

1. **LLM** tab → set `--llm-backend` to `chat-completions` (NOT `responses-api` — Ollama serves the Chat Completions endpoint, not Responses).
2. `--model-name`: the exact tag Ollama knows, e.g. `gemma3:4b` (NOT a HuggingFace repo id).
3. `--responses-api-base-url`: `http://192.168.1.42:11434/v1`
4. `--responses-api-api-key`: empty string (Ollama ignores it).
5. `--responses-api-stream`: enabled.
6. Click **Save Settings**, then **Control → Start Pipeline**.

That's it. The pipeline will talk to Ollama over HTTP for the LLM, while STT and TTS run locally on the machine running the dashboard.

---

# Pointing at OpenAI / HF Inference Providers

Both work with the OpenAI-compatible API slot. The only difference is the `base_url` and which environment variable holds the API key.

| Provider | `--llm-backend` | `--responses-api-base-url` | API key env var |
|---|---|---|---|
| OpenAI | `responses-api` or `chat-completions` | (leave default) | `OPENAI_API_KEY` |
| HF Inference Providers | `responses-api` or `chat-completions` | `https://router.huggingface.co/v1` | `HF_TOKEN` |
| OpenRouter | `responses-api` or `chat-completions` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| vLLM (local) | `responses-api` or `chat-completions` | `http://localhost:8000/v1` | (empty) |
| llama.cpp server | `responses-api` or `chat-completions` | `http://127.0.0.1:8080/v1` | (empty) |

For `chat-completions` you may also want to set `--responses-api-reasoning-effort none` if the model gets chatty with its chain-of-thought and you want snappy voice responses.

API keys are read from the process environment of the dashboard. Add them in the **Settings → Environment Variables** editor (Save, then Restart) or export them before running `start_web_ui.sh`.

---

# What each backend does

## VAD (Voice Activity Detection)

Silero VAD v5 detects when the user is speaking and when they're silent. The pipeline uses the silence gaps to know when to send audio to STT. The most useful knobs:

- `--thresh` (0.6): how confident the model must be that audio is speech. Lower = more sensitive (more false positives like background noise).
- `--min-speech-ms` (384) and `--min-silence-ms` (64): the minimum durations to commit to speech / silence. Recommended pairing is `384 / 64`.
- `--min-speech-continuation-ms` (192): hysteresis for re-opening a soft-ended turn. Set to `0` to disable.

## STT (Speech to Text)

| Backend | When to pick it |
|---|---|
| `parakeet-tdt` (default) | Multilingual, 25 European languages, runs on CUDA / CPU / Apple Silicon. Best general choice. |
| `parakeet-onnx` | NVIDIA Parakeet TDT via `onnx-asr` (pure ONNX, no PyTorch). Three variants: `v2` (English-only), `v3` (multilingual 25 EU langs, default), `v3sq` (int8 SmoothQuant, best for long audio). CPU / CUDA only — Apple Silicon users keep `parakeet-tdt`. Requires the `parakeet-onnx` pip extra. |
| `whisper` | Hugging Face Transformers Whisper. Heavier than Parakeet but very accurate. |
| `whisper-mlx` | Apple Silicon only. Fast on M-series. |
| `mlx-audio-whisper` | Apple Silicon only. Uses `mlx-audio`. |
| `faster-whisper` | CTranslate2, CPU-friendly. Requires the `faster-whisper` pip extra. |
| `paraformer` | FunASR's Paraformer. Chinese-oriented by default. Requires the `paraformer` pip extra. |

### Parakeet ONNX (`parakeet-onnx`)

Three NVIDIA Parakeet TDT variants, all loaded through the pure-ONNX
`onnx-asr` library (no PyTorch, no NeMo):

| Variant | HuggingFace repo | Language | Notes |
|---|---|---|---|
| `v2` | `istupakov/parakeet-tdt-0.6b-v2-onnx` | English only | Smallest, fastest. For pure-English deployments. |
| `v3` (default) | `istupakov/parakeet-tdt-0.6b-v3-onnx` | 25 EU languages | Same language coverage as `parakeet-tdt`. |
| `v3sq` | `Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx` | 25 EU languages | int8 SmoothQuant rebuild. Best for long audio (>20 s) — calibrated to avoid the long-audio accuracy collapse that the stock int8 encoder hits. |

**Limitations** (honest, by design):

- **CPU / CUDA only.** The handler refuses `--device mps` with a clear error. Apple Silicon users keep using `parakeet-tdt` (the MLX path).
- **Final-only transcription.** There is no live / progressive transcription — `onnx-asr` does not expose a streaming API and the ONNX export is a single monolithic file (no encoder/decoder/joiner split). The conversation still works: the final transcript is sent to the LLM when the user stops speaking. If you want live sentence-by-sentence updates, use `parakeet-tdt`.
- **The `--enable-live-transcription` global flag is silently ignored** for this backend.

**Auto-detect & CPU fallback** — the handler must "just work" regardless of what GPU/CUDA situation you have. The parakeet-onnx sub-form has a `device` dropdown right next to `variant` so you can override the default:

- `auto` (default): queries `onnxruntime.get_available_providers()`. If CUDA is advertised, uses it. Otherwise uses CPU. No probe needed.
- `cuda` (explicit): requests CUDA. If the load or the first warmup call fails (broken CUDA install, missing libcudnn, onnxruntime GPU wheel mismatch, unsupported compute capability), the handler logs a warning and reloads on **CPUExecutionProvider**. The pipeline always starts; you get a working STT just slower on the first utterance.
- `cpu` (explicit): CPU only.

**Install** (one-time):

```bash
uv pip install "speech-to-speech[parakeet-onnx]"
# or, if you use the dashboard's install endpoint:
# install the "parakeet-onnx" extras group and restart the pipeline.
```

**Language dropdown** (visible when `--stt parakeet-onnx` is selected):

- For `v2`: the dropdown is greyed out and locked to `en`. v2 is an English-only model.
- For `v3` / `v3sq`: the dropdown is enabled, with `auto` (default — model auto-detects) plus the 25 EU language codes. Picking a specific code forces the model to that language.

**Reverting** — if you want to go back to the default pipeline:

```bash
git checkout main
git branch -D feature/parakeet-onnx
# then in the dashboard, set --stt back to "parakeet-tdt"
```

The model files in `~/.cache/huggingface/hub/` can be removed manually:

```bash
rm -rf ~/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v2-onnx
rm -rf ~/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx
rm -rf ~/.cache/huggingface/hub/models--Olicorne--parakeet-tdt-0.6b-v3-smoothquant-onnx
```

## LLM (Language Model)

- `responses-api` (default): OpenAI's `/v1/responses` endpoint. Use with OpenAI directly, or any provider that implements that path.
- `chat-completions`: OpenAI's `/v1/chat/completions` endpoint. **This is the one Ollama supports.** Also good for vLLM and llama.cpp servers.
- `transformers`: local in-process. Needs a CUDA GPU and a HuggingFace model id.
- `mlx-lm`: local in-process, Apple Silicon only. Fastest for M-series Macs.

## TTS (Text to Speech)

| Backend | When to pick it |
|---|---|
| `qwen3` (default) | Multilingual, GGML on Linux, mlx-audio on Apple Silicon. Best general choice. |
| `kokoro` | Lightweight English-focused. CPU-friendly. |
| `pocket` | CPU, voice cloning with preset voices. |
| `chatTTS` | English/Chinese, expressive. |
| `facebookMMS` | Multilingual via MMS checkpoints. |
| `chatterbox` | Voice cloning from a short reference audio. English (`chatterbox`), Turbo (`chatterbox-turbo`), Nano (`chatterbox-nano`), and Multilingual (`chatterbox-multilingual`) variants. CPU works. |

The Qwen3-TTS backend needs a specific `qwentts-cpp-python` wheel matching your CUDA runtime on Linux. If you see import errors, see "Common issues" below.

## Voice library (Chatterbox)

The `chatterbox` TTS is a *voice cloning* model: you upload a short reference audio (5-30 seconds of clean speech, WAV preferred), name the voice, and the robot will speak in that voice for every reply. The cloned voice is stored as a small `.pt` file in `voices/<name>.pt` at the repo root.

**In the dashboard, when `--tts chatterbox` is selected:**

1. Scroll down to the **Voice Library** section in the **TTS** tab.
2. Click **+ Clone new voice**, pick a WAV file from your disk, type a name (letters, digits, `_` and `-` only; up to 64 chars), pick the chatterbox model variant, submit.
3. The new card appears in the grid. The reference audio is **discarded** after cloning — only the speaker embedding is stored.
4. Click **Set active** on the card to wire it into `--chatterbox-voice`. Click **Test** to synthesize a short preview using your current parameter values; if you don't like how it sounds, tweak the parameters in the form above and click **Test** again.
5. **Save Settings** then start the pipeline — the robot speaks in your cloned voice.

You can clone as many voices as you like; switch between them with **Set active** or by typing the name directly into `--chatterbox-voice` in the form. Use **×** on a card to delete a voice (the `.pt` and the manifest entry are both removed; if it was the active one, `--chatterbox-voice` reverts to empty).

### Chatterbox model variants

| Variant | Size | Notes |
|---|---|---|
| `chatterbox` (English 500M) | ~1 GB | Best quality, slowest. `exaggeration`, `cfg_weight`, `repetition_penalty`, `min_p` all apply. |
| `chatterbox-turbo` (350M) | ~700 MB | Single-step decoder, low latency. `exaggeration`/`cfg_weight` are no-ops (logged as ignored). |
| `chatterbox-nano` (110M) | ~250 MB | Smallest, fastest. CPU-friendly. Same restrictions as Turbo. |
| `chatterbox-multilingual` | ~1 GB | Adds `language_id` (en, fr, de, es, it, pt, ja, zh, ko, hi, ar). Slower than English. |

Variant-conditional parameters are **grayed out** in the form when they don't apply — they're not errors, just no-ops on that variant.

The Qwen3-TTS backend needs a specific `qwentts-cpp-python` wheel matching your CUDA runtime on Linux. If you see import errors, see "Common issues" below.

---

# Run modes

The pipeline can take audio from different sources and deliver audio to different sinks.

- `realtime` (default): OpenAI Realtime WebSocket API at `/v1/realtime`. **Any OpenAI Realtime-compatible client can connect.** This is the mode for building apps and devices against the standard voice API.
- `local`: Your machine's microphone and speakers. Talk directly to the pipeline, no client needed. Best for trying things out.
- `websocket`: Raw PCM over WebSocket. Minimal custom client.
- `socket`: Raw PCM over TCP. Pipeline runs on a server, mic/playback on a client.

The companion scripts in `scripts/` provide clients for each mode:

```bash
uv run python scripts/listen_and_play.py --host <server-ip>             # socket mode
uv run python scripts/listen_and_play_realtime.py --host 127.0.0.1       # realtime mode
```

---

# Common issues

## "Address already in use" / port 8050 taken

Another program is using port 8050. Either close it, or set `SPEECH_TO_SPEECH_WEB_PORT` before launching:

```bash
SPEECH_TO_SPEECH_WEB_PORT=8090 ./start_web_ui.sh
```

## Qwen3-TTS: "no module named 'qwentts_cpp_python'" or wheel mismatch

The default PyPI wheel for the Qwen3-TTS GGML backend targets CUDA 12.8. If your CUDA is older or newer, install the matching wheel from the Hugging Face wheelhouse before installing `speech-to-speech`:

```bash
# CUDA 13.x
pip install "qwentts-cpp-python==0.3.1+cu130" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu130

# CUDA 12.4
pip install "qwentts-cpp-python==0.3.1+cu124" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cu124

# CPU only
pip install "qwentts-cpp-python==0.3.1+cpu" \
  -f https://huggingface.co/datasets/andito/qwentts-cpp-python-wheels/tree/main/whl/cpu

uv sync
```

## Optional backend not installed

If you select `--tts kokoro` and you don't have the `kokoro` extra, the pipeline will fail at import time. Install it:

```bash
uv pip install "speech-to-speech[kokoro]"
```

Same pattern for `pocket`, `chattts`, `facebook-mms`, `faster-whisper`, `paraformer`, `whisper-mlx`.

## Chatterbox TTS: install / runtime issues

**Install.** The `chatterbox` extra pulls in the Chatterbox TTS package plus the few odd small deps it needs (s3tokenizer, conformer, resemble-perth, diffusers, omegaconf, pykakasi, pyloudnorm). The dashboard's "Install chatterbox" modal runs the exact same command — if you want to do it manually:

```bash
uv pip install --no-deps chatterbox-tts==0.1.7 \
  s3tokenizer conformer==0.3.2 resemble-perth \
  diffusers omegaconf pykakasi pyloudnorm
```

(We pass `--no-deps` for `chatterbox-tts` itself because its strict pin on `transformers==5.2.0` conflicts with the rest of the project's pins. The packages listed after it are Chatterbox's actual runtime deps.)

**First-run downloads.** On the very first pipeline start with `--tts chatterbox`, the model weights download from Hugging Face into the standard cache (`~/.cache/huggingface/...`). ~700 MB for Turbo, ~1 GB for the English variant, ~250 MB for Nano. Subsequent starts are fast.

**"RuntimeError: expected scalar type Double but found Float".** This is a chatterbox-tts 0.1.7 dtype bug. The dashboard and the pipeline handler both patch the s3tokenizer and voice_encoder mel-spectrogram functions at import time to force float32 — so if you see this error, the patch didn't run. Make sure you import via the dashboard (or `from speech_to_speech.TTS.chatterbox_tts_handler import _patch_chatterbox_mel_spectrogram; _patch_chatterbox_mel_spectrogram()` before any other chatterbox import).

**Cloned voice doesn't sound like the reference.** Two common causes:
- **Reference audio too short.** Chatterbox needs at least ~5 seconds of clean speech to extract a usable embedding. 10-30 seconds is the sweet spot.
- **Reference audio noisy / music / multiple speakers.** Chatterbox can latch onto background music or the wrong speaker. Use a clean recording of just the target voice.

**Turbo / Nano variant — `exaggeration` and `cfg_weight` are grayed out.** These are no-ops on the Turbo decoder (the dashboard shows the warning "ignored by Turbo" in the chatterbox logs). The values you set are still saved, but they don't affect output. If you need `exaggeration` / `cfg_weight`, switch the variant dropdown to `chatterbox` (English 500M) or `chatterbox-multilingual`.

## Audio device not found (`local` mode)

`sounddevice` (used in `local` mode) needs a working audio system. On Linux you may need `libportaudio2` (`sudo apt install libportaudio2`). On a headless server, use `--mode websocket` or `--mode socket` instead.

## Ollama: "model not found" or empty response

- The model name in `--model-name` must be the exact Ollama tag, e.g. `gemma3:4b`, not a HuggingFace id.
- Run `ollama list` on the Ollama machine to see what tags are pulled.
- Test connectivity from the dashboard machine: `curl http://<ip>:11434/v1/models`.

## Ollama: keep the LLM model loaded between requests

Ollama unloads a model from VRAM **5 minutes** after the last request by
default. For a voice-agent pipeline that's painful — the next user
utterance after a 5-minute pause pays a ~20 s reload before the first
token. The dashboard exposes an **`--llm-keepalive`** dropdown on the
**LLM** tab (visible only when `--llm-backend` is `chat-completions` or
`responses-api`, i.e. an OpenAI-compat backend pointing at Ollama /
vLLM / llama.cpp). Pick how long Ollama should hold the model:

| Option | Behavior |
|---|---|
| `(off)` | Don't refresh — Ollama's native unload timer applies (default 5 min). |
| `5m` / `15m` / `30m` | Send a tiny `/v1/chat/completions` ping at half the interval so Ollama's idle timer keeps resetting. |
| `1h` / `2h` / `12h` | Same, for long-idle sessions. |
| `Forever (-1)` | Ollama-specific "never unload" sentinel. |
| `Custom (type your own)` | Any Ollama duration: `45m`, `90m`, `4h`, `0`, `-1`. |

The pinger runs as a dashboard-side daemon thread — it doesn't touch
`src/speech_to_speech/` (CLAUDE.md forbids editing the pipeline) and
it only activates when the selected backend is Ollama-style. It stops
automatically when you Stop or Restart the pipeline, and on Save
Settings. Pick a value, click **Save Settings**, then **▶ Start
Pipeline** (or **↻ Restart Pipeline** if it's already running) — the
pinger comes up alongside the pipeline.

## LLM request timeout: cut-off replies on the first model load

The pipeline's OpenAI-compatible LLM handler hardcodes a **20 second
read timeout** for every chat / response call. If your LLM takes
longer than that to start streaming — typical for a cold Ollama load,
a local 70 B model on a CPU box, a slow vLLM with a busy GPU, or
the first call right after `Stop Pipeline` + `Start Pipeline` — the
request is aborted and the robot replies with the canned
*"Wow I'm a bit slow today, could you repeat that?"* fallback
forever, because every subsequent attempt hits the same tight timeout.

The dashboard exposes a per-backend **LLM request timeout (s)**
dropdown on the **LLM** tab, right under `--ollama-load-timeout-seconds`.
Picks:

| Option | Behavior |
|---|---|
| `0 s — no timeout (wait forever)` | Recommended for local models and Hermes. The pipeline never aborts on silence; a truly hung LLM will stall the pipeline until it dies. |
| `30 s` / `60 s` | For fast hosted backends (OpenAI, HF Inference) where 20 s is usually enough but you'd like a small buffer. |
| `120 s` (default for local backends) | Covers most cold loads on a local Ollama / vLLM / llama.cpp server. |
| `300 s` / `600 s` | For very large models on slow links (70 B+ over LAN, etc.). |
| `Custom (type your own)` | Any integer 0-3600 s. Pick this for the long tail. |

The value is stored **per backend** — switching from `Hermes Agent`
(`0 = no timeout`) to `Ollama` (`120 s`) keeps each backend's tuned
value. The default for `responses-api` stays at `20 s` (hosted OpenAI
first-byte is fast; the pipeline's hardcoded value is correct there).
Defaults for `ollama`, `vllm`, `llama.cpp`, `mlx-lm`, and `transformers`
are `120 s` because those are the backends that get bitten by this.

Implementation: a small monkey-patch
(`web_ui/_openai_timeout_patch.py`) is auto-imported via
`sitecustomize.py` in the venv before the pipeline's `OpenAI(...)`
constructor runs. The patch only replaces the openai SDK's read
timeout when it sees the pipeline's hardcoded 20 s shape, so it
never clobbers a deliberate timeout the user or another tool has
set. No changes to `src/speech_to_speech/` (CLAUDE.md hard
constraint).

## Pipeline won't start, exits immediately

Open the **Status & Logs** tab and look at the error. Common causes:
- Missing pip extra for the chosen backend.
- CUDA wheel mismatch.
- Invalid model name.
- API key missing in the environment (use Settings → Environment Variables, then Save and Restart).

The **Toggle Verbose** button on the Status tab restarts the pipeline with `--log-level debug`, which surfaces everything including import errors and download progress.

## "I'm running out of RAM" / "Unload the TTS model"

The pipeline keeps the TTS model loaded in RAM while running. Chatterbox Turbo is ~700 MB, the full English variant is ~1 GB, parakeet is another ~600 MB — the Python interpreter baseline adds ~300 MB, so a full chatterbox pipeline sits at 1.5-2 GB before any of the LLM-related caches.

To free RAM **without stopping the pipeline** (so the VAD, STT, and LLM stay warm), open the **Control** tab and click **🧹 Unload TTS Model**. The TTS handler drops the model in place and runs `gc.collect()`. The next TTS request reloads the model — ~15-20s on CPU, ~5s on CUDA. The robot will pause for that long on the very first reply after unloading, then behave normally.

If you want to free **everything**, click **■ Stop Pipeline**. The subprocess exits and the OS reclaims all of it. Click **▶ Start Pipeline** to start over (model load is again ~15-20s on CPU).

## Ollama model lifecycle: num_ctx doesn't always stick

The `--responses-api-num-ctx` setting tells Ollama what context window
to use when it loads the model. The dashboard saves the value, the
pipeline forwards it on every chat-completions call, and the
dashboard's own warmup sends it at pipeline startup. **All the
plumbing is correct.**

**The catch:** Ollama ignores `num_ctx` on requests for a model that
is **already loaded** in its KV cache at a smaller context. Once a
model is resident at, say, 4096 ctx, every subsequent request with
`num_ctx=16000` is silently dropped — `ollama ps` keeps showing 4096
until the model is fully unloaded. The only way to apply a new
`num_ctx` is to unload first, then let the next request load it
fresh at the new size.

The dashboard has two coordinated ways to handle this:

1. **Auto-unload on `num_ctx` change.** When you edit
   `--responses-api-num-ctx` and the value differs from what Ollama
   currently has loaded, the dashboard silently sends `keep_alive=0`
   to unload the model. The next request loads it fresh at the new
   context. The lifecycle badge updates to show the new size. Silent
   — no toast, no confirmation.
2. **Manual "🔄 Unload & reload Ollama model" button.** In the new
   **Ollama model lifecycle** section at the bottom of the LLM tab
   (visible only when your LLM URL looks like Ollama). Click this
   whenever you want to force a reload at the current `num_ctx`,
   regardless of whether anything changed. Useful for:
   - First run after upgrading to 0.4.3 (the model may already be
     loaded at the wrong context from before).
   - After running `ollama run <model>` in your terminal (which
     loads the model at Ollama's default 4096 context).
   - Test-driving different context sizes without a full pipeline
     restart.

The section also shows live status (model name, endpoint, current
context from `ollama ps`) polled every 2 seconds — same cadence as
the Hermes tab. If Ollama is unreachable, the badge turns red and
the button is disabled.

What the section **does not** do: it doesn't change `--llm-keepalive`
behavior, doesn't restart the pipeline, doesn't touch the running
session's chat history. It only reloads the model at the user's
chosen context window so the **next** request — including the
pipeline's chat-completions stream — lands on the freshly-loaded
model.

## Where are my settings saved?

`web_ui_settings.json` in the repo root. It's gitignored by default. You can edit it directly, import / export it via the **Settings** tab, or delete it to reset to defaults.

A working example configuration is checked in as `web_ui_settings.example.json`. From the **Settings** tab, click **Import JSON** and pick that file to start with a real configuration (Ollama + qwen3-TTS + parakeet STT, realtime mode) instead of bare defaults.

## How do I get the realtime WebSocket URL?

When the pipeline is running in `realtime` mode, it listens on `ws://<host>:8765/v1/realtime` by default. The dashboard proxies its `/v1/pool` status endpoint, visible in the Status tab. You can connect any OpenAI Realtime-compatible client (browser, app, robot) to that URL.

## Qwen3-TTS: voice selection in Realtime clients

When you connect an OpenAI Realtime client to the pipeline, the client
sends `voice` in `session.update`. qwen3-TTS doesn't recognize OpenAI's
`alloy` / `echo` / `shimmer` / `ash` / `ballad` / `coral` / `sage` /
`verse` / `marin` / `cedar` voice names — those are OpenAI TTS voices,
not qwen3 speakers.

The dashboard **silently ignores the client's `voice` field** and uses
whatever you set in **Settings → TTS → Qwen3-TTS → Speaker**. To change
the voice:

1. Open the dashboard's **Settings** tab → **TTS** → **Qwen3-TTS** subgroup.
2. Pick a speaker from the **Speaker** dropdown. The 9 CustomVoice
   presets are listed (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan,
   Aiden, Ono_Anna, Sohee).
3. Click **Save Settings**, then **▶ Start Pipeline** (or restart if
   it's already running).

The voice library panel under the TTS tab also has a **Set active**
button on each card — click it for a one-click "use this voice".
Each card has a **Test** button that synthesizes a sample sentence
in the speaker's native language so you can hear it before committing.

## Qwen3-TTS: voice cloning (Base) and VoiceDesign — ABI v2 limitation

The bundled `qwentts-cpp-python` wheel exposed to the dashboard on
Pascal (sm_61) GPUs is **ABI v1 only**. The voice-cloning path
(`ref_audio` + `ref_text` on a Base model) and the voice-design path
(`instruct` text on a VoiceDesign model) need ABI v2 symbols
(`qt_extract_voice_ref`, `qt_voice_ref_free`) that are not in the
public `andimarafioti/qwentts.cpp` source — they're in an unreleased
private commit the PyPI wheel was built against.

What this means for you on this Pascal GPU:

- ✅ **CustomVoice models work fully end-to-end.** The 9 preset speakers
  are baked into the model weights; no enumeration or reference audio
  is required at runtime. Pick one from the dashboard, save, the
  Realtime pipe speaks in that voice.
- ⚠ **Base models** (voice cloning via `ref_audio` + `ref_text`) load
  fine but synthesize fails with
  `QwenTTSError: qt_extract_voice_ref is unavailable; voice reference
  extraction requires qwentts.cpp ABI v2`. The dashboard shows a
  yellow banner when you select a Base model. Reference audio uploads
  still work (files are saved under `voices/qwen3_refs/`) — they're
  ready for the day upstream publishes ABI v2 source.
- ⚠ **VoiceDesign models** (`instruct` text) have the same ABI v2
  limitation; same banner, same "wait for upstream" path.

On a different GPU (e.g. an RTX 30/40 series) where the upstream wheel
ships ABI v2 symbols, all 5 qwen3-TTS models work fully — voice cloning
and voice design light up automatically with the same dashboard
configuration, no extra dashboard work needed.

## Qwen3-TTS: which model to pick?

Pick **`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`** unless you have a
specific reason. It has the highest audio quality, supports all 9
preset speakers on GPU, and is the version with the most usage in
production.

The 0.6B CustomVoice model is ~3× smaller and noticeably faster on
Pascal (sm_61) cards; quality is still good but slightly below the
1.7B. Worth switching to if you're CPU-constrained or want faster
time-to-first-audio.


# Connecting to Hermes Agent

The dashboard can route the voice pipeline through [Hermes Agent](https://github.com/NousResearch/hermes-agent)
instead of Ollama. Hermes becomes the brain (skills, memory, Home
Assistant control); the dashboard's existing pipeline stays the body
and voice. You can still swap STT/TTS/VAD freely — only the LLM slot
changes.

## 0.5.0 (this version)

- **Parakeet ONNX STT backend** (`--stt parakeet-onnx`). Three NVIDIA
  Parakeet TDT variants via `onnx-asr` (pure ONNX, no PyTorch needed
  for STT): `v2` (English-only), `v3` (multilingual 25 EU langs,
  default), `v3sq` (int8 SmoothQuant rebuild, best for long audio).
  Exposes `--parakeet-onnx-variant`, `--parakeet-onnx-num-threads`,
  and `--parakeet-onnx-language` (dropdown: `auto` + 25 EU languages,
  greyed out and locked to `en` when v2 is selected). CPU / CUDA only —
  Apple Silicon users keep `parakeet-tdt`. Final-only transcription
  (no live / progressive updates — `onnx-asr` exposes no streaming
  API). Install with `uv pip install "speech-to-speech[parakeet-onnx]"`.
  See the "Parakeet ONNX" section under "STT" above.

## 0.4.3

- **Ollama model lifecycle section** at the bottom of the LLM tab
  (visible only when the LLM URL looks like Ollama). Shows live model
  name + endpoint + current context from `ollama ps` (polled every
  2s), plus a **🔄 Unload & reload Ollama model** button that
  forces the model to reload at the user's chosen `--responses-api-num-ctx`.
  Fixes the "num_ctx doesn't stick" bug where Ollama ignores
  `num_ctx` on requests for a model already resident in its KV cache.
  Also auto-unloads silently when `--responses-api-num-ctx` changes
  if Ollama is currently holding the model at a different context.
  See the "Ollama model lifecycle" section below.

## 0.4.2

- **Per-backend LLM request timeout (s)** dropdown on the LLM tab
  (under `--ollama-load-timeout-seconds`). Overrides the pipeline's
  hardcoded 20 s openai-SDK read timeout so the first reply after a
  cold model load isn't cut off ("Wow I'm a bit slow today...").
  Presets: 0 (no timeout), 30, 60, 120, 300, 600 + Custom. Stored per
  backend so switching from Hermes (0) to Ollama (120 s) keeps each
  backend's tuned value. Default for `responses-api` stays at 20 s
  (hosted OpenAI fast first-byte). See the "LLM request timeout"
  section below for details.

## 0.4.0

- **Hermes tab** in the sidebar: start, polite-stop, cancel, kill, plus
  a text console and a filler-phrase config (10-line box).
- **LLM tab → "Backend type"** dropdown: pick *Hermes Agent* and the
  dashboard auto-fills the LLM URL with the dashboard's reverse-proxy
  base URL (`http://<dashboard-host>:8050/hermes-proxy/v1`) and the
  api_key from the Hermes tab.
- **Reverse proxy** lives inside the dashboard's own uvicorn — no
  extra port, no extra process, no extra dependency. Resource cost is
  negligible: it forwards bytes + injects two headers
  (`Authorization`, `X-Hermes-Session-Id`).
- **Session id** is one per pipeline lifetime. New uuid on pipeline
  start / restart / kill. Polite stop and cancel preserve the id so
  the conversation continues on the next start.

## First-time setup

1. Install hermes-agent and pick a model inside hermes (the model is
   picked by hermes, not by the dashboard — see
   `ollama launch hermes` or `hermes` in your terminal).
2. Open the dashboard, click **Hermes** in the sidebar.
3. Click **Start**. The dashboard spawns `hermes gateway run
   --accept-hooks` on `127.0.0.1:8642` with an auto-generated API key
   persisted in `web_ui_settings.json["hermes"]["api_key"]`.
4. Wait for the badge to turn **● running** (green dot).
5. Open the **LLM** tab, change **Backend type** from *Direct backend*
   to *Hermes Agent*. The URL and api_key fields auto-fill. Pick
   the same model name you configured in hermes.
6. **Start Pipeline** (Control tab).

The pipeline now talks to the dashboard's `/hermes-proxy/v1/*`, which
forwards to hermes with the right session header. Voice → STT → LLM
(hermes) → TTS → speaker, just like before.

## How the proxy works (one diagram)

```
pipeline (LLM slot)
   │ POST /v1/chat/completions
   ▼
http://127.0.0.1:8050/hermes-proxy/v1/chat/completions   ← dashboard uvicorn
   │ adds Authorization: Bearer <api_key>
   │ adds X-Hermes-Session-Id: <uuid>
   │ forwards to:
   ▼
http://127.0.0.1:8642/v1/chat/completions              ← hermes subprocess
```

The proxy is on the hot path of every LLM token chunk. To keep it
cheap: the hermes config block is cached in memory and only refreshed
when the dashboard writes it (no per-request disk I/O). The session
id is held in a slot — no allocation per request.

## Common Hermes tasks

- **Reset the conversation without killing hermes:** click
  **↻ Reset session** in the Emergency section of the Hermes tab.
  Mints a new session id; hermes treats the next request as a fresh
  conversation. Faster than kill+start, no subprocess churn.
- **Stop a long-running tool mid-flight:** click **⏸ Cancel**. Sends
  `/v1/runs/stop` to hermes. The session id stays; you can resume.
- **Hard kill (last resort):** click **✖ Kill Hermes**, confirm.
  Herme context lost; pipeline id resets automatically.
- **Open hermes's own dashboard:** the **↗ Open Hermes Dashboard**
  link points at the upstream web UI on port 9119.

## Filler audio

When hermes is mid-tool-call (>1.5s without producing text), the
robot can play a short phrase so the user knows it's still working.
Uses the pipeline's existing TTS — no extra config. Edit the
10-line box on the Hermes tab, or toggle the checkbox off if you
don't want it.

## Out of scope (deferred)

- **Voice-activated cancel** ("stop", "halt"): discussed and dropped
  for v1 — would require either editing `src/speech_to_speech/` (forbidden)
  or running a separate wake-word model. May return later.
- **Auto `compress_context` scheduler:** hermes-agent does not
  currently expose a documented `compress_context` tool (verified
  2026-07). When upstream ships one, swap the *Reset session*
  endpoint to call it instead of (or in addition to) the id rotation.
