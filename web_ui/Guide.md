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
| `whisper` | Hugging Face Transformers Whisper. Heavier than Parakeet but very accurate. |
| `whisper-mlx` | Apple Silicon only. Fast on M-series. |
| `mlx-audio-whisper` | Apple Silicon only. Uses `mlx-audio`. |
| `faster-whisper` | CTranslate2, CPU-friendly. Requires the `faster-whisper` pip extra. |
| `paraformer` | FunASR's Paraformer. Chinese-oriented by default. Requires the `paraformer` pip extra. |

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

## Audio device not found (`local` mode)

`sounddevice` (used in `local` mode) needs a working audio system. On Linux you may need `libportaudio2` (`sudo apt install libportaudio2`). On a headless server, use `--mode websocket` or `--mode socket` instead.

## Ollama: "model not found" or empty response

- The model name in `--model-name` must be the exact Ollama tag, e.g. `gemma3:4b`, not a HuggingFace id.
- Run `ollama list` on the Ollama machine to see what tags are pulled.
- Test connectivity from the dashboard machine: `curl http://<ip>:11434/v1/models`.

## Pipeline won't start, exits immediately

Open the **Status & Logs** tab and look at the error. Common causes:
- Missing pip extra for the chosen backend.
- CUDA wheel mismatch.
- Invalid model name.
- API key missing in the environment (use Settings → Environment Variables, then Save and Restart).

The **Toggle Verbose** button on the Status tab restarts the pipeline with `--log-level debug`, which surfaces everything including import errors and download progress.

## Where are my settings saved?

`web_ui_settings.json` in the repo root. It's gitignored by default. You can edit it directly, import / export it via the **Settings** tab, or delete it to reset to defaults.

## How do I get the realtime WebSocket URL?

When the pipeline is running in `realtime` mode, it listens on `ws://<host>:8765/v1/realtime` by default. The dashboard proxies its `/v1/pool` status endpoint, visible in the Status tab. You can connect any OpenAI Realtime-compatible client (browser, app, robot) to that URL.
