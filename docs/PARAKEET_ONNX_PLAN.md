# Parakeet ONNX STT backend — plan

**Goal.** Add a new STT backend to the pipeline that exposes three NVIDIA Parakeet TDT variants via ONNX, so the Dashboard STT tab can offer them as a dropdown alongside the existing `parakeet-tdt`. Apple Silicon users keep using `parakeet-tdt` (MLX). The new backend is CPU/CUDA only.

**Scope.** Add the backend. **Do not** touch `parakeet_tdt_handler.py`, `smart_progressive_streaming.py`, or any other existing handler. Do not remove or rename anything. Revert = `git revert` of the feature branch.

**Status.** Locked plan, awaiting user approval before any code is written.

---

## 1. The three variants

| Variant | HuggingFace repo | Language | Notes |
|---|---|---|---|
| `v2`     | `istupakov/parakeet-tdt-0.6b-v2-onnx` | English only | Smaller, fastest. For pure-English deployments. |
| `v3`     | `istupakov/parakeet-tdt-0.6b-v3-onnx` | 25 EU languages | Default. Same multilingual set as the current `parakeet-tdt`. |
| `v3sq`   | `Olicorne/parakeet-tdt-0.6b-v3-smoothquant-onnx` | 25 EU languages | int8 build, calibrated for long audio (>20s). Best for noisy / long utterances. |

The variant key is the only thing the user sees in the Dashboard. The repo id is hidden inside the handler.

---

## 2. What gets added

### 2.1 New files

| Path | Purpose |
|---|---|
| `src/speech_to_speech/arguments_classes/parakeet_onnx_arguments.py` | New dataclass with `--parakeet-onnx-variant` (one of `v2`, `v3`, `v3sq`) and `--parakeet-onnx-num-threads` (default 0 = auto). |
| `src/speech_to_speech/STT/parakeet_onnx_handler.py` | New `BaseSTTHandler` subclass. Lazy-loads `onnx_asr.load_model(repo_id)`. Final-only transcription. No live transcription. |
| `tests/test_parakeet_onnx_handler.py` | Skip-by-default tests (run only when `onnx-asr` is installed). Covers: enum validation, dtype conversion (int16 → float32), lazy-load on first transcribe, variant swap unloads prior model, device validation rejects MPS. |
| `docs/PARAKEET_ONNX_PLAN.md` | This document. |

### 2.2 Files that are edited (small, isolated, revertible)

| Path | Edit |
|---|---|
| `src/speech_to_speech/arguments_classes/module_arguments.py` | Add `"parakeet-onnx"` to the `--stt` enum. Update the help string. |
| `src/speech_to_speech/s2s_pipeline.py` | Add `parakeet_onnx_stt_handler_kwargs: ParakeetOnnxSTTHandlerArguments` to `ParsedArguments`, `prepare_all_args`, both `_build_pipeline_*` functions, `get_stt_handler`, and both `prepare_args` / `prepare_args_websocket` calls. Add one `elif module_kwargs.stt == "parakeet-onnx"` branch in `get_stt_handler`. Add one `rename_args(..., "parakeet_onnx")` call. |
| `pyproject.toml` | Add a new optional-dependency group `[parakeet-onnx]` with `onnx-asr>=0.6` and `onnxruntime>=1.18`. Same pins as the reachy_conv_app. |
| `web_ui/Guide.md` | New section: "Parakeet ONNX STT backend" — explains the 3 variants, the CPU/CUDA-only limitation, the final-only transcription behavior, and how to revert. |

### 2.3 Files that are NOT touched

- `src/speech_to_speech/STT/parakeet_tdt_handler.py` (untouched)
- `src/speech_to_speech/STT/smart_progressive_streaming.py` (untouched)
- All other STT handlers (`whisper*`, `paraformer`, `faster_whisper`)
- `src/speech_to_speech/arguments_classes/parakeet_tdt_arguments.py` (untouched)
- Anything in `web_ui/` other than `Guide.md` (dashboard auto-renders the new field via `settings_schema.py` introspection)

---

## 3. The new handler — behavior

### 3.1 `setup()` — auto-detect GPU/CPU with CPU fallback

```
Args:
    variant: Literal["v2", "v3", "v3sq"] = "v3"
    device:  Literal["auto", "cuda", "cpu"] = "auto"   # dropdown in the dashboard
    num_threads: int = 0
    language: Optional[str] = "auto"  # see §4.1 for the dropdown behavior
```

- Validate `variant` is in the known set. Raise `ValueError` with a helpful message if not.
- Resolve `variant` → HF repo id via a private `_VARIANTS` dict inside the handler.
- **Reject `--device mps`**: if the resolved device is `mps`, raise `RuntimeError("parakeet-onnx does not support MPS (Apple Silicon). Use --stt parakeet-tdt instead.")`. CPU and CUDA are OK.
- **Auto-detect + CPU fallback (user-approved requirement):** The handler must work out-of-the-box on every box, even ones whose CUDA install is broken (e.g. the 1080Ti box used during development — old GPU + missing libcudnn) or absent. Resolution rules:

  | User passes | What happens |
  |---|---|
  | `device="auto"` (default — what the dashboard passes) | `onnxruntime.get_available_providers()` is consulted. If `"CUDAExecutionProvider"` is advertised, request it. Otherwise request `CPUExecutionProvider`. No try/load needed at this stage — the availability check is cheaper and more accurate than a probe-and-fallback (it reflects what's actually loadable, not what torch sees). |
  | `device="cuda"` (explicit) | Request `CUDAExecutionProvider`. If `onnx_asr.load_model()` raises (broken CUDA install, missing libcudnn, ORT GPU wheel mismatch, unsupported compute capability), log a WARNING and reload with `CPUExecutionProvider`. The pipeline always starts. Conversation still works, just slower on the first utterance. |
  | `device="cuda"` + CUDA load OK but first warmup call fails | Same fallback: drop the cached model, set providers to CPU, reload, re-warmup. |
  | `device="cpu"` (explicit) | CPU only. Load failure re-raises (genuine broken install → loud failure is better than silent skip). |
  | `device="mps"` | Loud RuntimeError with a pointer to `--stt parakeet-tdt`. |

- The chosen `providers` list is logged at startup: `providers=['CUDAExecutionProvider']`. After a fallback, the log line says `"CUDA load failed (<err>); falling back to CPU."` and the second log line shows `providers=['CPUExecutionProvider']`.
- **Eager load + warmup in `setup()`**: same pattern as `parakeet_tdt_handler.setup()` (the pipeline assumes the model is hot when the handler is up, so the first user utterance is not slow). Variant changes at runtime use the lazy `_ensure_loaded` pattern from §3.4.

### 3.2 `process()`

```python
def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
    audio = vad_audio.audio
    # Convert int16 PCM → float32 in [-1, 1] if needed
    if audio.dtype == np.int16:
        pcm = audio.astype(np.float32) / 32768.0
    else:
        pcm = audio.astype(np.float32)
    pcm = np.ascontiguousarray(pcm)

    text = self._ensure_loaded().recognize(pcm, sample_rate=16000).strip()

    # Language detection — same lingua-py post-hoc path as parakeet_tdt_handler
    if self.variant_language_hint in ("multi",):
        language_code = self._detect_language_from_text(text) or self.last_language
    else:
        language_code = "en"  # v2 is English-only

    yield Transcription(
        text=text,
        language_code=language_code,
        turn_id=vad_audio.turn_id,
        turn_revision=vad_audio.turn_revision,
        speech_stopped_at_s=vad_audio.created_at_s,
    )
```

- **No live transcription.** The handler never yields `PartialTranscription`. The `BaseSTTHandler.should_emit_output` / `before_emit_output` speculative-turn logic still works correctly because it gates on `Transcription` and `PartialTranscription` types — final-only is well-supported.
- `enable_live_transcription` (the global `--enable-live-transcription` flag) is **ignored** for this backend. The Dashboard's help text will say so explicitly.

### 3.3 `cleanup()`

Drop the in-memory model. Standard `del self.model` pattern.

### 3.4 `_ensure_loaded()` — variant swap + provider fallback

```python
def _ensure_loaded(self):
    if self._model is not None and self._loaded_variant == self.variant:
        return self._model
    import onnx_asr
    providers = self.providers
    try:
        self._model = onnx_asr.load_model(_VARIANTS[self.variant], providers=providers)
    except Exception as e:
        if providers == ["CUDAExecutionProvider"]:
            logger.warning("CUDA load failed (%s); falling back to CPU", e)
            providers = ["CPUExecutionProvider"]
            self.providers = providers
            self._model = onnx_asr.load_model(_VARIANTS[self.variant], providers=providers)
        else:
            raise
    self._loaded_variant = self.variant
    return self._model
```

Two triggers for reload:
- Variant changed in the dropdown (e.g. `v3` → `v3sq`): unloads the old model, loads the new one.
- CUDA load failed during `setup()`: same reload path, just with `providers` switched to CPU.

### 3.5 Warmup

The current `parakeet_tdt_handler.py` runs a 1-second silence warmup in `setup()` (lines 213-234). **The new handler does the same**: in `setup()`, after `onnx_asr.load_model()` returns, run `model.recognize(np.zeros(16000, dtype=np.float32), sample_rate=16000)` and discard the result. This matches the existing pattern exactly so the first user utterance is not slow.

If the warmup itself fails on CUDA (some CUDA installs let the model load but blow up on the first inference), the handler drops the cached model, switches to CPU, reloads, and re-warms up. If that also fails, the warning is logged and the handler will retry on the next user utterance — the user gets a working pipeline either way.

One difference: `onnx-asr` may take noticeably longer than `nano-parakeet` to do the first inference (one-time CUDA kernel JIT, memory allocator warmup, etc.), so the warmup is what makes the pipeline feel responsive on the first user turn. Without warmup, the user would see a 2-5 second pause before the first STT result returns.

---

## 4. Locked CLI surface (user-approved)

| Flag | Type | Default | Dashboard widget | Notes |
|---|---|---|---|---|
| `--parakeet-onnx-variant` | enum `[v2, v3, v3sq]` | `v3` | Dropdown: `v2` / `v3` / `v3sq` | The model selector. |
| `--parakeet-onnx-num-threads` | int | `0` | Number input | `0` = onnxruntime decides. `>0` = force thread count. |
| `--parakeet-onnx-language` | enum (see §4.1) | `auto` | Dropdown (see §4.1) | See §4.1. |

`rename_args(..., "parakeet_onnx")` strips the prefix on the way into the handler constructor, exactly the way the existing `parakeet_tdt` works.

### 4.1 The `--parakeet-onnx-language` dropdown

**Choices (default `auto`):** `auto` plus the 25 EU languages supported by Parakeet TDT v3, in this order:

```
auto, en, de, fr, es, it, pt, nl, pl, ru, uk, cs, sk, hu, ro, bg, hr, sl, sr, da, no, sv, fi, et, lv, lt
```

**Behavior per variant:**

| Variant | Dropdown state | What the dropdown does |
|---|---|---|
| `v2` | **Greyed out, locked to `en`** | v2 is an English-only model. The user can see the field but cannot change it. The pipeline never sends a language hint to v2. |
| `v3` | **Enabled** | `auto` = onnx-asr auto-detects (no language hint sent). Any of the 25 codes = the user picks one language and the model restricts to it. |
| `v3sq` | **Enabled** | Same as v3. |

**Implementation note:** the disabled-when-v2 behavior is enforced in the Dashboard by inspecting the current value of `--parakeet-onnx-variant` and setting `disabled` on the language field. The handler itself does NOT need to guard — it just passes whatever the user picked to `onnx_asr.load_model().recognize(...)`. If the user somehow sends a non-English language to v2 via raw CLI, `onnx-asr` will ignore it (v2's vocabulary is English-only) and the model will still produce English output. No crash.

**Why the user wants this:** matches the existing `parakeet-tdt` handler's `language` field (parity), and gives users a way to force a language on the multilingual variants if they want consistent language detection behavior across calls. The `auto` default is the normal path.

---

## 5. Pinned dependencies

```toml
parakeet-onnx = [
    "onnx-asr>=0.6",
    "onnxruntime>=1.18",
]
```

These pins match the reachy_conv_app (`pyproject.toml` lines for `onnx-asr>=0.6` and `onnxruntime>=1.18`). No platform markers — the backend is CPU/CUDA only, but installing on Mac is harmless (the runtime just refuses to use MPS).

`onnx-asr` pulls in `onnxruntime` transitively on most platforms, but we pin it explicitly so the user can install just `onnxruntime` for CPU-only boxes without the full `onnx-asr` if they want a smaller install.

`nano-parakeet`, `mlx-audio`, and `transformers` are **not** required for this backend. The new handler does not import them. This means the new backend has a much smaller install footprint than the existing `parakeet-tdt`.

---

## 6. Dashboard behavior

The Dashboard's `web_ui/settings_schema.py` auto-introspects the new dataclass. After the change, the STT tab will show:

1. `--stt` dropdown → gets one new value `parakeet-onnx`
2. Once the user picks `parakeet-onnx`, the existing `--parakeet-tdt-*` fields disappear and the new `--parakeet-onnx-variant`, `--parakeet-onnx-num-threads`, `--parakeet-onnx-language` fields appear — exactly the same way the existing `parakeet-tdt` and `whisper` blocks work today.

**No dashboard code changes needed.** The introspection handles it. This is the design goal of constraint #2 in `CLAUDE.md`.

**Edge case:** if the user picks `--stt parakeet-onnx` and the dashboard is running on Mac, the help text in the variant field will say "CPU/CUDA only — use `parakeet-tdt` for Apple Silicon." The handler itself will fail loudly at startup if `--device mps` is selected, so the user can't accidentally get a non-working pipeline.

---

## 7. Tests

`tests/test_parakeet_onnx_handler.py` — new file, skip-by-default:

```python
pytest.importorskip("onnx_asr")
```

Covers:

1. **Variant validation** — `setup(variant="bogus")` raises `ValueError`.
2. **MPS rejection** — `setup(device="mps")` raises `RuntimeError` with a clear message.
3. **Lazy load** — `setup()` does NOT call `onnx_asr.load_model()` (verified by mocking).
4. **First transcribe loads** — calling `process()` with a dummy audio triggers `onnx_asr.load_model()` exactly once.
5. **Variant swap reloads** — calling `process()` with `variant="v3"`, then `process()` with `variant="v3sq"`, results in two `load_model` calls (different repo ids).
6. **Dtype conversion** — int16 input → float32 in `[-1, 1]` before `recognize()` (verified by patching `recognize` and inspecting the call args).
7. **Final-only** — handler never yields `PartialTranscription`.

Tests do **not** download any model. They mock `onnx_asr.load_model` and `recognize`. This keeps the test suite fast and offline.

Existing tests (`tests/test_chat.py`, `tests/` in general) must still pass without any change — the new files are additive.

---

## 8. End-to-end verification (manual, before declaring done)

Run in this order:

1. **Install the new extra, fresh shell:**
   ```bash
   cd /home/john/speech-to-speech-dashboard
   git checkout -b feature/parakeet-onnx
   uv sync --extra parakeet-onnx
   ```
   Confirm installation succeeds. No errors about conflicting dep versions.

2. **Run the existing test suite:**
   ```bash
   uv run pytest tests/ -x -q
   ```
   All existing tests pass. Run is offline (no model downloads).

3. **Run the new tests:**
   ```bash
   uv run pytest tests/test_parakeet_onnx_handler.py -v
   ```
   All 7 tests pass.

4. **Lint / format:**
   ```bash
   uv run ruff check src/speech_to_speech/arguments_classes/parakeet_onnx_arguments.py src/speech_to_speech/STT/parakeet_onnx_handler.py tests/test_parakeet_onnx_handler.py
   uv run ruff format --check src/speech_to_speech/arguments_classes/parakeet_onnx_arguments.py src/speech_to_speech/STT/parakeet_onnx_handler.py tests/test_parakeet_onnx_handler.py
   ```

5. **Mypy:**
   ```bash
   uv run mypy src/speech_to_speech/arguments_classes/parakeet_onnx_arguments.py src/speech_to_speech/STT/parakeet_onnx_handler.py
   ```
   No new errors.

6. **CLI smoke test (no model load):**
   ```bash
   uv run speech-to-speech --help | grep parakeet-onnx
   ```
   The new flag appears in the help. Pick `--stt parakeet-onnx --parakeet-onnx-variant v3` and verify the CLI parses without error.

7. **Dashboard round-trip:**
   ```bash
   ./start_web_ui.sh
   ```
   Open http://localhost:8050. Open the STT tab. Confirm the new variant dropdown appears next to the existing parakeet-tdt field. Save settings. Reload. Confirm round-trip via `web_ui_settings.json`.

8. **End-to-end pipeline test (CUDA box):**
   - Set `--stt parakeet-onnx --parakeet-onnx-variant v3` in the dashboard
   - Click Start
   - Speak in English
   - Confirm ONNX Parakeet v3 loads (`~/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx/`)
   - Confirm the LLM responds and TTS speaks
   - Inspect the pipeline logs for any warnings

9. **Switch back to the existing backend:**
   - Set `--stt parakeet-tdt` again
   - Click Start
   - Speak again
   - Confirm the original `parakeet_tdt_handler` works exactly as before

10. **Revert test:**
    ```bash
    git stash --include-untracked
    git checkout main
    ./start_web_ui.sh
    ```
    Confirm the new dropdown is gone and `--stt parakeet-tdt` is the only Parakeet option. Restore with `git checkout feature/parakeet-onnx && git stash pop`.

---

## 9. Revert procedure

If anything goes wrong at any point:

```bash
git checkout main
git branch -D feature/parakeet-onnx
```

Then clean up locally downloaded models:

```bash
rm -rf ~/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v2-onnx
rm -rf ~/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx
rm -rf ~/.cache/huggingface/hub/models--Olicorne--parakeet-tdt-0.6b-v3-smoothquant-onnx
```

And uninstall the opt-in extras:

```bash
uv pip uninstall onnx-asr onnxruntime
```

The `pyproject.toml` extras group is removed by the branch delete. No database migration. No settings migration. The `web_ui_settings.json` schema is unchanged — if a saved setting references the new backend, the dashboard simply shows an unknown-enum error and the user picks `parakeet-tdt` again.

---

## 10. What we explicitly do NOT do

- **No upstream PR yet.** This is a fork-only feature to start. After it stabilizes locally, we can upstream separately.
- **No live transcription for ONNX.** Documented limitation. See `web_ui/Guide.md` for the user-facing note.
- **No MPS support.** Documented limitation. Apple Silicon users keep using `parakeet-tdt`.
- **No new lint rules / no pyproject.toml refactors.** Minimal-touch change.
- **No dashboard code changes.** The existing `settings_schema.py` introspection handles the new field automatically.
- **No changes to the existing `parakeet-tdt` defaults.** Users who never pick `parakeet-onnx` see exactly the same UI they see now.

---

## 11. Resolved open questions

These were open in earlier drafts and are now locked:

- **Default variant** is `v3` (matches the existing `parakeet-tdt` default and the reachy_conv_app default).
- **Apple Silicon behavior** is fail-loud: the handler refuses `--device mps` with a clear `RuntimeError`. The user falls back to `parakeet-tdt` for MLX.
- **Live transcription** is not implemented in v1. The conversation works fine without it (final-only transcription is what triggers the LLM). The Dashboard's `--enable-live-transcription` flag is silently ignored for this backend.
- **Warmup** runs once per `setup()` call, matching the existing `parakeet_tdt_handler` pattern. One second of silence at 16 kHz fed through `model.recognize()` and discarded.
- **No `canary` variant.** Per user instruction: "exept the canary one i dont care". Locked list is v2, v3, v3sq. Adding canary later is one line in the `_VARIANTS` dict.
- **No `path` override** for the local model directory. The Dashboard does not expose it. Power users can pass raw CLI args if they really need to.
- **`pnc`, `target_language` RecognizeOptions kwargs** are ignored — they only apply to Whisper/Canary, not Parakeet.
- **GPU/CPU auto-detect + CPU fallback** (added after initial implementation). `--device auto` picks CUDA when onnxruntime reports it available, else CPU. `--device cuda` requests CUDA and falls back to CPU on load/warmup failure so the pipeline always starts. This means a box with a broken CUDA install (the developer's 1080Ti with missing libcudnn, ORT GPU wheel mismatches, unsupported compute capability, etc.) will start `--stt parakeet-onnx` on CPU without the user having to know. Covered by tests 6-13 in `tests/test_parakeet_onnx_handler.py`.

---

## 12. Files checklist (for the implementer)

When implementation begins, this is the exact diff manifest:

- [ ] Read `src/speech_to_speech/arguments_classes/parakeet_tdt_arguments.py` (already exists, will be mirrored)
- [ ] Create `src/speech_to_speech/arguments_classes/parakeet_onnx_arguments.py` (new file, ~40 lines)
- [ ] Read `src/speech_to_speech/STT/parakeet_tdt_handler.py` (already exists, will be mirrored structurally)
- [ ] Create `src/speech_to_speech/STT/parakeet_onnx_handler.py` (new file, ~250 lines)
- [ ] Edit `src/speech_to_speech/arguments_classes/module_arguments.py` (one line in the `--stt` enum)
- [ ] Edit `src/speech_to_speech/s2s_pipeline.py` (9 small touch points — see below)
- [ ] Edit `pyproject.toml` (one new optional-dependency group `[parakeet-onnx]`)
- [ ] Create `tests/test_parakeet_onnx_handler.py` (new file, ~150 lines)
- [ ] Edit `web_ui/Guide.md` (one new section, ~30 lines)

`s2s_pipeline.py` touch points (line numbers approximate, will be re-verified at edit time):

1. Import the new arguments class near line 40
2. Add field to `ParsedArguments` near line 103
3. Add to `by_type[...]` lookup near line 193
4. Add to `prepare_all_args` parameter list near line 299
5. Add `rename_args(parakeet_onnx_stt_handler_kwargs, "parakeet_onnx")` near line 330
6. Add to `_build_pipeline_handlers` parameter list (one of the 467 / 604 / 667 / 752 sites)
7. Add to `get_stt_handler` parameter list near line 776
8. Add the `elif module_kwargs.stt == "parakeet-onnx":` branch near line 842
9. Add to `prepare_args` / `prepare_args_websocket` near lines 1030 / 1075

Every touch point is mechanical. None are structural changes to existing logic.

---

## 13. Estimated effort

- Plan doc (this file): DONE
- Implementation: 1–2 days of focused work
- Test writing: 0.5 day
- End-to-end verification: 0.5 day
- **Total: 2–3 days**

This is intentionally a small, scoped change. It is *not* a rewrite of the STT pipeline. It is one new path that follows the existing dispatch pattern exactly.

---

## 14. Approval

I will not write any code until the user approves this plan. If the user wants changes, edits happen here, then re-approval, then code.
