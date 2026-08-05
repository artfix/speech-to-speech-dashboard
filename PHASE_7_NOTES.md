# Phase 7: pyproject.toml reconciliation notes

Compared local `pyproject.toml` against upstream `v0.2.12`. No changes required.

## Kept as-is (local/dashboard requirements)
- `version = "0.2.12"` (set in Phase 1).
- `nltk==3.10.0` (set in Phase 1).
- `psutil>=5.9.0` — dashboard GPU probing.
- `chatterbox` extra — dashboard voice library.
- `parakeet-onnx` extra — dashboard ONNX STT backend.
- `speech-to-speech-web = "web_ui.server:main"` console script.
- `where = ["src", "."]` and `include = ["speech_to_speech*", "web_ui*"]` for setuptools.
- `web_ui` package-data for static files, themes, and Guide.

## Intentionally skipped from upstream
- `huggingface-hub>=0.34.0` — not imported anywhere in `src/`, `tests/`, or `web_ui/`.
- `onnxruntime>=1.17.0` in core deps — kept only in `parakeet-onnx` extra for now. Will be added to core deps in Phase 6 only if Smart Turn VAD is merged.

## Verification
- `uv build` succeeds.
- `uvx twine check --strict dist/*` passes.
