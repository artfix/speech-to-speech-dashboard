"""qwentts-cpp-python Pascal (sm_61) wheel installer.

The upstream ``qwentts-cpp-python`` PyPI wheel ships ``libggml-cuda.so`` built
without Pascal SASS — loading it on a GTX 10XX card (sm_61) crashes the very
first CUDA dispatch with::

    CUDA_ERROR_NO_BINARY_FOR_GPU: no kernel image is available for execution
    on the device
    ggml_abort()

To make the dashboard usable on Pascal-class GPUs without forcing every
upstream wheel consumer to recompile, we ship a Pascal-compatible wheel. It
can come from either of two sources (checked in this order):

1. A wheel bundled at ``web_ui/wheels/qwentts_cpp_python-*.whl`` (when the
   repo was cloned with the wheel already in place — the historical
   pre-HF-Space arrangement).
2. A wheel downloaded on demand from the HF Dataset
   ``ArtFix0/speech-to-speech-wheels`` (the current HF Space deployment
   path: HF Spaces reject files over 10 MiB in regular git, so the wheel
   lives in a separate Dataset repo and the dashboard fetches it the
   first time a Pascal user runs the pipeline).

The install itself runs the first time the user picks ``--tts qwen3``
with ``--qwen3-tts-device cuda`` on a sm_61 box.

Behavior:

* Detect the GPU's compute capability via PyTorch (which always reports it
  correctly, even when nvidia-smi is flaky).
* If the GPU is NOT sm_61 → no-op (the upstream wheel is fine on Volta+).
* If the user picked a non-cuda qwen3 device (cpu/vulkan/sycl) → no-op.
* If a wheel matching the platform tag is bundled in ``web_ui/wheels/`` →
  install it. Otherwise, fetch from the HF Dataset → install.
* If both fail → log a warning and let upstream take over (the pipeline
  will crash with the original error, but the dashboard keeps working;
  user can report).

The check + install runs once per dashboard start. The result is cached in
``~/.cache/speech-to-speech-dashboard/qwentts_sm61_install.json`` so we don't
re-download / re-install on every pipeline restart.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Cache file marks "we already attempted the install on this machine; the
# installed package matches the bundled wheel". Subsequent starts just verify
# and skip.
_CACHE_DIR = Path(os.path.expanduser("~/.cache/speech-to-speech-dashboard"))
_CACHE_FILE = _CACHE_DIR / "qwentts_sm61_install.json"

# Only sm_61 (Pascal consumer, e.g. GTX 1060/1070/1080/1080 Ti) needs this
# wheel. sm_62 is the Tesla P40/P4 server variant — same family, same SASS
# size limits, so include it for completeness.
_PASCEL_CC = {"6.1", "6.2"}

# HuggingFace Dataset that holds wheels too large for an HF Space repo
# (HF Spaces cap regular-git files at 10 MiB; the Pascal qwentts wheel is
# 126 MiB). The dataset is a public mirror maintained by the dashboard
# author; the dashboard downloads on first need and caches the file under
# ``web_ui/wheels/`` so subsequent runs don't hit the network.
_HF_WHEEL_DATASET = "ArtFix0/speech-to-speech-wheels"
# Wheel filenames in the dataset mirror the bundled naming so
# ``_matching_wheel()`` finds them with no changes.
_HF_WHEEL_FILENAME = "qwentts_cpp_python-0.3.1-py3-none-linux_x86_64.whl"
_HF_WHEEL_URL = (
    f"https://huggingface.co/datasets/{_HF_WHEEL_DATASET}/resolve/main/{_HF_WHEEL_FILENAME}"
)
_HF_DOWNLOAD_TIMEOUT_S = 600


def _bundled_wheel_dir() -> Path:
    """Directory where the Pascal-compatible wheel is shipped."""
    return Path(__file__).resolve().parent / "wheels"


def _detect_compute_capability() -> str | None:
    """Return the active CUDA device's compute capability as a string like '6.1'.

    Returns None if PyTorch isn't installed, CUDA isn't available, or no GPU
    is visible. We prefer PyTorch because it's a hard dashboard dependency.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    except Exception:
        return None


def _is_pascal() -> bool:
    cc = _detect_compute_capability()
    return cc is not None and cc in _PASCEL_CC


def _user_wants_qwen3_cuda(settings: dict[str, Any]) -> bool:
    """True iff the dashboard settings select qwen3 TTS on a CUDA device."""
    if (settings.get("--tts") or "").strip() != "qwen3":
        return False
    device = (settings.get("--qwen3-tts-device") or "cuda").strip().lower()
    return device in ("cuda", "gpu", "")


def _matching_wheel() -> Path | None:
    """Find the Pascal-compatible wheel bundled in web_ui/wheels/ matching this Python.

    Wheel filenames take one of three python tags:

    - ``py3-none-linux_x86_64.whl`` (universal py3 — matches any Python 3)
    - ``cp310-cp310-manylinux_2_28_x86_64.whl`` (CPython-specific)
    - ``cp311-cp311-manylinux_2_28_x86_64.whl``
    - ``cp312-cp312-manylinux_2_28_x86_64.whl``

    We prefer the py3 universal wheel if present (it always works), then
    fall back to the matching cp-specific tag.
    """
    wheel_dir = _bundled_wheel_dir()
    if not wheel_dir.is_dir():
        return None
    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    # 1. Universal py3 wheel — matches any Python 3.
    universal = sorted(wheel_dir.glob("qwentts_cpp_python-*-py3-none-*.whl"))
    if universal:
        return universal[-1]
    # 2. CPython-specific wheel for this interpreter.
    matches = sorted(wheel_dir.glob(f"qwentts_cpp_python-*{py_tag}-*{py_tag}*.whl"))
    if matches:
        return matches[-1]
    # 3. Broad fallback: any cp-specific linux wheel for this Python.
    matches = sorted(wheel_dir.glob(f"qwentts_cpp_python-*{py_tag}-*linux*.whl"))
    return matches[-1] if matches else None


def _download_pascal_wheel() -> Path | None:
    """Download the Pascal wheel from the HF Dataset to ``web_ui/wheels/``.

    Returns the local file path on success, ``None`` on failure. Called by
    ``install_pascal_wheel_if_needed`` when the wheel isn't already bundled
    (the typical case for a fresh clone of an HF Space deployment, where
    HF rejected the wheel for being too large for plain git).

    Streams the response to disk so we don't hold the whole 126 MiB in
    memory. Validates that the downloaded file is a real zip (a 404 page
    masquerading as a wheel is the most likely failure mode).
    """
    target_dir = _bundled_wheel_dir()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError:  # noqa: BLE001
        logger.exception("Could not create web_ui/wheels/ directory")
        return None

    target = target_dir / _HF_WHEEL_FILENAME
    if target.is_file() and target.stat().st_size > 50 * 1024 * 1024:
        # Sanity: anything under 50 MiB is almost certainly an HTML error
        # page saved by a previous failed download. Re-download.
        try:
            target.unlink()
        except OSError:  # noqa: BLE001
            pass

    try:
        import httpx  # local import — only needed for the download path.
    except ImportError:
        logger.warning(
            "httpx not installed; cannot download Pascal qwentts wheel from "
            "HF Dataset. Install httpx (already a dashboard dependency) and retry."
        )
        return None

    logger.info(
        "Downloading Pascal-compatible qwentts wheel from HF Dataset (%s) ...",
        _HF_WHEEL_URL,
    )
    try:
        with httpx.Client(timeout=_HF_DOWNLOAD_TIMEOUT_S, follow_redirects=True) as client:
            with client.stream("GET", _HF_WHEEL_URL) as resp:
                if resp.status_code != 200:
                    logger.error(
                        "HF Dataset wheel download failed: HTTP %s for %s",
                        resp.status_code,
                        _HF_WHEEL_URL,
                    )
                    return None
                # Stream to a .tmp file, then atomic rename so a partial
                # download doesn't leave a broken wheel behind.
                tmp = target.with_suffix(".whl.tmp")
                bytes_written = 0
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
                        bytes_written += len(chunk)
                # Final size sanity: <50 MiB is certainly not a real wheel.
                if bytes_written < 50 * 1024 * 1024:
                    logger.error(
                        "HF Dataset wheel download too small (%d bytes); "
                        "expected ~126 MiB. Treating as failure.",
                        bytes_written,
                    )
                    try:
                        tmp.unlink()
                    except OSError:  # noqa: BLE001
                        pass
                    return None
                tmp.replace(target)
    except httpx.HTTPError:  # noqa: BLE001
        logger.exception("HF Dataset wheel download raised")
        return None
    except OSError:  # noqa: BLE001
        logger.exception("Could not write downloaded wheel to %s", target)
        return None

    logger.info(
        "Downloaded Pascal-compatible qwentts wheel: %s (%d MiB)",
        target,
        target.stat().st_size // (1024 * 1024),
    )
    return target


def _already_installed(wheel: Path) -> bool:
    """True if the installed qwentts_cpp_python matches the bundled wheel."""
    try:
        from importlib.metadata import version, distribution  # type: ignore
    except Exception:
        return False
    try:
        dist = distribution("qwentts-cpp-python")
    except Exception:
        return False
    installed_version = version("qwentts-cpp-python")
    # wheel filename looks like qwentts_cpp_python-0.3.1-cp312-...whl
    stem = wheel.name.replace(".whl", "")
    parts = stem.split("-")
    if len(parts) < 2:
        return False
    bundled_version = parts[1]
    return installed_version == bundled_version


def _read_cache() -> dict[str, Any]:
    try:
        return json.loads(_CACHE_FILE.read_text("utf-8"))
    except Exception:
        return {}


def _write_cache(payload: dict[str, Any]) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(payload, indent=2), "utf-8")
    except Exception:
        logger.exception("Could not write qwentts installer cache")


def install_pascal_wheel_if_needed(settings: dict[str, Any]) -> dict[str, Any]:
    """Check & install the Pascal-compatible qwentts wheel.

    Safe to call on every pipeline start: no-op unless this is a sm_61 box,
    the user picked qwen3+CUDA, and the bundled wheel isn't already
    installed.

    Returns a dict ``{"status": ..., "wheel": ..., "message": ...}`` so the
    process manager can log it and the UI can surface it.
    """
    # Apply the dashboard's venv shim on every start. Idempotent: a no-op
    # when the file already has the patched body. Keeps the OpenAI Realtime
    # path from crashing with ``QwenTTSError: Speaker enumeration requires
    # qwentts.cpp ABI v2`` even after ``uv sync`` reinstalls the upstream
    # wheel. See ``_patch_faster_qwen3_tts_if_needed`` for the full rationale.
    try:
        shim_result = _patch_faster_qwen3_tts_if_needed()
        if shim_result.get("status") == "patched":
            logger.info(
                "Applied faster_qwen3_tts ABI v2 speaker-enumeration shim (%s)",
                shim_result.get("path"),
            )
    except Exception:  # noqa: BLE001
        logger.exception("faster_qwen3_tts shim patcher crashed; continuing")

    result: dict[str, Any] = {"status": "skipped", "reason": "no-op"}

    if not _is_pascal():
        result["reason"] = "not Pascal"
        return result

    if not _user_wants_qwen3_cuda(settings):
        result["reason"] = "qwen3 not selected on CUDA"
        return result

    wheel = _matching_wheel()
    if wheel is None:
        # No bundled wheel — typical case for a fresh clone of an HF Space
        # deployment, where plain git rejected the 126 MiB wheel. Try
        # downloading from the HF Dataset that holds it.
        logger.info(
            "No bundled Pascal qwentts wheel found in web_ui/wheels/. "
            "Attempting to download from HF Dataset %s ...",
            _HF_WHEEL_DATASET,
        )
        if _download_pascal_wheel() is not None:
            wheel = _matching_wheel()
    if wheel is None:
        logger.warning(
            "Pascal GPU detected but no qwentts Pascal wheel available (not "
            "bundled and HF Dataset download failed). qwen3-TTS on CUDA will "
            "likely fail with 'no kernel image is available for execution on "
            "the device'."
        )
        result.update({"status": "missing-wheel", "reason": "no bundled wheel"})
        return result

    if _already_installed(wheel):
        result.update({"status": "already-installed", "wheel": str(wheel)})
        return result

    cache = _read_cache()
    if cache.get("status") == "installed" and cache.get("wheel") == wheel.name:
        # We tried last time and it succeeded; double-check the import still works.
        try:
            import qwentts_cpp  # type: ignore # noqa: F401
            result.update({"status": "already-installed", "wheel": str(wheel)})
            return result
        except Exception:
            pass  # fall through to reinstall

    logger.info("Installing Pascal-compatible qwentts-cpp-python wheel: %s", wheel)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--force-reinstall",
                "--no-deps",
                "--no-index",
                str(wheel),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            logger.error(
                "pip install of Pascal qwentts wheel failed (rc=%s):\nstdout:\n%s\nstderr:\n%s",
                proc.returncode,
                proc.stdout,
                proc.stderr,
            )
            result.update({"status": "install-failed", "wheel": str(wheel)})
            return result
    except Exception:
        logger.exception("pip install of Pascal qwentts wheel raised")
        result.update({"status": "install-failed", "wheel": str(wheel)})
        return result

    # pip/uv install leaves a small upstream-style libggml-cuda.so behind
    # in the same directory even with --force-reinstall (RECORD-level
    # overwrite doesn't always replace files whose size differs). Hard-
    # replace from the wheel zip so the bundled Pascal SASS is actually
    # in place. Skipping this leaves the upstream 121MB lib (no sm_61)
    # sitting on disk and qwen3-TTS silently crashes on first CUDA
    # dispatch.
    try:
        import zipfile

        with zipfile.ZipFile(wheel) as zf:
            for name in zf.namelist():
                if not name.startswith("qwentts_cpp/lib/"):
                    continue
                rel = name[len("qwentts_cpp/lib/") :]
                target = _bundled_wheel_dir().parent / "qwentts_cpp" / "lib" / rel
                # Real install path is site-packages/qwentts_cpp/lib/, not
                # web_ui/wheels/. Walk up from the wheel directory.
                import site

                site_root = Path(next(p for p in site.getsitepackages() if (Path(p) / "qwentts_cpp").exists()))
                target = site_root / "qwentts_cpp" / "lib" / rel
                with zf.open(name) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                logger.info("Hard-replaced %s with bundled wheel file", target)
    except Exception:
        logger.exception(
            "Hard-replace of installed qwentts_cpp lib files failed; "
            "qwen3-TTS may not see the Pascal SASS until files are replaced manually."
        )

    # Verify the import works after install.
    try:
        import qwentts_cpp  # type: ignore # noqa: F401
    except Exception:
        logger.exception(
            "qwentts-cpp-python wheel installed but import still fails. "
            "The Pascal wheel may not match this Python version."
        )
        result.update({"status": "import-failed", "wheel": str(wheel)})
        return result

    _write_cache({"status": "installed", "wheel": wheel.name})
    result.update({"status": "installed", "wheel": str(wheel)})
    logger.info("qwentts-cpp-python Pascal wheel installed OK: %s", wheel.name)
    return result


# ----------------------------------------------------------------------
# Venv shim: rewrite faster_qwen3_tts.get_supported_speakers() to []
# ----------------------------------------------------------------------


# Marker comment we write at the top of the patched body so we can
# detect "already patched" without diff'ing the whole file.
_SHIM_MARKER = "# PATCH (dashboard v0.3.0): always return []."


def _faster_qwen3_tts_path() -> "Path | None":
    """Locate the installed faster_qwen3_tts/ggml_backend.py.

    Walks ``site.getsitepackages()`` for ``faster_qwen3_tts/ggml_backend.py``.
    Returns ``None`` when the package isn't importable (yet) — the caller
    treats that as a silent no-op so we never break a non-qwen3 user.
    """
    try:
        import site  # noqa: PLC0415
        sp_paths = [Path(p) for p in site.getsitepackages()]
    except Exception:  # noqa: BLE001
        sp_paths = [Path(p) for p in sys.path if "site-packages" in p]

    for sp in sp_paths:
        candidate = sp / "faster_qwen3_tts" / "ggml_backend.py"
        if candidate.is_file():
            return candidate
    return None


def _patch_faster_qwen3_tts_if_needed() -> dict[str, Any]:
    """Rewrite ``get_supported_speakers()`` in faster_qwen3_tts to ``return []``.

    The bundled Pascal ``qwentts-cpp-python`` wheel exposes ABI v1 only,
    so calling ``runtime.speaker_names()`` raises
    ``QwenTTSError: Speaker enumeration requires qwentts.cpp ABI v2``.
    The OpenAI Realtime path trips this on every ``session.update``
    because the upstream ``qwen3_tts_handler._apply_session_voice_override``
    consults ``get_supported_speakers()`` to decide whether to honor the
    client-sent ``voice`` field.

    Replacing the body with a constant ``[]`` makes the handler take its
    existing "ignore client voice, use dashboard-configured speaker" path
    (it already handles the empty-list case with a warning + return).

    Idempotent: re-runs are a no-op once the marker is present. Safe to
    call from every pipeline start: ``uv sync`` reinstalls upstream
    ``faster_qwen3_tts`` and reverts the shim; this re-applies it.

    The shim is intentionally applied to the venv-installed wheel file
    (not to ``src/speech_to_speech/``) per the dashboard's hard
    constraint: the pipeline module is treated as a black box.
    """
    target = _faster_qwen3_tts_path()
    if target is None:
        return {"status": "skipped", "reason": "faster_qwen3_tts not installed"}

    try:
        original = target.read_text(encoding="utf-8")
    except OSError as e:
        return {"status": "error", "reason": f"read failed: {e}"}

    if _SHIM_MARKER in original:
        return {"status": "already-patched", "path": str(target)}

    # Locate the ``def get_supported_speakers(self) -> list[str]:`` block
    # and replace its body with a constant ``return []``. The simplest
    # regex targets the whole function header line + leading docstring /
    # body up to the next top-level ``def`` or end-of-class indent. We
    # deliberately keep the patch narrow: if the function signature ever
    # changes upstream, this becomes a no-op rather than a syntax error.
    import re  # noqa: PLC0415

    pattern = re.compile(
        r"(    def get_supported_speakers\(self\)(?:\s*->\s*[^:]+)?\s*:\n)"
        r"(?:        [^\n]*\n)*?"  # any existing body lines (incl. docstring)
    )
    new_body = (
        "    def get_supported_speakers(self) -> list[str]:\n"
        f"        {_SHIM_MARKER}  The bundled Pascal qwentts-cpp-python wheel\n"
        "        # exposes ABI v1 only; ``runtime.speaker_names()`` raises\n"
        "        # ``QwenTTSError: Speaker enumeration requires ABI v2`` on\n"
        "        # every OpenAI Realtime ``session.update`` that sends a\n"
        "        # ``voice`` field. Returning ``[]`` makes the upstream\n"
        "        # handler take its existing \"ignore client voice\" branch\n"
        "        # (logged as a warning) and use the dashboard-configured\n"
        "        # ``--qwen3-tts-speaker`` instead. The 9 CustomVoice\n"
        "        # preset speakers are baked into the model weights and\n"
        "        # don't require enumeration at runtime.\n"
        "        return []\n"
    )

    new_src, n = pattern.subn(new_body, original, count=1)
    if n != 1:
        # The signature didn't match (upstream changed). Log and skip;
        # the worst case is the realtime path keeps crashing with the
        # QwenTTSError, which is exactly the behavior the user had
        # before this shim was added.
        logger.warning(
            "Could not find faster_qwen3_tts.get_supported_speakers() to patch "
            "(signature may have changed upstream). Realtime qwen3 path may "
            "crash with 'QwenTTSError: Speaker enumeration requires ABI v2'."
        )
        return {"status": "skipped", "reason": "signature mismatch"}

    try:
        # Atomic write: .tmp + rename so a crash mid-write can't corrupt
        # the upstream module (which would break ``import faster_qwen3_tts``
        # for everyone, including non-qwen3 users).
        tmp = target.with_suffix(".py.tmp")
        tmp.write_text(new_src, encoding="utf-8")
        tmp.replace(target)
    except OSError as e:
        logger.exception("Could not write faster_qwen3_tts shim: %s", e)
        return {"status": "error", "reason": f"write failed: {e}"}

    return {"status": "patched", "path": str(target)}
