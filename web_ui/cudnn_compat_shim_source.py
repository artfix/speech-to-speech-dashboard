"""GPU preflight shim — imported once at Python startup by the
``_speech_to_speech_cudnn_compat.pth`` site-packages file.

Why this exists
---------------

PyTorch wheels for CUDA 12.x (cu124, cu126) list sm_60 in their
compiled CUDA arch list, but the cuDNN 9.x versions they bundle
refuse to compile a fused ``F.conv1d`` kernel at runtime on Pascal
hardware (sm_60 / sm_61). The pipeline dies with::

    RuntimeError: GET was unable to find an engine to execute this computation

when chatterbox TTS tries to run inference on Pascal. This was verified
on a GTX 1080 Ti (CC 6.1) with torch 2.6.0+cu126 — the wheel installs
cleanly, the arch-list check passes, but the actual conv1d op fails.

The upstream-documented fix is to disable cuDNN before any inference
code runs::

    torch.backends.cudnn.enabled = False

This forces the cuDNN reference conv path, which works on all SM
versions including Pascal. The flag has to be set before torch's
first forward pass; setting it later is too late for the JIT cache.

How it's wired
--------------

The dashboard installs this package alongside its companion
``.pth`` file in the venv's ``site-packages``. CPython's site module
processes ``*.pth`` files at interpreter startup, executing any line
that starts with ``import``. Importing this package IS the install
hook. Nothing in ``src/`` is patched; chatterbox itself is unchanged.
"""

from __future__ import annotations

import sys
import warnings

# Idempotent: project site-processing can happen twice for some
# build hooks (uv, editable installs).
_ALREADY_RAN_ATTR = "_speech_to_speech_cudnn_compat_ran"


def _apply_if_needed() -> None:
    try:
        import torch
    except Exception:
        return

    if getattr(torch, _ALREADY_RAN_ATTR, False):
        return

    try:
        if not torch.cuda.is_available():
            return
        major, minor = torch.cuda.get_device_capability(0)
        if major < 7:  # Pascal / Maxwell / Kepler — pre-Volta.
            torch.backends.cudnn.enabled = False
            print(
                f"[cudnn-compat] Pascal-class GPU (sm_{major}{minor}) detected; "
                f"cuDNN disabled for cross-version conv safety.",
                file=sys.stderr,
            )
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"cudnn-compat shim no-op: {e!r}")
    finally:
        try:
            setattr(torch, _ALREADY_RAN_ATTR, True)
        except Exception:  # noqa: BLE001
            pass


_apply_if_needed()
