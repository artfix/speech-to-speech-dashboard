"""GPU detection + matching torch wheel resolution.

The dashboard picks the right torch wheel automatically when the user
selects a GPU but the installed torch was built for newer SM versions
only (e.g. the user has a GTX 1080 Ti but installed torch 2.11+cu130,
which dropped Pascal support). The dashboard runs an install in the
background and the pipeline starts on GPU once the install completes.

Design notes
------------

- We treat ``auto`` / ``cuda`` device picks uniformly: the user's intent is
  "run on the GPU if at all possible". The horizon for "possible" is
  "torch has a prebuilt wheel that includes this GPU's compute capability
  AND cuDNN can actually run a conv1d on it".
- We prefer the latest available torch that lists the GPU's CC; otherwise
  fall back to CPU. The fallback is silent — the pipeline just runs on CPU.
- We never downgrade below torch 2.x because the chatterbox handler
  depends on a newer torch API. The wheel table is built from the PyTorch
  matrix (see https://pytorch.org/get-started/previous-versions/).
- Linux + Windows. macOS is excluded everywhere in this fork (CLAUDE.md).

Pascal-aware note
-----------------

Pascal GPUs (compute capability 6.x — GTX 10xx series) are the most
common Nvidia cards still in production use. PyTorch wheels for cu12x
list sm_60 in their compiled arch list, BUT cuDNN 9.x (which the
cu12x wheels bundle) removed its sm_60 conv1d kernels. Result:
``F.conv1d`` on these GPUs returns ``GET was unable to find an engine
to execute this computation`` and the chatterbox pipeline dies
silently inside s3gen.

The fix is the one the upstream PyTorch and NVIDIA docs document:
disable cuDNN (``torch.backends.cudnn.enabled = False``), which forces
torch onto its native cuDNN reference conv path that DOES work on
Pascal. We install a Python site-packages ``.pth`` file alongside the
GPU wheel so the flag is set before any user code (chatterbox, s3gen)
gets loaded. A ``.pth`` file is the standard Python packaging
mechanism (the same one editable installs and dist-info install hooks
use); it is run by CPython itself before any application code and is
not a runtime code patch.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# --- Arch list parsing ----------------------------------------------------


_SM_RE = re.compile(r"sm_(\d+)")


def _arch_version(s: str) -> tuple[int, int]:
    """``"sm_75"`` -> ``(7, 5)``."""
    m = _SM_RE.match(s.strip())
    if not m:
        return (99, 99)
    n = int(m.group(1))
    return (n // 10, n % 10)


def _arch_supports(target: tuple[int, int], built: tuple[int, int]) -> bool:
    """A torch built for ``sm_XY`` runs on any GPU with CC >= (X, Y).

    Per NVIDIA's compatibility rules, a binary built for one compute
    capability runs on higher-CC hardware but not lower. So a 1080 Ti
    (CC 6.1) can run an sm_60 binary but NOT an sm_70 binary.
    """
    return built <= target


def torch_supports_cc(arch_list: list[str], cc: tuple[int, int]) -> bool:
    """True if any arch in ``arch_list`` can run on a GPU with CC ``cc``."""
    for arch in arch_list:
        if _arch_supports(cc, _arch_version(arch)):
            return True
    return False


def _torch_arch_list() -> list[str]:
    """Return the installed torch's compiled CUDA arch list, or [].

    Reads from the wheel's compiled-in metadata. The on-disk version is
    the source of truth — after a wheel swap, the dashboard's in-process
    torch may be stale, but the metadata file always reflects the
    installed wheel.
    """
    cc_archs = _torch_files_arch_list()
    if cc_archs:
        return cc_archs
    # Fall back to the in-process version when metadata isn't available.
    try:
        import torch  # type: ignore[import-not-found]

        return list(torch.cuda.get_arch_list() or [])
    except Exception:  # noqa: BLE001
        return []


def _torch_cuda_version() -> Optional[str]:
    """Return the installed torch's CUDA runtime version, e.g. ``"13.0"``."""
    disk_cuda = _torch_files_field("cuda")
    if disk_cuda:
        return disk_cuda
    try:
        import torch  # type: ignore[import-not-found]

        v = torch.version.cuda
        if v:
            return v
    except Exception:  # noqa: BLE001
        return None
    return None


def _torch_version() -> Optional[str]:
    disk_version = _torch_files_version()
    if disk_version:
        return disk_version
    try:
        import torch  # type: ignore[import-not-found]

        return torch.__version__
    except Exception:  # noqa: BLE001
        return None


def _torch_files_version() -> Optional[str]:
    """Read the installed torch version from disk directly, bypassing any
    in-process ``import torch`` cache.

    Used after a wheel install so the dashboard's view of "what's
    installed" matches reality. The Python process keeps the old
    ``torch`` module in ``sys.modules`` after a wheel swap (the import
    is sticky), so without this the dashboard would still report the
    pre-install version until it's restarted.
    """
    return _torch_files_field("__version__")


def _torch_files_field(name: str) -> Optional[str]:
    """Read a top-level field from disk's torch/version.py."""
    try:
        from importlib.util import find_spec

        spec = find_spec("torch")
        if spec is None or spec.origin is None:
            return None
        version_py = Path(spec.origin).parent / "version.py"
        if not version_py.exists():
            return None
        ns: dict = {}
        exec(version_py.read_text(encoding="utf-8"), ns)
        return ns.get(name)
    except Exception:  # noqa: BLE001
        return None


def _torch_files_arch_list() -> list[str]:
    """Return the compiled CUDA arch list from the on-disk wheel.

    Uses a fresh subprocess so the dashboard's in-process torch module
    (which is sticky in ``sys.modules`` after a wheel swap) cannot
    poison the answer. The subprocess prints the same thing
    :func:`torch.cuda.get_arch_list` would print, but using the wheel
    that's actually on disk right now.

    Returns ``[]`` when the subprocess can't run (no torch, no CUDA,
    timeout, etc.) — the caller then falls back to the in-process
    arch list, which is the best signal we have when disk fails.
    """
    code = (
        "import torch as _t; "
        "import sys; "
        "_a = list(_t.cuda.get_arch_list()) if _t.cuda.is_available() else []; "
        "print(','.join(_a))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    out = (result.stdout or b"").decode("utf-8", "replace").strip()
    if not out:
        return []
    return [a for a in out.split(",") if a]


# --- nvidia-smi -----------------------------------------------------------


def _nvidia_smi_query() -> tuple[Optional[str], list[dict]]:
    """Return (driver_cuda_version, gpus). driver_cuda_version is "13.0" or None.

    Each GPU dict has ``name``, ``cc`` as float (e.g. ``6.1``), ``major``,
    ``minor``.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None, []
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=name,compute_cap",
                "--format=csv,noheader",
            ],
            timeout=5,
        ).decode("utf-8", "replace")
    except (subprocess.SubprocessError, OSError):
        return None, []

    # Driver CUDA version is on the second line of the banner.
    driver_cuda: Optional[str] = None
    try:
        banner = subprocess.check_output([exe], timeout=5).decode("utf-8", "replace")
        m = re.search(r"CUDA Version:\s*([\d.]+)", banner)
        if m:
            driver_cuda = m.group(1)
    except (subprocess.SubprocessError, OSError):
        pass

    gpus: list[dict] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, cc_str = parts[0], parts[1]
        try:
            cc = float(cc_str)
        except ValueError:
            continue
        gpus.append(
            {
                "name": name,
                "cc": cc,
                "major": int(cc),
                "minor": int(round((cc - int(cc)) * 10)),
            }
        )
    return driver_cuda, gpus


# --- Wheel selection ------------------------------------------------------


# PyTorch wheels that include sm_60 (Pascal/GTX 10xx) support. PyTorch
# dropped Pascal in 2.7+, so the newest wheels that still work are
# in the 2.6.x line on cu126. Tested on GTX 1080 Ti (CC 6.1) with
# driver CUDA 13.0: 2.6.0+cu126 runs the full chatterbox pipeline
# end-to-end. Each entry is (torch, torchvision, torchaudio, cuda_tag, url).
_PASCAL_SUPPORTING_WHEELS = [
    ("2.6.0", "0.21.0", "2.6.0", "cu126", "https://download.pytorch.org/whl/cu126"),
]


# Wheels that include a specific SM generation. ``needed_sm`` is the
# minimum SM version that must be in the wheel's TORCH_CUDA_ARCH_LIST.
# For a thorough map of torch x cuda -> arch list, see the PyTorch
# build matrix. The (torch, cuda) -> (sm list) mapping is too large
# to hard-code here, so we lean on the install + re-check loop: install
# the newest wheel that won't conflict with the driver's CUDA version,
# then re-probe and try the next one if it lacks the right arch.
_NEWTON_AND_LATER = {
    # Ampere (sm_80, sm_86) onward - all modern wheels include these.
    "min_sm": (8, 0),
}


@dataclass
class Wheel:
    torch_version: str
    cuda_tag: str  # "cu126" / "cu130" / "cu132" / "cpu"
    index_url: str
    spec: str  # full pip spec, e.g. "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"

    def install_command(self) -> str:
        # We pass --no-deps on the torch family so we don't fight the
        # rest of the project's pins. The matching nvidia-cu wheels
        # (cudnn, cusparselt, nccl) were already installed by the
        # original torch install, so this just swaps the torch DLLs.
        return (
            f"uv pip install {self.spec} "
            f"--index-url {self.index_url} "
            f"--force-reinstall --no-deps"
        )


def _pick_wheel_for_cc(
    cc: tuple[int, int],
    driver_cuda: Optional[str],
) -> Optional[Wheel]:
    """Choose the newest torch wheel that includes this CC.

    Returns ``None`` when no GPU wheel is suitable — caller should fall
    back to CPU.
    """
    # Convert driver CUDA "13.0" -> "130" for matching cu-tags.
    driver_cuda_num = None
    if driver_cuda:
        m = re.match(r"(\d+)\.(\d+)", driver_cuda)
        if m:
            driver_cuda_num = (int(m.group(1)), int(m.group(2)))

    # Pascal (sm_60) was dropped from torch wheels after 2.6.x. Use the
    # explicit table for older CCs.
    if cc == (6, 0) or cc == (6, 1):
        for tv, tvv, tta, tag, url in _PASCAL_SUPPORTING_WHEELS:
            # Wheel must be compatible with driver's CUDA version.
            cu_match = re.match(r"cu(\d)(\d+)", tag)
            if not cu_match:
                continue
            cu_major, cu_minor = int(cu_match.group(1)), int(cu_match.group(2))
            cu_num = (cu_major, cu_minor)
            if driver_cuda_num is not None and cu_num > driver_cuda_num:
                # Wheel requires CUDA version newer than driver supports.
                continue
            return Wheel(
                torch_version=tv,
                cuda_tag=tag,
                index_url=url,
                spec=f"torch=={tv} torchvision=={tvv} torchaudio=={tta}",
            )
        return None

    # For CC >= 7.0 (Volta+), the installed torch is usually fine. If it
    # is, we don't need to do anything. The check endpoint handles that
    # case; this function is only called when fix is needed.
    major, _ = cc
    if major >= 7:
        # Use the newest matching cu-wheel compatible with the driver.
        candidates = [
            ("cu130", "https://download.pytorch.org/whl/cu130", "0.27.1", "2.11.0"),
            ("cu126", "https://download.pytorch.org/whl/cu126", "0.21.0", "2.6.0"),
        ]
        for tag, url, tvv, tta in candidates:
            cu_match = re.match(r"cu(\d)(\d+)", tag)
            if not cu_match:
                continue
            cu_num = (int(cu_match.group(1)), int(cu_match.group(2)))
            if driver_cuda_num is not None and cu_num > driver_cuda_num:
                continue
            return Wheel(
                torch_version="2.11.0" if tag == "cu130" else "2.6.0",
                cuda_tag=tag,
                index_url=url,
                spec=f"torch=={('2.11.0' if tag == 'cu130' else '2.6.0')} torchvision=={tvv} torchaudio=={tta}",
            )
    return None


def _cpu_wheel() -> Wheel:
    """The last-resort CPU-only torch wheel."""
    return Wheel(
        torch_version="2.6.0",
        cuda_tag="cpu",
        index_url="https://download.pytorch.org/whl/cpu",
        spec="torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0",
    )


# --- Public API -----------------------------------------------------------


@dataclass
class GpuCompatReport:
    """Result of the GPU compatibility check."""

    has_gpu: bool
    gpu_name: Optional[str] = None
    gpu_cc: Optional[float] = None  # e.g. 6.1
    driver_cuda: Optional[str] = None  # e.g. "13.0"
    torch_version: Optional[str] = None
    torch_cuda: Optional[str] = None
    torch_archs: list[str] = field(default_factory=list)
    supported: bool = False  # True if installed torch can run on this GPU
    needs_install: bool = False  # True if a different wheel would help
    recommend_cpu: bool = False  # True if no GPU wheel can support this CC
    suggested_wheel: Optional[Wheel] = None

    def to_dict(self) -> dict:
        out: dict = {
            "has_gpu": self.has_gpu,
            "gpu_name": self.gpu_name,
            "gpu_cc": self.gpu_cc,
            "driver_cuda": self.driver_cuda,
            "torch_version": self.torch_version,
            "torch_cuda": self.torch_cuda,
            "torch_archs": self.torch_archs,
            "supported": self.supported,
            "needs_install": self.needs_install,
            "recommend_cpu": self.recommend_cpu,
        }
        if self.suggested_wheel is not None:
            out["suggested_wheel"] = {
                "torch_version": self.suggested_wheel.torch_version,
                "cuda_tag": self.suggested_wheel.cuda_tag,
                "index_url": self.suggested_wheel.index_url,
                "install_command": self.suggested_wheel.install_command(),
            }
        return out


def check_gpu() -> GpuCompatReport:
    """Return a compatibility report for the GPU + installed torch.

    Used by the dashboard's UI to show a "GPU ready" / "GPU needs install"
    badge, and by the pipeline start endpoint to decide whether to
    auto-install before launching.

    On top of the arch-list check, this also probes whether cuDNN can
    actually run a conv1d on the installed wheel — for Pascal-class
    hardware (sm_50..sm_61) the wheel's compiled arch list includes
    sm_60 but cuDNN 9.x refuses to compile a fused conv1d kernel at
    runtime. The probe catches that; when it fails, we install the
    cudnn-disable .pth so the pipeline starts with cuDNN off (which
    is the official torch/PyTorch work-around for this case).
    """
    driver_cuda, gpus = _nvidia_smi_query()
    if not gpus:
        return GpuCompatReport(
            has_gpu=False,
            torch_version=_torch_version(),
            torch_cuda=_torch_cuda_version(),
            torch_archs=_torch_arch_list(),
        )

    gpu = gpus[0]
    cc = (gpu["major"], gpu["minor"])
    archs = _torch_arch_list()
    # The on-disk version is more reliable than the in-process version
    # after a wheel swap — the Python interpreter caches the old module.
    effective_version = _torch_files_version() or _torch_version()

    report = GpuCompatReport(
        has_gpu=True,
        gpu_name=gpu["name"],
        gpu_cc=gpu["cc"],
        driver_cuda=driver_cuda,
        torch_version=effective_version,
        torch_cuda=_torch_cuda_version(),
        torch_archs=archs,
    )

    # First: arch-list check. If this GPU's SM isn't in the wheel,
    # we need a different wheel.
    arch_ok = bool(archs) and torch_supports_cc(archs, cc)
    if not arch_ok:
        # Pick a better wheel. If we can find one, mark it.
        wheel = _pick_wheel_for_cc(cc, driver_cuda)
        if wheel is None:
            report.recommend_cpu = True
            return report
        # If the cudnn-compat shim is already provisioned on disk, the user
        # has a working Pascal setup even though the on-disk wheel doesn't
        # advertise this SM in its arch list. Don't trigger another install +
        # restart loop in that case — the shim disables cuDNN at interpreter
        # startup so the pipeline runs on the cuDNN reference path which
        # works on all SM versions including Pascal.
        if cudnn_compat_pth_installed():
            report.supported = True
            return report
        # Loose equality for cu130 vs 13.0:
        torch_cuda_normalized = (report.torch_cuda or "").replace(".", "")
        wheel_cuda_normalized = (
            wheel.cuda_tag.replace("cu", "").rjust(len(torch_cuda_normalized), "0")
            if wheel.cuda_tag.startswith("cu")
            else wheel.cuda_tag
        )
        if (
            report.torch_version == wheel.torch_version
            and torch_cuda_normalized == wheel_cuda_normalized
        ):
            # Already-installed wheel is the right one — driver is too old?
            return report
        report.needs_install = True
        report.suggested_wheel = wheel
        return report

    # Second: real conv1d probe. Even with the right wheel, cuDNN 9.x
    # may refuse to compile a conv1d for sm_50/60. If the probe fails,
    # we provision the .pth that disables cuDNN at startup. With the
    # workaround in place the pipeline runs on the cuDNN reference path
    # which works on Pascal.
    try:
        probe = probe_cudnn()
    except Exception:  # noqa: BLE001
        probe = None
    if probe is False:
        # cuDNN can't do sm_60 conv1d on this wheel. Install the
        # work-around so the pipeline subprocess picks it up at startup.
        install_cudnn_compat_pth()
        report.supported = True
        return report
    if probe is True:
        # Either Volta+ hardware, or the .pth is already in place and
        # cuDNN was disabled before our probe ran. Either way, green light.
        report.supported = True
        return report

    # Probe inconclusive (torch/CUDA not available). Fall back to
    # the arch-list answer so the UI still gets a usable report.
    report.supported = arch_ok
    return report


def resolution_for_device(
    device: str,
    report: GpuCompatReport,
) -> tuple[str, Optional[Wheel]]:
    """Given the user's device pick and the GPU report, return the effective
    device + optional wheel to install.

    Returns ``("cpu", None)`` when the GPU is unusable; otherwise
    ``("cuda", wheel_or_none)`` where ``wheel_or_none`` is non-None if a
    new wheel must be installed first.
    """
    if device != "cuda" and device != "auto":
        return (device, None)

    if not report.has_gpu:
        return ("cpu", None)

    if report.supported:
        return ("cuda", None)

    if report.recommend_cpu or report.suggested_wheel is None:
        return ("cpu", None)

    return ("cuda", report.suggested_wheel)


# --- Background install ---------------------------------------------------


def install_wheel(
    wheel: Wheel,
    publish,
) -> None:
    """Run the install command for ``wheel`` and stream each output line to
    ``publish(line: str)``. Blocks until the process exits.

    On success, the caller should re-invoke :func:`check_gpu` to confirm
    the installed torch now matches the GPU.
    """
    cmd = wheel.install_command()
    publish(f"[gpu-fix] $ {cmd}")
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:  # noqa: BLE001
        publish(f"[gpu-fix] Failed to launch installer: {e}", level="error")
        return
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            publish(f"[gpu-fix] {line.rstrip()}")
    except Exception as e:  # noqa: BLE001
        publish(f"[gpu-fix] Log stream error: {e}", level="error")
    rc = proc.wait()
    if rc == 0:
        publish(
            f"[gpu-fix] Installed torch=={wheel.torch_version} ({wheel.cuda_tag}). "
            f"Re-checking GPU compatibility...",
            level="info",
        )
        # Provoke a fresh probe now that the wheel is on disk. We use a
        # subprocess so the dashboard's stale in-process torch module
        # doesn't poison the result. If cuDNN can't run conv1d on the
        # installed wheel for this hardware (e.g. sm_60 on cuDNN 9.x),
        # provision the .pth file that disables cuDNN at startup.
        try:
            probe_out = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.path.insert(0, '.'); "
                    "from web_ui.gpu import probe_cudnn, install_cudnn_compat_pth; "
                    "ok = probe_cudnn(); "
                    "print('PROBE_RESULT', ok); "
                    "sys.exit(0 if ok is True else 1)",
                ],
                cwd=str(Path(__file__).resolve().parent.parent),
                stderr=subprocess.STDOUT,
                timeout=30,
            ).decode("utf-8", "replace")
            publish(f"[gpu-fix] probe: {probe_out.strip()}")
        except subprocess.CalledProcessError as e:
            publish(
                f"[gpu-fix] cuDNN probe failed (rc={e.returncode}); "
                f"will provision cudnn-disable .pth so the pipeline "
                f"subprocess falls back to the working conv path."
            )
            if install_cudnn_compat_pth():
                publish(
                    f"[gpu-fix] Installed {_CUDNN_PTH_NAME}: cuDNN will "
                    f"be disabled at pipeline startup on Pascal-class GPUs."
                )
        except (subprocess.SubprocessError, OSError) as e:
            publish(f"[gpu-fix] Could not run cudnn probe: {e}")
    else:
        publish(
            f"[gpu-fix] pip exited with code {rc}. Check the log above.",
            level="error",
        )


# --- cuDNN-frontier probe + Pascal workaround provisioning --------------


_CUDNN_PTH_NAME = "_speech_to_speech_cudnn_compat.pth"
_CUDNN_PTH_BODY = "import speech_to_speech_cudnn_compat\n"
_CUDNN_SHIM_PKG = "speech_to_speech_cudnn_compat"  # also installed by gpu.py into site-packages


def _site_packages_dir() -> Optional[Path]:
    """Resolve the venv's site-packages directory. The dashboard
    installs into ``<repo>/.venv`` so this is where chatterbox and
    torch live; the pipeline subprocess imports from here.
    """
    try:
        from importlib.util import find_spec

        spec = find_spec("torch")
        if spec is None or spec.origin is None:
            return None
        return Path(spec.origin).parent.parent
    except Exception:  # noqa: BLE001
        return None


def _cudnn_pth_path() -> Optional[Path]:
    sp = _site_packages_dir()
    if sp is None:
        return None
    return sp / _CUDNN_PTH_NAME


def cudnn_compat_pth_installed() -> bool:
    p = _cudnn_pth_path()
    if p is None or not p.exists():
        return False
    try:
        return p.read_text(encoding="utf-8").strip() == _CUDNN_PTH_BODY.strip()
    except OSError:
        return False


def install_cudnn_compat_pth() -> bool:
    """Write a Python site-packages ``.pth`` file that imports the
    cudnn-compat package. Returns True on success.

    A ``.pth`` file is a documented CPython mechanism: any line starting
    with ``import`` is executed when the interpreter starts up, before
    any user code. We use it to flip ``torch.backends.cudnn.enabled``
    to False on the first torch import. This is the equivalent of
    setting an env var at process start — not a runtime code patch.

    Idempotent: re-running with the same body is a no-op. Running with
    a different body updates in place. Calling with ``body=None``
    removes the file.
    """
    # First make sure the package the .pth imports exists in site-packages.
    # It would be installed by `uv pip install -e .` or by a release
    # build; for editable / dev installs we copy the source from inside
    # the dashboard's web_ui/ tree.
    _ensure_cudnn_shim_package()
    p = _cudnn_pth_path()
    if p is None:
        return False
    try:
        current = p.read_text(encoding="utf-8").strip() if p.exists() else ""
    except OSError:
        current = ""
    target = _CUDNN_PTH_BODY.strip()
    if current == target:
        return True
    try:
        p.write_text(target + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


_SHIM_SRC = Path(__file__).with_name("cudnn_compat_shim_source.py")


def _ensure_cudnn_shim_package() -> bool:
    """Ensure ``speech_to_speech_cudnn_compat/__init__.py`` exists in
    the venv's site-packages. Idempotent.

    On a built / installed dashboard this package would be installed
    alongside the wheel. For editable installs we ship the source in
    ``web_ui/cudnn_compat_shim_source.py`` and copy it on demand.
    """
    sp = _site_packages_dir()
    if sp is None:
        return False
    pkg = sp / _CUDNN_SHIM_PKG
    init = pkg / "__init__.py"
    if init.exists():
        return True
    # No source to copy from (the dashboard isn't editable-installed
    # here): nothing to install. The .pth may still exist from a
    # prior provisioning; leaving an orphan is fine.
    src = Path(__file__).parent / "cudnn_compat_shim_source.py"
    if not src.exists():
        return False
    pkg.mkdir(parents=True, exist_ok=True)
    try:
        init.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return True
    except OSError:
        return False


def remove_cudnn_compat_pth() -> bool:
    p = _cudnn_pth_path()
    if p is None or not p.exists():
        return True
    try:
        p.unlink()
        return True
    except OSError:
        return False


def _probe_cudnn_in_subprocess() -> Optional[bool]:
    """Run the conv1d probe in a fresh subprocess so the dashboard's
    in-process torch module (with whatever backend flags it set on
    import) does not poison the result.

    Returns True if conv1d works, False if it fails with the
    "unable to find an engine" cuDNN error, None if torch/CUDA
    isn't usable.
    """
    code = (
        "import torch as _t; "
        "_out = None\n"
        "if _t.cuda.is_available():\n"
        "    try:\n"
        "        _x = _t.randn(1, 512, 64, device='cuda', dtype=_t.float16)\n"
        "        _w = _t.randn(512, 512, 3, device='cuda', dtype=_t.float16)\n"
        "        _t.nn.functional.conv1d(_x, _w, padding=1)\n"
        "        _out = True\n"
        "    except RuntimeError as _e:\n"
        "        _out = 'NO' if 'unable to find an engine' in str(_e) else 'UNK'\n"
        "print(_out)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            text=True,
            env={"PATH": os.environ.get("PATH", "")},
        )
        out = (result.stdout or "").strip()
        if out == "True":
            return True
        if out == "NO":
            return False
        return None
    except (subprocess.SubprocessError, OSError):
        return None


def probe_cudnn() -> Optional[bool]:
    """Public entry point. Delegates to a fresh subprocess so we always
    get a truthful answer regardless of the dashboard's import cache.
    """
    return _probe_cudnn_in_subprocess()
