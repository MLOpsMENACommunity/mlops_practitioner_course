"""Pre-flight check. Run it BEFORE the session, not during it.

    make doctor

Each FAIL prints the command that fixes it. WARN lines are optional stages that will
record `not_run` instead of crashing — a CPU laptop is expected to show several.
Exits non-zero only if something required for `make profile` is broken.
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys

from src import config

_results: list[tuple[bool | None, str, str]] = []


def check(ok: bool, label: str, fix: str = "") -> bool:
    _results.append((ok, label, fix))
    return ok


def warn(label: str, fix: str = "") -> None:
    _results.append((None, label, fix))


def has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def main() -> None:
    check(sys.version_info[:2] == (3, 12), f"Python {platform.python_version()} (3.12 required)", "make setup")
    for module in ("torch", "torchvision", "onnx", "onnxruntime", "torch_pruning", "psutil", "matplotlib"):
        check(has(module), f"import {module}", "make setup")
    check(not (config.ROOT / "triton").exists(), "no top-level triton/ folder shadowing the triton package",
          "rename it: torch._dynamo imports `triton` and finds the folder instead")  # fmt: skip
    for module, stage in (("openvino", "s09"), ("nncf", "s09"), ("tritonclient", "s11")):
        if not has(module):
            warn(f"{module} missing — {stage} will not run", "pip install -r requirements-openvino.txt")
    edge = config.ROOT / ".venv-edge" / "bin" / "python"
    if not edge.exists():
        warn("no .venv-edge — s10 (TFLite) will not run", "make setup-edge")

    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        check(True, f"CUDA device: {name} (compute capability {cap[0]}.{cap[1]})")
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()  # fmt: skip
        check(bool(driver) and int(driver.split(".")[0]) >= 580, f"NVIDIA driver {driver} (R580+ for CUDA 13.x)",
              "upgrade the host driver")  # fmt: skip
        if has("tensorrt"):
            import tensorrt

            check(tensorrt.__version__.startswith("10.16"), f"TensorRT {tensorrt.__version__} (10.16.x pinned)",
                  "pip install -r requirements-gpu.txt — TensorRT 11 removed the INT8 calibrator s08 teaches")  # fmt: skip
        else:
            warn("tensorrt missing — s08 will record not_run", "make setup-gpu")
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            warn(f"onnxruntime providers {providers}: no CUDA EP", "make setup-gpu (replaces onnxruntime with onnxruntime-gpu)")
    else:
        warn("no CUDA device — GPU rows (s00 gpu-decode, s04 CUDA/TensorRT EPs, s08) will record not_run")

    if shutil.which("docker") and subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
        from src.stages.s11_triton import IMAGE

        present = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0
        if not present:
            warn(f"{IMAGE} not pulled — s11 will record not_run", f"docker pull {IMAGE}")
    else:
        warn("Docker not available — s11 (Triton) will record not_run")

    data = config.data_dir() / "manifest.json"
    check(data.exists(), f"dataset for profile {config.profile().name}", "make data")
    for name in ("detector_baseline", "ocr_baseline"):
        if not (config.artifacts_dir() / f"{name}.pt").exists():
            warn(f"checkpoint {name} missing", "make train")
    print(f"\n  threads: ANPR_THREADS={config.THREADS} (every runtime is pinned to this)\n")

    failed = 0
    for ok, label, fix in _results:
        print(f"    {'WARN' if ok is None else 'PASS' if ok else 'FAIL'}  {label}")
        if not ok and fix:
            print(f"          -> {fix}")
        failed += ok is False
    print(f"\n  {sum(1 for ok, _, _ in _results if ok)} passed, {failed} failed, {sum(1 for ok, _, _ in _results if ok is None)} warnings\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
