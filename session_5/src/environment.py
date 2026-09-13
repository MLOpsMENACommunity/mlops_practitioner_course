"""What a number was measured on. Without this block a latency is unreproducible.

Thread counts are recorded next to the hardware because they move CPU results as
much as the hardware does: the same model on the same laptop can differ several-fold
between OMP_NUM_THREADS=1 and the library default.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess

import psutil

from src import config

LIBRARIES = ("torch", "torchvision", "onnx", "onnxruntime", "onnxruntime-gpu", "openvino", "nncf",
             "tensorrt", "tensorrt-cu13", "ai-edge-litert", "onnx2tf", "tritonclient", "numpy", "pillow")  # fmt: skip


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _gpu() -> dict:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {}
    if not out:
        return {}
    name, driver, mem = (s.strip() for s in out[0].split(","))
    return {"gpu": name, "driver": driver, "gpu_memory": mem}


def _versions() -> dict[str, str]:
    found = {}
    for lib in LIBRARIES:
        try:
            found[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def capture() -> dict:
    """Hardware, library versions and every thread knob that was in effect."""
    env = {
        "cpu": _cpu_model(),
        "cpu_cores_logical": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
        "threads": {
            "ANPR_THREADS": config.THREADS,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
            "ort_intra_op_num_threads": config.THREADS,
            "ort_inter_op_num_threads": 1,
        },
        "libraries": _versions(),
        # Load average when the row was measured. A latency taken while something else
        # was compiling, pulling an image or training is not that runtime's latency.
        "load_avg_1m": round(psutil.getloadavg()[0], 2),
        **_gpu(),
    }
    try:
        import torch

        if torch.cuda.is_available():
            env["cuda_runtime"] = torch.version.cuda
        env["threads"]["torch_num_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return env


def hardware_id(env: dict) -> str:
    """Short, stable key for 'the same machine at the same thread count'."""
    key = f"{env['cpu']}|{env.get('gpu', 'no-gpu')}|threads={env['threads']['ANPR_THREADS']}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
