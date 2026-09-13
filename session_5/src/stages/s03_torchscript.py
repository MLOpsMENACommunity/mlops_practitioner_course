"""Stage 3 — TorchScript: an export and serving-compatibility format, not an optimizer.

torch.jit is deprecated in current PyTorch; torch.compile (s02) is the speed path and
torch.export / ONNX (s04) is the portable one. TorchScript still matters because you
will meet it: Triton's PyTorch backend and many existing services load .ts files, and
it is the only way to run a model from C++ LibTorch with no Python at all.

    make s03
"""

from __future__ import annotations

import warnings

import torch

from src import config
from src.backends import TorchBackend, TorchScriptBackend
from src.benchmark import RunSpec, run
from src.models import io
from src.models.detector import DetectorExport
from src.stages import common


# --- snippet:jit-trace ---
def trace(model: torch.nn.Module, example: torch.Tensor, name: str) -> list[str]:
    """Trace and save. Returns the warnings PyTorch raised, so the deprecation is shown, not claimed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        traced = torch.jit.trace(model.eval(), example)  # records ONE execution path: no data-dependent control flow
        torch.jit.save(traced, str(config.artifacts_dir() / name))
    return sorted({f"{w.category.__name__}: {str(w.message).splitlines()[0]}" for w in caught})


# --- end-snippet ---


def main() -> None:
    common.banner("s03 TorchScript")
    common.require(common.DET_BASE, common.OCR_BASE)
    dev = common.device()
    det = DetectorExport(io.load(common.DET_BASE, dev))
    seen = trace(det, torch.rand(2, 3, config.DET_H, config.DET_W, device=dev), "detector_baseline.ts")
    seen += trace(io.load(common.OCR_BASE, dev), torch.rand(3, 1, config.OCR_H, config.OCR_W, device=dev), "ocr_baseline.ts")
    for message in sorted(set(seen)):
        print(f"  warning raised by torch.jit: {message}")

    spec = RunSpec("s03_torchscript", "jit-trace-fp32", common.BASELINE_ROW, "torchscript",
                   {"detector": "detector_baseline.ts", "ocr": "ocr_baseline.ts", "device": dev},
                   branch="export", device=dev, notes="; ".join(sorted(set(seen)))[:500])  # fmt: skip
    ref = TorchBackend(common.DET_BASE, common.OCR_BASE, device=dev)
    common.gate(spec, ref, TorchScriptBackend(**spec.options))
    run(spec)


if __name__ == "__main__":
    main()
