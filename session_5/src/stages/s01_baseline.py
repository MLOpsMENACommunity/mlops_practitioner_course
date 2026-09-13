"""Stage 1 — the baseline every other row is compared against.

Eager PyTorch, FP32: ResNet-34 detector + BiLSTM CRNN, NMS in Python, Pillow
bilinear resize. This is the "research prototype that works on the server" the
session starts from.

    make s01
"""

from __future__ import annotations

from src.benchmark import RunSpec, run
from src.stages import common


def spec(device: str | None = None) -> RunSpec:
    dev = device or common.device()
    variant = "eager-fp32" if dev == common.device() else f"eager-fp32-{dev}"
    return RunSpec("s01_baseline", variant, parent=None, backend="torch", device=dev,
                   options={"detector": common.DET_BASE, "ocr": common.OCR_BASE, "device": dev})  # fmt: skip


def main() -> None:
    common.banner("s01 baseline")
    common.require(common.DET_BASE, common.OCR_BASE)
    run(spec())
    if common.has_cuda():
        run(spec("cpu"))  # the same pipeline on the CPU of the GPU box: the no-accelerator reference


if __name__ == "__main__":
    main()
