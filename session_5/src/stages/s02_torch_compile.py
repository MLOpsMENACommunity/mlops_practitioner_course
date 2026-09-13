"""Stage 2 — torch.compile: the first thing to try, because it is one line.

TorchDynamo captures the Python forward as a graph; Inductor fuses the operators
into generated kernels (C++/OpenMP on CPU, Triton on CUDA). No export, no new file
format, same checkpoint. What it costs is measured, not hidden: the first call
compiles (see the row's `process_first_call_ms`), and a new input shape can compile again.

    make s02
"""

from __future__ import annotations

from src.backends import TorchBackend
from src.benchmark import RunSpec, not_run, run
from src.stages import common


def main() -> None:
    common.banner("s02 torch.compile")
    common.require(common.DET_BASE, common.OCR_BASE)
    dev = common.device()
    opts = {"detector": common.DET_BASE, "ocr": common.OCR_BASE, "device": dev}
    spec = RunSpec("s02_torch_compile", "inductor-fp32", common.BASELINE_ROW, "torch",
                   opts | {"compile": True}, device=dev)  # fmt: skip
    try:
        # --- snippet:torch-compile ---
        compiled = TorchBackend(**opts, compile=True)  # torch.compile(model) on both models, inside
        # --- end-snippet ---
        common.gate(spec, TorchBackend(**opts), compiled)  # compiling must not change a single plate
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — no C++ compiler, unsupported platform: skip, do not crash
        not_run(spec, f"torch.compile failed here: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}")
        return
    run(spec)


if __name__ == "__main__":
    main()
