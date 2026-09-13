"""Stage 4 — ONNX: a file FORMAT, executed by a RUNTIME, on a BACKEND.

Exports both models once, checks the graphs, gates them against PyTorch, then runs
the same .onnx files through every ONNX Runtime execution provider this machine has:
CPU always; CUDA and TensorRT on an NVIDIA box with onnxruntime-gpu; OpenVINO with
onnxruntime-openvino. Providers that are not installed are recorded as not_run.

    make s04
"""

from __future__ import annotations

from pathlib import Path

import onnx
import torch

from src import config
from src.backends import OrtBackend, TorchBackend
from src.benchmark import RunSpec, not_run, run
from src.models import io
from src.models.detector import DetectorExport
from src.stages import common

DET_ONNX, DET_NMS_ONNX, OCR_ONNX = "detector_baseline.onnx", "detector_baseline_nms.onnx", "ocr_baseline.onnx"


# --- snippet:onnx-export ---
def export(model: torch.nn.Module, example: torch.Tensor, path: Path, inp: str, outs: list[str]) -> None:
    """Dynamo-based export with a symbolic batch dimension, then the ONNX checker."""
    batch = torch.export.Dim("batch", min=1, max=64)
    torch.onnx.export(
        model.eval(), (example,), str(path),
        input_names=[inp], output_names=outs,
        dynamic_shapes={"x": {0: batch}},  # the successor of dynamic_axes; omit it and batch is frozen at 2
        opset_version=config.ONNX_OPSET, dynamo=True, external_data=False,
    )  # fmt: skip
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)  # structure + shape inference


# --- end-snippet ---


def export_all(name: str = common.DET_BASE, ocr: str = common.OCR_BASE, suffix: str = "baseline") -> None:
    """Shared with later stages: pruned, quantized-aware and distilled models export the same way."""
    out = config.artifacts_dir()
    det = io.load(name)
    # Example batch of 2, not 1: torch.export specializes a dimension whose example
    # value is 1, and the "dynamic" batch silently becomes a constant.
    x = torch.rand(2, 3, config.DET_H, config.DET_W)
    export(DetectorExport(det), x, out / f"detector_{suffix}.onnx", "images", ["scores", "boxes"])
    export(DetectorExport(det, with_nms=True), x, out / f"detector_{suffix}_nms.onnx", "images", ["detections"])
    export(io.load(ocr), torch.rand(2, 1, config.OCR_H, config.OCR_W), out / f"ocr_{suffix}.onnx", "crops", ["logits"])


def provider_rows() -> list[tuple[RunSpec, str | None]]:
    """(spec, reason-if-unavailable) for every execution provider worth measuring."""
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    base = {"detector": DET_ONNX, "ocr": OCR_ONNX}
    cache = str(config.artifacts_dir() / "trt_ep_cache")
    rows = [
        (RunSpec("s04_onnx_export", "ort-cpu-fp32", common.BASELINE_ROW, "ort", base, branch="export"), None),
        (RunSpec("s04_onnx_export", "ort-cpu-fp32-nms-in-graph", "s04_onnx_export:ort-cpu-fp32", "ort",
                 base | {"detector": DET_NMS_ONNX, "nms_in_graph": True}, branch="export"), None),
        (RunSpec("s04_onnx_export", "ort-cuda-fp32", common.BASELINE_ROW, "ort",
                 base | {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"], "device": "cuda"},
                 branch="export", device="cuda"),
         None if "CUDAExecutionProvider" in have else "CUDAExecutionProvider not available (needs onnxruntime-gpu + NVIDIA GPU)"),
        # --- snippet:trt-execution-provider ---
        (RunSpec("s04_onnx_export", "ort-tensorrt-ep-fp16", common.BASELINE_ROW, "ort",
                 base | {"device": "cuda", "providers": [
                     ["TensorrtExecutionProvider", {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                                                    "trt_engine_cache_path": cache}],
                     "CUDAExecutionProvider", "CPUExecutionProvider"]},
                 branch="export", device="cuda", precision="fp16"),
         None if "TensorrtExecutionProvider" in have else "TensorrtExecutionProvider not available (onnxruntime-gpu + TensorRT 10 libs)"),
        # --- end-snippet ---
        (RunSpec("s04_onnx_export", "ort-openvino-ep-cpu", common.BASELINE_ROW, "ort",
                 base | {"providers": [["OpenVINOExecutionProvider", {"device_type": "CPU"}]]}, branch="export"),
         None if "OpenVINOExecutionProvider" in have else "OpenVINOExecutionProvider not available (install onnxruntime-openvino in its own venv; it replaces onnxruntime)"),
    ]  # fmt: skip
    return rows


def main() -> None:
    common.banner("s04 ONNX + execution providers")
    common.require(common.DET_BASE, common.OCR_BASE)
    export_all()
    for spec, reason in provider_rows():
        if reason:
            not_run(spec, reason)
            continue
        nms = spec.options.get("nms_in_graph", False)
        ref = TorchBackend(common.DET_BASE, common.OCR_BASE, nms_in_graph=nms)
        common.gate(spec, ref, OrtBackend(**spec.options), strict=spec.precision == "fp32")
        run(spec)


if __name__ == "__main__":
    main()
