"""Stage 6a — post-training quantization: dynamic, static, and the calibration set.

Rows (branches off the ONNX Runtime CPU FP32 row unless noted):
  torch-dynamic-int8-ocr        torch.ao.quantization.quantize_dynamic on the recognizer's LSTM + Linear
  ort-dynamic-int8              weights INT8, activation scales computed per call at runtime
  ort-static-int8-stratified    QDQ, per-channel weights, calibrated on all five conditions
  ort-static-int8-daytime       identical, calibrated on daytime frames only — the worked failure
  ort-static-int8-per-tensor    one scale per weight tensor instead of one per channel
Mixed precision lives in `make int8-debug` (src/quant_debug.py): it needs the sensitivity scan first.

    make s06a
"""

from __future__ import annotations

from pathlib import Path

import torch
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_dynamic, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn

from src import config
from src.backends import OrtBackend, TorchBackend, select_quantized_engine
from src.benchmark import RunSpec, run
from src.datasets import calibration
from src.models import io
from src.stages import common

ORT_PARENT = "s04_onnx_export:ort-cpu-fp32"


def art(name: str) -> Path:
    return config.artifacts_dir() / name


# --- snippet:ort-static-quant ---
def static_int8(src: str, dst: str, input_name: str, batches: list, per_channel: bool = True, **kw) -> None:
    """QDQ static quantization. Signature verified against onnxruntime 1.30's quantize_static."""
    pre = art(src.replace(".onnx", "_pre.onnx"))
    # Graph cleanup + ONNX shape inference. Symbolic shape inference is skipped: on these
    # dynamo-exported graphs it raises "Exception: Incomplete symbolic shape inference".
    # Node names and their module-scope metadata survive either way (quant_debug relies on that).
    quant_pre_process(str(art(src)), str(pre), skip_symbolic_shape=True)
    quantize_static(
        model_input=str(pre),
        model_output=str(art(dst)),
        calibration_data_reader=calibration.OrtReader(input_name, batches),
        quant_format=QuantFormat.QDQ,  # QuantizeLinear/DequantizeLinear pairs: what TensorRT and OpenVINO read too
        per_channel=per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,  # min/max: one outlier frame sets the whole range
        **kw,
    )


# --- end-snippet ---


def torch_dynamic() -> None:
    """PyTorch-native dynamic quantization — it only touches LSTM and Linear, so only the recognizer moves."""
    select_quantized_engine()
    # --- snippet:torch-dynamic-quant ---
    ocr = torch.ao.quantization.quantize_dynamic(io.load(common.OCR_BASE), {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    # --- end-snippet ---
    io.save(ocr, "ocr_dynamic_int8", "ocr", "crnn", whole=True)
    spec = RunSpec("s06a_ptq", "torch-dynamic-int8-ocr", common.BASELINE_ROW, "torch",
                   {"detector": common.DET_BASE, "ocr": "ocr_dynamic_int8"}, branch="quantization", precision="int8-dyn",
                   notes="conv layers untouched: dynamic quantization has no activation ranges to give a conv")  # fmt: skip
    common.gate(spec, TorchBackend(common.DET_BASE, common.OCR_BASE), TorchBackend(**spec.options), strict=False)
    run(spec)


def ort_rows() -> None:
    quantize_dynamic(str(art("detector_baseline.onnx")), str(art("detector_dyn.onnx")), weight_type=QuantType.QInt8)
    quantize_dynamic(str(art("ocr_baseline.onnx")), str(art("ocr_dyn.onnx")), weight_type=QuantType.QInt8)
    variants = {"ort-dynamic-int8": ("detector_dyn.onnx", "ocr_dyn.onnx", "int8-dyn", "weights only; activations scaled per call")}
    for strategy in calibration.STRATEGIES:
        det_cal = calibration.detector_batches(calibration.frames(strategy))
        static_int8("detector_baseline.onnx", f"detector_int8_{strategy}.onnx", "images", det_cal)
        static_int8("ocr_baseline.onnx", f"ocr_int8_{strategy}.onnx", "crops", calibration.ocr_batches(strategy))
        variants[f"ort-static-int8-{strategy}"] = (f"detector_int8_{strategy}.onnx", f"ocr_int8_{strategy}.onnx", "int8",
                                                   f"calibration: {strategy}, {len(det_cal)} frames, per-channel")  # fmt: skip
    det_cal = calibration.detector_batches(calibration.frames("stratified"))
    static_int8("detector_baseline.onnx", "detector_int8_pertensor.onnx", "images", det_cal, per_channel=False)
    static_int8("ocr_baseline.onnx", "ocr_int8_pertensor.onnx", "crops", calibration.ocr_batches("stratified"), per_channel=False)
    variants["ort-static-int8-per-tensor"] = ("detector_int8_pertensor.onnx", "ocr_int8_pertensor.onnx", "int8",
                                              "calibration: stratified, per-tensor")  # fmt: skip

    ref = OrtBackend("detector_baseline.onnx", "ocr_baseline.onnx")
    for variant, (det, ocr, precision, notes) in variants.items():
        spec = RunSpec("s06a_ptq", variant, ORT_PARENT, "ort", {"detector": det, "ocr": ocr},
                       branch="quantization", precision=precision, notes=notes)  # fmt: skip
        common.gate(spec, ref, OrtBackend(det, ocr), strict=False)
        run(spec)


def main() -> None:
    common.banner("s06a post-training quantization")
    common.require(common.DET_BASE, common.OCR_BASE)
    common.require_files("detector_baseline.onnx", "ocr_baseline.onnx", fix="make s04")
    torch_dynamic()
    ort_rows()


if __name__ == "__main__":
    main()
