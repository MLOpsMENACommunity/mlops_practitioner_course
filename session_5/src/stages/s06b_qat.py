"""Stage 6b — quantization-aware training (QAT) on the recognizer.

The s06a rows show where INT8 hurts; for this pipeline that is the question QAT
answers. FakeQuantize modules simulate INT8 rounding and clamping during fine-tuning,
so the weights learn values that survive it. It costs a training run — PTQ costs a
calibration pass — which is why it only runs on the model that needs it.

Three traps, each hit while building this stage and fixed below:
  1. the default QAT qconfig uses FusedMovingAvgObsFakeQuantize, and the ONNX exporter
     has no translation for `aten::fused_moving_avg_obs_fake_quant` -> non-fused FakeQuantize;
  2. the dynamo exporter cannot trace FakeQuantize's `if self.observer_enabled[0] == 1`
     (a data-dependent branch) -> this one export uses the TorchScript-based exporter;
  3. FX graph mode has no QAT module for nn.LSTM -> the LSTM stays FP32 (mixed precision).

Rows:
  ort-qat-int8-ocr        s06a's stratified INT8 detector + the QAT recognizer exported as Q/DQ ONNX
  torch-qat-converted-ocr convert_fx -> real PyTorch INT8 kernels on CPU (x86 or qnnpack)

    make s06b
"""

from __future__ import annotations

import os

import torch
from torch import nn
from torch.ao.quantization import (
    FakeQuantize,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    QConfig,
    QConfigMapping,
    disable_observer,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_qat_fx

from src import config
from src.backends import OrtBackend, TorchScriptBackend, select_quantized_engine
from src.benchmark import RunSpec, run
from src.models import io
from src.stages import common
from src.train import fit_ocr, quick_ocr_accuracy

# --- snippet:qat-qconfig ---
QAT_QCONFIG = QConfig(
    # Non-fused FakeQuantize: the fused default trains the same but cannot be exported to ONNX.
    activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, dtype=torch.quint8),
    weight=FakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver, quant_min=-128, quant_max=127,
                                  dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0),
)  # fmt: skip


def qat_finetune(model: nn.Module, epochs: int) -> torch.fx.GraphModule:
    """Insert fake quantization, fine-tune through it, then freeze the learned scales."""
    select_quantized_engine()
    mapping = QConfigMapping().set_global(QAT_QCONFIG).set_object_type(nn.LSTM, None)  # no QAT LSTM in FX mode
    prepared = prepare_qat_fx(model.train(), mapping, (torch.rand(2, 1, config.OCR_H, config.OCR_W),))
    fit_ocr(prepared, epochs, lr=2e-4, tag="qat")  # a low LR: nudge the weights, do not retrain them
    prepared.cpu().eval()
    prepared.apply(disable_observer)  # scales stop moving; export sees the trained ones
    return prepared


# --- end-snippet ---


def export_qdq(prepared: torch.fx.GraphModule, name: str) -> None:
    # --- snippet:qat-export ---
    torch.onnx.export(
        prepared, (torch.rand(2, 1, config.OCR_H, config.OCR_W),), str(config.artifacts_dir() / name),
        input_names=["crops"], output_names=["logits"], opset_version=config.ONNX_OPSET,
        dynamo=False,  # dynamo cannot trace FakeQuantize's observer_enabled branch
        dynamic_axes={"crops": {0: "batch"}, "logits": {0: "batch"}},  # the TorchScript exporter's spelling
    )  # fmt: skip
    # --- end-snippet ---


def main() -> None:
    common.banner("s06b quantization-aware training")
    common.require(common.OCR_BASE)
    common.require_files("detector_int8_stratified.onnx", "ocr_int8_stratified.onnx", fix="make s06a")
    common.require_files("detector_baseline.ts", "ocr_baseline.ts", fix="make s03")
    if not common.has_cuda():
        os.environ["ANPR_TRAIN_DEVICE"] = "cpu"  # MPS has no fake-quantize kernels
    prepared = qat_finetune(io.load(common.OCR_BASE), 3 * config.profile().finetune_epochs)
    print(f"  QAT recognizer, fake-quant exact match on GT crops: {quick_ocr_accuracy(prepared):.3f}")
    export_qdq(prepared, "ocr_qat_int8.onnx")

    ptq_detector = "detector_int8_stratified.onnx"
    spec = RunSpec("s06b_qat", "ort-qat-int8-ocr", "s06a_ptq:ort-static-int8-stratified", "ort",
                   {"detector": ptq_detector, "ocr": "ocr_qat_int8.onnx"}, branch="quantization", precision="int8-qat",
                   notes="LSTM kept FP32 (no FX QAT module); convs + fc fake-quantized during fine-tuning")  # fmt: skip
    common.gate(spec, OrtBackend(ptq_detector, "ocr_int8_stratified.onnx"), OrtBackend(**spec.options), strict=False)
    run(spec)

    # --- snippet:qat-convert-deploy ---
    converted = convert_fx(prepared)  # FakeQuantize -> real quantized kernels (CPU only)
    # torch.save(converted) fails: pickling an FX GraphModule of quantized modules raises
    # "AttributeError: 'str' object has no attribute '__name__'". TorchScript is how it ships.
    traced = torch.jit.trace(converted, torch.rand(3, 1, config.OCR_H, config.OCR_W))
    torch.jit.save(traced, str(config.artifacts_dir() / "ocr_qat_converted.ts"))
    # --- end-snippet ---
    spec = RunSpec("s06b_qat", "torch-qat-converted-ocr", "s03_torchscript:jit-trace-fp32", "torchscript",
                   {"detector": "detector_baseline.ts", "ocr": "ocr_qat_converted.ts"}, branch="quantization",
                   precision="int8-qat", notes="convert_fx -> torch.jit.trace; LSTM stays FP32")  # fmt: skip
    common.gate(spec, TorchScriptBackend("detector_baseline.ts", "ocr_baseline.ts"), TorchScriptBackend(**spec.options), strict=False)
    run(spec)


if __name__ == "__main__":
    main()
