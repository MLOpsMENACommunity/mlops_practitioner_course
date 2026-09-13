"""Stage 5 — pruning. Masks make nothing smaller or faster. Slicing does.

Rows, all branches off the baseline detector (the recognizer is unchanged):
  masked-structured-50   prune.ln_structured on 50% of filters -> fine-tune WITH masks -> prune.remove
                         Same tensor shapes, so the same file size and the same dense kernel work.
  sliced-oneshot-50      torch-pruning physically removes 50% of channels at once, then fine-tunes
  sliced-iterative-50    the same 50%, in three prune -> fine-tune rounds, same total epochs
  sliced-student-budget  sliced until the detector is as small as the s07 student, for the
                         "distillation vs pruning at the same size" comparison
Each torch row is also exported to ONNX so the file-size claim is measured on the real artifact.

    make s05
"""

from __future__ import annotations

import copy

import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp
from torch import nn

from src import config
from src.backends import TorchBackend
from src.benchmark import RunSpec, run
from src.models import io
from src.models.detector import OUTPUT_LAYERS, DetectorExport, PlateDetector
from src.stages import common
from src.stages.s04_onnx_export import export
from src.train import fit_detector


# --- snippet:never-prune-output-layers ---
def prunable_convs(model: PlateDetector) -> list[nn.Conv2d]:
    """Every dense conv EXCEPT the output layers.

    A naive `isinstance(m, nn.Conv2d)` loop also hits `obj` and `box`: pruning filters
    there deletes output channels — box coordinates — not redundant features.
    """
    outputs = {getattr(model, name) for name in OUTPUT_LAYERS}
    return [m for m in model.modules() if isinstance(m, nn.Conv2d) and m not in outputs and m.groups == 1]


# --- end-snippet ---


def sparsity(model: nn.Module) -> dict:
    """Real accounting: parameters that exist, and how many of them are exactly zero."""
    total = sum(p.numel() for p in model.parameters())
    zeros = sum(int((p == 0).sum()) for p in model.parameters())
    return {"params": total, "zero_params": zeros, "zero_fraction": round(zeros / total, 4)}


# --- snippet:pruning-mask-lifecycle ---
def prune_and_finetune(model: PlateDetector, amount: float, epochs: int, remove_first: bool = False) -> PlateDetector:
    """The correct order: prune -> fine-tune WITH masks attached -> prune.remove -> export."""
    convs = prunable_convs(model)
    for conv in convs:
        prune.ln_structured(conv, "weight", amount=amount, n=1, dim=0)  # 1. mask whole filters by L1 norm
    if remove_first:  # the silent failure: the mask is gone, so Adam refills the zeros
        for conv in convs:
            prune.remove(conv, "weight")
    fit_detector(model, epochs, lr=5e-4, tag="masked-finetune")  # 2. weight = weight_orig * mask, every step
    if not remove_first:
        for conv in convs:
            prune.remove(conv, "weight")  # 3. bake the zeros into a plain weight tensor
    return model  # 4. only now is it safe to export


# --- end-snippet ---


# --- snippet:physical-slicing ---
def slice_channels(model: PlateDetector, ratio: float, steps: int, epochs_per_step: int) -> PlateDetector:
    """Remove channels for real: weight tensors shrink, and every dependent layer shrinks with them."""
    example = torch.rand(1, 3, config.DET_H, config.DET_W)
    pruner = tp.pruner.MagnitudePruner(
        model, example, importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=ratio, iterative_steps=steps,
        ignored_layers=[model.obj, model.box],  # same rule as above: never the output layers
        round_to=8,  # channel counts in multiples of 8 are what dense kernels are tuned for
    )  # fmt: skip
    for step in range(steps):
        pruner.step()  # DepGraph: prune a residual block's conv and its coupled layers follow
        print(f"  slice step {step + 1}/{steps}: {sparsity(model)['params']:,} params", flush=True)
        fit_detector(model.cpu(), epochs_per_step, lr=5e-4, tag=f"slice-{step + 1}")
        model.cpu()
    return model


# --- end-snippet ---


def ratio_for_budget(budget: int) -> float:
    """Smallest channel ratio whose sliced detector fits the parameter budget (no training)."""
    lo, hi = 0.0, 0.95
    for _ in range(8):
        mid = (lo + hi) / 2
        trial = slice_channels_dry(copy.deepcopy(io.load(common.DET_BASE)), mid)
        lo, hi = (mid, hi) if sparsity(trial)["params"] > budget else (lo, mid)
    return round(hi, 3)


def slice_channels_dry(model: PlateDetector, ratio: float) -> PlateDetector:
    example = torch.rand(1, 3, config.DET_H, config.DET_W)
    tp.pruner.MagnitudePruner(model, example, importance=tp.importance.MagnitudeImportance(p=1), pruning_ratio=ratio,
                              ignored_layers=[model.obj, model.box], round_to=8).step()  # fmt: skip
    return model


def measure(name: str, variant: str, whole: bool, model: PlateDetector, notes: str) -> None:
    # Fine-tuning leaves the model on the training device (MPS/CUDA). torch.export with a CPU
    # example then fails: "Tensor on device mps:0 is not on the expected device cpu!"
    model = model.cpu().eval()
    io.save(model, name, "detector", model.backbone_name, whole=whole, **sparsity(model))
    export(DetectorExport(model), torch.rand(2, 3, config.DET_H, config.DET_W),
           config.artifacts_dir() / f"{name}.onnx", "images", ["scores", "boxes"])  # fmt: skip
    torch_spec = RunSpec("s05_pruning", variant, common.BASELINE_ROW, "torch",
                         {"detector": name, "ocr": common.OCR_BASE, "device": common.device()},
                         branch="pruning", device=common.device(), notes=notes)  # fmt: skip
    common.gate(torch_spec, TorchBackend(common.DET_BASE, common.OCR_BASE), TorchBackend(name, common.OCR_BASE), strict=False)
    run(torch_spec)
    measure_onnx(name, variant, notes)


def measure_onnx(name: str, variant: str, notes: str) -> None:
    """The pruned model's ONNX export, gated STRICTLY against the pruned PyTorch model it came from."""
    from src.backends import OrtBackend

    ort_spec = RunSpec("s05_pruning", f"{variant}-onnx", f"s05_pruning:{variant}", "ort",
                       {"detector": f"{name}.onnx", "ocr": "ocr_baseline.onnx"}, branch="pruning", notes=notes)  # fmt: skip
    common.gate(ort_spec, TorchBackend(name, common.OCR_BASE), OrtBackend(**ort_spec.options))
    run(ort_spec)


PRUNED = {"detector_masked50": "masked-structured-50", "detector_sliced50_oneshot": "sliced-oneshot-50",
          "detector_sliced50_iterative": "sliced-iterative-50", "detector_sliced_budget": "sliced-student-budget"}  # fmt: skip


def main() -> None:
    import sys

    common.banner("s05 pruning")
    common.require(common.DET_BASE, common.OCR_BASE)
    common.require_files("ocr_baseline.onnx", fix="make s04")
    if "--onnx-only" in sys.argv:  # re-gate and re-measure the exported rows from saved checkpoints, no training
        for name, variant in PRUNED.items():
            common.require(name)
            measure_onnx(name, variant, str(io.meta(name)))
        return
    prof = config.profile()
    base_params = sparsity(io.load(common.DET_BASE))["params"]

    wrong = prune_and_finetune(io.load(common.DET_BASE), 0.5, prof.finetune_epochs, remove_first=True)
    right = prune_and_finetune(io.load(common.DET_BASE), 0.5, prof.finetune_epochs)
    order = (f"after fine-tuning, zero fraction is {sparsity(wrong)['zero_fraction']:.1%} when prune.remove ran FIRST "
             f"vs {sparsity(right)['zero_fraction']:.1%} when masks stayed attached")  # fmt: skip
    print(f"  mask lifecycle: {order}")
    measure("detector_masked50", "masked-structured-50", False, right, f"{sparsity(right)} of {base_params:,}; {order}")

    budget_epochs = 3 * prof.finetune_epochs  # identical training budget for one-shot and iterative
    oneshot = slice_channels(io.load(common.DET_BASE), 0.5, 1, budget_epochs)
    measure("detector_sliced50_oneshot", "sliced-oneshot-50", True, oneshot, str(sparsity(oneshot)))
    iterative = slice_channels(io.load(common.DET_BASE), 0.5, 3, prof.finetune_epochs)
    measure("detector_sliced50_iterative", "sliced-iterative-50", True, iterative, str(sparsity(iterative)))

    student_params = sparsity(PlateDetector("mobilenet_v3_small"))["params"]
    ratio = ratio_for_budget(student_params)
    budget = slice_channels(io.load(common.DET_BASE), ratio, 4, prof.finetune_epochs)
    measure("detector_sliced_budget", "sliced-student-budget", True, budget,
            f"ratio {ratio} to reach the student's {student_params:,} params: {sparsity(budget)}")  # fmt: skip


if __name__ == "__main__":
    main()
