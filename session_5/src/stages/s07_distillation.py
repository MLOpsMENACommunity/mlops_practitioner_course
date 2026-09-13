"""Stage 7 — knowledge distillation: a small DENSE student, taught by the baseline.

Rows (branches off the baseline):
  student-scratch     MobileNetV3-Small detector + conv-only recognizer, trained on labels alone
  student-distilled   the same architectures and epochs, plus the teacher's signals
  student-distilled-onnx   the distilled pair exported — the artifacts s08-s11 deploy
Read student-distilled against s05's sliced-student-budget row: the same parameter budget,
one network dense and small by design, the other a large one cut down to size.

    make s07
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src import config
from src.backends import OrtBackend, TorchBackend
from src.benchmark import RunSpec, run
from src.models import io
from src.stages import common
from src.stages.s04_onnx_export import export_all
from src.train import fit_detector, fit_ocr, quick_map, quick_ocr_accuracy

T = 4.0  # softening temperature for the losses that have a softmax (or sigmoid) under them


# --- snippet:ocr-kd-loss ---
def ocr_kd_loss(student_logits: Tensor, teacher_logits: Tensor, t: float = T) -> Tensor:
    """KL between softened per-time-step character distributions — a real classification.

    The T*T factor restores the gradient scale that dividing logits by T shrinks
    (Hinton et al., https://arxiv.org/abs/1503.02531).
    """
    c = student_logits.shape[-1]
    log_p = F.log_softmax(student_logits.reshape(-1, c) / t, dim=-1)
    q = F.softmax(teacher_logits.reshape(-1, c) / t, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean") * t * t


# --- end-snippet ---


# --- snippet:detector-kd-loss ---
class DetectorKD:
    """Three teacher signals for the detector, and only one of them has a temperature.

    features   hint loss (FitNets, https://arxiv.org/abs/1412.6550): a 1x1 adapter maps the
               student's stride-8 map into the teacher's, and MSE pulls them together
    objectness sigmoid is a two-class softmax, so softening with T is meaningful here
    boxes      plain MSE on the raw box maps, weighted by teacher confidence. NO temperature:
               mse(s/T, t/T) * T**2 == mse(s, t) exactly, so a T there is a no-op with a comment
    """

    def __init__(self, teacher: nn.Module, student: nn.Module) -> None:
        self.teacher = teacher.eval()
        # Attached to the student so fit_detector's optimizer trains it; removed before export.
        student.kd_adapter = nn.Conv2d(student.neck[0].out_channels, teacher.neck[0].out_channels, 1)

    def __call__(self, x: Tensor, feats: Tensor, obj: Tensor, box: Tensor, student: nn.Module) -> Tensor:
        with torch.no_grad():
            t_feats, t_obj, t_box = self.teacher.to(x.device).forward_with_features(x)
        hint = F.mse_loss(student.kd_adapter(feats), t_feats)
        soft_obj = F.binary_cross_entropy_with_logits(obj / T, torch.sigmoid(t_obj / T)) * T * T
        weight = torch.sigmoid(t_obj)  # only where the teacher sees a plate do its boxes carry information
        box_match = (weight * (box - t_box).pow(2)).sum() / weight.sum().clamp(min=1.0)
        return hint + soft_obj + 0.1 * box_match


# --- end-snippet ---


def train_students(distill: bool) -> tuple[str, str]:
    prof, tag = config.profile(), "distilled" if distill else "scratch"
    det_name, ocr_name = f"detector_student_{tag}", f"ocr_student_{tag}"
    torch.manual_seed(config.SEED)
    det = io.build("detector", "mobilenet_v3_small")
    extra = None
    if distill:
        kd = DetectorKD(io.load(common.DET_BASE), det)
        extra = lambda x, f, o, b: kd(x, f, o, b, det)  # noqa: E731
    fit_detector(det, prof.det_epochs, extra_loss=extra, tag=det_name)
    if hasattr(det, "kd_adapter"):
        del det.kd_adapter  # training-only: the deployed student has no idea it was taught
    io.save(det, det_name, "detector", "mobilenet_v3_small", val_map50_quick=round(quick_map(det), 4))

    torch.manual_seed(config.SEED)
    ocr = io.build("ocr", "conv_ctc")
    extra_ocr = None
    if distill:
        teacher = io.load(common.OCR_BASE)

        def extra_ocr(x: Tensor, logits: Tensor) -> Tensor:
            with torch.no_grad():
                t_logits = teacher.to(x.device)(x)
            return ocr_kd_loss(logits, t_logits)

    fit_ocr(ocr, prof.ocr_epochs, extra_loss=extra_ocr, tag=ocr_name)
    io.save(ocr, ocr_name, "ocr", "conv_ctc", val_gt_crop_exact_match=round(quick_ocr_accuracy(ocr), 4))
    return det_name, ocr_name


def main() -> None:
    common.banner("s07 knowledge distillation")
    common.require(common.DET_BASE, common.OCR_BASE)
    dev = common.device()
    for distill in (False, True):
        det, ocr = train_students(distill)
        spec = RunSpec("s07_distillation", f"student-{'distilled' if distill else 'scratch'}", common.BASELINE_ROW,
                       "torch", {"detector": det, "ocr": ocr, "device": dev}, branch="distillation", device=dev)  # fmt: skip
        common.gate(spec, TorchBackend(common.DET_BASE, common.OCR_BASE, device=dev), TorchBackend(**spec.options), strict=False)
        run(spec)

    export_all("detector_student_distilled", "ocr_student_distilled", suffix="student")
    spec = RunSpec("s07_distillation", "student-distilled-onnx", "s07_distillation:student-distilled", "ort",
                   {"detector": "detector_student.onnx", "ocr": "ocr_student.onnx"}, branch="distillation")  # fmt: skip
    ref = TorchBackend("detector_student_distilled", "ocr_student_distilled")
    common.gate(spec, ref, OrtBackend(**spec.options))  # strict: an export must not change the student
    run(spec)


if __name__ == "__main__":
    main()
