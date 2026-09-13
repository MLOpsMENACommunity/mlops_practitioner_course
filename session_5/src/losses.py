"""Training losses for the two models."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

from src.models.detector import STRIDE, PlateDetector


def assign(centres: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
    """FCOS-style assignment: a cell is positive if its centre is inside a plate AND near its middle.

    "Near" is 1.5 strides vertically but a third of the plate horizontally — plates
    are ~5x wider than tall, and a square radius would leave most of a wide plate
    with no positive cells.
    """
    if gt.numel() == 0:
        return torch.zeros(len(centres), dtype=torch.bool, device=centres.device), torch.zeros_like(centres).repeat(1, 2)
    cx, cy = centres[:, 0:1], centres[:, 1:2]
    x0, y0, x1, y1 = gt.unbind(-1)
    inside = (cx > x0) & (cx < x1) & (cy > y0) & (cy < y1)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    near = ((cx - mx).abs() < torch.clamp((x1 - x0) / 3, min=1.5 * STRIDE)) & ((cy - my).abs() < 1.5 * STRIDE)
    area = torch.where(inside & near, ((x1 - x0) * (y1 - y0)).expand_as(inside), torch.inf)
    best = area.argmin(1)  # a cell inside two plates belongs to the smaller one
    return torch.isfinite(area.min(1).values), gt[best]


def detector_loss(model: PlateDetector, obj: Tensor, box: Tensor, gts: list[Tensor]) -> Tensor:
    """Focal loss on objectness (every cell) + GIoU on boxes (positive cells only)."""
    logits = obj.flatten(1)
    _, boxes = model.decode(obj, box)
    pos, matched = zip(*(assign(model.centres, g.to(obj.device)) for g in gts))
    pos, matched = torch.stack(pos), torch.stack(matched)
    n_pos = pos.sum().clamp(min=1)
    cls = sigmoid_focal_loss(logits, pos.float(), reduction="sum") / n_pos
    reg = generalized_box_iou_loss(boxes[pos], matched[pos], reduction="sum") / n_pos if pos.any() else logits.sum() * 0
    return cls + 2.0 * reg


def ocr_loss(logits: Tensor, targets: Tensor, target_lengths: Tensor) -> Tensor:
    """CTC over the recognizer's 32 time steps."""
    log_probs = logits.log_softmax(-1).transpose(0, 1)  # CTC wants [T, B, C]
    if log_probs.device.type == "mps":
        log_probs = log_probs.cpu()  # no MPS kernel for ctc_loss; autograd moves the gradient back
    input_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, zero_infinity=True)
