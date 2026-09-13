"""Plate detector: single-class, anchor-free, one output level (FCOS-style).

Two backbones share one neck and head:
  - ``resnet34``            the baseline and teacher: what you run on a server GPU
  - ``mobilenet_v3_small``  the distillation student: what the edge box can afford

Written here instead of imported from a detection library. Ultralytics YOLO is
AGPL-3.0 including trained weights, and owning ~100 lines of architecture is what
makes channel slicing, feature distillation and QAT teachable at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models

from src import config

STRIDE = 8  # one output level: plates are small, stride 8 keeps 2-4 cells per plate
OUTPUT_LAYERS = ("obj", "box")  # never pruned, never quantized first — see guides/06


def _conv_bn_relu(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def _resnet(depth: int) -> tuple[nn.Module, nn.Module, int, int]:
    net = {18: models.resnet18, 34: models.resnet34}[depth](weights=None)
    stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2)
    return stem, net.layer3, 128, 256


def _mobilenet() -> tuple[nn.Module, nn.Module, int, int]:
    feats = models.mobilenet_v3_small(weights=None).features
    return feats[:4], feats[4:9], 24, 48


# weights=None everywhere: torchvision's ImageNet weights carry dataset-derived
# terms (see data/README.md), and synthetic plates train fine from scratch.
BACKBONES = {"resnet18": lambda: _resnet(18), "resnet34": lambda: _resnet(34), "mobilenet_v3_small": _mobilenet}


class PlateDetector(nn.Module):
    """Frame tensor [B,3,384,640] -> raw objectness [B,1,48,80] and box maps [B,4,48,80]."""

    def __init__(self, backbone: str = "resnet34", width: int = 96) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.c3, self.c4, ch3, ch4 = BACKBONES[backbone]()
        self.lat3, self.lat4 = nn.Conv2d(ch3, width, 1), nn.Conv2d(ch4, width, 1)
        self.neck = _conv_bn_relu(width, width)
        self.tower = nn.Sequential(_conv_bn_relu(width, width), _conv_bn_relu(width, width))
        self.obj = nn.Conv2d(width, 1, 3, padding=1)
        self.box = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.constant_(self.obj.bias, -4.6)  # sigmoid(-4.6) ~ 1%: focal loss starts stable
        ys, xs = torch.meshgrid(torch.arange(config.DET_H // STRIDE), torch.arange(config.DET_W // STRIDE), indexing="ij")
        centres = (torch.stack([xs, ys], -1).reshape(-1, 2).float() + 0.5) * STRIDE
        self.register_buffer("centres", centres, persistent=False)

    def features(self, x: Tensor) -> Tensor:
        """The stride-8 feature map the head reads — and the one distillation matches."""
        c3 = self.c3(x)
        p3 = self.lat3(c3) + F.interpolate(self.lat4(self.c4(c3)), scale_factor=2.0, mode="nearest")
        return self.neck(p3)

    def forward_with_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(features, objectness, boxes) in one pass — distillation needs all three."""
        feats = self.features(x)
        t = self.tower(feats)
        return feats, self.obj(t), self.box(t)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        _, obj, box = self.forward_with_features(x)
        return obj, box

    def decode(self, obj: Tensor, box: Tensor) -> tuple[Tensor, Tensor]:
        """Raw maps -> scores [B,N] and xyxy boxes [B,N,4] in letterboxed pixels."""
        scores = obj.flatten(1).sigmoid()
        ltrb = F.softplus(box.flatten(2).transpose(1, 2)) * STRIDE
        return scores, torch.cat([self.centres - ltrb[..., :2], self.centres + ltrb[..., 2:]], -1)


def pairwise_iou(b: Tensor) -> Tensor:
    """IoU between every pair of boxes in [B,K,4] -> [B,K,K], plain tensor ops only."""
    lt = torch.maximum(b[:, :, None, :2], b[:, None, :, :2])
    rb = torch.minimum(b[:, :, None, 2:], b[:, None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area[:, :, None] + area[:, None, :] - inter + 1e-6)


class DetectorExport(nn.Module):
    """What gets exported: decode (and optionally NMS) inside the graph.

    ``with_nms=False`` returns scores + boxes and leaves top-K and NMS to Python.
    ``with_nms=True`` runs top-K + Fast NMS as tensor ops, so every runtime
    executes the post-processing and the output is a fixed [B, TOPK, 5] tensor.
    """

    def __init__(self, detector: PlateDetector, with_nms: bool = False) -> None:
        super().__init__()
        self.detector, self.with_nms = detector, with_nms
        # Constant strictly-upper-triangular mask: box j can only be suppressed by a
        # higher-scoring box i < j. A constant avoids the Trilu op, which not every
        # converter handles.
        self.register_buffer("upper", torch.ones(config.TOPK, config.TOPK).triu(1), persistent=False)

    # --- snippet:nms-in-graph ---
    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        scores, boxes = self.detector.decode(*self.detector(x))
        if not self.with_nms:
            return scores, boxes
        top, idx = scores.topk(config.TOPK, dim=1)
        cand = boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
        suppressed = (pairwise_iou(cand) * self.upper).amax(dim=1) >= config.NMS_IOU
        return torch.cat([cand, (top * (~suppressed)).unsqueeze(-1)], dim=-1)

    # --- end-snippet ---
