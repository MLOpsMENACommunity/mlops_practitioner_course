"""Accuracy: detector mAP@0.5 and plate-level OCR exact match — always both.

Why both, always: INT8 can leave the boxes exactly where they were while the
characters inside them stop reading correctly. A table that reports only mAP
reports that model as healthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import config


@dataclass
class FrameResult:
    condition: str
    gt_boxes: np.ndarray  # [G, 4] in frame pixels
    gt_texts: list[str]
    dets: np.ndarray  # [M, 5] in frame pixels, score last
    texts: list[str] = field(default_factory=list)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[M,4] x [G,4] -> [M,G]."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: np.clip(x[:, 2:4] - x[:, :2], 0, None).prod(-1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(frames: list[FrameResult]) -> float:
    """Single-class AP at IoU 0.5, all-point interpolation (the VOC2010+ definition)."""
    scored, n_gt = [], sum(len(f.gt_boxes) for f in frames)
    for fi, f in enumerate(frames):
        scored += [(float(d[4]), fi, di) for di, d in enumerate(f.dets)]
    if n_gt == 0:
        return float("nan")
    scored.sort(key=lambda s: -s[0])
    taken = [np.zeros(len(f.gt_boxes), bool) for f in frames]
    tp = np.zeros(len(scored))
    for k, (_, fi, di) in enumerate(scored):
        f = frames[fi]
        if len(f.gt_boxes) == 0:
            continue
        overlaps = iou(f.dets[di : di + 1], f.gt_boxes)[0]
        overlaps[taken[fi]] = 0  # a plate can only be found once
        j = int(overlaps.argmax())
        if overlaps[j] >= config.MATCH_IOU:
            taken[fi][j], tp[k] = True, 1
    recall = np.concatenate([[0], np.cumsum(tp) / n_gt, [1]])
    precision = np.concatenate([[1], np.cumsum(tp) / np.arange(1, len(tp) + 1), [0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def exact_matches(frame: FrameResult) -> int:
    """Plates read completely correctly: matched at IoU >= 0.5 AND every character right."""
    if len(frame.gt_boxes) == 0 or len(frame.dets) == 0:
        return 0
    overlaps, hits = iou(frame.dets, frame.gt_boxes), 0
    for g, text in enumerate(frame.gt_texts):
        d = int(overlaps[:, g].argmax())
        read = frame.texts[d] if d < len(frame.texts) else ""  # detector-only evaluations carry no text
        hits += int(overlaps[d, g] >= config.MATCH_IOU and read == text)
        overlaps[d, :] = 0  # one detection reads one plate
    return hits


def summarize(frames: list[FrameResult]) -> dict:
    """mAP@0.5 + end-to-end plate exact match, overall and per condition."""

    def block(fs: list[FrameResult]) -> dict:
        n_plates = sum(len(f.gt_boxes) for f in fs)
        return {
            "map50": round(average_precision(fs), 4),
            "ocr_exact_match": round(sum(exact_matches(f) for f in fs) / max(n_plates, 1), 4),
            "n_frames": len(fs),
            "n_plates": n_plates,
        }

    out = block(frames)
    out["per_condition"] = {c: block([f for f in frames if f.condition == c]) for c in config.CONDITIONS}
    return out
