"""Detector and recognizer outputs -> boxes and strings. numpy only, like preprocess.py."""

from __future__ import annotations

import numpy as np

from src import config


# --- snippet:greedy-nms ---
def greedy_nms(scores: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """One image: top-K, score threshold, greedy NMS -> [M, 5] (x0, y0, x1, y1, score)."""
    order = np.argsort(-scores)[: config.TOPK]
    order = order[scores[order] >= config.SCORE_THRESHOLD]
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        lt = np.maximum(boxes[i, :2], boxes[rest, :2])
        rb = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.clip(rb - lt, 0, None).prod(1)
        area = lambda b: np.clip(b[..., 2:] - b[..., :2], 0, None).prod(-1)  # noqa: E731
        iou = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-6)
        order = rest[iou < config.NMS_IOU]
    return np.concatenate([boxes[keep], scores[keep, None]], 1).astype(np.float32)


# --- end-snippet ---


def filter_fixed(dets: np.ndarray) -> np.ndarray:
    """One image of in-graph NMS output [TOPK, 5] -> the rows above threshold, [M, 5]."""
    return dets[dets[:, 4] >= config.SCORE_THRESHOLD].astype(np.float32)


# --- snippet:ctc-greedy-decode ---
def ctc_decode(logits: np.ndarray) -> list[str]:
    """[B, T, C] -> strings: argmax per step, collapse repeats, drop blanks (index 0)."""
    out = []
    for path in logits.argmax(-1):
        chars = [c for t, c in enumerate(path) if c != 0 and (t == 0 or c != path[t - 1])]
        out.append("".join(config.CHARSET[c - 1] for c in chars))
    return out


# --- end-snippet ---


def encode_text(text: str) -> list[int]:
    """'ABC123' -> CTC target indices (blank is 0, so characters start at 1)."""
    return [config.CHARSET.index(c) + 1 for c in text]
