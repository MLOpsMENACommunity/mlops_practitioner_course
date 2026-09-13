"""The ANPR pipeline as six separately callable phases.

    decode -> preprocess -> detect -> nms -> crop -> ocr

There is no timing code in this file. src/benchmark.py calls the phases one at a
time and owns the clock, which is how the per-phase breakdown is measured on the
exact code path that produces the accuracy numbers — not on a copy of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config
from src.backends import Backend
from src.postprocess import ctc_decode, filter_fixed, greedy_nms
from src.preprocess import Letterbox, crop_plates, decode_jpeg, letterbox, to_nchw


@dataclass
class Plate:
    box: np.ndarray  # x0, y0, x1, y1, score — in original frame pixels
    text: str


class Pipeline:
    PHASES = ("decode", "preprocess", "detect", "nms", "crop", "ocr")

    def __init__(self, backend: Backend, fast_resize: bool = False) -> None:
        self.backend, self.fast_resize = backend, fast_resize

    def decode(self, jpegs: list[bytes]) -> list[np.ndarray]:
        return [decode_jpeg(j) for j in jpegs]

    def preprocess(self, frames: list[np.ndarray]) -> tuple[np.ndarray, list[Letterbox]]:
        boxed = [letterbox(f, self.fast_resize) for f in frames]
        return to_nchw([c for c, _ in boxed]), [lb for _, lb in boxed]

    def detect(self, batch: np.ndarray):
        return self.backend.detect(batch)

    def nms(self, raw) -> list[np.ndarray]:
        """Per-frame [M,5] detections in letterboxed pixels. A no-op filter if NMS ran in the graph."""
        if self.backend.nms_in_graph:
            return [filter_fixed(d) for d in raw]
        scores, boxes = raw
        return [greedy_nms(s, b) for s, b in zip(scores, boxes)]

    def crop(self, frames: list[np.ndarray], dets: list[np.ndarray], lbs: list[Letterbox]) -> np.ndarray:
        crops = [crop_plates(f, d, lb) for f, d, lb in zip(frames, dets, lbs)]
        return np.concatenate(crops) if crops else np.zeros((0, 1, config.OCR_H, config.OCR_W), np.float32)

    def ocr(self, crops: np.ndarray) -> list[str]:
        return ctc_decode(self.backend.ocr(crops)) if len(crops) else []

    def run(self, jpegs: list[bytes]) -> list[list[Plate]]:
        """Untimed end-to-end call, used by the accuracy pass and the parity gate."""
        frames = self.decode(jpegs)
        batch, lbs = self.preprocess(frames)
        dets = self.nms(self.detect(batch))
        texts = iter(self.ocr(self.crop(frames, dets, lbs)))
        return [[Plate(to_frame(d, lb), next(texts)) for d in frame_dets] for frame_dets, lb in zip(dets, lbs)]


def to_frame(det: np.ndarray, lb: Letterbox) -> np.ndarray:
    """One detection from letterboxed pixels back to original frame pixels."""
    out = det.copy()
    out[[0, 2]] = (det[[0, 2]] - lb.pad_x) / lb.scale
    out[[1, 3]] = (det[[1, 3]] - lb.pad_y) / lb.scale
    return out
