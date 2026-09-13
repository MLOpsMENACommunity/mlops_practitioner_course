"""Load a split — and refuse a validation set that changed since it was generated.

Every stage must be scored on the identical frames, or the journey table compares
two different exams. The manifest written by synthetic.py holds a sha256 over the
annotations and JPEG bytes; `load_split(..., verify=True)` recomputes it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src import config


@dataclass(frozen=True)
class Sample:
    path: Path
    condition: str
    boxes: list[list[int]]
    texts: list[str]


class DatasetChanged(RuntimeError):
    """The frames on disk are not the frames the results were measured on."""


def load_split(split: str, verify: bool = False) -> list[Sample]:
    root = config.data_dir()
    ann = root / f"{split}.jsonl"
    if not ann.exists():
        raise FileNotFoundError(f"{ann} is missing — run: make data")
    lines = ann.read_text().splitlines()
    samples = []
    for line in lines:
        rec = json.loads(line)
        boxes = [p["box"] for p in rec["plates"]]
        samples.append(Sample(root / split / rec["file"], rec["condition"], boxes, [p["text"] for p in rec["plates"]]))
    if verify:
        digest = hashlib.sha256()
        for line, sample in zip(lines, samples):
            digest.update(line.encode())
            digest.update(sample.path.read_bytes())
        if digest.hexdigest() != fingerprint(split):
            raise DatasetChanged(f"{split} split differs from manifest.json — regenerate with: make data")
    return samples


def gt_crop(crop: np.ndarray, inner: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """Cut the recognizer input out of a cached crop, with per-side margins as box fractions."""
    h, w = crop.shape
    x0, y0, x1, y1 = inner * [w, h, w, h]
    bw, bh = x1 - x0, y1 - y0
    region = (max(0, x0 - margins[0] * bw), max(0, y0 - margins[1] * bh),
              min(w, x1 + margins[2] * bw), min(h, y1 + margins[3] * bh))  # fmt: skip
    img = Image.fromarray(crop).crop(region).resize((config.OCR_W, config.OCR_H), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def fingerprint(split: str = "val") -> str:
    """The content hash every result records, so rows from different data never share a table."""
    manifest = json.loads((config.data_dir() / "manifest.json").read_text())
    return manifest["splits"][split]["sha256"]
