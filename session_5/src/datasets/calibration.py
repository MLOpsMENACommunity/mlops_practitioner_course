"""Calibration sets: which frames INT8 learns its value ranges from.

The quantizer sets each tensor's scale from the activations it sees here, once.
A night plate, a rain-streaked frame or a low-contrast plate that never appears in
calibration gets squeezed into ranges that were measured on bright daylight — and
nothing warns you. Calibration frames come from their own split, never from val:
calibrating on the frames you are about to score is leaking the test set.
"""

from __future__ import annotations

import numpy as np

from src import config
from src.datasets.splits import Sample, gt_crop, load_split
from src.preprocess import decode_jpeg, letterbox, to_nchw

STRATEGIES = ("stratified", "daytime")


# --- snippet:calibration-set ---
def frames(strategy: str, n: int | None = None) -> list[Sample]:
    """`stratified`: equal frames per condition. `daytime`: the tempting shortcut."""
    n = n or config.profile().n_calib
    pool = load_split("calib")
    if strategy == "daytime":
        return [s for s in pool if s.condition == "day"][:n]
    per = n // len(config.CONDITIONS)
    return [s for c in config.CONDITIONS for s in [p for p in pool if p.condition == c][:per]]


# --- end-snippet ---


def detector_batches(samples: list[Sample]) -> list[np.ndarray]:
    """[1,3,384,640] float32 per frame, preprocessed exactly as the pipeline does."""
    return [to_nchw([letterbox(decode_jpeg(s.path.read_bytes()))[0]]) for s in samples]


def ocr_batches(strategy: str, n: int | None = None) -> list[np.ndarray]:
    """[1,1,32,128] ground-truth plate crops drawn with the same condition strategy."""
    n = (n or config.profile().n_calib) * 2
    cache = np.load(config.data_dir() / "calib_ocr.npz")
    conds = cache["conditions"]
    if strategy == "daytime":
        idx = np.flatnonzero(conds == "day")[:n]
    else:
        idx = np.concatenate([np.flatnonzero(conds == c)[: n // len(config.CONDITIONS)] for c in config.CONDITIONS])
    return [gt_crop(cache["crops"][i], cache["inner"][i], np.full(4, 0.08))[None, None] for i in idx]


class OrtReader:
    """onnxruntime.quantization.CalibrationDataReader over a list of arrays."""

    def __init__(self, input_name: str, arrays: list[np.ndarray]) -> None:
        self.name, self.it = input_name, iter(arrays)

    def get_next(self) -> dict[str, np.ndarray] | None:
        arr = next(self.it, None)
        return None if arr is None else {self.name: arr}

    def rewind(self) -> None:  # part of the CalibrationDataReader protocol; one pass is enough here
        pass
