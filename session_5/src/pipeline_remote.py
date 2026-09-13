"""The pipeline when Triton runs it: the client decodes the JPEG, the server does the rest.

The phases keep their names so the harness can still time them, but only two carry
work: `decode` on the client and `detect`, which is the whole round trip to the `anpr`
model (letterbox, detector, crop, recognizer, CTC decode — server side).
"""

from __future__ import annotations

import numpy as np

from src.pipeline import Pipeline, Plate


class RemotePipeline(Pipeline):
    def preprocess(self, frames: list[np.ndarray]):
        return frames, None

    def detect(self, frames: list[np.ndarray]):
        return [self.backend.pipeline(f) for f in frames]

    def nms(self, raw) -> list[np.ndarray]:
        self._texts = [texts for _, texts in raw]
        return [dets for dets, _ in raw]

    def crop(self, frames, dets, lbs):
        return self._texts

    def ocr(self, texts) -> list[str]:
        return [t for frame_texts in texts for t in frame_texts]

    def run(self, jpegs: list[bytes]) -> list[list[Plate]]:
        out = []
        for frame in self.decode(jpegs):
            dets, texts = self.backend.pipeline(frame)
            out.append([Plate(d, t) for d, t in zip(dets, texts)])
        return out
