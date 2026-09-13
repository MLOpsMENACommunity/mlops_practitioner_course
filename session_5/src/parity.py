"""The export parity gate: a converted artifact must compute what its parent computed.

Three levels, because each one alone has a blind spot:
  1. tensors — np.testing.assert_allclose on raw detector outputs and OCR logits
  2. boxes   — post-NMS detections matched one-to-one by IoU
  3. strings — decoded plate text, character for character

Logits can agree to 1e-4 while NMS keeps a different box, or a near-tie in CTC
decoding flips a character. Only the last two levels are what a user sees.

Precision-changing stages (FP16, INT8, pruned, distilled) are *expected* to move
the tensors, so they run with strict=False: the report is recorded, and their
accuracy is judged by the benchmark instead.
"""

from __future__ import annotations

import numpy as np

from src.metrics import iou
from src.pipeline import Pipeline


class ParityError(AssertionError):
    """A converted artifact disagrees with its parent beyond tolerance."""


def _as_list(raw) -> list[np.ndarray]:
    return [raw] if isinstance(raw, np.ndarray) else list(raw)


# --- snippet:parity-gate ---
def check(ref: Pipeline, new: Pipeline, jpegs: list[bytes], strict: bool = True,
          rtol: float = 1e-3, atol: float = 1e-5, min_box_iou: float = 0.99) -> dict:  # fmt: skip
    frames = ref.decode(jpegs)
    batch, lbs = ref.preprocess(frames)
    ref_raw, new_raw = ref.detect(batch), new.detect(batch)
    ref_dets, new_dets = ref.nms(ref_raw), new.nms(new_raw)
    crops = ref.crop(frames, ref_dets, lbs)  # identical crops into both recognizers
    ref_logits, new_logits = ref.backend.ocr(crops), new.backend.ocr(crops)

    report = {"frames": len(jpegs), "strict": strict, "rtol": rtol, "atol": atol}
    tensors = list(zip(_as_list(ref_raw), _as_list(new_raw))) + [(ref_logits, new_logits)]
    report["max_abs_diff"] = float(max(np.abs(a - b).max() for a, b in tensors if a.size))
    report["box_count_equal"] = all(len(a) == len(b) for a, b in zip(ref_dets, new_dets))
    ious = [iou(a[:, :4], b[:, :4]).max(1).min() for a, b in zip(ref_dets, new_dets) if len(a) and len(b)]
    report["min_matched_box_iou"] = float(min(ious)) if ious else 1.0
    ref_text, new_text = ref.ocr(crops), new.ocr(crops)
    report["strings_equal"] = float(np.mean([a == b for a, b in zip(ref_text, new_text)])) if ref_text else 1.0

    if strict:
        try:
            for a, b in tensors:
                np.testing.assert_allclose(b, a, rtol=rtol, atol=atol)
        except AssertionError as exc:
            raise ParityError(f"tensor parity failed:\n{exc}") from None
        if not report["box_count_equal"] or report["min_matched_box_iou"] < min_box_iou:
            raise ParityError(f"box parity failed: {report}")
        if report["strings_equal"] < 1.0:
            raise ParityError(f"string parity failed: {report}")
    report["passed"] = True if strict else None
    return report


# --- end-snippet ---
