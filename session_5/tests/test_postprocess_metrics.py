"""NMS, CTC decoding and the accuracy metrics — the code every number depends on."""

from __future__ import annotations

import numpy as np

from src import config
from src.metrics import FrameResult, average_precision, exact_matches
from src.postprocess import ctc_decode, encode_text, greedy_nms


def test_greedy_nms_keeps_the_best_of_overlapping_boxes() -> None:
    boxes = np.array([[0, 0, 100, 20], [2, 1, 101, 21], [300, 300, 400, 320]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    kept = greedy_nms(scores, boxes)
    assert len(kept) == 2 and kept[0, 4] == np.float32(0.9)


def test_greedy_nms_drops_scores_below_threshold() -> None:
    boxes = np.array([[0, 0, 10, 10]], np.float32)
    assert len(greedy_nms(np.array([config.SCORE_THRESHOLD - 0.01], np.float32), boxes)) == 0


def test_ctc_decode_collapses_repeats_and_drops_blanks() -> None:
    path = [0] + encode_text("A") * 3 + [0] + encode_text("A") + encode_text("7") * 2 + [0] * 24
    logits = np.full((1, len(path), config.NUM_CLASSES), -9.0, np.float32)
    logits[0, np.arange(len(path)), path] = 9.0
    assert ctc_decode(logits) == ["AA7"]  # the blank between the two As is what keeps them apart


def test_average_precision_perfect_and_empty() -> None:
    gt = np.array([[0, 0, 50, 10]], np.float32)
    perfect = FrameResult("day", gt, ["X"], np.array([[0, 0, 50, 10, 0.9]], np.float32))
    missed = FrameResult("day", gt, ["X"], np.zeros((0, 5), np.float32))
    assert average_precision([perfect]) == 1.0
    assert average_precision([missed]) == 0.0


def test_exact_match_needs_both_the_box_and_every_character() -> None:
    gt = np.array([[0, 0, 50, 10]], np.float32)
    det = np.array([[0, 0, 50, 10, 0.9]], np.float32)
    assert exact_matches(FrameResult("day", gt, ["ABC123"], det, ["ABC123"])) == 1
    assert exact_matches(FrameResult("day", gt, ["ABC123"], det, ["ABC128"])) == 0
