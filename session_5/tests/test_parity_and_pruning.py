"""The parity gate fails loudly, and pruning never touches output layers."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src import config
from src.backends import Backend
from src.models.detector import OUTPUT_LAYERS, PlateDetector
from src.parity import ParityError, check
from src.pipeline import Pipeline
from src.stages.s05_pruning import prunable_convs


class FakeBackend(Backend):
    """Deterministic outputs with one plate-shaped box, plus an optional perturbation."""

    def __init__(self, noise: float = 0.0) -> None:
        super().__init__("fake", "fake")
        self.noise = noise

    def detect(self, x):
        n = (config.DET_H // 8) * (config.DET_W // 8)
        scores = np.full((len(x), n), 0.01, np.float32)
        scores[:, 100] = 0.9 + self.noise
        boxes = np.tile(np.array([100, 150, 220, 180], np.float32), (len(x), n, 1)) + self.noise
        return scores, boxes

    def ocr(self, crops):
        logits = np.zeros((len(crops), config.OCR_SEQ_LEN, config.NUM_CLASSES), np.float32)
        logits[:, :, 1] = 5.0 + self.noise
        return logits


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (config.FRAME_W, config.FRAME_H), (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def test_identical_artifacts_pass_strict_parity() -> None:
    report = check(Pipeline(FakeBackend()), Pipeline(FakeBackend()), [jpeg(), jpeg()])
    assert report["passed"] and report["max_abs_diff"] == 0.0


def test_a_perturbed_artifact_fails_strict_parity() -> None:
    with pytest.raises(ParityError):
        check(Pipeline(FakeBackend()), Pipeline(FakeBackend(noise=0.05)), [jpeg()])


def test_report_only_mode_never_raises() -> None:
    report = check(Pipeline(FakeBackend()), Pipeline(FakeBackend(noise=0.05)), [jpeg()], strict=False)
    assert report["passed"] is None and report["max_abs_diff"] > 0


def test_pruning_never_selects_output_layers() -> None:
    model = PlateDetector("resnet18")
    chosen = prunable_convs(model)
    for name in OUTPUT_LAYERS:
        assert getattr(model, name) not in chosen
    assert len(chosen) > 10
