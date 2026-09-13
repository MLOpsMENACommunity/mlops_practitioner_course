"""Single source of truth: paths, geometry, thresholds, the SLA, thread counts.

Every stage imports from here. A number that changes a result and lives anywhere
else is a number two stages can silently disagree about — and then the journey
table compares two different experiments while looking like one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
# Not `triton/`: a top-level folder of that name shadows the `triton` package
# that torch._dynamo imports, and `import torchvision` dies with
# "module 'triton' has no attribute 'language'" when run from this directory.
TRITON_REPO = ROOT / "triton_serving" / "model_repository"

# --- camera + model geometry -------------------------------------------------
FRAME_W, FRAME_H = 1280, 720  # what the roadside camera hands us, as JPEG
DET_W, DET_H = 640, 384  # letterboxed detector input; both divisible by 32
OCR_W, OCR_H = 128, 32  # grayscale plate crop fed to the recognizer
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES = len(CHARSET) + 1  # +1: index 0 is the CTC blank
OCR_SEQ_LEN = 32  # time steps out of the recognizer (OCR_W / 4)
MAX_PLATES = 4  # fixed OCR batch for static-shape runtimes (TFLite, TRT profile max)
TOPK = 50  # detector candidates kept before NMS
# Opset 18: accepted by every runtime in this session's matrix (ORT 1.24 inside
# Triton 26.05, TensorRT 10.16's parser, OpenVINO 2026, onnx2tf). A newer opset buys
# nothing for Conv/LSTM/TopK and loses you whichever converter lags behind.
ONNX_OPSET = 18

# --- decision thresholds -----------------------------------------------------
SCORE_THRESHOLD = 0.35
NMS_IOU = 0.5
MATCH_IOU = 0.5  # the ".5" in mAP@0.5

CONDITIONS = ("day", "night", "rain", "motion_blur", "low_contrast")


@dataclass(frozen=True)
class SLA:
    """The deployment contract. The CI gate reads these, not a copy of them."""

    p95_ms: float = 30.0  # per frame, end to end, on the roadside box
    batch_size: int = 1  # a camera delivers one frame at a time
    max_map50_drop: float = 0.02  # absolute, vs the parent stage
    max_ocr_em_drop: float = 0.02  # absolute, vs the parent stage


SLA_TARGET = SLA()


@dataclass(frozen=True)
class Profile:
    """How much data and training a run gets.

    `quick` is what `make all` uses on a CPU laptop; `full` is for the RTX 3090.
    Every result records its profile, and the journey table never puts rows from
    two profiles side by side — a quick-profile mAP is not comparable to a full one.
    """

    name: str
    n_train: int
    n_val: int
    n_calib: int
    det_epochs: int
    ocr_epochs: int
    finetune_epochs: int
    latency_runs: int
    throughput_batches: tuple[int, ...]
    throughput_iters: int


PROFILES = {
    "ci": Profile("ci", 160, 48, 32, 1, 3, 1, 40, (1, 2), 6),
    "quick": Profile("quick", 1600, 240, 128, 8, 30, 2, 200, (1, 4, 8), 20),
    "full": Profile("full", 12000, 1000, 512, 40, 40, 8, 500, (1, 2, 4, 8, 16, 32), 50),
}


def profile() -> Profile:
    """The active profile, from ANPR_PROFILE (default: quick)."""
    return PROFILES[os.environ.get("ANPR_PROFILE", "quick")]


def data_dir() -> Path:
    return DATA_DIR / "synthetic" / profile().name


def artifacts_dir() -> Path:
    path = ARTIFACTS_DIR / profile().name
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- benchmark settings ------------------------------------------------------
WARMUP_RUNS = 20  # first calls pay for lazy allocation, kernel selection, caches
# Thread counts alone swing CPU latency several-fold, so they are fixed and
# recorded, never left to library defaults that differ per machine. 4 matches
# the core count of the edge boxes we target (Raspberry Pi 5, Jetson Orin Nano).
THREADS = int(os.environ.get("ANPR_THREADS", "4"))
SEED = 1234
