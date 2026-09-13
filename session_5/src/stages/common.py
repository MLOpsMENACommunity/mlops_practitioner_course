"""Helpers every stage script shares: artifact names, device checks, the parent rows."""

from __future__ import annotations

import importlib.util

from src import config
from src.models import io

DET_BASE, OCR_BASE = "detector_baseline", "ocr_baseline"
BASELINE_ROW = "s01_baseline:eager-fp32"


def has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def device() -> str:
    """Where benchmarks run: the GPU if there is one, otherwise CPU — never a crash."""
    return "cuda" if has_cuda() else "cpu"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def require(*names: str) -> None:
    """Fail with the fix, not a traceback, when a stage runs before its inputs exist."""
    missing = [n for n in names if not io.path_for(n).exists()]
    if missing:
        raise SystemExit(f"missing checkpoints {missing} — run: make train")


def require_files(*names: str, fix: str) -> None:
    """Stages run alone (`make s09`) instead of re-running everything upstream, so each checks its inputs."""
    missing = [n for n in names if not (config.artifacts_dir() / n).exists()]
    if missing:
        raise SystemExit(f"missing artifacts {missing} — run: {fix}")


def sample_jpegs(n: int = 4) -> list[bytes]:
    """A fixed handful of validation frames that contain plates — the parity batch."""
    from src.datasets.splits import load_split

    picked = [s for s in load_split("val") if s.boxes][:n]
    return [s.path.read_bytes() for s in picked]


def gate(spec, ref, new, strict: bool = True) -> dict:
    """Run the parity gate for `spec`. A strict failure records the row and stops the stage loudly."""
    from src.benchmark import failed
    from src.parity import ParityError, check
    from src.pipeline import Pipeline

    try:
        report = check(Pipeline(ref), Pipeline(new), sample_jpegs(4), strict=strict)
    except ParityError as exc:
        spec.parity = {"passed": False, "error": str(exc)[:2000]}
        failed(spec, "parity gate failed")
        raise SystemExit(f"\n  PARITY GATE FAILED for {spec.id}\n{exc}\n") from None
    spec.parity = report
    print(f"  parity {spec.id}: max|diff| {report['max_abs_diff']:.2e}  boxes equal {report['box_count_equal']}  "
          f"min IoU {report['min_matched_box_iou']:.4f}  strings equal {report['strings_equal']:.0%}"
          f"{'  (strict: PASSED)' if strict else '  (report only)'}", flush=True)  # fmt: skip
    return report


def banner(title: str) -> None:
    print(f"\n=== {title}  (profile={config.profile().name}, threads={config.THREADS}) ===", flush=True)
