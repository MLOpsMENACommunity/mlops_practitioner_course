"""The CI gate's decision logic: accuracy vs the baseline, latency vs the row's own record, SLA only when asked."""

from __future__ import annotations

from ci.perf_gate import check
from src import config


def row(row_id: str, map50: float = 1.0, em: float = 0.9, p95: float = 20.0) -> dict:
    return {"id": row_id, "accuracy": {"map50": map50, "ocr_exact_match": em}, "latency": {"total_ms": {"p95": p95}}}


def test_identical_rows_pass() -> None:
    base = row("s01_baseline:eager-fp32")
    assert check(row("cand"), base, recorded=row("cand"), max_p95_ratio=1.15, enforce_sla=False) == []


def test_accuracy_drop_beyond_the_sla_fails() -> None:
    too_low = config.SLA_TARGET.max_ocr_em_drop + 0.01
    failures = check(row("cand", em=0.9 - too_low), row("base"), recorded=None, max_p95_ratio=1.15, enforce_sla=False)
    assert any("ocr_exact_match" in f for f in failures)


def test_latency_regression_is_judged_against_the_rows_own_record() -> None:
    recorded, remeasured = row("cand", p95=20.0), row("cand", p95=30.0)
    failures = check(remeasured, row("base", p95=5.0), recorded=recorded, max_p95_ratio=1.15, enforce_sla=False)
    assert len(failures) == 1 and "recorded p95" in failures[0]


def test_a_slower_runtime_than_the_baseline_is_not_a_regression() -> None:
    """ONNX Runtime slower than eager PyTorch on some CPU is a hardware fact, not a code regression."""
    assert check(row("cand", p95=90.0), row("base", p95=50.0), recorded=row("cand", p95=90.0),
                 max_p95_ratio=1.15, enforce_sla=False) == []  # fmt: skip


def test_absolute_sla_only_when_enforced() -> None:
    slow = row("cand", p95=config.SLA_TARGET.p95_ms + 5)
    assert check(slow, row("base"), recorded=None, max_p95_ratio=1.15, enforce_sla=False) == []
    assert any("SLA" in f for f in check(slow, row("base"), recorded=None, max_p95_ratio=1.15, enforce_sla=True))
