"""Gate stage: fail the pipeline when RAGAS scores drop below thresholds.

Classical ML equivalent: an MAE regression gate ("fail the build if error
got worse"). This script is designed to be the last step of the DVC pipeline
and to be called from GitHub Actions / any CI system: a non-zero exit code
fails the CI job, blocking merges/deploys of a degraded RAG system.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger()

PARAMS_PATH = Path("params.yaml")
REPORT_PATH = Path("reports/ragas.json")


def load_thresholds() -> dict[str, float]:
    """Read ragas_thresholds from params.yaml."""
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)["ragas_thresholds"]


def load_report(path: Path) -> dict:
    """Read the metrics file produced by the evaluate stage."""
    if not path.is_file():
        log.error("report_missing", expected=str(path), hint="Run the evaluate stage first.")
        sys.exit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def check_thresholds(
    report: dict, thresholds: dict[str, float]
) -> list[tuple[str, float, float]]:
    """Return a list of (metric, actual, threshold) for every failing metric."""
    failures: list[tuple[str, float, float]] = []
    for metric, threshold in thresholds.items():
        actual = report.get(metric)
        if actual is None:
            log.error("metric_missing_in_report", metric=metric)
            failures.append((metric, float("nan"), threshold))
        elif actual < threshold:
            failures.append((metric, actual, threshold))
    return failures


def main() -> None:
    """Run the gate stage: exit 1 if any RAGAS metric is below its threshold."""
    thresholds = load_thresholds()
    report = load_report(REPORT_PATH)

    failures = check_thresholds(report, thresholds)
    if failures:
        # print() on top of structlog so the verdict is unmissable in CI logs.
        print("RAGAS gate FAILED")
        for metric, actual, threshold in failures:
            shortfall = threshold - actual
            log.error(
                "ragas_gate_failed_metric",
                metric=metric,
                actual=actual,
                threshold=threshold,
                shortfall=round(shortfall, 4),
            )
            print(
                f"  FAIL {metric}: {actual:.4f} < {threshold:.2f} "
                f"(short by {shortfall:.4f})"
            )
        sys.exit(1)

    print("RAGAS gate PASSED")
    for metric, threshold in thresholds.items():
        print(f"  PASS {metric}: {report[metric]:.4f} >= {threshold:.2f}")
    log.info("ragas_gate_passed", **{m: report[m] for m in thresholds})
    sys.exit(0)


if __name__ == "__main__":
    main()
