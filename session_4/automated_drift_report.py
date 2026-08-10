from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.tests import (
    TestShareOfDriftedColumns, TestColumnDrift,
    TestNumberOfMissingValues, TestColumnValueMin,
)

# ── 1. Report — rich HTML for humans ──────────────────────────
report = Report(metrics=[
    DataQualityPreset(),    # missing values, duplicates, type mismatches
    DataDriftPreset(),      # per-column drift with distribution plots
])
report.run(reference_data=ref, current_data=curr)
report.save_html("reports/weekly_drift.html")

# ── 2. Test Suite — machine-readable pass/fail for CI/CD ──────
suite = TestSuite(tests=[
    TestShareOfDriftedColumns(lt=0.3),       # fail if > 30% columns drift
    TestColumnDrift("distance_km",  lt=0.1), # fail if KS statistic > 0.1
    TestColumnDrift("pred",         lt=0.25),# fail if prediction PSI > 0.25
    TestNumberOfMissingValues(eq=0),         # fail if any nulls
    TestColumnValueMin("distance_km", gt=0), # fail if any negative distances
])
suite.run(reference_data=ref, current_data=curr)
suite.save_html("reports/test_suite.html")

# Machine-readable result — use in Airflow or CI
result = suite.as_dict()
if not result["summary"]["all_passed"]:
    failed = [t for t in result["tests"] if t["status"] == "FAIL"]
    raise ValueError(f"Data quality / drift tests failed: {len(failed)} failures")
