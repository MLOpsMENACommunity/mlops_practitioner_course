import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric

# ── Load reference (training) and current (production) data ──
reference = pd.read_parquet("data/train.parquet")
current   = pd.read_parquet(f"data/scoring/output/{today}.parquet")

# ── Evidently DataDrift report ─────────────────────────────────
report = Report(metrics=[
    DataDriftPreset(),                         # all features
    ColumnDriftMetric(column_name="distance_km",    stattest="ks"),
    ColumnDriftMetric(column_name="passengers",     stattest="chisquare"),
    ColumnDriftMetric(column_name="hour_of_day",    stattest="psi"),
    ColumnDriftMetric(column_name="pred",           stattest="psi"),
])
report.run(reference_data=reference, current_data=current)
report.save_html("reports/drift_report.html")     # rich visual HTML

# ── Extract results programmatically ──────────────────────────
results = report.as_dict()
for metric in results["metrics"]:
    if metric["metric"] == "ColumnDriftMetric":
        m = metric["result"]
        status = "🚨 DRIFT" if m["drift_detected"] else "✅ OK"
        print(f'{m["column_name"]:20s} | {m["stattest"]:12s} | p={m["p_value"]:.4f} | {status}')

# ── Airflow task: alert if > 30% of features drifted ──────────
drift_share = results["metrics"][0]["result"]["share_of_drifted_columns"]
if drift_share > 0.3:
    raise ValueError(f"Data drift alert: {drift_share:.0%} of features drifted!")
