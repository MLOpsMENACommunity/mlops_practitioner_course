from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset
from evidently.metrics import ColumnDriftMetric
import numpy as np

# ── PSI on prediction distribution ───────────────────────────
def psi(expected: np.ndarray, actual: np.ndarray, n_bins=10) -> float:
    """Population Stability Index. PSI < 0.1 stable, > 0.25 retrain."""
    expected_perc = np.histogram(expected, bins=n_bins)[0] / len(expected)
    actual_perc   = np.histogram(actual,   bins=np.histogram(expected, bins=n_bins)[1])[0] / len(actual)
    # Avoid log(0)
    expected_perc = np.where(expected_perc == 0, 1e-4, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-4, actual_perc)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))

# Load predictions from batch jobs
ref_preds  = pd.read_parquet("data/scoring/output/2024-01-01.parquet")["pred"]
curr_preds = pd.read_parquet(f"data/scoring/output/{today}.parquet")["pred"]

score = psi(ref_preds.values, curr_preds.values)
print(f"Prediction PSI: {score:.4f}")
if score > 0.25:
    raise ValueError(f"Prediction drift: PSI={score:.3f} > 0.25 — investigate!")

# ── Evidently TargetDrift (when ground truth eventually arrives) ──
report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=ref_with_labels, current_data=curr_with_labels)
report.save_html("reports/target_drift.html")
