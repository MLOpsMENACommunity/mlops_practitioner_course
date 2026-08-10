from river.drift import PageHinkley, ADWIN
import mlflow, pandas as pd

# ── Page-Hinkley: detects a permanent upward shift in error ───
ph = PageHinkley(
    min_instances=30,   # wait for 30 samples before testing
    delta=0.005,        # sensitivity — magnitude of tolerated change
    threshold=50,       # threshold λ: higher = less sensitive
    alpha=0.9999,       # forgetting factor — weight recent more
)

# ── ADWIN: adaptive windowing — auto-adjusts window size ──────
adwin = ADWIN(delta=0.002)   # delta = false positive rate

def monitor_live_predictions(model, stream):
    """Call this for every prediction in a streaming pipeline."""
    for X, y_true in stream:
        y_pred = model.predict([X])[0]
        error  = abs(y_pred - y_true)       # regression: absolute error
        # error = int(y_pred != y_true)     # classification: 0/1 error

        ph.update(error)
        adwin.update(error)

        if ph.drift_detected:
            print(f"[Page-Hinkley] Concept drift detected at sample {ph.n_samples}")
            mlflow.log_metric("concept_drift_ph", 1)
            trigger_retraining()

        if adwin.drift_detected:
            print(f"[ADWIN] Concept drift — window shrank to {adwin.width} samples")
            mlflow.log_metric("concept_drift_adwin", 1)
            trigger_retraining()

# ── Without ground truth: monitor prediction distribution ─────
# Shift in the prediction distribution (PSI > 0.25) is an early
# warning for concept drift when ground truth isn't available yet.
