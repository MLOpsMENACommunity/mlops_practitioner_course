# ── 1. Instrument your Litestar/FastAPI service ──────
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Custom ML metrics (beyond HTTP request counts)
PREDICTION_HISTOGRAM = Histogram(
    "model_prediction_duration_min",
    "Distribution of predicted ride durations",
    buckets=[0,5,10,15,20,30,45,60,90,120]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "End-to-end API latency",
    ["endpoint", "status"]
)
DRIFT_GAUGE = Gauge("feature_psi_score", "PSI drift score per feature", ["feature"])
MODEL_VERSION = Gauge("model_version_info", "Active model version", ["version", "stage"])

@app.post("/predict")
async def predict(req: PredictRequest):
    t0 = time.perf_counter()
    result = model.predict([[req.distance_km, req.passengers, req.hour_of_day]])
    latency = time.perf_counter() - t0

    # Record to Prometheus
    PREDICTION_HISTOGRAM.observe(float(result[0]))
    REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(latency)

    return {"duration_min": round(float(result[0]), 2)}

# ── 2. Record drift scores (call from Airflow drift check task) ──
def record_drift_scores(psi_scores: dict):
    for feature, score in psi_scores.items():
        DRIFT_GAUGE.labels(feature=feature).set(score)

# ── 3. Grafana dashboard queries (PromQL) ─────────────────────
# Panel 1: Request rate
#   rate(api_request_latency_seconds_count[5m])
# Panel 2: p95 latency
#   histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))
# Panel 3: Prediction distribution
#   rate(model_prediction_duration_min_bucket[1h])
# Panel 4: PSI drift scores (alert if > 0.25)
#   feature_psi_score > 0.25
