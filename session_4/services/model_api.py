"""The instrumented ride-duration service — P_G_monitoring.py, made runnable.

P_G_monitoring.py is a teaching snippet: it references an ``app`` that does not
exist. This is that snippet as a service the incidents can actually break, using
THE SAME metric names, because monitoring/grafana/dashboards/model-monitoring.json
and monitoring/prometheus/alert_rules.yml already query them.

    pip install -e ".[metrics]"
    python -m services.ride_model          # once — fit and freeze artifacts/
    make api                              # uvicorn on :8001, scraped by Prometheus

Metrics are regression metrics. There is no ml_prediction_score, no positive
rate and no outcome label: this model predicts minutes, so its output
distribution is a histogram of minutes and its error is MAE. A single uvicorn
worker keeps the Gauges and the prediction cache coherent — the multiprocess
collector is a separate lesson, noted in the README.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from incidents import state
from jobs import metrics_store
from services.ride_model import ARTIFACTS, DATA, FEATURES, true_duration

# psutil is what makes Row 2 of the dashboard real on this stack. The
# prometheus_client ProcessCollector that would normally supply
# process_cpu_seconds_total and process_resident_memory_bytes reads /proc,
# so it silently exports NOTHING on macOS — and node-exporter cannot help:
# it measures the Linux VM, while this API runs on the host. psutil is the
# only thing here that can see the process actually serving predictions.
try:
    import psutil

    _PROCESS: Any = psutil.Process()
except ModuleNotFoundError:  # pragma: no cover - the resource row degrades, nothing else
    psutil = None
    _PROCESS = None

MODEL_NAME = os.getenv("MODEL_NAME", "rf_model")
CHAMPION_VERSION = "v3"
CANARY_VERSION = "v4"

# ══════════════════════════════════════════════════════════════════════
#  Metrics — names fixed by the existing dashboard and alert rules
# ══════════════════════════════════════════════════════════════════════
PREDICTION_HISTOGRAM = Histogram(
    "model_prediction_duration_min",
    "Distribution of predicted ride durations",
    buckets=[0, 5, 10, 15, 20, 30, 45, 60, 90, 120],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "End-to-end API latency", ["endpoint", "status"]
)
DRIFT_GAUGE = Gauge("feature_psi_score", "PSI drift score per feature", ["feature"])
MODEL_VERSION = Gauge("model_version_info", "Active model version", ["version", "stage"])

# ── Added for session 4's incidents, all regression-shaped ────────────
FEEDBACK_TOTAL = Counter("feedback_total", "Late labels received")  # incident 15
SEGMENT_PSI = Gauge(
    "segment_psi_score", "PSI per feature, split by city", ["feature", "city"]
)  # incident 05 — the split that the global gauge above cannot show
SEGMENT_MAE = Gauge("model_mae_minutes", "MAE on late labels, by city", ["city"])
DISK_FREE = Gauge("disk_free_bytes", "Free space on the log volume", ["path"])
DEPLOY_INFO = Gauge("deploy_info", "Unix ts of the last change", ["component", "version"])

# ── Row 1: service. The latency histogram counts only what reaches the
# handler body, so a 422 is invisible to it and an error rate built on it is a
# flat zero forever. This counter is incremented by middleware, outside the
# handler, which is the only place that can see a request the handler rejected.
REQUESTS_TOTAL = Counter(
    "api_requests_total", "Every HTTP request, by outcome", ["endpoint", "status"]
)

# ── Row 2: resources of THIS process (see the psutil note above) ──────
PROCESS_CPU = Gauge("api_process_cpu_percent", "CPU used by the serving process")
PROCESS_MEMORY = Gauge("api_process_memory_bytes", "Process memory", ["type"])
PROCESS_OPEN_FDS = Gauge("api_process_open_fds", "Open file descriptors")
PROCESS_THREADS = Gauge("api_process_threads", "Thread count")

# ── Row 3: model behaviour ────────────────────────────────────────────
VALIDATION_FAILURES = Counter(
    "api_validation_failures_total", "Requests rejected by the schema", ["endpoint", "field"]
)

# ── Row 4: quality & drift. Counters, not gauges, so Grafana can draw an
# annotation the moment one increases — `changes()` on a gauge that is reset to
# the same value on every scrape never fires.
DEPLOY_EVENTS = Counter("deploy_events_total", "Deploys observed", ["component"])
RETRAIN_EVENTS = Counter("retrain_events_total", "Retrain decisions observed", ["decision"])
RETRAIN_LAST = Gauge(
    "retrain_last_timestamp_seconds", "Unix ts of the last decision", ["decision"]
)

#: Registered lazily, only while incident 04 is active — see _record_ride_id().
_ride_id_counter: Counter | None = None
#: Last deploy/retrain timestamps this process has already counted. Both start
#: as None so the FIRST scrape establishes a baseline instead of firing an
#: annotation for every row that was already in the store at boot.
_last_deploy_ts: float | None = None
_seen_retrain_ts: float | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the frozen artifacts and publish the champion version."""
    global _model, _scaler, _scaler_stale
    _model = joblib.load(ARTIFACTS / "rf_model.pkl")
    _scaler = joblib.load(ARTIFACTS / "scaler.pkl")
    _scaler_stale = joblib.load(ARTIFACTS / "scaler_stale.pkl")
    MODEL_VERSION.labels(version=CHAMPION_VERSION, stage="production").set(1)
    yield


app = FastAPI(title="Ride Duration API (session 4)", lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Count every request by its real status — the only honest error rate.

    The endpoint label is the matched ROUTE TEMPLATE, never request.url.path.
    A 404 sweep or a path parameter would otherwise mint one time series per
    URL, which is incident 04 by a different door; unmatched paths collapse to
    a single "unmatched" series on purpose.

    /metrics is skipped. Prometheus scrapes it every 15s, and counting those
    would put a floor of ~4 rpm under the request-rate panel that has nothing
    to do with anyone using the service.
    """
    response = await call_next(request)
    if request.url.path != "/metrics":
        route = request.scope.get("route")
        endpoint = getattr(route, "path", None) or "unmatched"
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=str(response.status_code)).inc()
    return response


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Record WHICH field was rejected, then answer exactly as FastAPI would.

    A bare count of 422s tells you that callers are unhappy. The field name
    tells you which release broke them, and that is the difference between a
    dashboard that raises a question and one that answers it. Field names come
    from the model definition, so the label space is bounded by the schema.
    """
    route = request.scope.get("route")
    endpoint = getattr(route, "path", None) or "unmatched"
    for error in exc.errors():
        field = ".".join(str(part) for part in error.get("loc", ()) if part != "body") or "body"
        VALIDATION_FAILURES.labels(endpoint=endpoint, field=field).inc()
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(exc.errors())})


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape target.

    An explicit route, not ``app.mount("/metrics", make_asgi_app())``. The mount
    is the idiomatic one-liner, but Starlette then answers ``GET /metrics`` with
    a 307 to ``/metrics/``. Prometheus follows redirects so scraping still works
    — but ``curl localhost:8001/metrics`` prints nothing, and that curl is the
    first thing anyone does when a target goes down. Costing a lab twenty minutes
    to save one line is a bad trade.
    """
    _refresh_process_gauges()
    _refresh_disk_gauge()
    _refresh_deploy_gauge()
    _refresh_retrain_metrics()
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


_model: Any = None
_scaler: Any = None
_scaler_stale: Any = None
#: request_id -> (predicted_min, city). Bounded: late labels arrive within hours,
#: not weeks, and an unbounded dict here would be its own incident.
_recent: OrderedDict[str, tuple[float, str]] = OrderedDict()


class PredictRequest(BaseModel):
    """One trip. `city` is a segment label, not a model feature (incident 05)."""

    distance_km: float = Field(gt=0, le=500)
    passengers: int = Field(ge=1, le=8)
    hour_of_day: int = Field(ge=0, le=23)
    city: str = "cairo"
    ride_id: str | None = None


class FeedbackRequest(BaseModel):
    """The actual duration, arriving late — this is what makes MAE possible."""

    request_id: str
    actual_duration_min: float


SERVING_LOG = DATA / "serving_log.jsonl"
_log_buffer: list[str] = []


def _log_serving(row: dict[str, Any]) -> None:
    """Append the exact feature vector the model scored, and what it returned.

    Incident 02 is invisible to every distribution monitor in this repo: the
    inputs are fine, the model is fine, and only the transform between them is
    stale. Nothing about the request looks wrong. The one thing that catches it
    is re-scoring these logged vectors through the offline pipeline and
    comparing (tools/replay.py) — which is what a serving log is FOR.
    """
    _log_buffer.append(json.dumps(row))
    if len(_log_buffer) >= 50:
        DATA.mkdir(parents=True, exist_ok=True)
        with open(SERVING_LOG, "a") as fh:
            fh.write("\n".join(_log_buffer) + "\n")
        _log_buffer.clear()


def _record_ride_id(ride_id: str) -> None:
    """Incident 04: mint one time series per ride. Registered on first use only,
    so the cardinality guard in tests/ passes on a healthy system and fails here."""
    global _ride_id_counter
    if _ride_id_counter is None:
        _ride_id_counter = Counter(
            "model_predictions_by_ride_total", "Predictions per ride id", ["ride_id"]
        )
    _ride_id_counter.labels(ride_id=ride_id).inc()


def _drop_ride_id_metric() -> None:
    """Unregister the exploded metric so `make heal` alone recovers the scrape.

    Removing the label from the code is not enough in a live process: the series
    already registered keep being exported, so the scrape stays over
    sample_limit and the target stays down. Real recovery means dropping the
    collector — which in production is the restart everyone forgets they need.
    """
    global _ride_id_counter
    if _ride_id_counter is not None:
        REGISTRY.unregister(_ride_id_counter)
        _ride_id_counter = None


def _active_version() -> str:
    """Incident 10: a canary that was never completed still splits traffic."""
    if not state.flag("SERVE_SECOND_VERSION"):
        return CHAMPION_VERSION
    MODEL_VERSION.labels(version=CANARY_VERSION, stage="canary").set(1)
    return CANARY_VERSION if uuid.uuid4().int % 2 else CHAMPION_VERSION


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """Predict ride duration in minutes and record the four teaching metrics."""
    t0 = time.perf_counter()
    version = _active_version()

    # Incident 02: the preprocessing from before the last retrain. Same model,
    # same inputs, different transform — and nothing in the request looks wrong.
    scaler = _scaler_stale if state.flag("USE_STALE_SCALER") else _scaler
    x = np.array([[req.distance_km, float(req.passengers), float(req.hour_of_day)]])
    minutes = float(_model.predict(scaler.transform(x))[0])
    if version == CANARY_VERSION:
        minutes *= 1.09  # the canary is a different fit, not a different model

    request_id = req.ride_id or uuid.uuid4().hex
    _recent[request_id] = (minutes, req.city)
    while len(_recent) > 20_000:
        _recent.popitem(last=False)

    _log_serving(
        {
            "request_id": request_id,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "distance_km": req.distance_km,
            "passengers": req.passengers,
            "hour_of_day": req.hour_of_day,
            "city": req.city,
            "served_min": minutes,
            "version": version,
        }
    )
    PREDICTION_HISTOGRAM.observe(minutes)
    REQUEST_LATENCY.labels(endpoint="/predict", status="200").observe(time.perf_counter() - t0)
    if state.flag("ENABLE_RIDE_ID_LABEL"):
        _record_ride_id(request_id)
    elif _ride_id_counter is not None:
        _drop_ride_id_metric()

    return {"duration_min": round(minutes, 2), "model_version": version, "request_id": request_id}


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict[str, str]:
    """Accept a late label. Incident 15 makes this silently do nothing."""
    if state.flag("FEEDBACK_DEAD"):
        return {"status": "accepted"}  # the lie that keeps every dashboard green
    predicted, city = _recent.get(req.request_id, (None, "cairo"))
    if predicted is None:
        return {"status": "unknown request_id"}
    FEEDBACK_TOTAL.inc()
    metrics_store.record_labels(
        [
            (
                req.request_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                city,
                req.actual_duration_min,
                predicted,
            )
        ]
    )
    return {"status": "recorded"}


@app.post("/internal/gauges")
def publish_gauges(payload: dict[str, Any]) -> dict[str, int]:
    """Publish drift/error numbers computed by jobs/daily_drift.py.

    P_G_monitoring.py's record_drift_scores() assumes the drift check runs
    in-process. It does not — it is a separate job — and adding a Pushgateway
    would mean a seventh container. So the job posts its numbers to the one
    process Prometheus already scrapes.
    """
    n = 0
    for feature, value in (payload.get("feature_psi") or {}).items():
        DRIFT_GAUGE.labels(feature=feature).set(float(value))
        n += 1
    for row in payload.get("segment_psi") or []:
        SEGMENT_PSI.labels(feature=row["feature"], city=row["city"]).set(float(row["value"]))
        n += 1
    for row in payload.get("mae") or []:
        SEGMENT_MAE.labels(city=row["city"]).set(float(row["value"]))
        n += 1
    return {"published": n}


def _refresh_deploy_gauge() -> None:
    """Publish the most recent deploy so Grafana can draw the annotation.

    RUNBOOK step 3 is "what changed, and when" — and the answer is almost always
    a deploy. The deploys table holds it, but Grafana's annotation layer reads
    time series, so the latest row is republished here as an info metric whose
    VALUE is the deploy's unix timestamp.

    .clear() first: a new deploy means new label values, and without it every
    past version would linger as its own series forever.
    """
    global _last_deploy_ts
    rows = metrics_store.latest_deploys(limit=1)
    if not rows:
        return
    row = rows[0]
    ts = datetime.fromisoformat(row["ts"]).timestamp()
    DEPLOY_INFO.clear()
    DEPLOY_INFO.labels(component=row["component"], version=row["version"]).set(ts)

    # DEPLOY_INFO alone cannot drive an annotation. .clear() above means it
    # holds only the newest deploy, so its VALUE changes but it never "appears"
    # — and Grafana draws a Prometheus annotation where a series increases.
    # This counter is that signal: one increment per newly observed deploy.
    if _last_deploy_ts is not None and ts > _last_deploy_ts:
        DEPLOY_EVENTS.labels(component=row["component"]).inc()
    _last_deploy_ts = ts


def _refresh_retrain_metrics() -> None:
    """Republish retrain DECISIONS from SQLite as counters Grafana can annotate.

    Retrain history lives in the store, not in Prometheus, and it cannot simply
    be replayed: a decision made on Tuesday would be recorded at today's scrape
    timestamp and draw its line in the wrong place. So only decisions this
    process has not seen yet are counted, which puts the annotation at the
    moment the gate actually ran — forward-looking, like every real deploy
    annotation. The seeded backlog sets the baseline without inventing lines.
    """
    global _seen_retrain_ts
    rows = metrics_store.latest_retrain_events(limit=50)
    if not rows:
        return
    first_pass = _seen_retrain_ts is None
    if first_pass:
        _seen_retrain_ts = 0.0
    for row in reversed(rows):  # oldest first, so the counter advances in order
        # Compared as instants, not as strings. Every row in the store today is
        # UTC-aware and fixed width, so a string compare would work — right up
        # until something writes a naive timestamp and this silently stops
        # counting. A missing annotation is not an error anyone would notice.
        ts = datetime.fromisoformat(row["ts"]).timestamp()
        if ts <= _seen_retrain_ts:
            continue
        _seen_retrain_ts = ts
        decision = row["decision"]
        RETRAIN_LAST.labels(decision=decision).set(ts)
        if not first_pass:
            RETRAIN_EVENTS.labels(decision=decision).inc()


def _refresh_process_gauges() -> None:
    """Sample CPU, memory, fds and threads for the process serving predictions.

    Row 2 of the dashboard is about saturation showing up BEFORE users feel it:
    memory climbing across a week is a leak you can schedule, memory climbing
    into an OOM kill at 3am is an incident. Sampled here for the same reason as
    the disk gauge — nothing else would ever move these numbers.

    cpu_percent(interval=None) is deliberate: it reports usage since the PREVIOUS
    call rather than blocking. Passing an interval would sleep the scrape, and a
    /metrics endpoint that blocks is a monitoring system that causes outages.
    """
    if _PROCESS is None:  # psutil not installed — the rest of the scrape is fine
        return
    with _PROCESS.oneshot():
        PROCESS_CPU.set(_PROCESS.cpu_percent(interval=None))
        memory = _PROCESS.memory_info()
        PROCESS_MEMORY.labels(type="rss").set(float(memory.rss))
        PROCESS_MEMORY.labels(type="vms").set(float(memory.vms))
        PROCESS_THREADS.set(float(_PROCESS.num_threads()))
        try:
            PROCESS_OPEN_FDS.set(float(_PROCESS.num_fds()))
        except (AttributeError, psutil.AccessDenied):  # not available on Windows
            pass


def _refresh_disk_gauge() -> None:
    """Recompute free space. Called on every SCRAPE, not just on /health.

    A gauge that is only refreshed by an endpoint nobody scrapes is a flat line
    that looks like a healthy disk forever. Prometheus reads /metrics, so the
    reading has to happen there.
    """
    usage = shutil.disk_usage(Path(__file__).resolve().parents[1])
    free = float(usage.free)
    if state.flag("SIMULATE_DISK_FILL"):
        # Simulated, not real: filling a student's laptop mid-session is not a
        # lesson, it is a support ticket. The gauge falls linearly, which is all
        # predict_linear needs to extrapolate a time-to-full.
        started = state.read_state().get("updated")
        if started:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
            free = max(0.0, free - elapsed * (usage.free / 3600.0) * 0.22)
    DISK_FREE.labels(path="/var/log").set(free)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness, and the version actually serving right now."""
    _refresh_disk_gauge()
    return {"status": "ok", "model": MODEL_NAME, "version": _active_version()}


def offline_score(distance_km: float, passengers: int, hour_of_day: int) -> float:
    """Score with the CURRENT preprocessing — the offline pipeline's answer.

    tools/replay.py compares this against what the API returned. When they differ
    on identical inputs, the difference is the serving path (incident 02).
    """
    x = np.array([[distance_km, float(passengers), float(hour_of_day)]])
    return float(_model.predict(_scaler.transform(x))[0])


__all__ = ["app", "FEATURES", "true_duration", "offline_score"]
