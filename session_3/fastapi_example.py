"""
Plain FastAPI, annotated — the CONTROL in the serving benchmark.

This file deliberately serves the SAME model, with the SAME response contract,
as `bentoml_example.py`. The only difference is everything a serving framework
would have given you for free. Each of those is marked below with the same
`# [N] …` numbering `bentoml_example.py` uses, so the two files read side by
side:

    1. Dynamic batching      NOT HERE — batch size is always 1, provably
    2. Request queueing      NOT HERE — no admission control, no queue timeout
    3. Pre/post-processing   present — pydantic in, derived fields out
    4. Model versioning      BARELY HERE — a string resolved once at startup
    5. Concurrency control   present, but only as "whatever the threadpool does"

The point of running both is that [1], [2] and [4] stop being slide bullets and
become numbers: hit `/metrics` after a load test and `avg_batch_size` is still
`1.0`, no matter how many concurrent callers you threw at it.

----------------------------------------------------------------------------
RUN IT
----------------------------------------------------------------------------
  # optional: pull the model from the registry rather than the local pickle
  export MLFLOW_TRACKING_URI=http://localhost:5000

  uvicorn fastapi_example:app --port 8000
  # ...or just: python fastapi_example.py

  curl -X POST localhost:8000/predict \
    -H 'content-type: application/json' \
    -d '{"distance_km": 12.5, "passengers": 2}'
  # → {"duration_min":25.98,"eta_band":"20-30 min","model_version":"...",
  #    "batch_size":1}

  curl localhost:8000/health
  curl localhost:8000/model_info
  curl localhost:8000/metrics
  open http://localhost:8000/docs

NOTE ON THE WIRE FORMAT: unlike the BentoML service, `/predict` here takes and
returns a FLAT object — there is no `{"inputs": [...]}` envelope and no
one-element array, because there is no batching to address. The *fields* are
identical, so one client can parse either server's rows; only the envelope
differs. `locustfile.py` encodes exactly that difference.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.train import FEATURES

#: Alias, not stage. src/register_model.py moves @production onto each version.
REGISTRY_URI = "models:/RideDurationModel@production"
REGISTRY_MODEL_NAME = "RideDurationModel"
PRODUCTION_ALIAS = "production"
PICKLE_PATH = Path(__file__).resolve().parent / "models" / "rf_model.pkl"

# [3] PRE-PROCESSING — the training feature list is the source of truth for the
# request schema below. Asserting it here means a rename in src/train.py breaks
# this file at import time, loudly, instead of at request time, silently.
assert FEATURES == ["distance_km", "passengers"], (
    f"src/train.py FEATURES changed to {FEATURES}; update PredictRequest to match"
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — schemas: identical fields to bentoml_example.py, flat envelope
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """[3] PRE-PROCESSING, part 1 — validation at the edge.

    Field constraints turn bad input into a 422 with a readable message *before*
    it reaches sklearn, instead of a 500 or (worse) a confident garbage
    prediction. The names are the TRAINING names — `distance_km`, not a
    friendlier `distance`. Renaming a field at the serving boundary is how
    training/serving schema drift gets introduced, and it is the single most
    common way a "working" model breaks in production. Don't be nice here; be
    identical.
    """

    distance_km: float = Field(gt=0, le=500, description="trip distance in km")
    passengers: int = Field(ge=1, le=8, description="passenger count")


class PredictResponse(BaseModel):
    """[3] POST-PROCESSING — byte-identical to the BentoML service's response.

    Same four fields, same types, same rounding. That is what lets one Locust
    scenario validate either server (see `locustfile.py`).
    """

    duration_min: float
    eta_band: str = Field(description="human-readable bucket, derived server-side")
    model_version: str = Field(description="[4] which artifact produced this")
    batch_size: int = Field(description="[1] how many requests were merged")


class ModelInfo(BaseModel):
    name: str
    version: str
    n_features: int
    features: list[str]


class Metrics(BaseModel):
    """What the benchmark actually reads back out of this server."""

    requests: int
    avg_batch_size: float
    avg_predict_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — mutable server state, and the lock that makes it safe
# ─────────────────────────────────────────────────────────────────────────────
# [5] CONCURRENCY CONTROL — `def` handlers (see /predict) run in Starlette's
# anyio threadpool, so several requests mutate these counters *at the same
# time*. `n += 1` is not atomic under the GIL: it is LOAD, ADD, STORE, and a
# thread switch between the load and the store loses an increment. A plain
# threading.Lock costs ~100ns and removes the whole class of bug.
_lock = threading.Lock()
_requests = 0
_batched_items = 0
_batches = 0
_predict_s = 0.0


def _band(minutes: float) -> str:
    """[3] Pure post-processing helper — identical to the BentoML service's."""
    lo = int(minutes // 10) * 10
    return f"{lo}-{lo + 10} min"


def _load_model() -> tuple[object, str]:
    """Load the trained model — registry first, local pickle second.

    Mirrors `bentoml_example.save_model()`. The preference order is the lesson:
    the MLflow registry is the answer to "which model is in production?" that
    survives someone's laptop. The DVC-tracked pickle is the offline fallback so
    this file still runs with no tracking server up.

    Returns:
        ``(model, version_string)`` — the string is what [4] stamps on every
        response so a caller can always tell which bytes answered them.
    """
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow.sklearn
            from mlflow import MlflowClient

            model = mlflow.sklearn.load_model(REGISTRY_URI)
            # [4] MODEL VERSIONING — resolve the alias to a concrete version
            # number. "@production" is a moving pointer; the number is not, and
            # the number is what you want stamped on a prediction you have to
            # explain three weeks later.
            version = MlflowClient().get_model_version_by_alias(
                REGISTRY_MODEL_NAME, PRODUCTION_ALIAS
            ).version
            source = f"{REGISTRY_MODEL_NAME}@{PRODUCTION_ALIAS}/{version}"
            print(f"loaded from MLflow registry: {REGISTRY_URI} -> v{version}")
            return model, source
        except Exception as exc:  # server down, alias missing, auth, …
            print(f"MLflow unavailable ({type(exc).__name__}), using local artifact")

    model = joblib.load(PICKLE_PATH)
    print(f"loaded from {PICKLE_PATH}")
    # [4] MODEL VERSIONING — and here is the honest version of "no versioning".
    # A filename is not a version. Two people with different pickles both report
    # `pickle:rf_model.pkl` and there is no way to tell their predictions apart.
    # BentoML mints an immutable content-addressed tag per save; this does not.
    return model, f"pickle:{PICKLE_PATH.name}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE per process, at startup — not per request.

    Loading inside the handler is the classic way to make a fast model look
    slow: a 40ms prediction behind a 900ms joblib.load. Anything expensive and
    reusable — model, tokenizer, DB pool — belongs here.
    """
    model, version = _load_model()
    app.state.model = model
    app.state.version = version
    app.state.model.predict(np.array([[5.0, 1]]))  # warm-up: first call is slowest
    yield


app = FastAPI(
    title="Ride Duration API — plain FastAPI baseline",
    version="0.1.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — the endpoint
# ─────────────────────────────────────────────────────────────────────────────
# [5] CONCURRENCY CONTROL — note `def`, NOT `async def`.
#   sklearn's predict() is blocking, CPU-bound C code. In an `async def` handler
#   it would run ON the event loop and stall every other connection for its
#   whole duration — the server would stop *accepting* sockets, not just stop
#   answering. Declaring it `def` makes FastAPI hand it to the anyio threadpool
#   (40 threads by default) and the loop stays free.
#   That is the whole of this server's concurrency story: a fixed thread pool
#   and the OS scheduler. Compare bentoml_example.py's `traffic={"concurrency":
#   64, "timeout": 30}` — an explicit admission limit and an explicit deadline.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """One request in, one model call out.

    Read this next to `RideDuration.predict` in bentoml_example.py. That one
    receives `list[PredictRequest]` and makes ONE vectorised call for the whole
    merged batch. This one receives exactly one row and makes one call for it.
    """
    global _requests, _batched_items, _batches, _predict_s

    # [3] PRE-PROCESSING, part 2 — typed object → the exact feature matrix the
    # model was trained on. Column ORDER is part of the contract; get it wrong
    # and sklearn happily returns nonsense with no error at all. FEATURES is
    # ["distance_km", "passengers"], so that is the order.
    #
    # [1] NO DYNAMIC BATCHING — this is the finding, not an oversight.
    # The array below is shape (1, 2) on every single request, no matter how
    # many callers are in flight. Per-call Python and serialisation overhead
    # dominates a RandomForest's actual compute, so every concurrent caller pays
    # that overhead separately instead of amortising it over a merged batch.
    # `avg_batch_size` on /metrics will read 1.00 forever; that is the number to
    # put next to BentoML's, which climbs to 16+ under the same load.
    features = np.array([[req.distance_km, req.passengers]], dtype=float)

    t0 = time.perf_counter()
    pred = float(app.state.model.predict(features)[0])
    dt = time.perf_counter() - t0

    # [2] NO REQUEST QUEUEING / ADMISSION CONTROL — also the finding.
    # There is no bound on how many requests may be in flight and no deadline
    # after which a waiting request is abandoned. Past ~40 concurrent callers
    # the extras queue invisibly inside anyio's threadpool with no timeout, so
    # under sustained overload every client eventually times out *after* the
    # server has already paid for the work. BentoML's `traffic.timeout` sheds
    # that load with a 504 instead. Bound the queue, or the queue is unbounded.

    with _lock:  # [5] see the note on _lock above
        _requests += 1
        _batches += 1
        _batched_items += 1  # always 1. That is the finding.
        _predict_s += dt

    # [3] POST-PROCESSING — rounding, derived fields, and stamping the version.
    # Doing this server-side means every client agrees on what an "eta band" is
    # instead of each reimplementing it.
    return PredictResponse(
        duration_min=round(pred, 2),
        eta_band=_band(pred),
        model_version=app.state.version,  # [4]
        batch_size=1,  # [1] observable proof batching did NOT happen
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe.

    Named `/health` because that is this server's convention. BentoML does NOT
    serve `/health` — it serves `/healthz`, `/readyz` and `/livez`. Assuming the
    two agree is a 404 and a red bar in Locust; `locustfile.py` keeps the path
    per-target for exactly this reason.
    """
    return {"status": "ok", "model_version": app.state.version}


@app.get("/model_info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    """[4] Ask a running service exactly what it is serving.

    Worth exposing in real deployments: when predictions look wrong, the first
    question is always "which version is actually live?". Note this is a GET;
    the BentoML sibling is a POST, because every `@bentoml.api` is a POST.
    """
    return ModelInfo(
        name="ride_duration",
        version=app.state.version,
        n_features=int(app.state.model.n_features_in_),
        features=FEATURES,
    )


@app.get("/metrics", response_model=Metrics)
def metrics() -> Metrics:
    """The benchmark's read-out. Run a load test, then read this.

    `avg_batch_size` is the whole exhibit: it is 1.00 here under any load, and
    climbs on the BentoML service under the same load. That single number is the
    difference between "a model behind HTTP" and "a serving framework".
    """
    with _lock:  # [5] read a consistent snapshot, not a torn one
        requests_, batches, items, predict_s = (
            _requests,
            _batches,
            _batched_items,
            _predict_s,
        )
    return Metrics(
        requests=requests_,
        avg_batch_size=round(items / max(batches, 1), 2),
        avg_predict_ms=round(predict_s / max(requests_, 1) * 1000, 3),
    )


if __name__ == "__main__":
    import uvicorn

    # Single worker on purpose: the counters above live in THIS process's
    # memory, so `--workers 4` would give you four disjoint /metrics views.
    # Shared metrics across workers need a shared store — another thing the
    # baseline does not have.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
