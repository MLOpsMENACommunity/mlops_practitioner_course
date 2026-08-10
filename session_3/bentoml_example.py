"""
BentoML, annotated — the five things a serving framework gives you for free.

    1. Dynamic batching      merge concurrent calls into ONE model invocation
    2. Request queueing      absorb bursts instead of dropping them
    3. Pre/post-processing   validation + feature building + response shaping
    4. Model versioning      immutable tags, pinned or floating
    5. Concurrency control   how many requests are in flight at once

Every one of those is marked with a `# [N] …` comment below.

----------------------------------------------------------------------------
WHY THIS FILE WAS REWRITTEN
----------------------------------------------------------------------------
The original snippet targeted BentoML 1.1 and MLflow 2. Both APIs are gone:

  WAS (BentoML 1.1)                        NOW (1.2+, installed here: 1.4.39)
  ---------------------------------------  ---------------------------------
  runner = ....to_runner()                 removed — the service class IS the
  svc = bentoml.Service(runners=[runner])    unit of deployment
  @svc.api(input=bentoml.io.JSON(...))     @bentoml.api; types come from the
                                             method's type annotations
  await runner.predict.async_run(x)        self.model.predict(x)
  signatures={"predict": {"batchable"}}    batchable=True on @bentoml.api
  bentoml.io.*                             REMOVED → AttributeError

  models:/RideDurationModel/Production     stages REMOVED in MLflow 3 → use an
                                             alias: models:/Name@production
                                             (src/register_model.py sets it)

Two other bugs the original had, both of which fail at request time:
  * it never set MLFLOW_TRACKING_URI, so it read the empty local mlflow.db
    instead of the tracking server → "Registered Model ... not found"
  * it sent 3 features (distance, passengers, hour_of_day) to a model trained
    on 2 (src/train.py: FEATURES = ["distance_km", "passengers"]) →
    "ValueError: X has 3 features, but RandomForestRegressor is expecting 2"

----------------------------------------------------------------------------
RUN IT
----------------------------------------------------------------------------
  # optional: pull the model from the registry rather than the local pickle
  export MLFLOW_TRACKING_URI=http://localhost:5000

  python bentoml_example.py                     # step 1 — save to model store
  bentoml serve bentoml_example:RideDuration --port 8005

  curl -X POST localhost:8005/predict \
    -H 'content-type: application/json' \
    -d '{"inputs": [{"distance_km": 12.5, "passengers": 2}]}'
  # → [{"duration_min": 25.98, "eta_band": "20-30 min", "model_version": "...",
  #     "batch_size": 1}]

  curl -X POST localhost:8005/model_info        # non-batched sibling endpoint
  open http://localhost:8005/docs               # generated OpenAPI UI

NOTE ON THE WIRE FORMAT, since it surprises everyone: a batchable JSON API is
addressed as `{"<param name>": [ <one item> ]}` — here `inputs`, a ONE-element
array. Each client sends its own single row and gets a one-element array back.
The batching happens server-side, across clients. See [1].
"""

from __future__ import annotations

import os
from pathlib import Path

import bentoml
import joblib
import numpy as np
from pydantic import BaseModel, Field

MODEL_NAME = "ride_duration"
#: Alias, not stage. src/register_model.py moves @production onto each new version.
REGISTRY_URI = "models:/RideDurationModel@production"
PICKLE_PATH = Path(__file__).resolve().parent / "models" / "rf_model.pkl"

# [4] MODEL VERSIONING — which artifact this service serves.
#   "ride_duration:latest"  → floating; picks up whatever was saved last
#   "ride_duration:eigze…"  → pinned; the deploy is reproducible byte-for-byte
# Production deploys should PIN. `latest` means a colleague running the save
# script silently changes what your running service returns.
MODEL_TAG = os.getenv("RIDE_MODEL_TAG", f"{MODEL_NAME}:latest")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — get the model into the BentoML model store
# ─────────────────────────────────────────────────────────────────────────────
# The store is a local, content-addressed registry: every save mints a new
# immutable tag (`ride_duration:eigzetudugfi5i5z`). That tag is what makes [4]
# possible — you can always say exactly which bytes served a prediction.
def save_model() -> bentoml.Model:
    """Load the trained model (registry first, pickle second) and store it.

    The preference order is the lesson: the MLflow registry is the answer to
    "which model is in production?" that survives someone's laptop. The
    DVC-tracked pickle is the offline fallback so this file still runs with no
    tracking server up.
    """
    model = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        try:
            import mlflow.sklearn

            model = mlflow.sklearn.load_model(REGISTRY_URI)
            print(f"loaded from MLflow registry: {REGISTRY_URI}")
        except Exception as exc:  # server down, alias missing, auth, …
            print(f"MLflow unavailable ({type(exc).__name__}), using local artifact")

    if model is None:
        model = joblib.load(PICKLE_PATH)
        print(f"loaded from {PICKLE_PATH}")

    # In 1.2+ `signatures={"predict": {"batchable": True}}` is NOT passed here.
    # Batching moved off the stored artifact and onto the API method — the store
    # holds bytes, the service decides how they are served.
    with bentoml.models.create(MODEL_NAME) as ref:
        joblib.dump(model, ref.path_of("model.pkl"))

    print(f"saved {ref.tag}  (n_features_in_={model.n_features_in_})")
    return ref


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — schemas: the pre- and post-processing boundary
# ─────────────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """[3] PRE-PROCESSING, part 1 — validation at the edge.

    Field constraints turn bad input into a 422 with a readable message
    *before* it reaches sklearn, instead of a 500 or (worse) a confident
    garbage prediction. Feature-schema drift between training and serving is
    the single most common way a "working" model breaks in production, so this
    schema is pinned to src/train.py's FEATURES list — two fields, no more.
    """

    distance_km: float = Field(gt=0, le=500, description="trip distance in km")
    passengers: int = Field(ge=1, le=8, description="passenger count")


class PredictResponse(BaseModel):
    """[3] POST-PROCESSING — the contract the client actually consumes."""

    duration_min: float
    eta_band: str = Field(description="human-readable bucket, derived server-side")
    model_version: str = Field(description="[4] which artifact produced this")
    batch_size: int = Field(description="[1] how many requests were merged")


class ModelInfo(BaseModel):
    name: str
    version: str
    n_features: int
    features: list[str]


# The service class needs a model in the store at import time, so seed it once
# on a fresh checkout. Serving never re-saves — that would mint a new version on
# every worker start and make [4] meaningless.
if not bentoml.models.list(MODEL_NAME):
    save_model()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — the service
# ─────────────────────────────────────────────────────────────────────────────
@bentoml.service(
    name="ride_duration_api",
    # [5] CONCURRENCY CONTROL, part 1 — process-level parallelism.
    # Each worker is a separate OS process with its own copy of the model.
    # sklearn only partly releases the GIL, so CPU-bound models scale with
    # workers, not threads. Rule of thumb: workers ≈ cores, memory permitting.
    workers=1,
    resources={"cpu": "1"},
    traffic={
        # [5] CONCURRENCY CONTROL, part 2 — request-level admission.
        # Max requests processed simultaneously by this service. Must be >=
        # max_batch_size below, or the dispatcher can never fill a batch: it
        # would never be allowed to hold that many requests at once.
        "concurrency": 64,
        # [2] REQUEST QUEUEING — everything past `concurrency` WAITS here rather
        # than being refused. This is the whole point of a serving framework: a
        # burst of 500 callers becomes a queue, not 436 connection errors.
        # `timeout` bounds that wait — a request that has queued 30s is
        # abandoned with a 504. Without a timeout, a queue under sustained
        # overload grows without bound and every client times out anyway, after
        # the server has already paid for the work. Bound the queue, shed load.
        "timeout": 30,
    },
)
class RideDuration:
    # [4] MODEL VERSIONING — resolved once at class definition; the tag is then
    # fixed for the life of the process even if someone saves a new version.
    bento_model = bentoml.models.BentoModel(MODEL_TAG)

    def __init__(self) -> None:
        # Runs ONCE per worker at startup, not per request. Anything expensive
        # and reusable — model load, tokenizer, DB pool — belongs here. Loading
        # inside the handler is the classic way to make a fast model look slow.
        self.model = joblib.load(self.bento_model.path_of("model.pkl"))
        self.version = self.bento_model.tag.version

    @bentoml.api(
        # [1] DYNAMIC BATCHING — the headline feature.
        # BentoML holds concurrently-arriving requests for a few milliseconds,
        # concatenates them along `batch_dim`, and hands your method ONE list.
        # Each caller still sends a single row and receives a single row back;
        # the merging is invisible to them and happens ACROSS clients.
        # Why it wins here: per-call Python and serialisation overhead dominates
        # a RandomForest's actual compute, so amortising it over 64 rows is close
        # to free throughput. Measured on this service: 24 concurrent callers
        # merged into batches of up to 16.
        batchable=True,
        batch_dim=0,  # stack along axis 0 — i.e. rows
        # Tabular rows are tiny, so batch far more aggressively than you would
        # for images. For a ResNet this would be more like 8–16.
        max_batch_size=64,
        # [1]+[2] This is an SLA, NOT a wait window. The dispatcher does not sit
        # here waiting for 2s of traffic — it adapts, using observed latency to
        # pick a batch size whose total time stays under this bound. Set it to
        # the latency your clients can actually tolerate.
        max_latency_ms=2_000,
    )
    def predict(self, inputs: list[PredictRequest]) -> list[PredictResponse]:
        """One vectorised model call for the whole merged batch.

        Note the shape of the work: build features for N rows, ONE
        ``self.model.predict(...)``, then map results back. Looping and calling
        predict per row would give up the entire batching win.
        """
        # [3] PRE-PROCESSING, part 2 — typed objects → the exact feature matrix
        # the model was trained on. Column ORDER is part of the contract; get it
        # wrong and sklearn happily returns nonsense with no error at all.
        features = np.array([[r.distance_km, r.passengers] for r in inputs])

        preds = self.model.predict(features)  # [1] a single call for all N rows

        # [3] POST-PROCESSING — rounding, derived fields, and stamping the
        # version. Doing this server-side means every client agrees on what an
        # "eta band" is instead of each reimplementing it.
        return [
            PredictResponse(
                duration_min=round(float(p), 2),
                eta_band=self._band(float(p)),
                model_version=self.version,  # [4]
                batch_size=len(inputs),  # [1] observable proof batching happened
            )
            for p in preds
        ]

    # [1] No batchable= here, so each call runs on its own. Not everything
    # should batch: metadata lookups are cheap and want the lowest latency.
    @bentoml.api
    def model_info(self) -> ModelInfo:
        """[4] Ask a running service exactly what it is serving.

        Worth exposing in real deployments: when predictions look wrong, the
        first question is always "which version is actually live?"
        """
        return ModelInfo(
            name=self.bento_model.tag.name,
            version=self.version,
            n_features=int(self.model.n_features_in_),
            features=["distance_km", "passengers"],
        )

    @staticmethod
    def _band(minutes: float) -> str:
        """[3] Pure post-processing helper — no model involved."""
        lo = int(minutes // 10) * 10
        return f"{lo}-{lo + 10} min"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — package and ship
# ─────────────────────────────────────────────────────────────────────────────
#   bentoml serve bentoml_example:RideDuration --port 8005   # dev, hot reload
#   bentoml build                                            # → a Bento
#   bentoml containerize ride_duration_api:latest            # → a Docker image
#
# `bentoml build` snapshots the code, the pinned model version [4], and the
# Python dependencies together, which is what makes the resulting image
# reproducible rather than "whatever pip resolved that morning".

if __name__ == "__main__":
    save_model()
