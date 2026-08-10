"""FastAPI inference service for the ride-duration model.

Session 2 trains a RandomForest and tracks it with MLflow; this module is the
*serving* side — it loads a fitted model and answers prediction requests over
HTTP. It's what the Docker image runs and what the CI/CD pipeline health-checks.

Endpoints:
    GET  /health   → liveness probe (used by the Docker HEALTHCHECK and CI)
    POST /predict  → ride-duration prediction from [distance_km, passengers]

The model file is chosen by the ``MODEL_PATH`` env var, so the same image can
serve a different model version without rebuilding. It defaults to the artifact
produced by the DVC ``train`` stage (``src/train.py``) / ``mflow_example.py``.

Run locally:
    MODEL_PATH=models/rf_model.pkl uvicorn serve:app --port 8000
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Where to load the pickled model from. Overridable so the container can point
# at a mounted/baked model without a code change.
MODEL_PATH = os.getenv("MODEL_PATH", "models/rf_model.pkl")

app = FastAPI(title="Ride Duration API", version="0.1.0")

# Load the model ONCE at startup (import time), not per request — unpickling a
# RandomForest on every call would be needlessly slow. If the file is missing or
# corrupt this raises immediately, so the container fails fast instead of
# serving a broken endpoint.
model = joblib.load(MODEL_PATH)


class RideRequest(BaseModel):
    """One ride to predict. Field constraints double as input validation:
    FastAPI returns HTTP 422 automatically if they're violated."""

    distance_km: float = Field(..., gt=0, description="Trip distance in km")
    passengers: int = Field(..., ge=1, le=6, description="Number of passengers")


class RideResponse(BaseModel):
    """The predicted trip duration in minutes."""

    duration_min: float


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns 200 as long as the process is up and the model
    was loaded (import would have failed otherwise)."""
    return {"status": "ok"}


@app.post("/predict", response_model=RideResponse)
def predict(req: RideRequest) -> RideResponse:
    """Predict ride duration for a single request.

    The feature column order MUST match training: ``src.train.FEATURES`` is
    ``["distance_km", "passengers"]`` — get this wrong and the model silently
    returns garbage, so we build the row explicitly in that order.
    """
    x = np.array([[req.distance_km, req.passengers]], dtype=float)
    prediction = float(model.predict(x)[0])
    return RideResponse(duration_min=prediction)
