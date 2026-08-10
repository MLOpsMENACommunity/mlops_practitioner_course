"""End-to-end integration tests for the FastAPI serving app (``serve.py``).

``serve.py`` loads the model at *import time* (``model = joblib.load(MODEL_PATH)``),
so importing it normally requires a real trained ``.pkl`` on disk. To keep these
tests hermetic — no DVC pipeline run, no model artifact needed — we patch
``joblib.load`` to return a stub model *before* importing ``serve``, then drive
the real app through FastAPI's ``TestClient``.

This tests the actual HTTP contract (status codes, validation, response shape),
not the model's numerical accuracy — the stub returns a fixed prediction.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Stub model whose .predict() returns a fixed value. serve.py does
    # `float(model.predict(x)[0])`, so a 1-element sequence is enough.
    stub = MagicMock()
    stub.predict.return_value = [15.5]

    # Ensure a clean import so the patched joblib.load is what runs at
    # serve.py's module top-level (drop any cached copy from another test).
    sys.modules.pop("serve", None)

    # Patch joblib.load for the duration of the import; the app object it
    # builds keeps the stub model even after the patch context exits.
    with patch("joblib.load", return_value=stub):
        import serve

    with TestClient(serve.app) as c:
        yield c


# ── Health probe ───────────────────────────────────────
def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ── Happy path ─────────────────────────────────────────
def test_predict_success(client):
    resp = client.post("/predict", json={"distance_km": 5.0, "passengers": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["duration_min"] == 15.5
    assert isinstance(body["duration_min"], float)


# ── Response contract ──────────────────────────────────
def test_response_schema(client):
    resp = client.post("/predict", json={"distance_km": 5.0, "passengers": 2})
    assert set(resp.json().keys()) == {"duration_min"}


# ── Validation errors (Field constraints in serve.py) ──
# distance_km > 0, 1 <= passengers <= 6; missing fields all yield HTTP 422.
@pytest.mark.parametrize(
    "payload",
    [
        {"distance_km": -1.0, "passengers": 2},  # distance must be > 0
        {"distance_km": 0.0, "passengers": 2},  # distance must be > 0 (not >=)
        {"distance_km": 5.0, "passengers": 0},  # passengers must be >= 1
        {"distance_km": 5.0, "passengers": 7},  # passengers must be <= 6
        {"passengers": 2},  # missing distance_km
        {"distance_km": 5.0},  # missing passengers
        {},  # empty body
    ],
)
def test_predict_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
