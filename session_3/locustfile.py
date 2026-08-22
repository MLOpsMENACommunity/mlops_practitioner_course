"""
Load-test scenario for the ride-duration model — ONE scenario, TWO servers.

Locust only GENERATES traffic. It never starts your API. If every request
fails, start there: is anything actually listening on the port the user class
targets? See README section 3, "Step 1 — start a server."

----------------------------------------------------------------------------
START A SERVER FIRST (separate terminal, one per target)
----------------------------------------------------------------------------
  # optional, for both: load from the registry instead of models/rf_model.pkl
  export MLFLOW_TRACKING_URI=http://localhost:5000

  # target A — plain FastAPI baseline, port 8000
  uvicorn fastapi_example:app --port 8000

  # target B — BentoML service, port 8005
  bentoml serve bentoml_example:RideDuration --port 8005

----------------------------------------------------------------------------
THEN PICK A TARGET BY NAMING ITS CLASS
----------------------------------------------------------------------------
  locust -f locustfile.py --list                    # what's in here
  locust -f locustfile.py FastAPIBaselineUser       # web UI on :8089
  locust -f locustfile.py BentoMLUser --headless -u 20 -r 10 -t 20s
  locust -f locustfile.py --class-picker             # or choose in the browser

Each class sets its own `host`, so you do NOT pass `--host`. That class
attribute is what removes the "You must specify the base host" startup error.

⚠️  RUNNING WITH NO CLASS NAME RUNS **BOTH** CLASSES AT ONCE.
    Locust then applies a single host to all of them, so one of the two is
    pointed at the wrong server and speaks the wrong wire format — roughly half
    your requests come back 422 and the run is meaningless. Always name exactly
    one class (or use --class-picker).
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Payloads are drawn INSIDE both servers' pydantic constraints (distance_km
# gt=0 le=500, passengers ge=1 le=8) and inside the range src/train.py actually
# trained on (distance 0.5–30 km, 1–4 passengers). That matters: it means a 422
# is always a genuine contract mismatch and never an unlucky random draw.
MIN_DISTANCE_KM, MAX_DISTANCE_KM = 0.5, 30.0
MIN_PASSENGERS, MAX_PASSENGERS = 1, 4

# The model is distance/0.5 + passengers*0.5 (+ noise), so a 30 km ride tops out
# near 62 minutes. Anything past this ceiling is not "slow traffic", it is a
# broken model or the wrong artifact loaded — a 200 that is still a bug.
MAX_PLAUSIBLE_MIN = 600.0


def _ride() -> dict:
    """One random ride, in TRAINING field names. Never rename these."""
    return {
        "distance_km": round(random.uniform(MIN_DISTANCE_KM, MAX_DISTANCE_KM), 2),
        "passengers": random.randint(MIN_PASSENGERS, MAX_PASSENGERS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# The shared scenario — everything that is the SAME for both servers
# ─────────────────────────────────────────────────────────────────────────────
class RideDurationScenario(HttpUser):
    """Simulates one concurrent user hitting the prediction endpoint.

    `abstract = True` means Locust will not instantiate this class itself — it
    only exists to be subclassed. Subclasses supply the four things that differ
    between the two servers:

        host          which server to hit
        health_path   its liveness endpoint (they do NOT agree)
        build_payload the request envelope
        parse_row     how to dig one prediction row out of the response

    Everything else — the think-time, the 10:1 task weighting, and the
    validation rules — is shared, which is the point: the two servers are
    compared under an identical scenario.
    """

    abstract = True

    wait_time = between(0.5, 2.0)  # think-time, so the load is realistic

    #: Overridden per target. BentoML and FastAPI genuinely disagree here.
    health_path: str = "/health"

    # ── the per-target hooks ────────────────────────────────────────────────
    def build_payload(self) -> dict:
        raise NotImplementedError

    def parse_row(self, body) -> dict:
        """Return the single prediction row from a decoded response body."""
        raise NotImplementedError

    # ── the shared tasks ────────────────────────────────────────────────────
    @task(weight=10)  # 10x more common than the health check
    def predict(self):
        with self.client.post(
            "/predict",
            json=self.build_payload(),
            catch_response=True,
            name="/predict",
        ) as resp:
            # 1. transport / contract failures
            if resp.status_code != 200:
                # 422 here means the payload shape no longer matches the
                # server's schema — check /openapi.json, not the network.
                resp.failure(f"Expected 200, got {resp.status_code}: {resp.text[:200]}")
                return

            # 2. unparseable body — a 200 that isn't the JSON we agreed on
            try:
                row = self.parse_row(resp.json())
                duration = float(row["duration_min"])
            except Exception as exc:
                resp.failure(f"Unparseable response ({type(exc).__name__}): {resp.text[:200]}")
                return

            # 3. semantic failures — a 200 OK that is still wrong. This is the
            # class of bug a status-code-only load tool reports as success.
            if duration < 0:
                resp.failure(f"Negative duration in response: {duration}")
            elif duration > MAX_PLAUSIBLE_MIN:
                resp.failure(f"Implausible duration in response: {duration} min")

    @task(weight=1)
    def health(self):
        with self.client.get(
            self.health_path, catch_response=True, name=self.health_path
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")


# ─────────────────────────────────────────────────────────────────────────────
# Target A — the plain FastAPI baseline (fastapi_example.py)
# ─────────────────────────────────────────────────────────────────────────────
class FastAPIBaselineUser(RideDurationScenario):
    """No batching, no queueing. `uvicorn fastapi_example:app --port 8000`.

    Wire format: a FLAT object in, a FLAT object out.
        POST /predict  {"distance_km": 12.5, "passengers": 2}
        200            {"duration_min": 25.98, "eta_band": "20-30 min", ...}
    """

    host = "http://localhost:8000"
    health_path = "/health"

    def build_payload(self) -> dict:
        return _ride()

    def parse_row(self, body) -> dict:
        return body


# ─────────────────────────────────────────────────────────────────────────────
# Target B — the BentoML service (bentoml_example.py)
# ─────────────────────────────────────────────────────────────────────────────
class BentoMLUser(RideDurationScenario):
    """Dynamic batching + a bounded queue. `bentoml serve bentoml_example:RideDuration --port 8005`.

    Wire format: a batchable `@bentoml.api` is addressed as
    `{"<param name>": [ <one item> ]}` — here `inputs`, a ONE-element array —
    and answers with a one-element array. Each client still sends and receives
    a single row; the merging happens server-side, across clients. Send the flat
    object the FastAPI baseline takes and you get a 422.

        POST /predict  {"inputs": [{"distance_km": 12.5, "passengers": 2}]}
        200            [{"duration_min": 25.98, ..., "batch_size": 7}]

    Health: BentoML serves /healthz, /readyz and /livez. It does NOT serve
    /health — asking for that is a 404 and a red bar that has nothing to do
    with the model.
    """

    host = "http://localhost:8005"
    health_path = "/healthz"

    def build_payload(self) -> dict:
        return {"inputs": [_ride()]}

    def parse_row(self, body) -> dict:
        return body[0]


# ── Run from terminal ──────────────────────────────────
# locust -f locustfile.py FastAPIBaselineUser \
#        --users 100           ← total concurrent users
#        --spawn-rate 10       ← add 10 users/sec until 100
#        --run-time 2m         ← stop after 2 minutes
#        --headless            ← no UI, print results to CSV
#        --csv=results/load_test
