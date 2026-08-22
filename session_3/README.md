# Ride Duration — Orchestration & Deployment (Session 3)

Session 2 stopped at a *trained, registered* model. Session 3 is about everything
that happens **after** the model exists: scheduling the pipeline that keeps it
fresh, choosing how predictions reach consumers, packaging the service as a
container, and proving it survives real traffic.

The model itself is unchanged — the same synthetic **ride-duration** regressor
(`src/train.py`) from session 2. What changes is the machinery around it.

**This README follows the order you'd actually build the stack:**

1. [Orchestration with Airflow](#1-orchestration-with-airflow) — schedule and retry the retraining pipeline
2. [Deployment strategies](#2-deployment-strategies) — [batch scoring](#21-batch-scoring) vs. [streaming inference](#22-streaming-inference), plus [release strategies](#23-release-strategies--roll-out-without-breaking-users) (blue/green, canary, A/B, shadow) and [Nginx](#24-nginx--the-traffic-layer-that-makes-23-possible), the traffic layer that implements them
3. [Load testing with Locust](#3-load-testing-with-locust) — find the breaking point before users do, on both servers at once
4. [BentoML serving](#4-bentoml-serving) — model → production API → Docker image

## Project structure

```
session_3/
├── docker-compose.yaml          # Airflow 3.3 local stack (LocalExecutor, Postgres)
├── Dockerfile                   # extends apache/airflow:3.3.0 with the DAG runtime deps
├── requirements.txt             # deps baked into the Airflow image
├── .env                         # Airflow local-dev env (UID, Fernet key, admin login)
├── pyproject.toml               # project deps + optional extras (tracking/streaming/gcp/serving/bentoml/load)
├── Dags/
│   └── example_1.py             # retraining DAG: extract → train → evaluate
├── src/
│   ├── config.py                # YAML config loader
│   ├── prepare.py               # seeds data/processed/train.parquet (synthetic)
│   ├── train.py                 # fits the RandomForest → models/rf_model.pkl
│   └── batch_scoring.py         # ── batch strategy: GCS parquet in → predictions out
├── consumer.py                  # ── streaming strategy: Kafka event-driven inference consumer
├── fastapi_example.py           # ── plain FastAPI baseline on :8000 — the control in the benchmark
├── bentoml_example.py           # BentoML service on :8005 (save → serve → build → containerize)
├── locustfile.py                # one load-test scenario, two targets (10:1 /predict vs health)
├── config/
│   └── config.yaml              # data + model hyperparameters
└── data/, logs/, plugins/       # mounted into the Airflow containers
```

## Setup

```bash
# Base training/scoring stack
pip install -e .

# Add only the extras you need for the section you're on:
pip install -e ".[tracking]"   # MLflow registry (batch, streaming, BentoML all load from it)
pip install -e ".[streaming]"  # confluent-kafka for consumer.py (pulls in tracking too)
pip install -e ".[gcp]"        # google-cloud-storage + firestore
pip install -e ".[serving]"    # plain FastAPI baseline (fastapi_example.py)
pip install -e ".[bentoml]"    # BentoML serving
pip install -e ".[load]"       # Locust
pip install -e ".[airflow]"    # only to parse/lint DAGs locally; they RUN in the image
```

Python **3.10–3.12** (Airflow 3.3 has no 3.13 wheels yet).

---

## 1. Orchestration with Airflow

### What it is

A model isn't a one-off script — it's a *recurring pipeline*: pull fresh data →
train → evaluate → promote if better. **Airflow** turns that chain into a DAG
(Directed Acyclic Graph) of tasks with a schedule, and takes over the boring,
critical parts: running it on time, retrying the flaky step, showing you which
task failed and why, and never re-running a task whose upstream dependency
didn't succeed.

[`Dags/example_1.py`](Dags/example_1.py) is the retraining DAG: `extract` pulls
new rides from GCS, `train` runs `src/train.py` inside an MLflow run and pushes
the run ID to **XCom**, `evaluate` pulls that run ID back and compares MAE
against the current Production model.

```python
dag = DAG(
    dag_id="ride_duration_retrain",
    schedule="@weekly",           # every Monday 00:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                # don't backfill every missed week since 2024
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)

t_extract >> t_train >> t_evaluate   # the dependency chain
```

### Why use it

- **Scheduling without cron sprawl.** One `schedule="@weekly"` instead of a crontab entry nobody can find.
- **Retries and alerting are free.** `retries: 2, retry_delay: 5min` handles the transient GCS timeout that would otherwise page you.
- **Dependencies are explicit.** `train` cannot run if `extract` failed. A cron chain of scripts gives you no such guarantee.
- **Observability.** The UI shows every run, every task, every log line, and lets you re-run a single failed task instead of the whole pipeline.
- **Backfills.** Flip `catchup=True` and Airflow replays history for you.
- **XCom passes small values between tasks** (here: the MLflow run ID) without you inventing a temp-file protocol.

### The local stack

[`docker-compose.yaml`](docker-compose.yaml) runs Airflow 3.3 with the
**LocalExecutor** — tasks are subprocesses of the scheduler, so there's no
Redis/Celery/Flower to babysit. Services: `postgres` (metadata DB),
`airflow-apiserver` (UI + REST API — Airflow 3 renamed the old *webserver*),
`airflow-scheduler`, `airflow-dag-processor` (a standalone component in Airflow 3),
and `airflow-triggerer`.

The DAG's runtime deps (`google-cloud-storage`, `mlflow`, scikit-learn, …) are
baked into the image by [`Dockerfile`](Dockerfile) + [`requirements.txt`](requirements.txt),
rather than reinstalled on every container start via `_PIP_ADDITIONAL_REQUIREMENTS`.

### Most important commands

```bash
# ── Start the stack ───────────────────────────────────
docker compose build                     # build the extended Airflow image
docker compose up -d                     # postgres + apiserver + scheduler + dag-processor + triggerer
open http://localhost:8080               # login: airflow / airflow

docker compose ps                        # service health
docker compose logs -f airflow-scheduler # follow the scheduler
docker compose down -v                   # stop and wipe the metadata DB

# ── Seed the local training data (once) ───────────────
docker compose run --rm airflow-cli python src/prepare.py

# ── Airflow CLI (the `debug` profile) ─────────────────
docker compose --profile debug run --rm airflow-cli airflow dags list
docker compose --profile debug run --rm airflow-cli airflow dags list-import-errors
docker compose --profile debug run --rm airflow-cli airflow tasks list ride_duration_retrain

# Unpause and trigger a run by hand
docker compose --profile debug run --rm airflow-cli airflow dags unpause ride_duration_retrain
docker compose --profile debug run --rm airflow-cli airflow dags trigger ride_duration_retrain

# Test ONE task in isolation — no scheduler, no DB state written
docker compose --profile debug run --rm airflow-cli \
  airflow tasks test ride_duration_retrain train 2024-01-01
```

> `airflow tasks test` is the command you'll actually live in while developing a
> DAG: it executes a single task's Python immediately and prints the traceback.

---

## 2. Deployment strategies

Before writing any serving code, answer one question: **how fresh does a
prediction need to be?** The honest answer decides the architecture, and it is
usually *far less fresh* than people assume.

| | Batch scoring | Streaming inference | Online (REST) |
|---|---|---|---|
| **Latency** | hours / days | seconds | milliseconds |
| **Trigger** | schedule (Airflow) | an event arrives | a user request |
| **Cost** | lowest — one job, no idle server | low — pay per invocation | highest — always-on |
| **Use when** | predictions are consumed from a table | reacting to events as they happen | a user is waiting for the answer |

Section 4 covers the *online* column. This section covers the other two.

### 2.1 Batch scoring

#### What it is

Score a whole file of rows on a schedule and write the predictions somewhere
your consumers already read from (a bucket, a warehouse table). Nobody calls an
API; they `SELECT` yesterday's predictions.

[`src/batch_scoring.py`](src/batch_scoring.py) reads a parquet from GCS, loads
the **Production** model from the MLflow registry, scores every row, stamps a
`run_date`, and writes the result back to GCS.

```python
def load_production_model():
    """Always load from MLflow Production stage — no hardcoded paths."""
    return mlflow.sklearn.load_model("models:/RideDurationModel/Production")

df["pred"]     = model.predict(features)
df["run_date"] = date.today().isoformat()
bucket.blob(output_blob).upload_from_string(out_buffer.getvalue())
```

#### Why use it

- **Cheapest thing that works.** No server sitting idle at 3 a.m. waiting for traffic.
- **Throughput over latency.** Vectorised `model.predict()` over 10M rows beats 10M HTTP round-trips by orders of magnitude.
- **Trivially retryable.** The job is idempotent — same input file, same output file. Re-run it and nothing breaks.
- **Failure is not an outage.** A late batch is an inconvenience; a down API is an incident.
- **The registry decouples deploys.** Promoting a new model to `Production` in MLflow changes tomorrow's predictions with *zero* code changes — note the deliberate absence of a hardcoded model path.

#### Most important commands

```bash
# Run the scoring job locally (needs the [gcp,tracking] extras + object-store creds)
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/batch_scoring.py
```

In production you don't run this by hand — you make it the last task of the
Airflow DAG from section 1:

```python
PythonOperator(task_id="batch_score", python_callable=batch_score, dag=dag)
```

### 2.2 Streaming inference

#### What it is

An event lands on a message topic; a function wakes up, scores it, and writes the
prediction to a store. No polling, no server you manage, no fixed schedule —
the *event* is the trigger.

Two variants of the same idea live in this session:

- **Serverless** — a GCP Cloud Function triggered by the `ride-events` Pub/Sub
  topic (code inline below). GCP runs the consumer *for* you: scaling,
  retries, and the subscription loop are all managed.
- **Self-hosted** — [`consumer.py`](consumer.py), a long-running process that
  subscribes to a **Kafka** topic. Same handler shape, but *you* own the loop —
  which is exactly what makes it runnable (and testable) entirely on your laptop.

The Cloud Function decodes the base64 payload, scores it, and appends the
result to a Firestore collection:

```python
MODEL = None   # module-level: loaded ONCE per container, not per message

def get_model():
    global MODEL
    if MODEL is None:                                  # cold start only
        MODEL = mlflow.sklearn.load_model(os.environ["MODEL_URI"])
    return MODEL

def predict_on_pubsub(event, context):
    payload = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    ...
    db.collection("ride-predictions").add(result)
```

That `global MODEL` is the single most important line in the file. Loading the
model inside the handler would re-download it from the registry **on every
message**. Loading it at module level means the warm container reuses it, and
only a cold start pays the cost.

#### Why use it

- **Reacts in seconds, not hours** — the gap batch scoring can't close.
- **Scales to zero.** No messages, no containers, no bill.
- **Auto-scales up.** Pub/Sub backpressure spawns more function instances; you wrote no scaling logic.
- **Producer and consumer are decoupled.** The app publishes a ride event and moves on. If the model is down, messages queue in Pub/Sub instead of erroring out.
- **Pub/Sub retries on failure** — a raised exception means the message is redelivered, not lost.

#### Run it locally — `consumer.py` + Kafka

[`consumer.py`](consumer.py) is the self-hosted twin: the same lazy
`get_model()` + `predict_on_event()` handler, wrapped in a
`confluent-kafka` poll loop that subscribes to the `ride-events` topic and
appends every prediction to `predictions.jsonl` (the swap-point for
PostgreSQL/Redis in real life). A malformed message is logged and skipped —
one bad event must never kill the consumer.

Two contracts to respect, both learned the hard way:

- The registered model expects **exactly 2 features** —
  `["distance_km", "passengers"]` (see `FEATURES` in
  [`src/train.py`](src/train.py)). No `hour_of_day`.
- The registry stores `mlflow-artifacts:/` URIs, so the model **must** be
  loaded through the running MLflow server (`http://localhost:5000`) —
  pointing `MLFLOW_TRACKING_URI` straight at `sqlite:///mlflow.db` fails.

```bash
# ── One-time setup ────────────────────────────────────
pip install -e ".[streaming]"                     # confluent-kafka + mlflow

# ── Start the broker (single-node KRaft, no ZooKeeper) ─
docker run -d --name kafka -p 9092:9092 apache/kafka
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --create --topic ride-events

# ── Run the consumer (blocks, handling messages forever) ─
export MLFLOW_TRACKING_URI="http://localhost:5000"          # the mlflow server
export MODEL_URI="models:/RideDurationModel@production"     # alias, not stage
python consumer.py

# ── Publish a test event from another shell ───────────
docker exec -i kafka /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 --topic ride-events <<< \
  '{"ride_id": "r1", "distance_km": 3.2, "passengers": 1}'

# consumer prints:   processed: {'ride_id': 'r1', 'prediction': 8.21}
# and appends it to: predictions.jsonl
```

Gotchas seen in practice:

- **Port 9092 already taken?** (e.g. the session_4 Langfuse MinIO container
  maps it) — run the broker on another host port and point the consumer at it
  with `KAFKA_BOOTSTRAP=localhost:<port>`.
- **Restarted the consumer and messages seem stuck?** The old instance's group
  membership blocks partition assignment until the broker's ~45 s session
  timeout expires. Wait it out — the backlog then replays in order
  (`auto.offset.reset=earliest` + committed offsets = at-least-once delivery,
  for free).

### 2.3 Release strategies — roll out without breaking users

A new model that passes all tests can still fail in production: the offline eval
set is not live traffic. These four patterns make the rollout itself safe and
measurable. They answer a different question than 2.1/2.2 — not *how is the model
served*, but *how does version N+1 replace version N*.

#### Blue / Green — instant switch

Run two identical environments. Route 100% of traffic to **Blue** (current).
Deploy the new model to **Green**, test it there, then flip the load balancer.
Used by Netflix and most SaaS B2B products for zero-downtime deploys.

| ✓ Pros | ✗ Cons |
|---|---|
| Instant rollback — flip back to Blue | Costs 2× infra while both envs run |
| Zero downtime during the switch | The flip is all-or-nothing: 100% of users hit the new model at once |
| Green can be tested against production infra before any user sees it | A bug that only shows under full traffic still reaches everyone |

#### Canary release — gradual rollout

Route a small slice (say 5%) of real traffic to the new model. Watch MAE and
error rate. If healthy, widen: 5% → 20% → 50% → 100%. If metrics degrade,
auto-roll back. Amazon runs hundreds of canary deploys simultaneously — a model
degrading add-to-cart by more than 0.01% is rolled back automatically within
15 minutes.

| ✓ Pros | ✗ Cons |
|---|---|
| Real traffic validates the model gradually — blast radius starts at 5% | Needs traffic splitting + monitoring infrastructure |
| Rollback affects only the canary slice | Small slices need time to accumulate statistical signal |
| Automatable: metric threshold → auto-rollback, no human in the loop | Two model versions live at once — feature pipelines must support both |

#### A/B testing — experiment

Split **users** (not requests) by segment: model A serves group A, model B serves
group B, for days or weeks. Measure business metrics — CTR, revenue, retention —
not model metrics like MAE. Booking.com runs 1000+ A/B tests at once; ranking
model improvements are validated by booking conversion, not NDCG.

| ✓ Pros | ✗ Cons |
|---|---|
| Proves actual business value, not proxy-metric improvement | Needs many users and a long run time for significance |
| Consistent per-user experience (sticky assignment, unlike a canary) | An experimentation platform is real infrastructure to build/buy |
| Can conclude "better offline metrics ≠ better business outcome" — the finding that matters | Both variants must stay deployed for the whole experiment |

#### Shadow mode — zero-risk eval

The new model receives a copy of **every** request in parallel, but its
predictions are logged and **never returned to users**. Compare shadow vs
production offline. Meta runs shadow mode before every major Feed-ranking swap —
48 hours of logged comparison before any traffic shifts.

| ✓ Pros | ✗ Cons |
|---|---|
| Absolutely zero risk to users | Doesn't measure user impact — nobody ever saw the predictions |
| Full production traffic distribution, not a sample | Double compute on every request |
| Catches serving-time failures (latency spikes, schema drift, crashes) before launch | Useless for models whose value depends on user reaction (ranking, recommendations) |

#### Picking one

They compose rather than compete — a common production sequence is:

1. **Shadow** the new model until it is stable on real traffic (catches crashes and latency, risk-free).
2. **Canary** it at 5% → 100% (catches degradation on live outcomes, small blast radius).
3. **A/B test** only when the question is *business* value, not model health.
4. **Blue/Green** is the mechanics of the final swap — and your instant-rollback lever.

Rule of thumb: shadow answers *"does it run?"*, canary answers *"does it behave?"*,
A/B answers *"does it pay?"*.

### 2.4 Nginx — the traffic layer that makes 2.3 possible

**Nginx** is a high-performance, event-driven server that sits *in front of* your
model API. It is not part of the model stack at all — it is the general-purpose
traffic layer the release strategies above are built on. One process, four jobs:

- **Web server** — serves static files (HTML, images, CSS) very efficiently,
  thousands of concurrent connections per worker without a thread per client.
- **Reverse proxy** — sits in front of application servers (uvicorn, BentoML,
  Triton) and forwards requests to them; clients only ever see Nginx.
- **Load balancer** — distributes traffic across multiple backend servers:
  round-robin by default, weighted, least-connections, or IP-hash for stickiness.
- **API gateway / TLS termination** — handles HTTPS, rate limiting, and routing,
  so your Python server speaks plain HTTP on localhost and never touches
  certificates or abuse traffic.

Why it matters for this session: **the weighted load balancer *is* the canary and
blue/green mechanism.** The whole of 2.3 reduces to one `upstream` block:

```nginx
upstream ride_api {
    server 127.0.0.1:8000 weight=95;   # blue  — model v1
    server 127.0.0.1:8005 weight=5;    # green — model v2, the 5% canary
}
# ↑ illustrative: two ports running the SAME service at two model versions.
#   In this repo 8000 and 8005 happen to run two *different* servers
#   (fastapi_example.py and bentoml_example.py) — see section 3.

server {
    listen 443 ssl;                     # TLS terminated here, not in Python
    location /predict {
        proxy_pass http://ride_api;     # reverse proxy to whichever upstream wins
    }
    location /static/ {
        root /var/www;                  # web server: files served directly
    }
}
```

- Canary 5% → 20%: change `weight=5` to `weight=20`, `nginx -s reload` (zero
  downtime — old workers finish their requests, new workers get the new config).
- Blue/green flip: swap which server carries the weight, same reload.
- Instant rollback: swap it back.

The model servers never know any of this is happening — which is exactly the
point. Traffic policy lives in the traffic layer, not in your Python code.

---

## 3. Load testing with Locust

### What it is

Your API answers one request in 40 ms. What does it do with **500 concurrent
users**? *"It'll probably be fine"* is a guess, and it's the guess that turns
into an incident. **Locust** answers it empirically: you describe one simulated
user in Python, it spawns thousands of them and reports latency percentiles and
failure rates.

[`locustfile.py`](locustfile.py) describes one scenario — hit `/predict` ten
times as often as the health check, with realistic think-time, and validate the
*content* of every response — and points it at **two different servers**, so the
load test doubles as a benchmark of what a serving framework is worth.

### Two targets, one scenario

| | `FastAPIBaselineUser` | `BentoMLUser` |
|---|---|---|
| **Serves** | [`fastapi_example.py`](fastapi_example.py) — plain FastAPI, the control | [`bentoml_example.py`](bentoml_example.py) — the serving framework |
| **Port** | `8000` | `8005` |
| **Start command** | `uvicorn fastapi_example:app --port 8000` | `bentoml serve bentoml_example:RideDuration --port 8005` |
| **Request body** | flat object:<br>`{"distance_km": 12.5, "passengers": 2}` | batch envelope:<br>`{"inputs": [{"distance_km": 12.5, "passengers": 2}]}` |
| **Response shape** | flat object:<br>`{"duration_min": 25.98, "eta_band": "20-30 min", "model_version": "...", "batch_size": 1}` | **one-element array**:<br>`[{"duration_min": 26.03, "eta_band": "20-30 min", "model_version": "...", "batch_size": 7}]` |
| **Health endpoint** | `/health` | `/healthz` (also `/readyz`, `/livez`) — **no `/health`** |
| **Batching** | none — `batch_size` is `1` forever | `batchable=True`, `max_batch_size=64` |
| **Queueing** | none — extras pile up in anyio's threadpool with no deadline | `traffic={"concurrency": 64, "timeout": 30}` → queue, then shed with a 504 |

The four *fields* are identical on both, on purpose. Only the envelope differs,
which is exactly what the two `locustfile.py` subclasses encode: `host`,
`health_path`, `build_payload()`, `parse_row()`. Everything else — the
think-time, the 10:1 weighting, the validation rules — lives in the shared
`abstract = True` base class, so the two servers are compared under a genuinely
identical scenario.

```python
class RideDurationScenario(HttpUser):
    abstract = True                     # never instantiated; only subclassed
    wait_time = between(0.5, 2.0)       # think-time, so the load is realistic

    @task(weight=10)                    # 10x more common than the health check
    def predict(self):
        with self.client.post("/predict", json=self.build_payload(),
                              catch_response=True, name="/predict") as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}: ...")
            ...
            elif duration < 0:
                resp.failure(...)       # a 200 OK that is still wrong


class FastAPIBaselineUser(RideDurationScenario):
    host = "http://localhost:8000"      # ← this line is why you never pass --host
    health_path = "/health"
    def build_payload(self): return _ride()
    def parse_row(self, body): return body


class BentoMLUser(RideDurationScenario):
    host = "http://localhost:8005"
    health_path = "/healthz"
    def build_payload(self): return {"inputs": [_ride()]}
    def parse_row(self, body): return body[0]
```

### Why use it

- **Scenarios are plain Python.** Random payloads, weighted task mixes, auth flows, stateful sessions — anything you can write, you can simulate. No XML, no clicking through a GUI recorder.
- **`catch_response` catches *semantic* failures.** A `200 OK` carrying a negative duration is a bug; Locust marks it as a failure. HTTP-only tools would report 100% success.
- **Percentiles, not averages.** p95/p99 is what your users feel. The mean hides the tail.
- **Finds the knee in the curve.** Ramp users up until p95 latency spikes — that's your real capacity, and now you can size the box you deploy on with a number instead of a vibe.
- **`--headless --csv` makes it a CI gate.** Fail the build if p95 regresses.
- **Subclassing makes it a benchmark.** One scenario, two servers, one apples-to-apples number.

### Step 1 — start a server. Locust never does this for you.

**This is the number one cause of a 100%-failure run.** Locust is a traffic
*generator*. It does not build, start, containerise or health-check your API; it
opens sockets to a host you named and reports what came back. If nothing is
listening, every request fails with `ConnectionRefusedError` and the scenario
itself is irrelevant.

Open a terminal per target and leave it running:

```bash
# optional, for both — load from the registry instead of models/rf_model.pkl
export MLFLOW_TRACKING_URI=http://localhost:5000

# terminal 1 — the plain FastAPI baseline
uvicorn fastapi_example:app --port 8000
# → "loaded from MLflow registry: models:/RideDurationModel@production -> v1"

# terminal 2 — the BentoML service
bentoml serve bentoml_example:RideDuration --port 8005
# → "Starting production HTTP BentoServer ... listening on http://localhost:8005"
```

Both print which model source won at startup — registry first, `models/rf_model.pkl`
as the offline fallback.

### Step 2 — prove the service is actually up.

Do this *before* blaming the load test. Two commands, ten seconds.

```bash
# ── is anything bound to the port at all? ─────────────
lsof -nP -iTCP:8000 -sTCP:LISTEN     # FastAPI baseline
lsof -nP -iTCP:8005 -sTCP:LISTEN     # BentoML
# no output = nothing is listening = go back to Step 1

# ── does the health endpoint answer? (note: different paths!) ──
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/health    # → 200
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8005/healthz   # → 200
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8005/health    # → 404, on purpose

# ── send the EXACT body locust will send ──────────────
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"distance_km": 12.5, "passengers": 2}'
# → {"duration_min":25.98,"eta_band":"20-30 min",
#    "model_version":"RideDurationModel@production/1","batch_size":1}

curl -s -X POST http://localhost:8005/predict \
  -H 'content-type: application/json' \
  -d '{"inputs": [{"distance_km": 12.5, "passengers": 2}]}'
# → [{"duration_min": 26.03, "eta_band": "20-30 min",
#     "model_version": "342pdzudugsbncmm", "batch_size": 1}]
#   ...an ARRAY. Sending the flat body here is a 422.

# ── when the shape is unclear, ask the server ─────────
open http://localhost:8000/docs           # or :8005/docs — interactive OpenAPI UI
curl -s http://localhost:8000/openapi.json | python -m json.tool | less
```

**If curl works and Locust still fails, the bug is in the scenario** (payload
shape, response parsing, health path). **If curl fails too, fix the server
first** — no locustfile edit will help.

### Step 3 — choose the target and run.

```bash
locust -f locustfile.py --list
# Available Users:
#     FastAPIBaselineUser
#     BentoMLUser

locust -f locustfile.py FastAPIBaselineUser     # web UI at http://localhost:8089
locust -f locustfile.py BentoMLUser             # the other target

locust -f locustfile.py --class-picker          # or pick in the browser at start
```

Two mistakes worth naming:

- **Running with no class name runs *both* classes at once.** Nothing errors — each class keeps its own `host` — so it looks like a clean run. It isn't: your users are split across two different servers and both servers' numbers are merged into one report. Useless as a benchmark.
- **Adding `--host` on top of that overrides *every* class**, pointing one of them at a server that speaks the other wire format. Measured here: `47.8%` failures — half of `/predict` returning `422`, and every `/healthz` a `404`. That is also why **`"You must specify the base host"`** is gone: each class now carries its own `host`, so naming the class *is* choosing the host, and you should not pass `--host` at all.

### Step 4 — the comparison.

Same scenario, same load (20 users, spawn rate 10, 20 seconds), both servers:

```bash
mkdir -p results
locust -f locustfile.py FastAPIBaselineUser --headless -u 20 -r 10 -t 20s \
  --csv=results/fastapi
locust -f locustfile.py BentoMLUser        --headless -u 20 -r 10 -t 20s \
  --csv=results/bentoml
```

Measured on this machine (Apple silicon, 1 worker each, model loaded from the
MLflow registry, second — warmed-up — run of each):

| `POST /predict` | `FastAPIBaselineUser` :8000 | `BentoMLUser` :8005 |
|---|---|---|
| requests | 274 | 283 |
| **failures** | **0 (0.00%)** | **0 (0.00%)** |
| req/s | 14.41 | 14.96 |
| median | 7 ms | 9 ms |
| p90 | 25 ms | 23 ms |
| p95 | 47 ms | **29 ms** |
| p99 | 62 ms | **53 ms** |
| max | 67 ms | 69 ms |

At 20 think-timed users the two look almost identical at the median — the
baseline is even marginally *faster* there, because batching costs a few
milliseconds of dispatcher wait that a lightly-loaded server doesn't need. That
is the honest result, and it is why you read the **tail**: BentoML's p95 is
already 40% lower, and the gap widens with concurrency.

Now ask the baseline what it actually did:

```bash
curl -s http://localhost:8000/metrics
# → {"requests":562,"avg_batch_size":1.0,"avg_predict_ms":7.965}
```

`avg_batch_size` is `1.0`. It stays `1.0` at 20 users, and it stays `1.0` when
you drop the think-time entirely and fire 300 requests from 32 concurrent
threads:

```
fastapi batch_size: {1: 300}   mean=1.00
```

Three hundred requests, three hundred model calls. Under that exact same burst,
BentoML ships the merge size back in every response's `batch_size` field:

```
bento batch_size: {1: 11, 7: 7, 8: 8, 10: 10, 11: 22, 12: 24, 13: 13,
                   19: 19, 20: 20, 21: 21, 23: 23, 29: 29, 31: 93}
max=31  mean=21.26  n=300      →  ~14 actual model calls for 300 requests
```

**That is the whole exhibit.** Identical clients, identical model, identical
response fields — one server made 300 `predict()` calls, the other made about
14. No client changed a line to get it; `batchable=True` did it server-side,
across callers. Everything the baseline is missing is marked `# [1] …`–`# [5] …`
in [`fastapi_example.py`](fastapi_example.py) against the same numbering
[`bentoml_example.py`](bentoml_example.py) uses, so the two files read side by
side.

Read the summary right to left: **failure count first** (any non-zero means stop
tuning and start fixing), then **p95/p99**, then RPS. A high RPS with a
12-second p99 is not a passing test.

### Locust commands cheat-sheet

All commands run from this directory with the session venv
(`.venv/bin/locust`, or just `locust` with the venv activated). Every one of
them names a user class instead of passing `--host` — the class carries its own.

```bash
# ── Sanity checks ─────────────────────────────────────
locust --version                    # confirm the install (2.45.0 here)
locust -f locustfile.py --list      # list the user classes/tasks without running

# ── Interactive: web UI at http://localhost:8089 ──────
locust -f locustfile.py FastAPIBaselineUser
# pick users + spawn rate in the browser; charts update live
locust -f locustfile.py BentoMLUser --web-port 8090
# ...if 8089 is already taken
locust -f locustfile.py --class-picker
# ...or choose the target in the browser instead of on the command line

# ── Headless: scripted run, no UI ─────────────────────
locust -f locustfile.py FastAPIBaselineUser \
  --users 100 \          # total concurrent users
  --spawn-rate 10 \      # add 10 users/sec until 100
  --run-time 2m \        # then stop
  --headless

# ── Results to files ──────────────────────────────────
locust -f locustfile.py BentoMLUser \
  --users 100 --spawn-rate 10 --run-time 2m --headless \
  --csv=results/load_test \        # 4 CSVs: stats, history, failures, exceptions
  --csv-full-history \             # keep every interval row, not just the last
  --html=results/load_test.html    # self-contained report with the charts

# ── CI gate: fail the build on regressions ────────────
locust -f locustfile.py FastAPIBaselineUser \
  --users 200 --spawn-rate 20 --run-time 3m --headless \
  --only-summary \                 # skip the per-interval console spam
  --exit-code-on-error 1           # non-zero exit if any request failed

# ── Useful knobs ──────────────────────────────────────
#   --stop-timeout 10s      let in-flight requests finish on shutdown
#   --autostart             web UI, but start the run immediately
#   --autoquit 5            exit 5s after the run finishes (pairs with --autostart)
#   --loglevel DEBUG        see every request/response while debugging a scenario
#   --config locust.conf    put any of these flags in a file instead

# ── Scale the load generator itself ───────────────────
# One Python process ≈ one core. When the *generator* is the bottleneck
# (CPU pegged, RPS plateaus while the server is idle), fan out:
locust -f locustfile.py BentoMLUser --processes 4     # 4 local workers, one master
locust -f locustfile.py BentoMLUser --processes -1    # one worker per core
# ...or across machines:
locust -f locustfile.py BentoMLUser --master                    # on the coordinator
locust -f locustfile.py BentoMLUser --worker --master-host <ip> # on each generator
```

**Ports in this repo:** `8000` plain FastAPI ride-duration baseline
(`fastapi_example.py`), `8005` BentoML ride-duration (`bentoml_example.py`) —
those are the two `locustfile.py` targets. The `serving_levels/` demos are a
different model (ResNet-50, image uploads) on their own ports: `8001` level-1
FastAPI, `8002` level-2 batching, `8004` BentoML ResNet. They speak a completely
different request shape, so they need their own scenario, not `--host`.

### When every request fails — a checklist

| Symptom in the Locust output | Cause | Fix |
|---|---|---|
| `ConnectionRefusedError` / `ConnectionError`, 100% failures, 0 ms latency | Nothing is listening. Locust never starts your API. | Step 1. Confirm with `lsof -nP -iTCP:8000 -sTCP:LISTEN`. |
| `You must specify the base host` before any request runs | A `User` class with no `host` attribute and no `--host` on the CLI | Name a class that sets `host` (`FastAPIBaselineUser` / `BentoMLUser`). |
| `Expected 200, got 422` on every `/predict` | Payload shape ≠ the server's schema — wrong field names (`distance` vs `distance_km`), or the flat body sent to the batch endpoint (or vice versa) | Compare against `/openapi.json`; check `build_payload()` in the subclass. Step 2's curl reproduces it outside Locust. |
| `Expected 200, got 422` on *half* the `/predict` requests, plus `404` on every `/healthz` | You ran with no class name **and** passed `--host`, which overrode both classes onto one server | Drop `--host` and name exactly one class, or use `--class-picker`. |
| No failures, but the report mixes `/health` and `/healthz` rows | You ran with no class name (and no `--host`), so both classes ran against their own hosts and the stats were merged | Name exactly one class — one report per server. |
| `Expected 200, got 404` on the health task only | Wrong health path — BentoML has no `/health`, it has `/healthz` | Set `health_path` on the subclass. |
| `Unparseable response (AttributeError)` / `(KeyError)` | The response is an array and you treated it as an object (or vice versa) | Fix `parse_row()`; BentoML returns `[ {...} ]`, FastAPI returns `{...}`. |
| `Expected 200, got 504` appearing only under load | BentoML's `traffic.timeout` (30 s) shed a request that queued too long. **This is the framework working**, not a bug. | Lower the load, raise `concurrency`/`workers`, or accept it as your capacity number. |
| 0 failures but `Implausible duration in response` | A 200 carrying nonsense — usually the wrong artifact loaded, or feature columns swapped | Check `/model_info` on the running server: `n_features` and `features` must match `src/train.py`'s `FEATURES`. |

---

## 4. BentoML serving

### Why `async def` alone doesn't save you: the GIL

#### The setup

When you write a FastAPI endpoint like this:

```python
@app.post("/predict")
async def predict(file: UploadFile):
    img = preprocess(await file.read())
    logits = model(img)   # ← this is the problem
    return {"class": int(logits.argmax())}
```

The `async def` makes it look non-blocking. But `model(img)` — the actual
PyTorch forward pass — is not async. It's a regular blocking call.

#### What the GIL is

Python has a rule called the **Global Interpreter Lock (GIL)**. It says: only
one thread can execute Python code at a time.

This means when `model(img)` starts running — which might take 80ms — it holds
the GIL. No other code in Python can run during those 80ms. Not even the event
loop that's supposed to be listening for new incoming requests.

#### What happens with 100 concurrent users

```
User 1   → request arrives → model(img) starts → 80ms blocked
User 2   → request arrives → has to WAIT (GIL held)
User 3   → request arrives → has to WAIT
...
User 100 → request arrives → has to WAIT for all 99 before it
```

So instead of 100 users getting served in parallel in 80ms each, they queue up.
User 100 waits 100 × 80ms = 8 seconds before their request even starts. That's
why the benchmark shows 4,200ms p95 at 100 users even though one user gets 85ms.

#### What BentoML Runner fixes

BentoML moves the model into a completely separate Python **process** — not a
thread, a full process. A separate process has its own GIL. So:

```
HTTP server process          Model Runner process
(handles all requests)       (owns the model)
        │                           │
User 1 arrives ──────────── async call ──→ model(img) runs here
User 2 arrives ──────────── async call ──→ queued by runner
        │                           │
HTTP server never blocks    Model runs independently
```

The HTTP server's event loop is free to accept requests 2, 3, 4... while
request 1's model inference is running in the other process. That's what
"truly async" means — the HTTP layer is no longer frozen waiting for the model.

#### One-line summary

`async def` alone doesn't fix the GIL. You need the model in a separate
process. BentoML Runner does exactly that with one line: `.to_runner()`.

> **Note:** `.to_runner()` is the legacy (pre-1.2) spelling — see the version
> note below. In the current `@bentoml.service` API this repo uses, the same
> separation comes from `workers` (each worker is its own process with its own
> GIL) plus the dispatcher's request queue.

### What it is

Between `model.pkl` and *a production API* sits a pile of undifferentiated work:
an HTTP layer, request validation, batching, health checks, a Dockerfile, the
right CUDA base image. **BentoML** generates all of it from a model reference
and a Python function.

[`bentoml_example.py`](bentoml_example.py) walks the full path: pull the
production model out of the MLflow registry, save it into BentoML's model store,
wrap it in a Service, then package it as a Docker image. Each of the five things
a serving framework buys you is marked `[1]`–`[5]` in the file.

```python
@bentoml.service(
    workers=1,                                    # [5] concurrency: process-level
    traffic={"concurrency": 64, "timeout": 30},   # [5] admission + [2] queue bound
)
class RideDuration:
    bento_model = bentoml.models.BentoModel(MODEL_TAG)   # [4] versioning

    def __init__(self) -> None:                          # runs once per worker
        self.model = joblib.load(self.bento_model.path_of("model.pkl"))

    @bentoml.api(batchable=True, batch_dim=0,            # [1] dynamic batching
                 max_batch_size=64, max_latency_ms=2_000)
    def predict(self, inputs: list[PredictRequest]) -> list[PredictResponse]:
        features = np.array([[r.distance_km, r.passengers] for r in inputs])  # [3] pre
        preds = self.model.predict(features)             # ONE call for all N rows
        return [PredictResponse(duration_min=round(float(p), 2), ...)         # [3] post
                for p in preds]
```

### Why use it

- **[1] Dynamic batching for free.** `batchable=True` makes BentoML merge concurrent requests into one `predict()` call — invisible to callers, who each send and receive a single row. This is the single biggest throughput win available to an ML API. Section 3 measures it: under 32 concurrent callers this service merged 300 requests into ~14 model calls (mean batch 21.3, max 31), while the plain-FastAPI baseline stayed at exactly 1.0 for all 300.
- **[2] Request queueing.** Traffic past `concurrency` waits instead of being refused, and `timeout` bounds that wait so an overloaded service sheds load rather than melting down.
- **[3] Pydantic in, Pydantic out.** Malformed requests are rejected with a clean 4xx before they ever touch the model — and post-processing (rounding, derived fields) happens once, server-side, instead of in every client.
- **[4] The model store is versioned.** Every save mints an immutable tag (`ride_duration:rlkriaedugl2xuiy`); rollback is redeploying a previous tag. Pin it in production — `:latest` means a colleague's save silently changes what you serve.
- **[5] Concurrency control.** `workers` sets process-level parallelism, `traffic.concurrency` sets admission. Note the constraint: `concurrency` must be ≥ `max_batch_size`, or the dispatcher can never hold enough requests to fill a batch.
- **`bentoml containerize` writes the Dockerfile.** Correct Python version, correct deps, correct model — no hand-maintained image.
- **Batteries included:** OpenAPI docs at `/docs`, Prometheus metrics at `/metrics`, health checks — all out of the box.

> **Version note:** this targets the `@bentoml.service` class API (BentoML 1.2+,
> tested on 1.4.39). The legacy `bentoml.io` + `bentoml.Service(runners=[...])`
> API was **removed** in 1.2 — `to_runner()` and `bentoml.io` no longer exist.
> Likewise, the model is loaded by MLflow *alias* (`models:/Name@production`),
> since MLflow 3 removed the `Production`/`Staging` stages.

### Most important commands

```bash
# ── Save the model into the BentoML store ─────────────
export MLFLOW_TRACKING_URI=http://localhost:5000   # else falls back to models/rf_model.pkl
python bentoml_example.py
bentoml models list                      # see the tags
bentoml models get ride_duration:latest

# ── Dev server (hot reload) ───────────────────────────
bentoml serve bentoml_example:RideDuration --port 8005 --reload

# NOTE the wire format: a batchable API takes {"<param>": [ <one item> ]}
curl -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"distance_km": 8.4, "passengers": 2}]}'
# → [{"duration_min": 17.68, "eta_band": "10-20 min", "model_version": "...", "batch_size": 1}]

curl -X POST http://localhost:8005/model_info -d '{}'   # [4] what's actually live?
open http://localhost:8005/docs

# ── Package & containerize ────────────────────────────
bentoml build                                    # → a versioned Bento
bentoml list                                     # list built Bentos
bentoml containerize ride_duration_api:latest    # → a Docker image

docker run -p 3000:3000 ride_duration_api:latest # the same image you'd ship anywhere
```

---

## 5. Accelerated serving — Triton (GPU) & OpenVINO (CPU)

BentoML solved the *packaging* problem, but the request path is still Python.
The [`serving_levels/`](serving_levels/) folder climbs one rung further: hand
the model to a dedicated **inference runtime** that executes an optimized graph
instead of eager PyTorch. Two names dominate that rung — one per kind of
hardware.

### 5.1 NVIDIA Triton Inference Server — the GPU-accelerated path

#### What it is

**Triton** is NVIDIA's open-source, C++ **inference server**. Its job is to be
the production serving layer itself: you don't write an API around your model —
you drop the model file into a `model_repository/` directory, describe it in a
`config.pbtxt` text file, and Triton provides the HTTP/gRPC endpoints, dynamic
batching, versioning, and Prometheus metrics. Everything
[`level_2_batching.py`](serving_levels/level_2_batching.py) built by hand
(~90 lines of `asyncio.Queue` + futures) becomes a `dynamic_batching { ... }`
stanza in that config file.

#### Why it matters

- **No GIL anywhere on the request path** — the server is C++, so section 4's
  entire GIL discussion evaporates.
- **Multiple frameworks in one process**: a TensorRT vision model, an ONNX
  ranker and a PyTorch encoder behind one endpoint, one GPU, one deployment.
- **Concurrent model execution**: `instance_group { count: 2 }` runs genuine
  parallel copies per device (Level 3, Lever C — impossible in the Python levels
  without a rewrite).
- **Model ensembles**: chain preprocess → infer → postprocess as a server-side
  DAG, the proper fix for the preprocessing bottleneck Level 3 uncovers.
- **Hot reload**: drop a `2/` version directory in and it loads live.

#### When to use it

A GPU fleet serving several models under a latency SLA. The cost is a model
*conversion* step (PyTorch → ONNX/TensorRT), another config format to get
right, and a ~15 GB runtime image. For one sklearn model in a Python shop,
BentoML is the better trade — Triton is what you graduate to when the GPU bill
and the latency SLA justify it.

### 5.2 OpenVINO — the CPU-accelerated path

#### What it is

**OpenVINO** is Intel's open-source **inference toolkit**. Its job is Level 3's
"Lever A" (graph optimization) for machines *without* a GPU: it compiles a
trained model (ONNX, PyTorch, TensorFlow) into an optimized graph — fusing
Conv+BN+ReLU into single kernels, folding constants, planning memory ahead of
time — and executes it with kernels tuned for Intel CPUs, integrated GPUs and
NPUs. Pair it with INT8 quantization (Lever B) and a CPU commonly gains 2–4×
throughput with little to no accuracy loss.

#### Why it matters

Most real deployments never see a GPU — the economics of batch scoring and
low-QPS APIs are CPU economics. Runtime optimization is **the only lever that
still pays off on CPU**: batching (Level 2) barely helps there because a CPU is
already compute-saturated by one input, as
[`level_3_optimize.py`](serving_levels/level_3_optimize.py) demonstrates with
real numbers.

#### When to use it

CPU-only or edge deployments, especially on Intel hardware. It sits in the same
family as ONNX Runtime (portable, multi-vendor) and TensorRT (NVIDIA GPUs,
maximum speed) — same lever, different hardware target. Measure on *your*
target machine before committing: Level 3's core lesson is that a lever that
helps on one device can hurt on another.

### Where they live in this repo

| File | What it shows |
|---|---|
| [`serving_levels/level_3_optimize.py`](serving_levels/level_3_optimize.py) | The CPU lever, measured: eager PyTorch vs `torch.compile` vs ONNX Runtime FP32/INT8 on your machine. OpenVINO is a drop-in alternative runtime for the same lever (same ONNX export, different engine). |
| [`serving_levels/level_4_triton_export.py`](serving_levels/level_4_triton_export.py) | Exports ResNet-50 to ONNX with a dynamic batch axis and **verifies** the output against PyTorch — the step people skip. |
| [`serving_levels/level_4_triton/`](serving_levels/level_4_triton/) | A real, ready-to-mount Triton `model_repository/` with a fully commented [`config.pbtxt`](serving_levels/level_4_triton/model_repository/resnet50/config.pbtxt). Its `platform:` line accepts `openvino` too — inside Triton, OpenVINO is just another backend. |
| [`serving_levels/level_4_triton/README.md`](serving_levels/level_4_triton/README.md) | The `docker run` command, health/metrics checks, a line-by-line comparison with the hand-written Level 2 batcher — and the honest caveat that Triton's image is `linux/amd64` only, so an Apple Silicon Mac can read the config but not benchmark it. |
| [`serving_levels/level_5_vllm_llm.py`](serving_levels/level_5_vllm_llm.py) | The rung above both: LLMs break the fixed-shape batching model entirely, so vLLM uses continuous batching + PagedAttention (CUDA GPU required). |

The full ladder — with the eager-mode explanation, the chef analogy, and the
benchmark methodology — is in
[`serving_levels/README.md`](serving_levels/README.md).

---

## Where this leaves you

You now have all three deployment shapes for the same model, and the tools to
choose between them: **Airflow** schedules the retraining, **batch scoring**
serves the cheap/high-throughput case, **streaming inference** serves the
event-driven case, **BentoML** serves the online case, **Locust** tells you
which of them will hold up — and when Python serving itself becomes the
bottleneck, **Triton** (GPU) and **OpenVINO** (CPU) are the next rung.

The lesson worth keeping: the model was the easy part.
