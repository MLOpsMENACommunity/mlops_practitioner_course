# Ride Duration — Monitoring & Drift Detection (Session 4)

Session 3 ended with the model **deployed** — scheduled by Airflow, scored in batch,
served over HTTP. Session 4 is about the question that follows deployment:
**is the model still right?**

A model does not fail loudly. It keeps returning `200 OK` and plausible-looking
numbers long after the world it was trained on has moved on. Monitoring is how you
notice. This session covers the three things that can drift, in the order you'd
detect them in practice:

| # | Script | What it watches | Needs ground truth? |
|---|--------|-----------------|---------------------|
| 1 | [`data_drift_evidently.py`](data_drift_evidently.py) | **Input** distributions (features) | No |
| 2 | [`label_predicion_drift.py`](label_predicion_drift.py) | **Output** distribution (predictions), then labels | No, then yes |
| 3 | [`hinkley_adwin.py`](hinkley_adwin.py) | **Error** over time (concept drift) | Yes |
| 4 | [`P_G_monitoring.py`](P_G_monitoring.py) | All of the above, as **live metrics** | — |
| 5 | [`ollama_langfuse_rag.py`](ollama_langfuse_rag.py) | The same discipline applied to an **LLM** | Judge scores instead |

The ordering matters: features drift first and are observable immediately;
predictions drift next; error drift is the ground truth of "the model is broken"
but it's the *last* signal to arrive, because labels lag predictions by hours or
weeks. You monitor 1 and 2 so you're not blind while waiting for 3.

Scripts 1–3 answer *"has it drifted?"* as a batch question. Script 4 and the
Docker stack answer *"how is it doing right now?"* — the same signals, exposed
continuously and drawn on a dashboard. Script 5 is the same idea again, on an LLM,
where the output is text and the score comes from a judge rather than a label.

### Contents

1. [`data_drift_evidently.py` — feature drift](#1-data_drift_evidentlypy--feature-drift)
2. [`label_predicion_drift.py` — prediction drift](#2-label_predicion_driftpy--prediction-drift)
3. [`hinkley_adwin.py` — concept drift](#3-hinkley_adwinpy--concept-drift-on-a-live-stream)
4. [`P_G_monitoring.py` — metrics for Prometheus](#4-p_g_monitoringpy--metrics-for-prometheus)
5. [The Docker stack](#5-the-docker-stack--prometheus-grafana-langfuse)
6. [`ollama_langfuse_rag.py` — LLM observability](#6-ollama_langfuse_ragpy--llm-observability)
7. [Operating the stack](#7-operating-the-stack)
8. [Troubleshooting](#8-troubleshooting)

## Project structure

```
session_4/
├── pyproject.toml                # deps + optional extras (dev/metrics/llm/model)
├── docker-compose.yaml           # Prometheus + Grafana, and Langfuse behind a profile
├── .env.example                  # ports + secrets for the compose stack
├── .env                          # your real values — gitignored, never commit
├── .gitignore                    # ignores .env, reports/, data/, .venv/
├── data_drift_evidently.py       # feature drift — Evidently DataDriftPreset + per-column tests
├── label_predicion_drift.py      # prediction drift — hand-rolled PSI + Evidently TargetDrift
├── hinkley_adwin.py              # concept drift — river's Page-Hinkley + ADWIN on live error
├── P_G_monitoring.py             # Prometheus instrumentation for the model API
├── ollama_langfuse_rag.py        # LLM observability — traced RAG on a local Ollama model
└── monitoring/
    ├── prometheus/
    │   ├── prometheus.yml        # scrape config (targets the host-run model API)
    │   └── alert_rules.yml       # PSI > 0.25, p95 latency, API down
    └── grafana/
        ├── provisioning/         # datasource + dashboard providers (as code)
        └── dashboards/
            └── model-monitoring.json
```

## Setup

```bash
pip install -e .              # pandas, pyarrow, evidently<0.7, river, mlflow
pip install -e ".[dev]"       # + pytest, ruff
pip install -e ".[metrics]"   # + prometheus-client, fastapi, uvicorn  (script 4)
pip install -e ".[llm]"       # + langfuse SDK v4, ollama  (Langfuse tracing)

docker compose up -d          # Prometheus + Grafana — see section 5
```

> **Why `evidently<0.7` is pinned.** These scripts use the legacy API —
> `from evidently.report import Report` and `evidently.metric_preset`.
> Evidently 0.7 replaced it with `evidently.Report` / `evidently.presets`.
> Installing the latest release will break every Evidently import here.

---

## 1. `data_drift_evidently.py` — feature drift

**The question:** are today's inputs shaped like the data the model was trained on?

It compares two dataframes:

- **reference** — `data/train.parquet`, the training set. This is the baseline the
  model actually learned; it never changes until you retrain.
- **current** — `data/scoring/output/{today}.parquet`, the batch that session 3's
  scoring job just produced.

### What it computes

```python
report = Report(metrics=[
    DataDriftPreset(),                                        # every feature, auto-chosen test
    ColumnDriftMetric(column_name="distance_km", stattest="ks"),
    ColumnDriftMetric(column_name="passengers",  stattest="chisquare"),
    ColumnDriftMetric(column_name="hour_of_day", stattest="psi"),
    ColumnDriftMetric(column_name="pred",        stattest="psi"),
])
```

`DataDriftPreset()` sweeps *all* columns and picks a statistical test per column
based on its type and cardinality. The four explicit `ColumnDriftMetric` entries
override that choice where you know better — and the choice of test is the actual
teaching point:

| Test | Use it for | Intuition |
|------|-----------|-----------|
| **KS** (Kolmogorov–Smirnov) | continuous features (`distance_km`) | largest gap between the two cumulative distributions |
| **chi-square** | low-cardinality categoricals (`passengers`) | do the category counts differ more than chance allows? |
| **PSI** (Population Stability Index) | binned numerics, and anything you want a *magnitude* for (`hour_of_day`, `pred`) | weighted log-ratio of bin shares; the credit-risk industry standard |

KS and chi-square give you a **p-value**, which is sample-size sensitive — with a
million rows, a meaningless shift becomes "significant". PSI gives you an
**effect size** that doesn't inflate with volume, which is why it's the one used
for alerting thresholds.

Note that `pred` is checked here too: the model's own output is treated as just
another column to watch, which overlaps with script 2 on purpose.

### The two outputs

1. **`reports/drift_report.html`** — the rich visual report, one panel per feature
   with reference-vs-current distributions overlaid. This is what a human opens.
2. **`report.as_dict()`** — the same results as nested dicts, which is what the
   pipeline reads. The loop prints one line per column:

   ```
   distance_km          | ks           | p=0.0001 | 🚨 DRIFT
   passengers           | chisquare    | p=0.4210 | ✅ OK
   ```

### The gate

```python
drift_share = results["metrics"][0]["result"]["share_of_drifted_columns"]
if drift_share > 0.3:
    raise ValueError(f"Data drift alert: {drift_share:.0%} of features drifted!")
```

`results["metrics"][0]` is the `DataDriftPreset` result (it was listed first).
Raising is deliberate: as an Airflow task, an exception marks the task **failed**,
which fires the DAG's alerting and stops downstream tasks from consuming a
scoring run built on inputs the model has never seen. One drifted feature out of
ten is noise; a third of your schema moving at once is usually an upstream data
bug, not the world changing.

---

## 2. `label_predicion_drift.py` — prediction drift

> Filename has a typo — `predicion` → `prediction`. Renaming it means updating any
> Airflow/CI reference; noted here so it isn't mistaken for a different concept.

**The question:** the inputs may be fine, but has the *output* distribution moved?

This is the early-warning layer. If the model suddenly predicts 40-minute rides
where it used to predict 15-minute ones, something is wrong — and you know it
today, without waiting for a single actual ride to complete.

### Part A — PSI implemented from scratch

```python
def psi(expected, actual, n_bins=10) -> float:
    expected_perc = np.histogram(expected, bins=n_bins)[0] / len(expected)
    actual_perc   = np.histogram(actual, bins=np.histogram(expected, bins=n_bins)[1])[0] / len(actual)
    expected_perc = np.where(expected_perc == 0, 1e-4, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-4, actual_perc)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))
```

Three details worth reading closely:

- **The bin edges come from `expected`, and are reused for `actual`.**
  `np.histogram(expected, bins=n_bins)[1]` returns the edges; passing them into the
  second call forces both histograms onto the same grid. Letting numpy re-bin
  `actual` independently would compare incomparable buckets and quietly return
  nonsense.
- **Empty bins are floored at `1e-4`.** PSI takes `log(actual/expected)`; a bin
  that's empty on either side would produce `log(0)` → `-inf` or a divide-by-zero.
  The floor is the standard epsilon trick.
- **The formula is symmetric-ish by construction:** `(actual − expected) · log(actual / expected)`
  is always ≥ 0 per bin, so PSI accumulates rather than cancelling out.

Reading the score:

| PSI | Verdict |
|-----|---------|
| < 0.10 | stable — no action |
| 0.10 – 0.25 | moderate shift — investigate, watch it |
| > 0.25 | significant — retrain |

The script raises above `0.25`, same Airflow-gate pattern as script 1.

Note the reference here is **not** the training set — it's a past scoring run
(`2024-01-01.parquet`). You're comparing production-to-production, so a stable
model on stable traffic gives PSI ≈ 0 regardless of how the training data looked.

### Part B — Evidently `TargetDriftPreset`

```python
report = Report(metrics=[TargetDriftPreset()])
report.run(reference_data=ref_with_labels, current_data=curr_with_labels)
report.save_html("reports/target_drift.html")
```

This runs **later**, once ground truth has landed and you can join actual ride
durations back onto the predictions. `TargetDriftPreset` looks at the target
column and the prediction column together — it will show you target drift,
prediction drift, and the relationship between them (whether the model's errors
correlate with the shift).

Part A is what you run every day. Part B is what you run when labels arrive.

---

## 3. `hinkley_adwin.py` — concept drift on a live stream

**The question:** the inputs are fine and the outputs look normal — but is the
*relationship* between them still what the model learned?

That's concept drift, and it is invisible to scripts 1 and 2 by definition. A new
toll road opens: same distances, same passenger counts, same predicted durations —
but every prediction is now 6 minutes too high. Only the **error** reveals it.

Both detectors here are **online**: they consume one observation at a time and hold
O(1)-ish state, so they run inside a streaming consumer (session 3's Pub/Sub
function) rather than as a nightly batch job.

### Page-Hinkley

```python
ph = PageHinkley(
    min_instances=30,   # don't test until 30 samples have accumulated
    delta=0.005,        # magnitude of change tolerated before it counts
    threshold=50,       # λ — the alarm bar; higher = less sensitive
    alpha=0.9999,       # forgetting factor — weight recent observations more
)
```

Page-Hinkley is a **cumulative sum test**. It tracks the running mean of the error
and accumulates how far each new observation falls above it (minus `delta`, the
tolerance). When that accumulated excess crosses `threshold`, it declares drift.

The consequence: it detects a **sustained, directional shift**, not a spike. One
catastrophic prediction won't trip it; a permanent +6-minute bias will, after
enough samples to be sure. `delta` and `threshold` together are the
sensitivity/false-alarm dial — lower both to react faster and cry wolf more.

### ADWIN

```python
adwin = ADWIN(delta=0.002)   # delta = false-positive rate
```

ADWIN (ADaptive WINdowing) keeps a window of recent errors and repeatedly asks:
*can I split this window into two halves whose means differ more than chance
allows?* If yes, it declares drift and **drops the old half** — the window shrinks
to only post-drift data automatically.

Its `delta` means something different from Page-Hinkley's: here it's the
**bound on the false-positive rate**, so `0.002` ≈ "0.2% chance of a spurious alarm".
The big advantage is no window size to tune — ADWIN grows the window while things
are stable and shrinks it the moment they aren't. `adwin.width` after an alarm
tells you how far back the new regime starts.

### The monitoring loop

```python
for X, y_true in stream:
    y_pred = model.predict([X])[0]
    error  = abs(y_pred - y_true)       # regression: absolute error
    # error = int(y_pred != y_true)     # classification: 0/1 error

    ph.update(error)
    adwin.update(error)

    if ph.drift_detected:    ... trigger_retraining()
    if adwin.drift_detected: ... trigger_retraining()
```

Both detectors consume a **stream of scalars**, so the only real design decision is
what you feed them. For regression it's absolute error; for classification it's the
0/1 miss indicator (the commented line) — feed that and ADWIN is effectively
tracking rolling accuracy. Running both side by side is intentional: Page-Hinkley
is better on gradual, one-directional decay; ADWIN reacts faster to abrupt regime
changes.

Each alarm logs `concept_drift_ph` / `concept_drift_adwin` to MLflow so drift
events land on the same timeline as the training runs they'll trigger, and calls
`trigger_retraining()`.

### The closing comment

```python
# ── Without ground truth: monitor prediction distribution ─────
# Shift in the prediction distribution (PSI > 0.25) is an early
# warning for concept drift when ground truth isn't available yet.
```

This is the loop back to script 2, and the point of the whole session: this file
needs `y_true`, which you often don't have for hours or weeks. Until it arrives,
prediction PSI is the proxy.

---

## 4. `P_G_monitoring.py` — metrics for Prometheus

Scripts 1–3 run on a schedule and either pass or raise. That's fine for a nightly
gate, but it gives you no *history* — no way to ask "when did latency start
climbing?" or "did PSI creep up before or after last Tuesday's deploy?"

Prometheus answers those. It **pulls** (scrapes) a `/metrics` endpoint your app
exposes, stores every sample as a time series, and lets you query across time.
Grafana draws the result.

### The four metric types

```python
PREDICTION_HISTOGRAM = Histogram(
    "model_prediction_duration_min", "Distribution of predicted ride durations",
    buckets=[0,5,10,15,20,30,45,60,90,120]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "End-to-end API latency", ["endpoint", "status"]
)
DRIFT_GAUGE   = Gauge("feature_psi_score", "PSI drift score per feature", ["feature"])
MODEL_VERSION = Gauge("model_version_info", "Active model version", ["version", "stage"])
```

- **Histogram vs Gauge.** A histogram accumulates observations into buckets and
  only goes up; a gauge is a single value that moves both ways. Latency and
  predicted duration are histograms because you want *distributions* (p95, shape).
  PSI is a gauge because it's one number per feature that's recomputed each run.
- **The bucket boundaries are the design decision.** `[0,5,10,...,120]` is chosen
  to match how ride durations actually distribute — dense where most rides land,
  sparse in the tail. Prometheus can only ever tell you "how many predictions fell
  between 10 and 15 minutes", so buckets you pick badly are resolution you can
  never recover.
- **`PREDICTION_HISTOGRAM` is prediction drift, live.** It's the same signal
  script 2 computes as PSI, except continuous. Watch the heatmap shift downward
  over a week and you've seen drift without running a single batch job.
- **Labels multiply series.** `["endpoint", "status"]` means one series per
  combination. Keep label values low-cardinality — never put a user ID or
  timestamp in a label, or you'll create millions of series and take Prometheus
  down.

### The bridge from batch to live

```python
def record_drift_scores(psi_scores: dict):
    for feature, score in psi_scores.items():
        DRIFT_GAUGE.labels(feature=feature).set(score)
```

This is the seam between the two halves of the session: the Airflow drift task
computes PSI with script 2's function, then calls this to publish it. The batch
job produces the number; Prometheus keeps its history; Grafana plots it against
the deploy that caused it.

> **A caveat for batch jobs:** a short-lived Airflow task may exit before
> Prometheus ever scrapes it. For those, push to a **Pushgateway** rather than
> exposing a `/metrics` server the scraper will always miss.

### Exposing it

`start_http_server(8001)` (imported at the top of the file) starts a metrics
server on its own port. **8001, not 8000** — session 3's nginx already publishes
8000 on the host, and `monitoring/prometheus/prometheus.yml` is configured to
scrape 8001 accordingly.

---

## 5. The Docker stack — Prometheus, Grafana, Langfuse

```bash
cp .env.example .env

docker compose up -d                       # Prometheus + Grafana  (2 containers)
docker compose --profile langfuse up -d    # + Langfuse            (6 more)
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Prometheus | http://localhost:9095 | — |
| Langfuse | http://localhost:3001 | `admin@example.com` / `langfuse123` |
| MinIO console | http://localhost:9093 | `minio` / `miniosecret` |

Ports are picked around what the other sessions already run: 5000 (MLflow,
session 2), 8000 (session 1 API), 8080 (Airflow, session 3), and 9090/9091 — held
by a pre-existing Langfuse/MinIO stack, which is why **Prometheus is on 9095**
rather than its conventional 9090. All are overridable in `.env`.

Langfuse is behind a **compose profile** because it drags in Postgres, ClickHouse,
Redis and MinIO — six containers. You shouldn't need a ClickHouse cluster running
to look at a Grafana dashboard, so the default `up` gives you just the metrics
stack.

### Everything is provisioned as code

Nothing here requires click-through setup. On first boot Grafana already has the
Prometheus datasource wired and the dashboard loaded, because
`monitoring/grafana/provisioning/` declares both. Edit the JSON, wait 30s, refresh.

The dashboard renders exactly the PromQL sketched in the comments at the bottom of
`P_G_monitoring.py`, plus a few additions:

| Panel | Query |
|-------|-------|
| Request rate | `sum by (endpoint, status) (rate(api_request_latency_seconds_count[5m]))` |
| Latency p50/p95/p99 | `histogram_quantile(0.95, sum by (le) (rate(api_request_latency_seconds_bucket[5m])))` |
| Prediction distribution | `sum by (le) (rate(model_prediction_duration_min_bucket[1h]))` — heatmap |
| PSI per feature | `feature_psi_score`, red above 0.25 |
| Active model version | `model_version_info` |

Two idioms worth internalising: a histogram's `_count` series gives you a request
counter for free (no separate `Counter` needed), and `histogram_quantile` over
`_bucket` is *the* way to get percentiles — never average a latency.

### Alerts mirror the script thresholds

`monitoring/prometheus/alert_rules.yml` encodes the same numbers the scripts use,
so the batch gate and the live alert can't drift apart:

- `FeatureDriftHigh` — `feature_psi_score > 0.25` for 30m (script 2's retrain line)
- `FeatureDriftModerate` — the 0.10–0.25 watch band
- `PredictLatencyP95High` — p95 above 500ms for 10m
- `NoPredictionTraffic` — API is up but serving nothing
- `ModelAPIDown` — scrape target unreachable

The `for:` durations matter: without them a single noisy scrape pages someone.
No Alertmanager is wired up, so alerts are visible in the Prometheus UI but
nothing is delivered. Add an `alerting:` block pointing at an Alertmanager
service when you want Slack or email.

### Why the scrape target is `host.docker.internal`

Prometheus runs in a container; your model API runs on the host. Inside the
container, `localhost` means *the container*, so the config targets
`host.docker.internal:8001` (with an `extra_hosts` entry so this also works on
Linux, where it isn't built in). Containerise the API later and this becomes a
plain service name.

### Langfuse — and when it actually applies

Langfuse is **LLM observability**: traces of prompts, completions, token counts,
cost per call, and evaluation scores. It is not a replacement for Prometheus —
it answers a different question, and the ride-duration RandomForest has no LLM
in it at all.

It's included because it's the tool you reach for the moment the pipeline grows
an LLM component — an agent, a RAG step, an LLM-as-judge evaluator. The parallel
to this session is direct: prompt drift and response-quality drift are the same
problem as feature and prediction drift, on a different data type.

```python
from langfuse import get_client

# Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL from .env
langfuse = get_client()
```

Those keys work immediately — the `LANGFUSE_INIT_*` variables auto-provision the
org, project, API keys and login on first boot, so there's no UI setup step.

The worked example is [`ollama_langfuse_rag.py`](ollama_langfuse_rag.py), covered
in full in [section 6](#6-ollama_langfuse_ragpy--llm-observability).

Two ports are deliberately remapped from Langfuse's upstream compose file:
**langfuse-web 3000 → 3001** (Grafana holds 3000) and **MinIO 9090 → 9092**
(Prometheus holds 9090). Postgres, ClickHouse and Redis publish no host ports at
all — only the Langfuse containers need them, and session 3's Airflow stack may
already want 5432.

> **Every secret in `.env.example` is a public default.** Before this runs
> anywhere but a laptop, regenerate `LANGFUSE_SALT` and `LANGFUSE_NEXTAUTH_SECRET`
> (`openssl rand -base64 32`) and `LANGFUSE_ENCRYPTION_KEY`
> (`openssl rand -hex 32` — it must be exactly 64 hex characters), along with the
> database passwords. `.env` should never be committed.

---

## Running these

**The Python files are teaching snippets, not runnable jobs as they stand.**
`pip install -e .` gets every import to resolve, but each script has undefined
names you must supply. Deliberately, so it's clear what belongs to *your* pipeline:

| Script | Undefined / missing |
|--------|---------------------|
| `data_drift_evidently.py` | `today`; the parquet files under `data/` |
| `label_predicion_drift.py` | `import pandas as pd`; `today`; `ref_with_labels`, `curr_with_labels` |
| `hinkley_adwin.py` | `trigger_retraining()`; a caller for `monitor_live_predictions(model, stream)` — the file only *defines* it |
| `P_G_monitoring.py` | `app` (the FastAPI instance), `model`, `PredictRequest`; a `start_http_server(8001)` call — it's imported but never invoked |

Two things here **do** run as-is:

- **`ollama_langfuse_rag.py`** — verified end to end against Ollama `llama3.1:8b`
  and the compose Langfuse stack. All four demos pass and write real traces.
- **The Docker stack** — though it has nothing to scrape until the model API is
  up, so `model-api` shows as DOWN in Prometheus targets until `P_G_monitoring.py`
  is wired up.

> **If Langfuse rejects your keys**, `.env` is probably missing the SDK-side
> variables. The server reads `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` / `..._SECRET_KEY`;
> the SDK reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_BASE_URL`.
> Both sets must be present and the key values must match. An `.env` copied from an
> older `.env.example` will have only the first set, and `auth_check()` fails with
> "initialized without public_key".

To wire the scripts up:

```bash
mkdir -p reports data/scoring/output
export TODAY=$(date +%F)   # then read it in-script, or replace `today` with a param
```

`today` is written as a bare name because in production it's an Airflow template —
`{{ ds }}` passed into the task — not something the script computes itself. Same
for `trigger_retraining()`: in a real DAG it's a `TriggerDagRunOperator` call, not
a local function.

## Where this fits

- **Session 2** — train, track, register the model
- **Session 3** — orchestrate and deploy it
- **Session 4** — watch it, and decide when session 2 needs to run again

The three scripts here close that loop: drift detected → retraining triggered →
new model registered → redeployed → monitored.
