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
| 6 | [`langfuse_workload.py`](langfuse_workload.py) | Nothing new — it *generates the traffic* the others are read against | — |

The ordering matters: features drift first and are observable immediately;
predictions drift next; error drift is the ground truth of "the model is broken"
but it's the *last* signal to arrive, because labels lag predictions by hours or
weeks. You monitor 1 and 2 so you're not blind while waiting for 3.

Scripts 1–3 answer *"has it drifted?"* as a batch question. Script 4 and the
Docker stack answer *"how is it doing right now?"* — the same signals, exposed
continuously and drawn on a dashboard. Script 5 is the same idea again, on an LLM,
where the output is text and the score comes from a judge rather than a label.

Knowing *what* the instruments measure is half of it. The other half is being
able to read them under pressure, which is what the second half of this session
is for:

| Section | What it adds |
|---|---|
| [9 — the incident engine](#9-the-incident-engine--find-it-from-the-evidence) | eleven failures, each injected with one command. The terminal prints the symptom and nothing else; you find the cause from the dashboard, the drift report and the traces. |
| [10 — evaluating the LLM](#10-evaluating-the-llm--and-evaluating-the-evaluator) | a bilingual corpus, a judge you can read, and a measured Cohen's κ that says whether to believe it. |
| [RUNBOOK.md](RUNBOOK.md) | the six-step diagnostic path every alert links into. |

The first half is instrumentation. The second half is the skill, and it is the
part that does not transfer from reading.

### Contents

**Hands-on — do this first, with every command and query written out:**

- [Guide A — Prometheus](#guide-a--prometheus): run the service that exposes `/metrics`, and the queries to test it with
- [Guide B — Grafana](#guide-b--grafana): add the datasource, build a dashboard panel by panel, pick the right visualisation
- [Guide C — Evidently](#guide-c--evidently): freeze a reference, run a drift report, read it, gate on it
- [Guide D — Langfuse](#guide-d--langfuse): trace an LLM call, read the observation tree, attach scores, prove it end to end
- [Guide E — RAGAS](#guide-e--ragas): freeze a testset, score it with library metrics, read the distribution, gate CI on it

**Reference — what each script is and why it is written that way:**

1. [`data_drift_evidently.py` — feature drift](#1-data_drift_evidentlypy--feature-drift)
2. [`label_predicion_drift.py` — prediction drift](#2-label_predicion_driftpy--prediction-drift)
3. [`hinkley_adwin.py` — concept drift](#3-hinkley_adwinpy--concept-drift-on-a-live-stream)
4. [`P_G_monitoring.py` — metrics for Prometheus](#4-p_g_monitoringpy--metrics-for-prometheus)
5. [The Docker stack](#5-the-docker-stack--prometheus-grafana-langfuse)
6. [`langfuse_workload.py` — filling Langfuse with data](#6-langfuse_workloadpy--filling-langfuse-with-data)
7. [What we actually put in Langfuse](#7-what-we-actually-put-in-langfuse)
8. [Langfuse v2 vs v3 — why this stack has ClickHouse](#8-langfuse-v2-vs-v3--why-this-stack-has-clickhouse)
9. [The incident engine](#9-the-incident-engine--find-it-from-the-evidence)
10. [Evaluating the LLM](#10-evaluating-the-llm--and-evaluating-the-evaluator)
11. [Running these](#11-running-these)

## Project structure

```
session_4/
├── pyproject.toml                # deps + optional extras (dev/metrics/llm/model)
├── docker-compose.yaml           # Prometheus + Grafana + node-exporter; Langfuse profiled
├── .env.example                  # ports + secrets for the compose stack
├── .env                          # your real values — gitignored, never commit
├── .gitignore                    # ignores .env, reports/, data/, .venv/
├── data_drift_evidently.py       # feature drift — Evidently DataDriftPreset + per-column tests
├── label_predicion_drift.py      # prediction drift — hand-rolled PSI + Evidently TargetDrift
├── hinkley_adwin.py              # concept drift — river's Page-Hinkley + ADWIN on live error
├── P_G_monitoring.py             # Prometheus instrumentation for the model API
├── ollama_langfuse_rag.py        # LLM observability — traced RAG on a local Ollama model
├── langfuse_workload.py          # drives that pipeline in bulk — traces, scores, datasets
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml        # scrape config — host-run model API + node-exporter
│   │   └── alert_rules.yml       # PSI, latency, API down + the incident alerts
│       └── dashboards/           # four rows: service, resources, behaviour, quality
│   └── grafana/
│       ├── provisioning/         # datasource + dashboard providers (as code)
│       └── dashboards/
│           └── model-monitoring.json
│
│   ── the incident engine (section 9) ──────────────────────────────
├── Makefile                      # every lab command: up, seed, incident-NN, heal
├── RUNBOOK.md                    # the six-step diagnostic path; alerts link into it
├── services/
│   ├── ride_model.py             # the session 1-3 regressor: fit, freeze, generate
│   └── model_api.py              # P_G_monitoring.py made runnable — same metric names
├── incidents/
│   ├── inject.py                 # one entry point; prints symptoms, never causes
│   ├── catalog.py                # the failures + their symptoms, causes and signals
│   ├── state.py                  # the flag file the services re-read every second
│   └── postmortems/              # five-sentence template + worked example (01)
├── jobs/
│   ├── daily_drift.py            # Evidently vs a FROZEN reference, results STORED
│   ├── retrain_gate.py           # the five gates — "should session 2 run again?"
│   ├── concept_drift.py          # hinkley_adwin.py's detectors over real late labels
│   └── metrics_store.py          # SQLite: drift, labels, deploys, retrain decisions
├── tools/
│   ├── traffic.py                # the ride feed — 3 incidents live here; --bad-rate 422s
│   ├── seed.py                   # 14 days of history; without it every trend is invisible
│   ├── replay.py                 # serving vs offline on identical inputs — finds skew
│   ├── cost_report.py            # incident 08 — amortised cost, min/max week
│   └── doctor.py                 # pre-flight checklist; run it BEFORE the session
├── tests/                        # cardinality guard, metric contract, gates, links, masking
│
│   ── evaluating the LLM (section 10) ──────────────────────────────
├── evals/
│   ├── knowledge_base.py         # 44 notes, English + Arabic; folding and tokenization
│   ├── judge.py                  # the judge: fingerprint, 3 rubrics, per-language report
│   ├── calibrate_judge.py        # Cohen's kappa vs labels, reported per language
│   ├── calibration_set.jsonl     # 50 labelled items, 25 Arabic — labels by construction
│   ├── build_testset.py          # the frozen 60-item set + its content hash
│   ├── testset_v1.jsonl          # 30 en / 30 ar, incl. no-answer and two-note items
│   └── run_ragas.py              # RAGAS on Ollama: gate / nightly / compare
└── docs/
    └── langfuse_capability_map.md  # which file+lines demo which Langfuse feature
```

## Setup

```bash
pip install -e .              # pandas, pyarrow, evidently<0.7, river, mlflow
pip install -e ".[dev]"       # + pytest, ruff
pip install -e ".[metrics]"   # + prometheus-client, fastapi, uvicorn, httpx, psutil
pip install -e ".[llm]"       # + langfuse SDK v4, ollama  (Langfuse tracing)
pip install -e ".[evals]"     # + ragas, langchain-ollama  (sections 10-11)

docker compose up -d          # Prometheus + Grafana + node-exporter — see section 5
```

For sections 9–11 you want **one environment with all of it**, on Python 3.10–3.12:

```bash
pip install -e ".[dev,metrics,llm,evals]"
make doctor                   # 26 checks — run this before the session, not during
```

> **Both halves must live in ONE interpreter.** The metrics half needs
> numpy/scikit-learn/evidently (which is why `requires-python` is `<3.13`); the
> LLM half needs langfuse and ollama. Split them across two virtualenvs and
> `make seed --llm` dies on an import, which is a confusing way to discover an
> environment problem. `make doctor` checks for exactly this.

> **Why `evidently<0.7` is pinned.** These scripts use the legacy API —
> `from evidently.report import Report` and `evidently.metric_preset`.
> Evidently 0.7 replaced it with `evidently.Report` / `evidently.presets`.
> Installing the latest release will break every Evidently import here.
> [`jobs/daily_drift.py`](jobs/daily_drift.py) uses the same legacy API on
> purpose, so the runnable job and the snippet you read agree.

> **Why the langchain stack is pinned too.** `ragas` 0.4.x imports
> `langchain_community.chat_models.vertexai`, which no longer exists in the
> langchain 1.x generation — the import fails before any of your code runs.
> `ragas` 0.2.x against langchain 0.3.x is the combination that resolves. Pin
> your evaluation stack as tightly as you pin your model: a judge that changes
> is a ruler that changes, and every score before and after it is incomparable.

---

## Hands-on guides — run it, query it, read it

Sections 1–11 explain what each script **is**. These five guides are what you
**do**, in order, with every command and every query written out.

Run them in sequence, because each needs the one before it: Prometheus needs the
API running, Grafana needs Prometheus scraping, Evidently needs the serving log
the API writes.

**Guides D and E are the exception.** They watch an *LLM*, which the
ride-duration model does not have — so they share no state with A–C and can be
run on their own, before or after them. Run D before E: E scores the pipeline D
instruments.

| Guide | You end up with |
|---|---|
| [A — Prometheus](#guide-a--prometheus) | a service exposing `/metrics`, scraped, and a dozen queries you can read |
| [B — Grafana](#guide-b--grafana) | a datasource you added by hand and a dashboard you built panel by panel |
| [C — Evidently](#guide-c--evidently) | a frozen reference, a drift report you can read, and its numbers on the dashboard |
| [D — Langfuse](#guide-d--langfuse) | a traced LLM call you can read span by span, scored, and a round-trip check that proves it |
| [E — RAGAS](#guide-e--ragas) | a frozen testset, four library metrics read as distributions, and a gate CI can fail on |

---

## Guide A — Prometheus

### A.1 What the service does, in simple words

[`services/model_api.py`](services/model_api.py) is a small web service with one
job: **guess how many minutes a taxi ride will take.** You POST it a distance, a
passenger count and an hour of day; it runs a RandomForest and returns a number
of minutes.

The monitoring part is the second thing it does while answering. It keeps a
running tally — how many predictions it has made, how long each took, what
durations it has been predicting, what the latest drift scores were — and
publishes that tally as a page of plain text at `GET /metrics`.

It does not send that anywhere. **Prometheus comes and reads the page every 15
seconds and remembers each reading with a timestamp.** That is the entire
mechanism: your app writes a page of numbers, a scraper reads it on a timer, and
the history is the scraper's, not yours.

> **Why pull and not push.** Because the scraper's failure to reach you is itself
> a signal. If the API dies, `up{job="model-api"}` goes to 0 and you get an alert.
> A service that pushes its own metrics goes quiet when it dies, and quiet looks
> exactly like healthy-but-idle.

The endpoints, and what each is for:

| Endpoint | What it does |
|---|---|
| `POST /predict` | the prediction, and where the latency and duration histograms are recorded |
| `POST /feedback` | the **true** duration arriving late — this is what makes MAE possible at all |
| `POST /internal/gauges` | where the batch drift job posts its PSI numbers so they reach Prometheus (Guide C) |
| `GET /metrics` | the scrape target |
| `GET /health` | liveness, and which version is actually serving right now |

> `/metrics` is a real route here, not `app.mount("/metrics", make_asgi_app())`.
> The mount is the idiomatic one-liner, but Starlette then answers
> `GET /metrics` with a 307 to `/metrics/` — Prometheus follows redirects so
> scraping works, while `curl localhost:8001/metrics` prints nothing. That curl
> is the first thing anyone does when a target goes down.

### A.2 Run it — the exact sequence

**Prerequisites:** Docker running, and Python **3.10–3.12**. Not 3.13+ — the
legacy Evidently API this repo uses isn't published for it, which is why
`requires-python` says `<3.13`.

**Once:**

```bash
cd session_4
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,metrics]"     # add ,llm,evals for sections 10–11
cp .env.example .env
make bootstrap                      # fits the model, freezes the drift reference
```

`make bootstrap` writes `artifacts/rf_model.pkl`, `artifacts/scaler.pkl`,
`artifacts/scaler_stale.pkl` and `data/reference.parquet` (50k rows). The API
will not start without those `.pkl` files — it loads them at startup.

**Then, one terminal each:**

```bash
# terminal 1 — the stack
make up                    # Prometheus :9095, Grafana :3000, node-exporter :9100

# terminal 2 — the service that exposes /metrics
make api                   # uvicorn on :8001

# terminal 3 — something to measure
make traffic               # ~20 rps of rides until Ctrl-C
```

> **`make traffic` sends a small share of malformed requests on purpose** — 2% by
> default, tunable with `python -m tools.traffic --bad-rate 0.1`, and `0` turns it
> off. Real callers send bad requests, and without them the error-rate panel is a
> flat zero and the validation-failures panel is empty. A monitoring lab where
> the error rate *cannot* move teaches the wrong thing about what a green
> dashboard means. Each bad payload violates a **different field**, which is what
> makes `api_validation_failures_total{field=...}` worth splitting.

> If `make` picks the wrong interpreter (it defaults to `python3`, and a pyenv
> shim can shadow your venv), pass it explicitly: `make api PY=.venv/bin/python`.
> Same for every other target.

**Confirm each link in the chain, in order:**

```bash
curl -s localhost:8001/health                    # {"status":"ok","model":"rf_model","version":"v3"}
curl -s localhost:8001/metrics | head -30        # the page Prometheus reads
curl -s localhost:8001/metrics | grep -c '^[a-z]'  # ~40 samples on a healthy scrape
curl -s localhost:8001/metrics | grep '^api_process_'  # empty? psutil is missing
```

Then open **http://localhost:9095/targets**. Both `model-api` and `node` must show
**UP**. If one shows DOWN, the error is printed right there on the same row — see
[A.5](#a5-when-the-target-is-down).

> **A target that is UP is not the same as a target exporting what you expect.**
> A long-running `make api` started before a metric was added answers `/health`
> perfectly, scrapes green, and leaves half the dashboard empty — the panels say
> "No data" and nothing anywhere says why. `make doctor` checks the metric
> **names**, not just the target, precisely because those two failures look
> identical from the outside. Restarting the API is the fix.

> **Prometheus is on 9095, not 9090.** A pre-existing Langfuse/MinIO stack holds
> 9090/9091 on this machine, so `.env.example` moves it. Everything in this guide
> uses 9095; change `PROMETHEUS_PORT` in `.env` if you want it back.

### A.3 What metrics we actually collect

Grouped by the question each one answers. Every name below is defined in
[`services/model_api.py`](services/model_api.py) and is already referenced by the
shipped dashboard and alert rules — **rename one and you break both**, which is
what `tests/test_metric_registry.py` guards.

**Traffic and speed**

- **`api_request_latency_seconds`** — Histogram, labels `endpoint`, `status`.
  End-to-end time to serve one prediction. A histogram is really three families
  of series: `_bucket` (cumulative counts per boundary), `_sum`, and `_count`.
  The `_count` gives you a request counter for free — you never need a separate
  `Counter` for "how many requests"… with one important exception, below.

- **`api_requests_total`** — Counter, labels `endpoint`, `status`. Every request,
  by its **real** status. This is the exception to "the histogram's `_count` is
  free": the histogram is observed *inside* the handler with `status="200"`
  written into the call, so a request rejected before the handler body — a 422 —
  never reaches it. An error rate derived from `_count` is therefore a flat zero
  no matter how badly the service is failing. This counter is incremented by
  **middleware**, which is the only place that can see a request the handler
  refused.

  Two details in that middleware are worth copying. The `endpoint` label is the
  matched **route template**, never `request.url.path` — otherwise a 404 sweep
  mints one series per URL, which is incident 04 through a different door, so
  unmatched paths collapse to a single `endpoint="unmatched"`. And `/metrics` is
  skipped, because Prometheus scraping every 15s would otherwise put a permanent
  floor under your request-rate panel that no user produced.

**What the model is saying**

- **`model_prediction_duration_min`** — Histogram, buckets
  `[0,5,10,15,20,30,45,60,90,120]`. The distribution of predicted minutes. This
  is prediction drift, live and continuous — the same signal section 2 computes
  as a batch PSI. The bucket boundaries are a permanent design decision:
  Prometheus can only ever tell you "how many predictions fell between 10 and 15
  minutes", so resolution you don't buy in the buckets is resolution you can
  never recover.
- **`model_version_info`** — Gauge, labels `version`, `stage`. Which model is
  serving. The value is always 1; **the information is in the labels.** This is
  the standard "info metric" pattern.
- **`api_validation_failures_total`** — Counter, labels `endpoint`, `field`.
  Requests rejected by the schema, **and which field was wrong**. A bare 422
  count tells you callers are unhappy; the field name tells you which release
  broke them, and that is the difference between a dashboard that raises a
  question and one that answers it. The label space is bounded by the pydantic
  model, so it cannot explode.

**Resources of the serving process** — sampled at scrape time, like the disk gauge

- **`api_process_cpu_percent`** — Gauge. CPU used by the process serving
  predictions, as a percentage of **one core**: 100% means one core saturated,
  and the box has more than one. Read with `psutil.cpu_percent(interval=None)`,
  which reports usage since the previous call instead of blocking — passing an
  interval would sleep the scrape, and a `/metrics` endpoint that blocks is a
  monitoring system that causes outages.
- **`api_process_memory_bytes`** — Gauge, label `type` (`rss` / `vms`). Chart
  `rss`. On macOS `vms` reads as hundreds of gigabytes of address space and means
  nothing. A line climbing steadily across a week is a leak you can schedule; the
  same line hitting the OOM killer at 3am is an incident.
- **`api_process_open_fds`**, **`api_process_threads`** — Gauges. The classic slow
  leak: descriptors that climb and never fall mean something is not being closed,
  and the failure surfaces hours later as "Too many open files" in code that has
  nothing to do with the bug.

**Change tracking** — what the dashboard annotations are drawn from

- **`deploy_events_total`** — Counter, label `component`. One increment per newly
  observed deploy. `deploy_info` below cannot drive an annotation on its own:
  `.clear()` means it only ever holds the *latest* deploy, so its value changes
  but the series never **appears** — and Grafana draws a Prometheus annotation
  where a series increases. This counter is that increase.
- **`retrain_events_total`** — Counter, label `decision` (`retrain` / `hold`), and
  **`retrain_last_timestamp_seconds`** — Gauge, same label. Republished from the
  `retrain_events` table in SQLite. Only decisions the process has not seen before
  are counted: replaying the backlog would stamp a decision made last Tuesday with
  today's scrape time and draw its line in the wrong place.

**Drift** — computed by the batch job, not in the request path (Guide C)

- **`feature_psi_score`** — Gauge, label `feature`. PSI per feature against the
  frozen reference.
- **`segment_psi_score`** — Gauge, labels `feature`, `city`. The same, split by
  city. It exists because the global number is a volume-weighted average, and a
  new region at 8% of traffic cannot move it (incident 05).
- **`model_mae_minutes`** — Gauge, label `city`. Mean absolute error against the
  late labels. This is the only metric here that measures whether the model is
  actually **right**; everything else is a proxy you watch while waiting for it.

**The label loop**

- **`feedback_total`** — Counter. Late labels received. Watch this one: if it
  flatlines while traffic continues, your ground-truth pipeline is dead, MAE
  freezes at its last value, and every accuracy dashboard stays reassuringly
  green forever (incident 15).

**Operational**

- **`disk_free_bytes`** — Gauge, label `path`. Recomputed on **every scrape**,
  not on `/health`. A gauge refreshed only by an endpoint nobody scrapes is a
  flat line that looks like a healthy disk indefinitely.
- **`deploy_info`** — Gauge, labels `component`, `version`, and the **value is
  the deploy's unix timestamp**. RUNBOOK step 3 is "what changed, and when";
  this is how that question gets onto a time-series dashboard.

**Only while an incident is active**

- **`model_predictions_by_ride_total`** — Counter labelled by `ride_id`,
  registered lazily by incident 04. One time series per ride: the cardinality
  bomb, in one line of code.

**From another target** — node-exporter, no instrumentation of ours

- **`node_cpu_seconds_total`**, **`node_memory_MemAvailable_bytes`**,
  **`node_load1`** and the rest of the `node_*` family. The machine *under* the
  service. Read `MemAvailable`, not `MemFree`: `MemFree` looks alarming on every
  healthy Linux box because the kernel spends spare memory on page cache, while
  `MemAvailable` is the kernel's own estimate of what a new process could get.
  `tests/test_metric_registry.py` allows the whole `node_` prefix by name —
  asserting that our Python process exports `node_memory_MemTotal_bytes` would be
  asserting the wrong thing entirely.

**For free, from Prometheus itself** — no instrumentation needed

- **`up{job="model-api"}`** — 1 or 0 per target, per scrape.
- **`scrape_duration_seconds`**, **`scrape_samples_scraped`** — how long the last
  scrape took and how many samples it returned. `scrape_samples_scraped` is your
  cardinality tripwire; `prometheus.yml` sets `sample_limit: 5000` and a scrape
  above it is rejected **whole**.

**Why the resource metrics are named `api_process_*` and not `process_*`.**
`prometheus_client` ships a `ProcessCollector` that registers
`process_resident_memory_bytes`, `process_cpu_seconds_total` and friends
automatically — **by reading `/proc`**. macOS has no `/proc`, so on a Mac it
silently registers nothing: curl `/metrics` there and the only default collectors
present are `python_gc_*` and `python_info`. That is not a bug and not a missing
import.

So the service samples the same facts with **psutil**, which works on macOS and
Linux alike. The names are prefixed deliberately: on Linux `ProcessCollector`
*does* register the `process_*` family, and registering our own metric under a
name already in the registry is a duplicate-registration error at import time.
A different prefix is the difference between code that runs everywhere and code
that runs on your laptop. See
[B.6](#b6-latency-memory-and-resources--what-you-can-actually-see-here) for the
full picture of what is measurable where.

### A.4 Queries to test, from trivial to useful

Open **http://localhost:9095/query**. Work down the list — each one adds one idea.

**Level 0 — is anything there at all**

```promql
up
```
Two series. `1` means the last scrape succeeded. This is the single most useful
query in Prometheus and it needs no instrumentation from you.

```promql
model_prediction_duration_min_count
```
One ever-increasing number: predictions since the process started. Absent → the
API was never scraped. Present but flat → the API is up and serving nothing.

**Level 1 — `rate()`, and why you can't skip it**

```promql
rate(model_prediction_duration_min_count[5m])
```
Predictions per second, averaged over a 5-minute window. With `make traffic`
running this sits near 20.

**This is the central idea in PromQL.** Counters only ever go up, so the raw
value is an accumulation since process start — meaningless on its own and reset
to zero by every restart. `rate()` turns it into "per second right now" and
handles the resets for you. You will wrap essentially every counter in it.

```promql
sum by (endpoint, status) (rate(api_request_latency_seconds_count[5m]))
```
The same, split by endpoint and status.

> The dashboard's request-rate panel uses **`api_requests_total`** rather than
> this `_count`, and the difference is not cosmetic: the histogram is only
> observed inside the handler, so requests rejected before it — every 422 — are
> missing here. Compare the two on a run with
> `make traffic` and the gap *is* your error rate:
> `sum(rate(api_requests_total[5m])) - sum(rate(api_request_latency_seconds_count[5m]))`.

> Give the rate window at least 4× your scrape interval — 15s here, so `[1m]` is
> the floor and `[5m]` is the safe default. Too short and the query silently
> returns nothing, which looks identical to "the service is down".

**Level 2 — percentiles out of a histogram**

```promql
histogram_quantile(0.95, sum by (le) (rate(api_request_latency_seconds_bucket[5m])))
```
p95 latency, in seconds. Read it inside out:

1. `rate(..._bucket[5m])` — how fast each bucket is filling.
2. `sum by (le)` — collapse `endpoint` and `status`, but **keep `le`**, the label
   holding the bucket boundary. Drop `le` and `histogram_quantile` has nothing to
   work with.
3. `histogram_quantile(0.95, ...)` — interpolate the 95th percentile from those
   buckets.

Swap `0.95` for `0.50` and `0.99` for the usual trio.

> Layer by layer — why `rate` has to come first, why `le` must survive the `sum`,
> and how much of the answer here is interpolation rather than measurement — is
> in [B.4](#b4-the-panels-to-build-and-their-queries), where this becomes a panel.

> **Never average a latency.**
> `api_request_latency_seconds_sum / api_request_latency_seconds_count` is a
> number no user experienced. It is dragged down by the fast majority and hides
> precisely the tail you are being paged about.

**Level 3 — the model's own output**

```promql
sum by (le) (rate(model_prediction_duration_min_bucket[1h]))
```
The predicted-duration distribution over time. Draw it as a heatmap (B.4) and
watch the shape move down over a week — that is prediction drift, seen without
running a single batch job.

```promql
histogram_quantile(0.5, sum by (le) (rate(model_prediction_duration_min_bucket[30m])))
```
Median predicted ride length right now. If this moves and your **inputs** did
not, the problem is in the serving path, not the world (incident 02).

**Level 4 — the drift gauges** (need `make drift` first — Guide C)

```promql
feature_psi_score
```
Three series, one per feature. Gauges need no `rate()` — the value is already
the value.

```promql
feature_psi_score > 0.25
```
A comparison used as a **filter**: it returns only the series above the line, and
returns *nothing* when all is well. That "nothing" is what an alert rule is built
on — `alert_rules.yml` uses exactly this expression.

```promql
topk(3, feature_psi_score)
count(feature_psi_score > 0.25)
```
The worst three features, and how many are over the line. `count(...) > 2` is the
`HighDriftedFeatures` alert.

```promql
max by (city) (segment_psi_score)
```
Worst drift per city. Compare it against `max(feature_psi_score)`: if one city
sits far above the global number, the global average is hiding a region
(incident 05). This is the query that turns "the average looks fine" into a
finding.

**Level 5 — operations**

```promql
count(count by (version) (model_version_info))
```
How many distinct versions are serving. **Should be 1.** Two means a canary was
started and never finished (incident 10). Inside out: `count by (version)` gives
one entry per version, the outer `count` counts those entries.

```promql
scrape_samples_scraped{job="model-api"}
```
Samples in the last scrape. Healthy is around 28 against a `sample_limit` of
5000 — roughly 180× headroom. Watch this climb during incident 04; when it
crosses the limit the scrape is rejected **entirely** and the target is marked
down, which turns a Prometheus-wide OOM into one loud, local, recoverable
failure that names itself on the Targets page.

```promql
rate(feedback_total[6h]) == 0
```
The ground-truth pipeline has gone silent.

```promql
predict_linear(disk_free_bytes[6h], 4 * 3600) < 0
```
Fit a line through the last 6 hours, extrapolate 4 hours forward, and fire if the
prediction is below zero — i.e. "you run out of disk within 4 hours". **Alerting
on a trend rather than a threshold** is the difference between a ticket and a
3am page.

**Instant vs range.** The **Table** tab evaluates the expression once, at *now*.
The **Graph** tab evaluates it repeatedly across the range. An expression that
looks empty on one may be fine on the other — check both before concluding the
metric is missing.

**Then look at the alerts.** http://localhost:9095/alerts lists all nine rules
from `monitoring/prometheus/alert_rules.yml`. Note the three states: **Inactive**,
**Pending** (the condition is true but the `for:` duration hasn't elapsed) and
**Firing**. The `for:` durations are the entire reason the alerts are usable —
without them one noisy scrape pages someone. No Alertmanager is wired up, so
alerts are visible here and delivered nowhere; add an `alerting:` block pointing
at an Alertmanager service when you want Slack or email.

### A.5 When the target is down

| Row on `/targets` says | Cause | Fix |
|---|---|---|
| `connection refused` | the API isn't running | `make api` in another terminal |
| `connection refused`, API *is* running | it bound to the wrong port, or `make` used the wrong Python | `make api PY=.venv/bin/python`; confirm with `lsof -nP -iTCP:8001 -sTCP:LISTEN` |
| `no such host: host.docker.internal` | Linux without the `extra_hosts` mapping | it's already in `docker-compose.yaml`; recreate with `docker compose up -d --force-recreate prometheus` |
| `sample limit exceeded` | cardinality explosion — incident 04 is active | `make heal`. Note the API must also **unregister** the collector, which it does; in production this is the restart everyone forgets they need |
| target isn't listed at all | config not reloaded | `docker compose restart prometheus`, then re-check |
| `Address already in use` on `make api` | something else holds 8001 | `lsof -nP -iTCP:8001` and kill it, or run uvicorn on another port and update `prometheus.yml` |

> **Why the target is `host.docker.internal:8001` and not `localhost:8001`.**
> Prometheus runs in a container; the API runs on your host. Inside the
> container, `localhost` means *the container itself*. Containerise the API and
> this becomes a plain service name.

---

## Guide B — Grafana

### B.1 Get in

**http://localhost:3000** — `admin` / `admin`. Skip the password prompt; it's local.

### B.2 The datasource

**It is already there.** `monitoring/grafana/provisioning/datasources/prometheus.yml`
is read on boot, so **Connections → Data sources** already lists *Prometheus*,
marked default. Open it and hit **Save & test** — you want
"Successfully queried the Prometheus API."

Now add one **by hand anyway**, because you will do it by hand on every real
Grafana you ever touch:

1. **Connections → Data sources → Add new data source**
2. Choose **Prometheus**.
3. **Prometheus server URL** — the only field people actually get wrong. The
   right answer depends on where Grafana is standing:

   | Grafana is... | Prometheus is... | URL |
   |---|---|---|
   | in this compose project | in this compose project | `http://prometheus:9090` |
   | native on your laptop | in Docker | `http://localhost:9095` |
   | in Docker | native on your host | `http://host.docker.internal:9095` |

   For our stack it is **`http://prometheus:9090`** — the **service name** and the
   **container's internal port**. Not `localhost` (inside the Grafana container
   that means Grafana), and **not 9095** (9095 is the host-side mapping;
   containers on the same network talk on the internal port and never see it).
4. **Interval behaviour → Scrape interval** → `15s`, matching `prometheus.yml`.
   This is what lets Grafana suggest sane `$__rate_interval` values later.
5. No auth — this Prometheus has none.
6. **Save & test.**

> **Don't delete the provisioned one.** It has a fixed `uid: prometheus`, and
> `monitoring/grafana/dashboards/model-monitoring.json` references the datasource
> by that uid. Recreate it by hand and Grafana assigns a random uid, at which
> point every panel in the shipped dashboard renders "Datasource not found".
> Fixed uids in provisioning are exactly why.

### B.3 Build a dashboard, step by step

The shipped one (**Ride Duration — Model Monitoring**) is already loaded. Build a
second from scratch — you learn nothing from a dashboard someone else's JSON drew.

1. **Dashboards → New → New dashboard → Add visualization.**
2. Pick **Prometheus**.
3. You land in the panel editor: preview on top, **query editor** bottom-left,
   **options sidebar** on the right.
4. In the query row, switch **Builder → Code**. The builder is fine for one bare
   metric; every query in this guide is faster typed.
5. Paste `sum by (endpoint, status) (rate(api_requests_total[$__rate_interval]))`
   and press **Shift+Enter**.
6. **Set the time range first** (top right) → **Last 15 minutes**, refresh **10s**.
   The default *Last 6 hours* makes a three-minute-old service look like a flat
   line hiding at the right edge, and people conclude the query is broken.
7. **Legend** field, under the query → `{{endpoint}} {{status}}`. Without it every
   series is labelled with its entire label set and the legend is unreadable.
8. Sidebar → **Visualization type** (top dropdown) → **Time series**.
9. Sidebar → **Panel options → Title** → `Request rate`.
10. Sidebar → **Standard options → Unit** → *Throughput → requests/sec (rps)*.
    Units are not decoration: an unlabelled latency panel reading `0.0234` gets
    read as milliseconds by everyone who glances at it.
11. **Save dashboard** (top right), name it, pick a folder.
12. Back on the dashboard: **Add → Visualization** for the next panel. Drag a
    panel's bottom-right corner to resize, its title to move it.

**Add a variable** once you have three or four panels — this is where a dashboard
stops being a picture and becomes a tool:

- **Dashboard settings → Variables → New variable**
- Type **Query**, datasource **Prometheus**, Query type **Label values**,
  label `city`, metric `segment_psi_score`. Name it `city`.
- Now `segment_psi_score{city="$city"}` follows the dropdown at the top of the
  dashboard, and one panel serves every region.

> **Anything you build in the UI lives only in Grafana's volume.**
> `docker compose down -v` deletes it without asking. Export it —
> **Dashboard settings → JSON Model**, or **Share → Export → Save to file** — and
> commit it next to `model-monitoring.json`. Dashboards are code or they are lost.

### B.4 The panels to build, and their queries

Build them in these four rows. Each row answers a different question, and the
order is the order you'd read them during an incident.

**Row 1 — Service: what the caller experiences** (build these first; they're the
ones you look at when paged). Everything here is a **symptom**, which is why the
alerts on this row page and the rest only warn.

| Panel | Query | Viz | Unit |
|---|---|---|---|
| Request rate | `sum by (endpoint, status) (rate(api_requests_total[$__rate_interval]))` | Time series | reqps |
| Latency p50 / p95 / p99 | three queries: `histogram_quantile(0.50, sum by (le) (rate(api_request_latency_seconds_bucket[$__rate_interval])))`, then `0.95`, then `0.99` | Time series | seconds (s) |
| Error rate | `100 * sum(rate(api_requests_total{status!~"2.."}[$__rate_interval])) / sum(rate(api_requests_total[$__rate_interval]))` | Time series | percent (0–100) |
| Target up | `up{job="model-api"}` | Stat + value mappings | — |

> **`$__rate_interval`, not a hardcoded `[5m]`.** Grafana expands it to at least
> four times the scrape interval it knows from the datasource, and widens it when
> you zoom out. A literal `[5m]` is correct at one zoom level and silently wrong
> at every other: zoom to seven days and a 5-minute window samples a tiny
> fraction of each pixel's worth of data, producing a spiky line that looks like
> instability and is really aliasing.

**What the latency query actually says.**

```promql
histogram_quantile(0.50, sum by (le) (rate(api_request_latency_seconds_bucket[5m])))
```

In one sentence: *over the last 5 minutes, half of all requests — every endpoint,
every status — finished faster than this many seconds.* That is the **median**,
and it is emphatically not the average.

It reads inside out, and every layer is load-bearing:

| Layer | What it does | What happens without it |
|---|---|---|
| `api_request_latency_seconds_bucket` | The raw histogram — not one series but **one counter per boundary**, labelled `le` ("less than or equal"). They are **cumulative**: `le="0.1"` counts every request that took ≤ 0.1s, *including* the ones already counted by `le="0.005"`. | — |
| `rate(…[5m])` | Turns those ever-climbing counters into per-second rates over a 5-minute sliding window. | You get the percentile **since process start** — a lifetime average that still carries the slow first request after boot and barely moves for the rest of the day. |
| `sum by (le)` | Adds those per-second rates across every `endpoint`/`status` series, **keeping `le`**. | `sum(…)` without `by (le)` hands the function a series with no bucket boundaries on it. Nothing to interpolate, empty panel. |
| `histogram_quantile(0.50, …)` | Walks the cumulative buckets to find where the 50% mark falls, then **linearly interpolates inside** that bucket. | — |

**Why the aggregation is on the inside.** The tempting shape is "compute p95 per
endpoint, then average". **Percentiles do not average.** The mean of two
endpoints' p95s is not the p95 of the service, and no amount of weighting fixes
it. Summing the *bucket rates* first rebuilds one combined histogram out of the
per-endpoint ones, and the percentile is then taken from that — which is the
real one. This is the whole reason `sum by (le)` sits where it does.

**`0.50` is the only thing that changes** between the three panel queries. p50,
p95 and p99 are the same expression with a different first argument — that is why
the table above says "then `0.95`, then `0.99`" instead of writing them out.
Name the series `p50` / `p95` / `p99` in each query's **Legend** field or the
legend shows the full expression three times.

**The interpolation is a guess, and here it is a coarse one.**
`histogram_quantile` assumes observations are spread *uniformly* within whichever
bucket the quantile lands in. They never are. Your precision is capped by the
boundaries chosen at instrumentation time — and `REQUEST_LATENCY` is declared
with no `buckets=` argument ([`services/model_api.py:61`](services/model_api.py#L61)),
so it inherits `prometheus_client`'s defaults, which are tuned for web requests
and measured in seconds:

```
.005  .01  .025  .05  .075  .1  .25  .5  .75  1  2.5  5  7.5  10  +Inf
```

A local scikit-learn predict is a couple of milliseconds, so on this API very
nearly every request lands in the **first** bucket, and `histogram_quantile` is
then drawing a straight line between 0 and 0.005 through a bucket that holds all
of your data. Check how bad it is on your machine:

```promql
rate(api_request_latency_seconds_bucket{le="0.005"}[5m])
  / rate(api_request_latency_seconds_count[5m])
```

If that sits near `1`, your p50 line is interpolation rather than measurement,
and the fix is **new buckets, not a better query**. Compare
`model_prediction_duration_min`, which declares its boundaries deliberately, and
the note in [A.3](#a3-what-metrics-we-actually-collect): resolution you don't buy
in the buckets is resolution you can never recover.

**p99 cannot exceed your largest finite bucket.** If the quantile falls in the
`+Inf` bucket there is no upper bound to interpolate toward, and Prometheus
returns the largest finite boundary — `10` here. A p99 sitting at exactly 10s is
not a measurement, it's the ceiling of the instrument.

**A blank panel is ambiguous.** No observations in the window means no
distribution to take a percentile of, which produces `NaN` and draws nothing —
identical on screen to the service being down. Same trap as the rate-window note
in [A.4](#a4-queries-to-test-from-trivial-to-useful): keep the window at least 4×
the scrape interval, and read **Target up** next to this panel, never alone.

For **Target up**, set sidebar → **Value mappings**: `1` → `UP` (green), `0` →
`DOWN` (red). A stat panel showing a bare `1` communicates nothing.

> **Why the error rate is built on `api_requests_total` and not on the
> histogram.** The obvious query is
> `sum(rate(api_request_latency_seconds_count{status!="200"}[5m])) / sum(rate(api_request_latency_seconds_count[5m]))`
> — and it is **always zero**, however badly the service is failing. Two
> independent reasons, and both are worth recognising in your own code:
>
> 1. `model_api.py` observes the histogram as
>    `REQUEST_LATENCY.labels(endpoint="/predict", status="200")` with the status
>    **hardcoded**. The instrumentation cannot express a failure.
> 2. Even with the real status substituted, the observation sits **inside the
>    handler body**. A request rejected by schema validation never gets there, so
>    a 422 is invisible to it — the handler is the wrong place to count requests
>    from.
>
> The fix that ships is the middleware described in
> [A.3](#a3-what-metrics-we-actually-collect): counting happens *outside* the
> handler, labelled with the real status code. **Instrumentation that can only
> report success is a common and very expensive bug** — and the version of it
> that survives longest is the one where somebody fixed reason 1 and never
> noticed reason 2.

**Row 2 — Resources: saturation, before anyone feels it**

Two sources on purpose, and the distinction is the lesson: `api_process_*` is the
**model server itself**, `node_*` is the **machine under it**. Only the first one
wakes you up — see [B.6](#b6-latency-memory-and-resources--what-you-can-actually-see-here).

| Panel | Query | Viz | Unit |
|---|---|---|---|
| Model process · CPU | `api_process_cpu_percent` | Time series | percent (0–100) |
| Model process · resident memory | `api_process_memory_bytes{type="rss"}` | Time series | bytes(IEC) |
| Host / VM · CPU busy | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[$__rate_interval])) * 100)` | Time series | percent |
| Host / VM · memory used | `100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)` | Time series | percent |
| Disk free + 4h projection | `disk_free_bytes` **and** `predict_linear(disk_free_bytes[6h], 4 * 3600)` | Time series | bytes(IEC) |
| File descriptors and threads | `api_process_open_fds` **and** `api_process_threads` | Time series | short |
| Samples per scrape | `scrape_samples_scraped{job="model-api"}` | Time series + threshold `5000` | short |

*Host CPU from idle time*: `mode="idle"` is the only mode guaranteed to exist on
every platform, so "busy" is computed as what is left over rather than by summing
the modes you hope are present.

*Disk free with a projection*: the second series **is** the `DiskFillPredicted`
alert, drawn. A disk at 85% tells you nothing; a disk losing 4 GB/hour tells you
when it dies. When the projection crosses zero you have four hours — a ticket
rather than an outage. Note it uses the service's own `disk_free_bytes` and not
`node_filesystem_avail_bytes`: inside Docker Desktop the latter reports the VM's
synthetic mounts, including a "40 TB" volume that does not exist.

*Samples per scrape*: the threshold at `5000` draws `sample_limit` right on the
chart, so the cardinality budget is visible instead of being a number in a YAML
file. A healthy scrape is ~30 samples; incident 04 walks this line to the limit
in about two minutes, and a scrape **above** it is rejected whole — the target
goes down and takes every other panel with it. This is saturation of the
monitoring system itself, which is why it belongs in this row.

**Row 3 — Model behaviour: what the model is saying**

This is a **regressor predicting minutes**. There is no score distribution and no
positive rate here because there is no score and no positive class — the output
distribution *is* a histogram of minutes, and its tail is the early warning.

| Panel | Query | Viz | Unit |
|---|---|---|---|
| Predicted duration distribution | `sum by (le) (rate(model_prediction_duration_min_bucket[$__rate_interval]))` | **Heatmap** | m |
| Long-ride rate (> 60 min) | `100 * (1 - (sum(rate(model_prediction_duration_min_bucket{le="60.0"}[$__rate_interval])) / sum(rate(model_prediction_duration_min_count[$__rate_interval]))))` | Time series | percent |
| Validation failures by field | `sum by (field) (rate(api_validation_failures_total[$__rate_interval]))` | Time series (stacked) | reqps |
| Predictions served (24h) | `sum(increase(model_prediction_duration_min_count[24h]))` | Stat | short |
| Active model version | `model_version_info` | Stat (Fields → version) | — |
| p99 predicted duration | `histogram_quantile(0.99, sum by (le) (rate(model_prediction_duration_min_bucket[$__rate_interval])))` | Stat | m |

> **`le="60.0"`, not `le="60"`.** `le` is a **string** label, and Prometheus
> renders the bucket boundary as the float `60.0`. `le="60"` matches nothing and
> the panel is empty — with no error, because an empty result is not an error.
> Check your own boundaries with `curl -s localhost:8001/metrics | grep _bucket`
> before writing an exact-match selector against them.

*Long-ride rate* is the regression answer to "positive rate". There is no positive
class to count, so the equivalent early-warning signal is the **share of
predictions in the tail** — computed from the cumulative bucket as
"100% minus everything at or below 60 minutes".

*Validation failures by field* is deliberately split by `field`. "422s are up"
starts an investigation; "422s are up on `passengers`" ends one. A spike on a
single field within minutes of a deploy is a contract change nobody announced.

For the heatmap, set the query row's **Format: Heatmap** and **Legend: `{{le}}`** —
that's how Grafana knows `le` is a bucket boundary rather than just another label.

For **Active model version**, the number is always 1, so use a **Stat** with
*Fields → version*, or a **Table** with the value column hidden. The lesson is
that info metrics carry their meaning in labels. Two entries here means an
unfinished canary (incident 10), and every aggregate on the dashboard is then a
blend of two models.

**Row 4 — Quality & drift: is it still right?** (needs Guide C)

Everything above this row can be green while this row rots. Drift is the leading
indicator, late-label MAE is the truth, and they arrive days apart.

| Panel | Query | Viz | Unit |
|---|---|---|---|
| Features above the PSI bands | `count(feature_psi_score > 0.25) or vector(0)` **and** `count(feature_psi_score > 0.10 and feature_psi_score <= 0.25) or vector(0)` | Time series (stacked) | short |
| MAE on late labels, by city | `model_mae_minutes{city=~"$city"}` | Time series | m |
| Retrain decisions in range | `sum by (decision) (increase(retrain_events_total[$__range]))` | Bar gauge | short |
| Since last promoted retrain | `time() - max(retrain_last_timestamp_seconds{decision="retrain"})` | Stat | seconds (s) |
| PSI per feature, now | `feature_psi_score` | Bar gauge, thresholds 0.10 / 0.25 | short |
| PSI by city | `segment_psi_score`, legend `{{feature}} · {{city}}` | Time series | short |
| Late-label feedback rate | `rate(feedback_total[$__rate_interval])` | Time series | reqps |

> **`or vector(0)` is not decoration.** `count()` over an empty set returns **no
> samples at all**, so without it the panel shows a gap — and a gap reads as
> "nothing drifted" when it actually means "nothing reported". Those two states
> need to look different, and this is the cheapest way to make them.

*Retrain decisions* shows **holds next to retrains** on purpose. A gate that only
displays its yeses cannot be audited after it promotes something it should not
have, and "why didn't we retrain" is the question the panel exists to answer.

On the two PSI panels, sidebar → **Thresholds** → add `0.10` (amber) and `0.25`
(red), then **Show thresholds: As filled regions and lines**. Now the bands from
section 1 are on the chart itself and nobody has to remember them.

Put **PSI per feature** and **PSI by city** side by side. That pairing *is*
incident 05: a flat global line next to one city climbing past 0.25.

**Useful panels that didn't make the four rows**, worth adding locally:

| Panel | Query | Viz | Unit |
|---|---|---|---|
| Hours until disk full | `disk_free_bytes / -deriv(disk_free_bytes[1h]) / 3600` | Stat | hours (h) |
| Scrape duration | `scrape_duration_seconds` | Time series | seconds (s) |
| Worst feature now | `topk(1, feature_psi_score)` | Stat | short |
| Last deploy | `deploy_info` | Table | Datetime → From Now |

*Hours until disk full*: `deriv()` gives bytes/second, negative while the disk
fills, so negating it and dividing free space by it gives seconds-to-zero. A
negative or absent result means the disk isn't filling — map it to "—" with a
value mapping rather than showing a confusing negative. The shipped dashboard
uses `predict_linear` instead, which answers the same question in the same units
as the series it sits on.

*Last deploy*: `deploy_info`'s **value is a unix timestamp**, so set the unit to
*Datetime → From Now* and the table reads "23 minutes ago".

**Overlay deploys and retrains on every panel** — this is RUNBOOK step 3, "what
changed and when", made visual. Both ship in the dashboard JSON:

- **Dashboard settings → Annotations → New annotation query**
- Data source **Prometheus**, expression
  `increase(deploy_events_total[$__rate_interval]) > 0`, Title `Deploy`, colour red.
- A second one for `increase(retrain_events_total[$__rate_interval]) > 0`,
  Title `Retrain`, colour blue.

> **Why a counter and not `deploy_info`.** The obvious expression is `deploy_info`
> — and it draws nothing useful. `_refresh_deploy_gauge()` calls `.clear()` before
> republishing, so the gauge only ever holds the **latest** deploy: its *value*
> changes on each new release but the series never **appears**, and Grafana draws
> a Prometheus annotation where a series **increases**. `deploy_events_total`
> is that increase — one increment per newly observed deploy.
>
> The `.clear()` is not the bug, it is the right call: without it every version
> ever deployed would linger as its own dead series forever. The lesson is that
> **"latest value" and "something happened" are different shapes of metric**, and
> an annotation needs the second one.

> **Annotations are forward-looking, and that is honest.** Both counters only
> count events the *running process* has not seen before, so the fourteen days of
> seeded history in `metrics_store.db` do not paint fourteen days of lines. They
> could not: a deploy made last Tuesday would be recorded at today's scrape
> timestamp and draw its line in the wrong place, which is worse than no line.
> The full history stays queryable in the `deploys` and `retrain_events` tables —
> a time-series database is the wrong place to keep a changelog, and this is a
> good illustration of why the repo keeps both.

**Add the `$model` variable** while you are in dashboard settings:

- **Settings → Variables → New → Query**, name `model`,
  query `label_values(up{job="model-api"}, model)`.
- Then select it in a panel query as `api_requests_total{model="$model"}`.

The label comes from the scrape target in `prometheus.yml`, not from the
instrumentation — so a second model becomes a second entry in the dropdown with
**no dashboard edit at all**. That is the difference between one dashboard and
twelve near-identical copies nobody maintains.

### B.5 Visualization types and aggregation

**Pick the visualization from the shape of the answer**, not from what looks nice:

| The answer is… | Use | Here |
|---|---|---|
| many values over time | **Time series** | latency, PSI, request rate |
| one number, right now | **Stat** | p95 now, API up, predictions today |
| one number against a known ceiling | **Gauge** | PSI against 0.25 |
| a distribution changing over time | **Heatmap** | predicted-duration buckets |
| a distribution right now | **Bar gauge** | MAE per city |
| labels, not numbers | **Table** | `model_version_info`, `deploy_info` |
| a handful of series where only the state matters | **State timeline** | `up` per target |

**The aggregation ladder.** Every query in this guide is four moves, always in
this order:

1. **Select.** `feature_psi_score{feature="distance_km"}`. Matchers are `=`,
   `!=`, `=~` (regex), `!~`.
2. **Rate.** `rate(x[5m])` for per-second, `increase(x[24h])` for a total over a
   window. **Counters only.** Gauges — every PSI and MAE metric here — skip this
   step entirely; rating a gauge is meaningless.
3. **Aggregate across series.** `sum`, `avg`, `max`, `min`, `count`, `topk`.
   Two forms, and the difference is worth internalising:
   - `sum by (city) (...)` — **keep** `city`, collapse everything else.
   - `sum without (instance) (...)` — **drop** `instance`, keep the rest. Safer
     when you don't control every label.
   - Bare `sum(...)` collapses to one line.

   **Always `rate()` inside `sum()`, never the reverse.** `rate(sum(...))` is
   wrong: summing counters first hides the resets, and `rate` then reads a
   restart as a huge negative jump.
4. **Post-process.** `histogram_quantile`, `predict_linear`, `deriv`, arithmetic
   between series, and comparisons used as filters (`> 0.25`).

**Grafana's aggregation is a separate, later thing.** The sidebar's
**Legend → Values** (Last, Max, Mean) computes over what is *already plotted* —
a display calculation, not part of the query. Use it to turn a legend into a
summary table. Do **not** use it to answer "what was p95 over the day": Grafana's
"Mean" of a p95 series is an average of percentiles, which is not a percentile of
anything.

**Two settings that make dashboards fast:**

- **Instant queries for Stat panels.** Query row → **Options → Type: Instant**.
  It evaluates once at *now* instead of computing the full range and discarding
  all but the last point.
- **`$__rate_interval` instead of a hardcoded `[5m]`.** Grafana computes it from
  the panel width and the datasource's scrape interval — which is why you set
  `15s` in B.2. Zoom out and the window widens automatically instead of the graph
  emptying out. Use it once you're past the learning stage.

### B.6 Latency, memory and resources — what you can actually see here

**Latency** is fully covered: the histogram is instrumented, and
`histogram_quantile` over `_bucket` is the answer. Nothing more to add.

**Memory and CPU need two different answers, and the Resources row draws both.**

`prometheus_client` ships a `ProcessCollector` that registers
`process_resident_memory_bytes`, `process_cpu_seconds_total`, `process_open_fds`
and friends **automatically** — by reading `/proc`. macOS has no `/proc`, so on a
Mac it silently registers nothing. Curl `/metrics` on macOS and the only default
collectors present are `python_gc_*` and `python_info`. Nothing errors; the
metrics are simply absent.

That is why this repo samples them with **psutil** instead, under the
`api_process_*` names ([A.3](#a3-what-metrics-we-actually-collect) explains why
the prefix differs). It is the smallest thing that works identically on macOS and
Linux, and it measures the process that actually matters.

| You want | Where it comes from here | Notes |
|---|---|---|
| API process RSS, CPU, fds, threads | **`api_process_*`** (psutil, in `/metrics`) | works on macOS and Linux alike; this is the model server itself |
| API process RSS/CPU, no code | `process_resident_memory_bytes{job="model-api"}` | Linux/containers only — free, but absent on macOS |
| Host (or VM) CPU/memory/load | **`node_*`** (node-exporter, `job="node"`) | on Docker Desktop this is the **LinuxKit VM**, not your laptop |
| Prometheus's own RSS | `process_resident_memory_bytes{job="prometheus"}` | works everywhere — it runs in a Linux container |
| Per-container CPU/memory | add **cAdvisor** | see below; same VM caveat |
| Disk on the real serving path | **`disk_free_bytes`** (our gauge, host path) | `node_filesystem_*` reports the VM's synthetic mounts |

**Why both `api_process_*` and `node_*` are on the same row.** node-exporter
answers *"is the box saturated"*; psutil answers *"is the model server
saturated"*. They disagree constantly and both are correct — a box at 30% CPU
with one pegged worker is a real outage, and only the second number shows it.
**Only the second one should wake you up.** The first tells you whether to fix
the service or ask for a bigger machine.

Further ways to get resource numbers, cheapest first:

**1. Watch Prometheus's own — already scraped, nothing to install.** It's also
the process here most likely to actually run out of memory, which is precisely
what `sample_limit` exists to prevent:

```promql
process_resident_memory_bytes{job="prometheus"}
prometheus_tsdb_head_series
rate(prometheus_tsdb_head_samples_appended_total[5m])
```

`prometheus_tsdb_head_series` is *the* number to watch during incident 04 — a
cardinality explosion in a single line, and the clearest possible demonstration
of why a `ride_id` label is a catastrophe rather than a nice-to-have.

**2. Containerise the API.** Run it inside the compose network and it executes on
Linux, so the whole `process_*` family appears with no code change at all — the
psutil gauges then sit alongside it rather than standing in for it. It also makes
`host.docker.internal` unnecessary: the scrape target becomes a plain service
name.

**3. Add cAdvisor** for per-container CPU and memory:

```yaml
  cadvisor:
    image: gcr.io/cadvisor/cadvisor       # pin whatever the current tag is
    container_name: session4-cadvisor
    ports: ["8081:8080"]
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
```

plus a scrape job in `prometheus.yml`:

```yaml
  - job_name: cadvisor
    static_configs:
      - targets: ["cadvisor:8080"]
```

then:

```promql
rate(container_cpu_usage_seconds_total{name=~"session4-.*"}[5m])
container_memory_working_set_bytes{name=~"session4-.*"}
```

> **On Docker Desktop for Mac, cAdvisor sees the LinuxKit VM, not your laptop.**
> Per-container numbers are real and useful; anything it reports as "host" is the
> VM's, not macOS's. Don't build a memory alert on it and expect it to describe
> your machine.

**The two saturation signals most specific to this service** are both on Row 2:
`scrape_samples_scraped` (the cardinality budget — incident 04 walks it to
`sample_limit` in about two minutes) and `disk_free_bytes` (the serving log is
append-only and will fill a volume). Neither is about CPU or memory, and both
will take the system down long before either runs out.

> **A GPU fleet adds one more exporter, not a different design.** There are no GPU
> panels here because this is a scikit-learn RandomForest on CPU — four
> permanently-empty `DCGM_FI_DEV_*` panels teach the wrong habit. On a GPU
> deployment you add `nvidia/dcgm-exporter` beside node-exporter and put
> utilisation, memory, temperature and clock in exactly this row. The split stays
> the same: the exporter tells you the card is saturated, and something like
> `api_process_*` tells you your server is the reason.

---

## Guide C — Evidently

### C.1 What it does, in simple words

Prometheus answers *"what is happening right now, and what did it look like an
hour ago?"* Evidently answers something different: **"is today's data shaped like
the data the model was trained on?"** — and it answers as a **report you read**,
not a number you plot.

Give it two dataframes:

- **reference** — the frozen training data. The baseline the model actually
  learned. It does not change until you retrain.
- **current** — what production actually sent.

For every column it picks an appropriate statistical test, decides whether that
column drifted, and renders an HTML page with a distribution plot per feature.
It is a **batch** tool: you run it on a schedule (an Airflow task, `make drift`,
a cron), never in the request path.

Three things it gives you that a gauge cannot:

- **A picture per feature.** "distance_km drifted" is a number. The overlaid
  histograms show you it drifted because a whole second mode appeared at 3× the
  old values — a completely different bug from a uniform 1.6× shift, and you
  cannot tell them apart from the scalar.
- **A test chosen per column.** KS for continuous, chi-square for low-cardinality
  categorical, PSI where you want an effect size instead of a verdict.
- **A share-of-drifted-columns summary** you can fail a pipeline on.

### C.2 Step 0 — install, and respect the pin

```bash
pip install -e .        # brings evidently>=0.4,<0.7
python -c "import evidently; print(evidently.__version__)"   # expect 0.4.x – 0.6.x
```

> **The upper bound is load-bearing.** Every Evidently file here uses the legacy
> API — `from evidently.report import Report`, `from evidently.metric_preset
> import DataDriftPreset`. **Evidently 0.7 deleted it** and replaced it with
> `evidently.Report` / `evidently.presets`. A plain `pip install evidently` gets
> you the latest and breaks every Evidently import in this repo at line 1. The
> legacy line is also not published for Python 3.13+, which is where
> `requires-python = ">=3.10,<3.13"` comes from.

### C.3 Step 1 — freeze a reference

```bash
make bootstrap
```

Among its four artifacts, the one that matters here is
**`data/reference.parquet`**: 50,000 training rows, carrying the model's **own
predictions** in a `pred` column.

> **The reference carries `pred`, not just the target, on purpose.** Prediction
> drift means "the model's output distribution moved", so the baseline has to be
> model *output* too. A RandomForest averages noise away, so its predictions are
> tighter than the target it was fit on — compare the two and you manufacture a
> permanent PSI of about 0.12 that never goes away and means nothing.

> **Freeze it, and never recompute it.** The most common way to build a drift
> monitor that never fires is to compare today against a *rolling recent window*.
> The baseline then chases the drift: every day looks like yesterday, the number
> stays reassuringly near zero, and six months later you're serving a completely
> different distribution behind a green dashboard. `jobs/daily_drift.py` names
> its baseline (`REFERENCE_ID = "train-v3-50k"`) and stamps that name on every
> stored row — a drift number measured against a different reference is a
> different measurement wearing the same name. Bump the id when you re-freeze,
> never silently.

### C.4 Step 2 — produce something to compare

Evidently needs *current* data. That's the serving log the API writes:

```bash
make api        # terminal 2
make traffic    # terminal 3 — give it a minute or two
```

Every `/predict` appends the exact feature vector it scored **and what it
returned** to `data/serving_log.jsonl`, flushed every 50 rows.

```bash
wc -l data/serving_log.jsonl
tail -1 data/serving_log.jsonl
```

> **The serving log is not redundant with the metrics.** Incident 02 — a stale
> scaler — is invisible to every distribution monitor in this repo: the inputs
> are fine, the model is fine, and only the transform between them is old.
> Nothing about the request looks wrong. The one thing that catches it is
> re-scoring these logged vectors through the current offline pipeline and
> comparing the answers (`make replay`). That is what a serving log is *for*.

### C.5 Step 3 — run the report

```bash
make drift          # one window, now
make drift-loop     # forever, one simulated day per minute
```

One run does six things, in order:

1. Loads `data/reference.parquet` and the **last 60 minutes** of the serving log
   (falling back to the last 2,000 rows if that window is thin).
2. Builds an Evidently `Report` — `DataDriftPreset()` plus three explicit
   `ColumnDriftMetric`s — and saves `reports/drift_<timestamp>.html`.
3. Computes PSI itself: globally, **per city**, and for the prediction
   distribution (model output now vs on the reference).
4. Writes every number as a row in `metrics_store.db` → `drift_metrics`, stamped
   with the window, segment, model version and reference id.
5. **POSTs the numbers to `http://localhost:8001/internal/gauges`**, which is how
   `feature_psi_score`, `segment_psi_score` and `model_mae_minutes` reach
   Prometheus at all.
6. Prints a band per feature:

```
  drift window: 2143 serving rows vs frozen train-v3-50k
    distance_km  PSI= 0.021  ok
    passengers   PSI= 0.008  ok
    hour_of_day  PSI= 0.014  ok
  report: reports/drift_20260909-185754.html   rows stored: 14
```

> **Step 5 is the seam between the two halves of this session.** The drift job is
> a separate, short-lived process — and a short-lived process usually exits
> before Prometheus ever scrapes it. The textbook answer is a **Pushgateway**,
> which is a seventh container. The answer here is cheaper: post the numbers to
> the one process Prometheus already scrapes. That is the entire reason
> `POST /internal/gauges` exists, and `P_G_monitoring.py`'s `record_drift_scores()`
> is the in-process version that only works if the drift check runs inside the
> API, which it doesn't.

> **Step 4 is why there's a SQLite file at all.** Prometheus is a ~15-day ring
> buffer of time series. "Which features drifted, against which reference, in the
> window ending last Tuesday" is a fact with provenance, and the retrain gate has
> to be able to ask about last month. Different questions, different stores.

### C.6 Step 4 — read the report

```bash
open reports/          # then the newest file
```

Top to bottom:

- **Dataset Drift** — the summary: how many columns, how many drifted, and the
  **share of drifted columns**. This is the number you gate a pipeline on.
- **Data Drift table** — one row per feature: the test used, the drift score,
  drifted yes/no. Click a row to expand it.
- **The expanded plot** — reference and current distributions overlaid. **This is
  the actual product.** The number tells you *that* something moved; the shape
  tells you *how*, which is the only thing that tells you *why*.

What the shapes mean here:

| Shape | Usually means | Incident |
|---|---|---|
| whole distribution shifted by a constant factor | a unit change upstream | 01 — miles into a km field |
| a new mode appeared | a new population arrived | 05 — a new city |
| mass moved between existing modes | behaviour changed | 06 — Ramadan hours |
| **distributions unchanged, but predictions moved** | **not data drift at all** — look at the serving path | 02 — stale scaler |

That last row is the one to internalise: **a clean Evidently report does not mean
the model is fine.** It means the inputs are fine. Those are different claims,
and the gap between them is where the expensive bugs live.

### C.7 Step 5 — read the numbers in code, and gate on them

```python
results = report.as_dict()
for metric in results["metrics"]:
    if metric["metric"] == "ColumnDriftMetric":
        m = metric["result"]
        status = "🚨 DRIFT" if m["drift_detected"] else "✅ OK"
        print(f'{m["column_name"]:20s} | {m["stattest"]:12s} | p={m["p_value"]:.4f} | {status}')

drift_share = results["metrics"][0]["result"]["share_of_drifted_columns"]
if drift_share > 0.3:
    raise ValueError(f"Data drift alert: {drift_share:.0%} of features drifted!")
```

**Raising is what makes it a gate.** In Airflow the task goes red, downstream
tasks don't run, and someone is told. A drift check that only prints is a drift
check nobody reads.

**Which test for which column** — what `DataDriftPreset()` picks automatically,
and what you override with `ColumnDriftMetric(stattest=...)`:

| Column | Test | Why |
|---|---|---|
| `distance_km` | `ks` | continuous and unbounded — Kolmogorov–Smirnov compares whole CDFs |
| `passengers` | `chisquare` | 1–8; categorical in practice, so compare bin counts |
| `hour_of_day` | `psi` | you want a magnitude, not a verdict |
| `pred` | `psi` | prediction drift is a *degree*, and PSI has agreed bands |

**What PSI actually measures.** Population Stability Index — one number for
"how far has this distribution moved from the baseline?". It is built in three
steps:

1. Cut the **reference** into 10 bins and record the *share* of rows in each —
   say `[0.10, 0.25, 0.30, 0.25, 0.10]`.
2. Bin the **current** window using the **reference's** bin edges, not its own.
   Same grid, or you are comparing buckets that aren't the same buckets.
3. Sum one term per bin: `(current% − reference%) × ln(current% / reference%)`.

That product is two things you already care about, multiplied:

- **the difference** — how much mass moved;
- **the log ratio** — how much that move matters *relative to what was already
  there*.

A bin going 1% → 2% moved one point and doubled. A bin going 30% → 31% moved one
point and barely twitched. The log ratio is what separates them. And the two
factors always share a sign, so every term is ≥ 0 — a bin that emptied can never
cancel out a bin that filled. Identical distributions give exactly 0, and it
climbs from there.

Worked, on those five bins:

| reference | current | difference | ln(ratio) | contribution |
|---|---|---|---|---|
| 0.10 | 0.05 | −0.05 | −0.693 | 0.0347 |
| 0.25 | 0.20 | −0.05 | −0.223 | 0.0112 |
| 0.30 | 0.30 | 0.00 | 0.000 | 0.0000 |
| 0.25 | 0.28 | +0.03 | +0.113 | 0.0034 |
| 0.10 | 0.17 | +0.07 | +0.531 | 0.0371 |
| | | | **PSI** | **0.086** |

Two things to read off that table. The small tail bin that went 0.10 → 0.17
contributes **ten times** what the 0.25 → 0.28 bin does, off a *smaller*
absolute move — small bins are where PSI looks. And a shift that looks alarming
when you plot it still lands under 0.10, i.e. "stable". PSI is more forgiving
than your eyes are. That is deliberate, and it is why the bands below aren't
stricter.

**PSI is unitless, and that's the operational point.** It is computed on bin
*shares*, so kilometres, hours-of-day and passenger counts all come out on one
scale. That is why a single threshold works for every feature, why
`feature_psi_score` is one Grafana panel carrying three series instead of three
panels with three different thresholds, and why `0.25` is hard-coded in
`alert_rules.yml` rather than tuned per feature.

**What PSI does not tell you: direction, or cause.** 0.30 says *this
distribution is a different shape now*. It does not say whether rides got longer
or shorter, and it certainly does not say why — a unit change upstream and a new
city arriving can produce the same number. That is what the overlaid plot in
[C.6](#c6-step-4--read-the-report) is for: the number tells you to look, the
shape tells you where.

**Why the alert threshold hangs off PSI and not the KS p-value.** A p-value
answers "could this difference be chance?" — and at 20 rps the answer is always
no. Feed a million rows to KS and a shift far too small to matter becomes
"statistically significant": the alert fires forever, everyone mutes it, and you
are worse off than with no alert. PSI answers "**how big** is the difference?",
and it does not inflate with sample size. The same 0.05σ shift — real, but far
too small to act on — measured on nested samples:

| rows compared | PSI | KS p-value |
|---|---|---|
| 1,000 | 0.020 | 1.5e-02 |
| 10,000 | 0.004 | 3.7e-04 |
| 100,000 | 0.002 | 5.6e-18 |
| 1,000,000 | 0.002 | 6.3e-180 |

Identical shift on every row. PSI sits in "stable" and stays there — noisier at
1,000 rows, but it does not *trend*. The p-value falls through the floor purely
because n grew. Gate on it and your drift alert is really a row-count alarm.

That is why PSI is the credit-risk industry's threshold metric, and why the
bands are fixed rather than tuned:

| PSI | Band | What you do |
|---|---|---|
| < 0.10 | stable | nothing |
| 0.10 – 0.25 | watch | look at the **trend**, not the value |
| > 0.25 | act | investigate, then consult the retrain gate |

> The "watch" band is where the actual skill is. The seeded `distance_km` PSI
> climbs from 0.02 past 0.25 over twelve days: a threshold alert fires on day 12,
> but the **trend** was obvious on day 6. That contrast is the point of
> `make seed`, and it's why the PSI panel is a time series and not a gauge.

### C.8 Step 6 — get it onto the dashboard

Once `make drift` has run at least once **with the API up**:

```promql
feature_psi_score          # three series in http://localhost:9095/query
```

Then build Row 3 from [B.4](#b4-the-panels-to-build-and-their-queries).

If those panels are empty, the drift job ran but couldn't reach the API — it says
so explicitly:

```
  (model API unreachable — gauges not published, rows still stored)
```

The rows are safely in SQLite; they just never reached Prometheus. Start
`make api` and re-run `make drift`.

> **`make seed` does not fill Prometheus.** It backfills ~14 simulated days into
> `metrics_store.db` so the retrain gate and drift-history queries have a past to
> read. **Prometheus's history starts when you start the API**, and not one second
> earlier. Don't go hunting for two weeks of PSI in Grafana after seeding — ask
> for it through `make gate` and `drift_history()` instead. Knowing which store
> answers which question is most of operating this stack.

### C.9 The whole loop, in one place

```bash
# once
pip install -e ".[dev,metrics]"
cp .env.example .env
make bootstrap                  # fit the model, freeze the reference
make seed                       # 14 days of drift history for the gate

# every session — one terminal each
make up                         # Prometheus :9095, Grafana :3000
make api                        # the /metrics endpoint, :8001
make traffic                    # ~20 rps
make drift-loop                 # a drift window every 60s

# then look
open http://localhost:9095/targets     # model-api must be UP
open http://localhost:9095/alerts      # nine rules, Inactive / Pending / Firing
open http://localhost:3000             # the dashboard
open reports/                          # the newest Evidently HTML

# and when you want to break something
make list-incidents
make incident-01
make heal
```

`make doctor` runs 26 checks over all of the above and prints a checklist — run
it **before** a session, not during one.

Once every panel above is drawing, you have the instruments. Reading them under
pressure is [section 9](#9-the-incident-engine--find-it-from-the-evidence) and
[RUNBOOK.md](RUNBOOK.md), and that is the part that doesn't transfer from reading.

---

## Guide D — Langfuse

### D.1 What it does, in simple words

The first three guides all watch the **ride-duration RandomForest**. This one
watches something that model does not have: an **LLM**. Nothing in Guide D
touches `rf_model.pkl`, and that is the point — it is the tool you reach for the
moment a pipeline grows a prompt.

The three tools answer three different questions:

| Tool | Unit | Question |
|---|---|---|
| Prometheus | a number over time | "What is happening right now, and an hour ago?" |
| Evidently | a batch of rows | "Is today's data shaped like the training data?" |
| **Langfuse** | **one request** | **"What happened inside *this* call, step by step — and was the answer any good?"** |

A Langfuse trace is a **tree of typed observations**, not a line in a log. One
`/ask` becomes a retrieval, a generation, a tool call and a judge, each with its
own latency, input, output and token count, nested the way they actually ran.
That shape is the product: a p95 of 12s tells you the request was slow, the tree
tells you 9 of those seconds were one generation and 3 were the judge you added
last week.

The second half — **"was the answer any good?"** — is the part with no
equivalent in Guides A–C. A latency is measured; **quality is not.** There is no
counter the LLM increments to tell you it hallucinated. Somebody has to attach a
judgement to the trace, and that is what scores are ([D.6](#d6-step-4--scores-the-part-nothing-computes-for-you)).

> **Prompt drift and response-quality drift are feature drift and prediction
> drift, on a different data type.** Same shape of problem: a frozen baseline,
> a moving present, and a threshold you argue about. That parallel is the reason
> Langfuse is in a session about drift at all.

### D.2 Step 0 — install, and mind *two* version numbers

```bash
pip install -e ".[llm]"                                    # langfuse>=4,<5 + ollama
python -c "import langfuse; print(langfuse.__version__)"   # expect 4.x
```

> **The SDK major and the server major are different numbers, and neither
> implies the other.** This repo runs **server `langfuse/langfuse:3.221.1`** and
> **SDK `langfuse>=4,<5`**. That looks like a mismatch and is not: v4 is the
> current client *for* the 3.x server. Do not "fix" it by pinning `langfuse<4`.

> **A healthy server tells you nothing about the client.** This is the failure
> mode to internalise, because every instinct points the wrong way:
>
> ```bash
> curl -s localhost:3001/api/public/health     # {"status":"OK"} — with no SDK installed at all
> ```
>
> All six containers can be green, the UI can load, the keys can be valid, and
> your app can still be recording **nothing** — because `pip install -e ".[llm]"`
> was never run in the interpreter that is actually executing. Worse, the SDK
> **degrades to a no-op instead of raising**: it prints one line to stderr and
> your code runs to completion, successfully, observing nothing. That is the
> classic observability failure — the thing that watches fails quietly, and the
> silence looks exactly like health. The only honest check is client-side:
>
> ```python
> from langfuse import get_client
> assert get_client().auth_check()      # True = keys AND reachability AND an installed SDK
> ```

> **A v3 tutorial fails at line 1 against this SDK.** v4 renamed most of the
> surface. Nearly everything written about Langfuse online is still v3:
>
> | v3 — what you'll find online | v4 — what this repo uses |
> |---|---|
> | `from langfuse.decorators import observe` | `from langfuse import observe` |
> | `langfuse_context.update_current_trace(...)` | `propagate_attributes(...)` |
> | `langfuse.start_as_current_span(name=…)` | `langfuse.start_as_current_observation(name=…, as_type="span")` |
> | `langfuse.start_as_current_generation(…)` | the same call, `as_type="generation"` |
> | `LANGFUSE_HOST` | **`LANGFUSE_BASE_URL`** |

> **That last row is not cosmetic.** `LANGFUSE_HOST` still works but is
> deprecated, and if **neither** variable is set the SDK does not fail — it
> defaults to **`https://cloud.langfuse.com`** and cheerfully ships your traces
> to the internet. Check `.env` has `LANGFUSE_BASE_URL=http://localhost:3001`
> before you run anything you would not want to send off the laptop.

### D.3 Step 1 — start the stack

```bash
cp .env.example .env        # if you haven't already
make up-langfuse            # Prometheus + Grafana + six more containers
```

Six containers is a lot for "some traces" —
[section 7](#why-six-containers-for-one-tool) explains why each one is there, and
[section 8](#8-langfuse-v2-vs-v3--why-this-stack-has-clickhouse) covers what to do
when ~3.8 GB is too much. Give ClickHouse a moment, then:

```bash
docker compose --profile langfuse ps      # all six, healthy
open http://localhost:3001                # admin@example.com / langfuse123
```

The login and the API keys already work. The `LANGFUSE_INIT_*` variables in
`.env` auto-provision the org, project, keys and user on **first boot** — there
is no click-through setup.

> **`LANGFUSE_INIT_*` only fires on an empty database.** Change
> `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` after the first boot and nothing happens:
> the row already exists, and your new key is simply wrong. To re-provision you
> have to destroy the volumes —
> `docker compose --profile langfuse down -v` — which also destroys every trace.

You also need a local model, since nothing here calls a paid API:

```bash
ollama serve &
ollama pull llama3.1:8b       # the tool-calling demo needs a tool-trained model
```

### D.4 Step 2 — make one trace you can read

```bash
python ollama_langfuse_rag.py --demo rag
```

```
✓ Ollama http://localhost:11434 · model llama3.1:8b
✓ Langfuse http://localhost:3001

Q: When should I retrain based on PSI, and why not use KS?
A: You should retrain when the PSI is above 0.25, as this indicates a significant
   shift in the distribution…
   retrieved: ['psi', 'retrain-gate']
   judge: faithfulness=0.8 relevance=0.9 — The answer accurately describes the
   threshold for retraining based on PSI, but fails to address why not to use KS.
  ↳ rag trace: http://localhost:3001/project/session-4/traces/fb760d46…

✓ Flushed. Open http://localhost:3001
```

Four demos, one concept each — run them all:

| `--demo` | Shows |
|---|---|
| `rag` | retrieval → generation → judge: the full tree |
| `tools` | tool calls as their own typed observations |
| `stream` | a streamed response, and where its latency is recorded |
| `error` | a **failure**, captured as an observation with `level=ERROR` |

> **`✓ Flushed` is doing real work.** The SDK batches on a background thread, so
> a short-lived script that exits without `langfuse.flush()` takes its last batch
> with it. Every script here flushes; yours must too.

> **The error demo is the one people skip, and it's the most useful.** A trace
> that only exists when the call succeeds is a trace you cannot debug with. The
> doomed generation lands with `level=ERROR` and the provider's actual message —
> `model 'this-model-does-not-exist:latest' not found (status code: 404)` —
> attached to the observation that failed, not to a log file on a machine you no
> longer have.

### D.5 Step 3 — read the observation tree

Open the trace URL the script printed. That `rag` run is this, exactly:

```
demo-rag  [SPAN]  12.61s
  rag-pipeline  [SPAN]  9.05s
    retrieve         [RETRIEVER]   0.00s
    generate-answer  [GENERATION]  9.00s   224 in / 82 out
  llm-judge  [EVALUATOR]  3.51s
    judge-generation [GENERATION]  3.51s   320 in / 54 out
```

Read it as a bill for the 12.61 seconds: retrieval was free, the answer cost 9s,
**and the judge you bolted on cost another 3.5s and 374 more tokens** — 28% of the
wall clock and more input tokens than the answer itself. That is a fact about
your evaluation strategy that no single latency number would ever have told you,
and it is visible at a glance because the tree is typed and nested.

The types are not decoration:

| Type | Used for | What the UI does with it |
|---|---|---|
| `SPAN` | any unit of work | plain nesting |
| `GENERATION` | an LLM call | token counts, cost, model, prompt/completion panes |
| `RETRIEVER` | a search step | shows the documents and scores |
| `EVALUATOR` | a judge | grouped with scores, not with your app's logic |
| `TOOL` / `AGENT` | tool calls, agent loops | the agent view |

> **Typing a step `GENERATION` is what makes cost exist.** Log the same call as a
> `SPAN` and you get a duration; Langfuse has nowhere to put tokens, so the cost
> page stays at $0.00 and you find out three months later. `as_type` is the
> difference between a trace and an invoice.

> **Cost is $0 here, on purpose.** `OLLAMA_USD_PER_1K_INPUT/OUTPUT` are `0` in
> `.env` because the model is local and genuinely free. The *token counts* are
> real — set those two variables to a provider's rates and every cost panel
> populates from the same traces.
> [`tools/cost_report.py`](tools/cost_report.py) (Incident 08) is the worked
> version.

### D.6 Step 4 — scores: the part nothing computes for you

Open the same trace's **Scores** tab. One `rag` run attaches five:

| Score | Type | Where it comes from |
|---|---|---|
| `faithfulness` | NUMERIC `0.8` | the LLM judge — is the answer grounded in what was retrieved? |
| `relevance` | NUMERIC `0.9` | the LLM judge — does it answer the question asked? |
| `grounded` | BOOLEAN `True` | derived: `faithfulness >= 0.7` |
| `retrieval_hit` | CATEGORICAL `"hit"` | a deterministic check on the top document's score |
| `user_feedback` | NUMERIC `1` | simulated 👍 — stands in for a real product signal |

Three different *kinds* of judgement, deliberately:

- **A model judging a model** (`faithfulness`, `relevance`) — flexible, and
  itself unreliable enough that [`evals/calibrate_judge.py`](evals/calibrate_judge.py)
  exists to measure it against human labels with Cohen's κ. *Evaluate your
  evaluator* is [section 10](#10-evaluating-the-llm--and-evaluating-the-evaluator).
- **A cheap deterministic heuristic** (`retrieval_hit`) — costs nothing, never
  flakes, and catches the failure that matters most in RAG: the model answering
  confidently with **nothing retrieved**.
- **A human** (`user_feedback`) — the only ground truth there is, and the
  scarcest.

> **A CATEGORICAL score reads back as `value=0`, and that is not a bug.** Through
> the API, `retrieval_hit` looks like this:
>
> ```
> retrieval_hit   dataType=CATEGORICAL   value=0   stringValue='hit'
> ```
>
> The label lives in **`stringValue`**; `value` is an unused numeric slot. Read
> `value` in a gate and every categorical score silently evaluates as zero —
> a green pipeline that has never once checked what it claims to check.

### D.7 Step 5 — get enough data to see a population

Four traces is the right size to *read* and the wrong size to *look at*. Every
chart is one dot and every filter returns everything:

```bash
python langfuse_workload.py --plan-only    # see the plan, call nothing
python langfuse_workload.py                # the default: ~31 sessions / ~39 traces
python langfuse_workload.py --traces 150   # a fuller population, closer to section 7
```

> **The default is deliberately small — `--traces` defaults to 40.** It finishes
> in a coffee break against a local 8B model. The inventory in
> [section 7](#7-what-we-actually-put-in-langfuse) was measured from a much larger
> run, so don't expect those counts from one default invocation.

Now the pages that only mean something over a population work — score trends,
cost per user, quality per prompt version, session threads.
[Section 6](#6-langfuse_workloadpy--filling-langfuse-with-data) is the script,
[section 7](#7-what-we-actually-put-in-langfuse) is the page-by-page inventory of
what lands. Re-running adds traffic rather than duplicating fixtures.

> **Traces cannot answer "did the new prompt help?"** They are a record of what
> production did, and production only ran one version per request. Answering it
> needs the same fixed inputs through both variants — a **dataset run**, which is
> the experiment half of [section 7](#5-datasets-and-experiments--what-traces-cannot-tell-you).

### D.8 Step 6 — prove it end to end

Guides A–C end on a dashboard you can see. The equivalent here is a
**round-trip**: ingest a trace, then read it back through the API.

```python
import os, time, uuid, httpx
from dotenv import load_dotenv
load_dotenv()
from langfuse import get_client, propagate_attributes

client = get_client()
assert client.auth_check(), "keys/URL/SDK — one of the three is wrong"

marker = f"verify-{uuid.uuid4().hex[:8]}"
with propagate_attributes(trace_name=marker, tags=["healthcheck"]):
    with client.start_as_current_observation(name=marker, as_type="span") as span:
        trace_id = span.trace_id
client.flush()

auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
base = os.environ["LANGFUSE_BASE_URL"]
for _ in range(30):                                    # ingestion is async
    r = httpx.get(f"{base}/api/public/traces/{trace_id}", auth=auth, timeout=15)
    if r.status_code == 200:
        print("queryable:", r.json()["name"]); break
    time.sleep(3)
else:
    raise SystemExit("ingested but never became queryable — check langfuse-worker")
```

> **The retry loop is not defensive padding — it is the architecture.** The SDK
> queues, the web tier returns immediately, Redis holds the event and
> **`langfuse-worker`** writes it to ClickHouse. A trace is typically queryable in
> **~3 seconds**, never instantly. Assert straight after `flush()` and you have
> written a test that fails on a perfectly healthy stack.
>
> It also localises the fault precisely: **200 from `/health` but a trace that
> never appears** means ingestion is fine and the *worker* is stuck — go read
> `docker compose --profile langfuse logs langfuse-worker`.

> **`load_dotenv()` searches from the calling file's directory, not your shell's
> `cwd`.** Move this script somewhere else and it silently finds no `.env`, the
> client disables itself, and — per [D.2](#d2-step-0--install-and-mind-two-version-numbers)
> — nothing raises. Run it from the project root, or pass an explicit path.

Then the cheap version, which checks this and 25 other things:

```bash
make doctor
```

> **Read the WARN lines, not just the tally.** `make doctor` deliberately does
> **not** fail when the LLM half is missing — Guides A–C don't need it, and a red
> line there would be noise. A missing SDK therefore shows up as a `WARN` beside
> a perfectly green `PASS Langfuse http://localhost:3001`, which is exactly the
> trap [D.2](#d2-step-0--install-and-mind-two-version-numbers) describes.

### D.9 The whole loop, in one place

```bash
# once
pip install -e ".[llm]"
cp .env.example .env
ollama pull llama3.1:8b

# every session
make up-langfuse                          # Prometheus, Grafana + 6 Langfuse containers
ollama serve                              # another terminal

# make something to look at
python ollama_langfuse_rag.py --demo rag  # one trace you can read end to end
python ollama_langfuse_rag.py             # all four demos
python langfuse_workload.py --traces 150   # a population, so the charts mean something

# then look
open http://localhost:3001                # Tracing → Sessions → Scores → Datasets
make doctor                               # 26 checks; mind the WARN lines

# evaluate, and evaluate the evaluator — Guide E
make judge                                # the judge on its two shipped samples
make calibrate                            # Cohen's κ vs human labels, per language
make ragas-gate                           # the CI quality gate

# when you're done with the RAM
docker compose --profile langfuse stop    # keeps every trace, frees ~3.8 GB
```

Which Langfuse feature is demonstrated in which file — and the order to project
them in — is [`docs/langfuse_capability_map.md`](docs/langfuse_capability_map.md).

Those last three commands have a guide of their own:
[Guide E — RAGAS](#guide-e--ragas) picks up exactly where this one stops, scoring
the pipeline you just instrumented against a frozen testset.

---

## Guide E — RAGAS

### E.1 What it does, in simple words

[Guide D](#guide-d--langfuse) ends with a score attached to a trace, written by a
judge prompt in [`ollama_langfuse_rag.py`](ollama_langfuse_rag.py). That is
**observability**: it records what production did, whatever production happened to
be asked. This guide is **evaluation**: the same questions every time, a reference
answer you wrote down, and a number you can compare across weeks.

The difference is the one Guide C already made:

| | Measures | Against | Runs on |
|---|---|---|---|
| Evidently | today's features | a **frozen** training reference | a schedule |
| Langfuse scores | whatever was asked | nothing — it describes | every request |
| **RAGAS** | **a frozen 60-item testset** | **a reference answer per item** | **a gate, in CI** |

RAGAS is a library of **reference implementations** of RAG metrics — faithfulness,
context precision, context recall, response relevancy — so you are not inventing
the definition of "grounded" yourself and then arguing about it in review.

> **A frozen testset is a frozen reference, for text.** Same discipline as
> [C.3](#c3-step-1--freeze-a-reference), same failure if you skip it: score
> against whatever questions came in this week and the baseline chases the
> quality, every week looks like last week, and a regression is invisible.
> [`evals/testset_v1.jsonl`](evals/testset_v1.jsonl) is 60 items with a
> `.sha256` beside it for exactly the reason `REFERENCE_ID = "train-v3-50k"`
> exists.

> **RAGAS is the *second* ruler, not the only one.**
> [`evals/judge.py`](evals/judge.py) is a hand-written judge with its mechanism
> showing; RAGAS is better engineered and hides it. Run both on the same items:
> **where they agree the number is probably real, where they diverge is the
> calibration lesson.** Which of them to believe is
> [`evals/calibrate_judge.py`](evals/calibrate_judge.py) and
> [section 10](#10-evaluating-the-llm--and-evaluating-the-evaluator).

### E.2 Step 0 — install, and the pins that are load-bearing

```bash
pip install -e ".[evals]"
python -c "import ragas, langchain; print(ragas.__version__, langchain.__version__)"
# expect 0.2.x 0.3.x
```

> **These pins are not cosmetic, and they are not conservatism.** `ragas 0.4.x`
> imports `langchain_community.chat_models.vertexai`, which **no longer exists**
> in the langchain 1.x generation — langchain-community is being sunset. The
> import fails before a line of your code runs. `ragas>=0.2,<0.3` against the
> `langchain*>=0.3,<0.4` line is the combination that actually resolves. A plain
> `pip install ragas` gets you the latest and a traceback.

> **If the resolver fails once, run it again.** Resolving this extra occasionally
> dies with `No matching distribution found for langchain-community<0.4,>=0.3`
> — for a version range that plainly exists on PyPI. It is a transient index
> hiccup during backtracking, not a broken pin; the identical command succeeds on
> a retry.

You also need the local judge model, and — for one of the four metrics — embeddings:

```bash
ollama pull llama3.1:8b
ollama serve --embeddings          # see E.5: without this, one metric is skipped
```

### E.3 Step 1 — the frozen testset

```bash
wc -l evals/testset_v1.jsonl        # 60
cat evals/testset_v1.sha256
head -1 evals/testset_v1.jsonl | python -m json.tool
```

Each item carries the fields that make it scorable, not just askable:

| Field | Why it's there |
|---|---|
| `question` | what to ask |
| `reference` | the answer a human says is right — **ground truth** |
| `lang` | `en` 30 / `ar` 30, deliberately balanced |
| `expects_refusal` | this question is **unanswerable** and refusing is the correct behaviour |
| `note_id`, `source` | which knowledge-base note should have been retrieved |

Rebuild and re-freeze with [`evals/build_testset.py`](evals/build_testset.py):

```bash
make testset        # rebuilds the 60 items and rewrites the .sha256
```

> **30/30 English/Arabic is the whole point of the segment gate.** A bilingual
> knowledge base is where quality collapses in one language while the global mean
> stays green — which is precisely how **incident 09** hid for four hours.
> A testset that is 90% English cannot detect it no matter how good the metrics
> are. See [E.7](#e7-step-5--gate-on-it).

> **`expects_refusal` items exist because a testset of answerable questions
> measures the wrong thing.** Production asks unanswerable questions constantly.
> An eval where every question has an answer rewards a model that always answers
> — which is the definition of the failure you are trying to catch.

### E.4 Step 2 — run it

```bash
make ragas                       # 10 stratified items from the frozen testset
python -m evals.run_ragas --source dataset --limit 4
```

It announces what it is scoring, makes a **second pass** for the metrics that
need ground truth, and only then names the judge it used:

```
  scoring 10 samples with: faithfulness, llm_context_precision_without_reference
  Evaluating: 100%|██████████| 20/20 [06:21<00:00, 19.05s/it]
  + 6/10 samples carry a reference: also scoring context_recall
  Evaluating: 100%|██████████| 6/6 [01:00<00:00, 10.01s/it]

  judge: llama3.1:8b @ num_ctx=8192 seed=42 workers=4
```

That second line is [E.5](#e5-step-3--the-four-metrics-and-what-each-one-needs)
happening in front of you, and the `19s/it` is not a typo — on a local 8B model
each item is several judge calls, so `make ragas` is a coffee-break command, not
an interactive one. That is also the argument for sampling in
[E.8](#e8-step-6--score-live-traffic-and-compare-runs).

> **`num_ctx=8192` is set explicitly, and that is a bug fix.** Ollama's default
> context is **2048 tokens**, which silently truncates the answer *plus* every
> retrieved chunk before the judge ever sees them. You would be scoring your own
> truncation — and because Arabic tokenises longer, **it hits the ceiling first**,
> so you'd measure a quality gap between languages that is really a context-window
> gap. Leave `num_ctx` to the default and you get a plausible, stable, completely
> fictional number.

> **`seed=42`, `temperature=0` — and it is still not deterministic.** Same seed,
> same prompt, same model can still land a different score, because the judge is
> an LLM. That is an argument for reading the distribution and for
> [`calibrate_judge.py`](evals/calibrate_judge.py), not for pretending the number
> is exact.

### E.5 Step 3 — the four metrics, and what each one needs

| Metric | Answers | Needs |
|---|---|---|
| `Faithfulness` | Is every claim in the answer supported by the retrieved context? | context + answer |
| `LLMContextPrecisionWithoutReference` | Were the retrieved chunks actually *useful*? | context + answer |
| `ResponseRelevancy` | Does the answer address the question asked? | **embeddings** |
| `ContextRecall` | Did retrieval find everything it should have? | **ground truth** |

Two of those requirements have consequences you will meet immediately.

**Embeddings.** `ollama serve` without `--embeddings` answers `501` to
`/api/embed`. [`run_ragas.py`](evals/run_ragas.py) probes for it and says so
rather than crashing — this is the real output on a server without it:

```
  NOTE: this Ollama server has embeddings disabled (POST /api/embed -> 501),
  so ResponseRelevancy is SKIPPED — it is the only metric here that needs them.
  Faithfulness and context precision are LLM-only and run normally.
  Fix: restart with `ollama serve --embeddings`, or set OLLAMA_EMBED_MODEL to a
  pulled embedding model such as nomic-embed-text.
```

> **Notice what it did *not* do: quietly report three metrics instead of four.**
> A harness that drops a metric without saying so is how you end up defending a
> quality number that never measured the thing you are being asked about.

**Ground truth.** `ContextRecall` needs a human-written reference, so it **can
never run on live production traffic** — production has no answer key. This is
why the script returns *two* metric lists and makes a second pass:

> **RAGAS validates required columns across the WHOLE dataset.** One adversarial
> item with no ground truth would make context recall reject the **entire run**
> — not skip the item, reject the run. So the reference-only metrics get their
> own pass over the referenced subset. That is what "only when the sample carries
> a reference" means in practice, and it is the kind of detail that turns a
> 20-minute eval into a 3-hour debugging session.

### E.6 Step 4 — read the output honestly

Per metric, per language, this is what a real run prints:

```
  llm_context_precision_without_reference
    ar:  mean 1.000  n=1    NaN 0/1 =   0%
    en:  mean 0.000  n=1    NaN 0/1 =   0%
    refusals: 2 unanswerable items excluded from the gate — 0 scored >0.5,
              i.e. the model answered anyway
    all: mean 0.500   distribution:
        0.0-0.1    1 ########################
        ...
        0.9-1.0    1 ########################
    worst 5:
      0.000  en-022     What does The frozen reference mean and when doe
      1.000  ar-041     ما معنى تعدد قيم الوسوم في المقاييس ومتى يكون مه
```

That `mean 0.500` describes **nothing that exists**. One item scored 0.0 and one
scored 1.0; no sample is anywhere near 0.5. The histogram is not decoration —
it is the difference between "the system is mediocre" and "the system is perfect
half the time and broken the other half", which have completely different fixes.

[`run_ragas.py`](evals/run_ragas.py) refuses to do three things, each because
doing them is how eval harnesses end up reporting numbers nobody should act on:

- **Print a bare mean.** A measured run: faithfulness mean **0.573** — from two
  items at 0.0 and two at 0.9–1.0. A bimodal 0.85 and a uniform 0.85 are
  different systems.
- **Ignore NaNs.** When the judge's output won't parse, RAGAS retries and then
  **drops** the sample. Dropped samples are the *hard* ones, so the surviving
  mean flatters you. Above a **10% NaN rate the run fails** rather than reports.
- **Gate on the global mean alone.** See [E.7](#e7-step-5--gate-on-it).

> **The refusal bucket is a finding, not bookkeeping.** Faithfulness over a
> refusal is meaningless — there are no claims to ground — so refusals are
> excluded from the gate and counted separately. But the count itself is the
> signal: a real run reported *"4 unanswerable items excluded — 1 scored >0.5,
> i.e. the model answered anyway"*. That is a **confabulation on a question with
> no answer**, and nothing else in this stack would have caught it.

### E.7 Step 5 — gate on it

```bash
make ragas-gate                              # exits non-zero below threshold
make ragas ARGS="--gate --threshold 0.75"    # same thing, stricter
```

Two conditions fail a run, and the second is the interesting one:

```
    SEGMENT GATE FAILED: en at 0.000 is more than 0.1 below the 0.7 threshold
    GATE FAILED: mean below 0.7
```

| Gate | Rule |
|---|---|
| global | mean below `--threshold` (default **0.70**) |
| **segment** | **any single language more than 0.10 below the threshold — even when the global mean passes** |

> **The segment gate is the entire reason the testset is 30/30.** A global mean
> of 0.78 with English at 0.95 and Arabic at 0.61 is a *passing* number hiding a
> completely broken language. That is incident 09. Averages are where regressions
> go to hide, and the fix is never a better average — it is refusing to average
> across the thing you care about.

Because it exits non-zero, this drops into CI unchanged — the repo runs it as its
own workflow, the same way `make test` guards the metric contract.

### E.8 Step 6 — score live traffic, and compare runs

The frozen testset tells you whether the system regressed. It cannot tell you
what production is actually being asked. For that, score real traces:

```bash
python -m evals.run_ragas --source langfuse --hours 24 --sample 0.03 --push
```

That samples 3% of the last 24 hours of Langfuse traces, scores them, and
`--push` writes the results **back onto those traces as Langfuse scores** — so
the RAGAS number and the trace that produced it live in the same place, and
[Guide D](#guide-d--langfuse)'s score trends now include a library metric.

> **3%, not 100%.** Every scored sample is several more LLM calls; judging all of
> production costs more than serving it. Sampling is not a compromise here, it is
> the design.

And to answer "did that change help?", save a run and diff it per item:

```bash
python -m evals.run_ragas --source dataset --save baseline     # once
# …change the prompt, the retriever, the model…
python -m evals.run_ragas --source dataset --compare baseline  # per-item diff
```

> **Per-item, not mean-to-mean.** Two runs with identical means can have half the
> items improved and half regressed. `evals/runs/baseline.json` ships with the
> repo, so `--compare baseline` works before you have saved anything.

### E.9 The whole loop, in one place

```bash
# once
pip install -e ".[evals]"       # retry if the resolver trips — see E.2
ollama pull llama3.1:8b
make testset                    # rebuild + re-freeze the 60 items (optional)

# every session
ollama serve --embeddings       # --embeddings or ResponseRelevancy is skipped
make up-langfuse                # only needed for --source langfuse

# measure
make ragas                      # 10 items, full reporting, no gate
make ragas-gate                 # the CI gate — exits non-zero
make judge                      # the hand-written judge, same idea
make calibrate                  # Cohen's κ — do we believe either of them?

# against production, and across changes
python -m evals.run_ragas --source langfuse --hours 24 --sample 0.03 --push
python -m evals.run_ragas --source dataset --compare baseline
```

[`ragas_example.py`](ragas_example.py) is the canonical five-line version of all
this — `evaluate(dataset, metrics=[faithfulness, …])` with RAGAS's module-level
metric singletons. It is a **teaching snippet and does not run**
([section 11](#11-running-these)): those singletons default to **OpenAI**, so
against a local model every metric has to be constructed with an injected LLM
(`Faithfulness(llm=LangchainLLMWrapper(ChatOllama(...)))`). Read the snippet for
the shape, read [`evals/run_ragas.py`](evals/run_ragas.py) for what running it
without an API key actually costs you.

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

> What PSI *means* — the formula in words, a worked example, why it's unitless
> and what it refuses to tell you — is in
> [C.7](#c7-step-5--read-the-numbers-in-code-and-gate-on-them). This is the
> implementation of it.

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

> These four are the **snippet's** metrics, and this file is unchanged. The
> runnable service in [`services/model_api.py`](services/model_api.py) keeps every
> one of these names and adds request, resource, validation and change-tracking
> metrics on top — the full catalogue is in
> [A.3](#a3-what-metrics-we-actually-collect).

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

docker compose up -d                       # Prometheus + Grafana + node-exporter  (3)
docker compose --profile langfuse up -d    # + Langfuse                           (6 more)
```

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | `admin` / `admin` |
| Prometheus | http://localhost:9095 | — |
| node-exporter | http://localhost:9100/metrics | — |
| Langfuse | http://localhost:3001 | `admin@example.com` / `langfuse123` |
| MinIO console | http://localhost:9093 | `minio` / `miniosecret` |

Ports are picked around what the other sessions already run: 5000 (MLflow,
session 2), 8000 (session 1 API), 8080 (Airflow, session 3), and 9090/9091 — held
by a pre-existing Langfuse/MinIO stack, which is why **Prometheus is on 9095**
rather than its conventional 9090. node-exporter keeps its conventional 9100.
All are overridable in `.env`.

> **What node-exporter measures here is not your laptop.** It reads `/proc` and
> `/sys`, and on Docker Desktop for macOS or Windows those belong to the Linux VM
> the containers run in — so `node_memory_MemTotal_bytes` is the VM's ceiling and
> the CPU count is the VM's allocation. It is a real machine, correctly measured;
> it is just not the machine you are sitting at, and it is never the process
> serving predictions, which runs on the host via `make api`. That is why the
> Resources row draws **both** `node_*` and the `api_process_*` gauges the API
> samples with psutil. On Linux the host mounts make it your actual host.

Langfuse is behind a **compose profile** because it drags in Postgres, ClickHouse,
Redis and MinIO — six containers. You shouldn't need a ClickHouse cluster running
to look at a Grafana dashboard, so the default `up` gives you just the metrics
stack.

### Everything is provisioned as code

Nothing here requires click-through setup. On first boot Grafana already has the
Prometheus datasource wired and the dashboard loaded, because
`monitoring/grafana/provisioning/` declares both. Edit the JSON, wait 30s, refresh.

The shipped dashboard is **four rows, read top to bottom in an incident**:

| Row | Question it answers | Panels |
|---|---|---|
| **1 · Service** | What is the caller experiencing? | request rate, p50/p95/p99, error rate, `up{}` |
| **2 · Resources** | What is saturating, before anyone feels it? | model-process CPU/RSS/fds, host CPU/RAM, disk + 4h projection, samples-per-scrape |
| **3 · Model behaviour** | What is the model saying? | prediction heatmap, long-ride rate, validation failures by field, version, p99 minutes |
| **4 · Quality & drift** | Is it still right? | PSI bands, PSI by city, MAE on late labels, retrain decisions, feedback rate |

It also ships a **`$model` variable** (one dashboard for every model, not twelve
near-identical copies nobody maintains) and two **annotation queries** that draw a
vertical line on every deploy and every retrain decision. Every query is written
out in [B.4](#b4-the-panels-to-build-and-their-queries).

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

The worked example is [`ollama_langfuse_rag.py`](ollama_langfuse_rag.py) — read the
file itself, it is written to be read. [Section 6](#6-langfuse_workloadpy--filling-langfuse-with-data)
then drives it in bulk, and [section 7](#7-what-we-actually-put-in-langfuse) is the
inventory of what that puts in Langfuse.

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

## 6. `langfuse_workload.py` — filling Langfuse with data

`ollama_langfuse_rag.py` writes four traces, one per concept. That is the right
size to **read** and the wrong size to **look at**. With four traces every chart
in Langfuse is a single dot, every filter returns everything, and the half of the
product that only means something over a population — score trends, session
threads, cost per user, quality per prompt version — has nothing to show.

This script drives that same pipeline in bulk:

```bash
python langfuse_workload.py                    # ~112 sessions / ~150 traces + 2 experiment runs
python langfuse_workload.py --plan-only        # print the plan, call nothing
python langfuse_workload.py --traces 40        # smaller
python langfuse_workload.py --no-experiment    # traffic only
python langfuse_workload.py --experiment-only  # just re-run the prompt comparison
```

Re-running is safe. The prompt versions and dataset items are created once and
reused, so a second run adds traffic instead of duplicating the fixtures.

### It is a traffic model, not new instrumentation

Every span, generation, tool call and score comes from `ollama_langfuse_rag.py`,
imported as a module. This file only decides **what to run and how often** —
which is the part a demo usually skips and a production system never gets to
skip.

| It generates | Which turns on |
|---|---|
| 5 request kinds, nested spans throughout | **Tracing** |
| Multi-turn threads sharing a `session_id` | **Sessions** |
| 6 personas, weighted so traffic is uneven | **Users**, cost-per-user |
| production / staging / development | **Environments** |
| 8 score names across 4 data types | **Scores**, and any chart built on them |
| Two live versions of one prompt | **Prompts** — "did the change help?" |
| An 8-item gold set, 2 of them unanswerable | **Datasets** |
| One run per prompt version | **Experiments**, side by side |

### The five request kinds

| Kind | Share | What it exercises |
|------|-------|-------------------|
| `rag` | 58% | retrieve → generate → judge → score, nested in one trace |
| `agent` | 20% | tool-calling loop; one observation per tool call |
| `stream` | 10% | time-to-first-token recorded apart from total latency |
| `guardrail` | 6% | prompt injection blocked *before* the model is called |
| `error` | 6% | a provider failure captured as `ERROR`, not a log line |

### Where the interesting data comes from

Three deliberate choices do most of the work, and each is a decision you face for
real the first time you build an eval set.

**One topic in six is unanswerable.** `out-of-scope` asks about XGBoost learning
rates and Kubernetes HPAs — nothing the corpus covers. These are the most useful
traces in the run, because they are the only ones that show what the assistant
does when retrieval finds nothing. A quality dashboard with no failures on it is
not measuring anything. Topics are drawn from a shuffled cycle rather than
sampled independently, precisely so a small run cannot skip this one by luck.

**The prompt A/B is randomised independently of the release version.** Traffic
carries two app versions (`1.0.0`, `1.1.0`) *and* two prompt versions. Shipping
the new prompt together with the new release is what a real team does, and it is
exactly what makes the result uninterpretable afterwards — you can no longer tell
the prompt apart from everything else in that release. Randomising the two
separately costs nothing up front and is unfixable later.

**Feedback is sparse and noisy.** Roughly 55% of traces get a thumbs score, and
it correlates with the judge only imperfectly. Both properties are the point: if
your real feedback rate is 2%, the judge is the only dense quality signal you
have — which is the entire argument for running one. And feedback that agreed
perfectly with the judge would make the judge redundant.

### The scores it writes

| Score | Type | Source |
|---|---|---|
| `faithfulness`, `relevance` | NUMERIC | LLM judge |
| `grounded` | BOOLEAN | judge, thresholded at 0.7 |
| `retrieval_hit` | CATEGORICAL | deterministic — `hit` / `weak` / `miss` |
| `user_feedback` | NUMERIC | simulated thumbs |
| `user_comment` | TEXT | simulated free text |
| `tool_calls`, `answered` | NUMERIC / BOOLEAN | deterministic, agent runs |
| `blocked` | BOOLEAN | input guardrail |

`TEXT` scores are not chartable, and that is what they are for. A NUMERIC score
tells you quality dropped on Tuesday; the TEXT ones are where the reason lives.

### The experiment answers a question the traces cannot

Traces tell you what production did. A dataset tells you what *should* happen, on
inputs you chose, and can be re-run against any change — which is the only way to
compare two prompts without waiting a week for traffic.

Variant B asks for inline `[doc-id]` citations and refuses harder on
out-of-corpus questions. It sounds like a straight improvement. On the gold set
it is not:

```
── experiment: prompt 'production' (v1) over 8 items
   mean_faithfulness      0.925
   mean_relevance         0.750
   mean_keyword_recall    1.000

── experiment: prompt 'experimental' (v2) over 8 items
   mean_faithfulness      0.825
   mean_relevance         0.750
   mean_keyword_recall    0.833
```

That is the whole argument for keeping prompts in Langfuse rather than in the
file. The change was plausible, it was worse, and it cost 90 seconds to find out
instead of a week of degraded answers.

> Run it twice before you believe a gap that size. Eight items is a small gold
> set, and run-to-run variance on a temperature-0 local model is not zero.

### The gotcha this script exists to demonstrate

Scores attach to the **currently active span**. Call `score_current_trace()`
after the pipeline's span has closed and the SDK logs
`No active span in current context` and silently drops the score.

Since judging and scoring necessarily happen *after* generation finishes,
something has to stay open across the whole request. That is why every request
kind here has its own `@observe`d wrapper (`rag-request`, `agent-request`, …)
rather than calling the imported pipeline directly. `ollama_langfuse_rag.py` gets
this for free from its `@observe`d `demo_*` functions, which is exactly why the
mistake is easy to make the moment you reuse its parts elsewhere.

### Two things it does not simulate

**Time.** Every trace lands inside the few minutes the script runs, so
time-series charts show one spike rather than a trend. Traces cannot be
backdated through the SDK. Run it on a few different days if you want history.

**Free inference.** `ollama_langfuse_rag.py` prices a local model at `$0`, which
is the honest per-token answer and makes every cost view read `$0.00`. This
script substitutes the amortised figure that file's own comment recommends —
GPU-hours × instance price ÷ tokens produced, about `$0.0052` per 1k output
tokens for an A10G at `$0.75/hour`. Pass `--no-price` to keep the zeros.


---

## 7. What we actually put in Langfuse

[Section 6](#6-langfuse_workloadpy--filling-langfuse-with-data) explains the
script that generates the data. This section is the **inventory**: after
`docker compose --profile langfuse up -d` and one `python langfuse_workload.py`,
this is what exists inside Langfuse, page by page, and the question each page can
now answer. Every number below is measured from a real run, not estimated.

### The instance

| | |
|---|---|
| URL | <http://localhost:3001> |
| Login | `admin@example.com` / `langfuse123` (set by `LANGFUSE_INIT_USER_*` in `.env`) |
| Organization | **MLOps MENA Community** |
| Project | **`session-4-llm-observability`** |
| Server | `langfuse/langfuse:3.221.1`, 6 containers |
| SDK | `langfuse>=4,<5` |

> The project is named per **session**, not per model. Langfuse only ever sees
> the LLM side of session 4. The ride-duration model itself is watched by
> Prometheus and Grafana — different tool, different signal,
> [section 5](#5-the-docker-stack--prometheus-grafana-langfuse).

### What landed

| Langfuse page | Contents | What it now answers |
|---|---|---|
| **Tracing** | **182 traces**, **996 observations**, 6 trace kinds | "What happened on this request, step by step?" |
| **Sessions** | **112 sessions**, 1–3 turns each | "Show me this user's whole conversation" |
| **Users** | **6 personas**, deliberately uneven (`amina.k` 53 → `omar.t` 10) | "What is *this* user seeing?" · cost per user |
| **Environments** | `production` 94 · `staging` 30 · `development` 26 · `sdk-experiment` 32 | "Only show me production" |
| **Scores** | **645 scores**, **14 names**, all 4 data types | "Is quality going up or down?" |
| **Prompts** | 1 prompt, **2 versions**, both live | "Which version answered this trace?" |
| **Datasets** | `session4-drift-qa`, **8 items** | "What *should* the answer be?" |
| **Experiments** | **4 dataset runs** | "Did the new prompt actually help?" |
| **Cost** | ~**$0.12** total | "What is this costing, and who is spending it?" |

Traces split by kind: `rag-request` 87 · `experiment-item-run` 32 ·
`agent-request` 30 · `stream-request` 15 · `guarded-request` 9 ·
`error-request` 9. Releases split `1.0.0` 91 / `1.1.0` 59.

### Why six containers for one tool

The usual reaction to `--profile langfuse` is that this is a lot of machinery to
look at some traces. The split is not arbitrary — it is the same OLTP/OLAP
separation you would end up building yourself:

| Container | Holds | Why it is there |
|---|---|---|
| **ClickHouse** | traces, observations, scores | Columnar. Append-only, high-volume, always read as aggregates — "p95 latency by model this week" is a column scan, not a row lookup |
| **Postgres** | projects, users, API keys, prompts, datasets, dataset runs, sessions | Relational and transactional. Things you *edit* and need consistent |
| **MinIO** | raw ingestion events, as blobs | Ingestion is durable *before* it is processed, so a worker crash loses nothing |
| **Redis** | the queue between `langfuse-web` and `langfuse-worker` | The API returns as soon as the event is queued |
| **web** / **worker** | the app, and the async processor | Split so ingest spikes cannot slow the UI |

The design point worth taking away: **your app is never blocked on the
observability backend.** The SDK batches on a background thread, the server
queues, and a worker does the expensive part later. That is also why
`langfuse.flush()` is mandatory in a short-lived script — without it the process
exits and the last batch dies with it.

It is not free. On this machine the six containers hold ~3.8 GB, of which
**ClickHouse alone is ~1.9 GB RAM and ~2.7 GB disk**. See
[section 8](#8-langfuse-v2-vs-v3--why-this-stack-has-clickhouse) for what to do if that is too much.

### 1. Tracing — the observation tree

A trace is a **tree of typed observations**, not a flat log. A real
`rag-request`, exactly as the UI renders it:

```
TRACE  rag-request              user amina.k · env production · session psi-thresholds-009
└─ SPAN       rag-request          15.22s
   ├─ SPAN       rag-pipeline        6.56s
   │  ├─ RETRIEVER  retrieve          0.00s   ← which docs came back, and their scores
   │  └─ GENERATION generate-answer   6.55s   ← model, params, tokens, cost, prompt version
   └─ EVALUATOR  llm-judge            8.66s
      └─ GENERATION judge-generation  8.66s
```

The types are not decoration — Langfuse renders and filters on them. Read that
tree and one thing jumps out immediately: **the judge cost more than the answer**
(8.66s vs 6.55s). Evaluation is not free, and that is the sort of fact you only
ever discover by looking at a trace.

The other four shapes:

```
agent-request    AGENT monitoring-agent ─┬─ GENERATION agent-turn-1
                                         ├─ TOOL       tool:get_psi
                                         └─ GENERATION agent-turn-2
stream-request   GENERATION ollama-stream       completion_start_time = TTFT
guarded-request  GUARDRAIL  input-guardrail     WARNING level when it blocks
error-request    GENERATION doomed-generation   ERROR level + the provider's message
```

Seven observation types are in use: `SPAN`, `GENERATION`, `RETRIEVER`,
`EVALUATOR`, `TOOL`, `AGENT`, `GUARDRAIL`.

### 2. Sessions, users, environments

These come from a single `propagate_attributes()` block wrapping each
conversation. They are cheap to set and **impossible to backfill** — traces
already written stay unattributed forever, so forgetting them is permanent.

- **`session_id`** groups turns into a thread. 112 of them. Without it a
  multi-turn conversation is just a pile of unrelated requests.
- **`user_id`** is what you need the day someone says *"it gave me nonsense this
  morning"*. Traffic is deliberately uneven — `amina.k` produces 53 traces to
  `omar.t`'s 10 — because that is what makes cost-per-user a chart worth having
  rather than a flat bar.
- **`environment`** keeps staging and student traffic out of production quality
  numbers. It is a first-class filter, not a tag. Note the fourth value:
  `sdk-experiment` is set by the SDK itself for experiment runs, so evaluation
  traffic never pollutes your production metrics.

Traces also carry `version` (`1.0.0` / `1.1.0`) and tags (`session-4`, `ollama`,
the request kind, and `prompt:<label>`).

### 3. Scores — 645 of them, across all four data types

Traces tell you *what happened*. **Scores are the only thing you can chart, alert
on, or compare between versions.**

| Score | Type | Written by | n | Observed |
|---|---|---|---|---|
| `faithfulness` | NUMERIC | LLM judge | 119 | mean **0.867**, range 0 → 1 |
| `relevance` | NUMERIC | LLM judge | 119 | |
| `grounded` | BOOLEAN | judge ≥ 0.7 | 119 | |
| `retrieval_hit` | CATEGORICAL | deterministic | 87 | `hit` 71 · `weak` 11 · `miss` 5 |
| `user_feedback` | NUMERIC | simulated thumbs | 63 | 75% positive |
| `user_comment` | TEXT | simulated free text | 25 | *"Invented a threshold I can't find anywhere."* |
| `tool_calls` | NUMERIC | deterministic | 30 | agent traces |
| `answered` | BOOLEAN | deterministic | 30 | agent traces |
| `blocked` | BOOLEAN | input guardrail | 9 | guardrail traces |
| `keyword_recall` | NUMERIC | deterministic | 24 | experiments |
| `correct_refusal` | BOOLEAN | deterministic | 8 | experiments |
| `mean_faithfulness` / `mean_relevance` / `mean_keyword_recall` | NUMERIC | run-level aggregate | 4 each | one per dataset run |

Four things this table is trying to teach:

- **Not every score needs a model.** `retrieval_hit`, `tool_calls`, `answered`,
  `blocked`, `keyword_recall` and `correct_refusal` are computed from facts
  already in hand. They cost nothing, never flake, and catch the failure that
  matters most in RAG — the model answering confidently with nothing retrieved.
- **`hit` 71 / `weak` 11 / `miss` 5 is a real distribution, not all-green.** That
  is only true because one topic in six is deliberately unanswerable. A quality
  dashboard with no failures on it is not measuring anything.
- **`faithfulness` ranges 0 → 1 with mean 0.867.** A judge that returns 1.0 for
  everything is a broken judge; the spread is what makes the metric usable.
- **TEXT scores are not chartable, and that is the point.** A NUMERIC score tells
  you quality dropped on Tuesday. The TEXT ones are where the *reason* lives.

### 4. Prompts — one prompt, two live versions

`session4-drift-assistant` lives in Langfuse rather than in the Python file:

| Version | Label | Difference |
|---|---|---|
| v1 | `production` | answer from context only, three sentences max |
| v2 | `experimental` | **also** demands inline `[doc-id]` citations, refuses harder |

Every `generate-answer` observation records which version produced it, because
the prompt object is passed into the generation call. That link is what makes
"quality by prompt version" answerable at all — and it is why promoting a prompt
is a label move in the UI rather than a redeploy.

### 5. Datasets and experiments — what traces cannot tell you

`session4-drift-qa`, 8 gold questions. **Two are unanswerable on purpose** and
expect the exact refusal string; a gold set made only of questions the system
handles well measures nothing.

Each dataset run scores every item with both an LLM judge (`faithfulness`,
`relevance`, `grounded`) and a deterministic check (`keyword_recall`, or
`correct_refusal` for the two unanswerable ones), then aggregates to `mean_*` at
the run level. The result is the most useful thing in the project:

```
prompt 'production'   (v1)   mean_faithfulness 0.925   mean_keyword_recall 1.000
prompt 'experimental' (v2)   mean_faithfulness 0.825   mean_keyword_recall 0.833
```

**The "improved" prompt is worse.** Asking for citations sounds like a free win
and measurably is not. Finding that out took 90 seconds against a fixed dataset.
Finding it out from production traffic would have taken a week of degraded
answers and an argument about whether anyone actually noticed.

> Run it more than once before believing a gap that size. Eight items is a small
> gold set and run-to-run variance is not zero, even at temperature 0.

### Regenerating and resetting

```bash
python langfuse_workload.py                    # add another batch of traffic
python langfuse_workload.py --experiment-only  # re-run just the prompt comparison
```

Traces are **append-only** — running again adds traffic rather than replacing it,
while prompt versions and dataset items are created once and reused. To start
completely clean:

```bash
docker compose --profile langfuse down -v      # -v also destroys the volumes
```

> `down -v` deletes Postgres, ClickHouse and MinIO. Everything above is gone,
> including your API keys — on the next `up`, the `LANGFUSE_INIT_*` variables
> re-provision the default org, project and keys from `.env`.

Deleting traces from the UI or API is **asynchronous**: the call returns `200`
immediately and a worker removes the rows up to a minute later. A browser tab
left open across that window will still show the deleted rows, and clicking one
gives `Trace not found — traces.byIdWithObservationsAndScores`. That is a stale
page, not a broken database. Refresh it.

### One thing this data is not

**It has no history.** Every trace lands inside the ~9 minutes the script runs,
so any chart with time on the x-axis shows a single spike rather than a trend.
Traces cannot be backdated through the SDK. Run the script on a few different
days if you want the time-series views to mean anything.

---

## 8. Langfuse v2 vs v3 — why this stack has ClickHouse

Six containers for a tracing tool invites an obvious question: **Langfuse v2 ran
on Postgres alone — two containers. Why not use that instead?**

It is a fair question, and the resource numbers back it up. Measured on this
machine while the stack was serving the run above:

| Container | RAM | Disk |
|---|---|---|
| `langfuse-clickhouse` | **1.87 GB** | **2.0 GB** + 0.7 GB logs |
| `langfuse-web` | 1.06 GB | — |
| `langfuse-worker` | 0.49 GB | — |
| `langfuse-minio` | 0.24 GB | 5 MB |
| `langfuse-postgres` | 0.10 GB | 47 MB |
| `langfuse-redis` | 0.02 GB | 0.5 MB |
| **total** | **~3.8 GB** | **~2.8 GB** |

That is roughly **half of Docker Desktop's default 7.6 GB VM**, and ClickHouse is
the single largest consumer. Dropping it is genuinely tempting.

### Why we don't

**The v4 Python SDK cannot send traces to a v2 server.** This is not a
preference, it is a wire-protocol mismatch — and it is worth being precise about,
because the SDK actually uses *two* transports:

| What | Endpoint | Since |
|---|---|---|
| spans / observations | `POST /api/public/otel/v1/traces` (OTLP HTTP) | **v3** |
| scores | `POST /api/public/ingestion` (batch events) | v2 |

```
langfuse/_client/span_processor.py         → OTLPSpanExporter(f"{base_url}/api/public/otel/v1/traces")
langfuse/_task_manager/score_ingestion_consumer.py → /api/public/ingestion
```

Langfuse only grew OpenTelemetry ingestion in v3, so a v2 server has no
`/api/public/otel/v1/traces` to post to. Point the v4 SDK at one and you get the
worst possible outcome: scores keep being accepted, while every span is rejected
— scores with no traces to attach to.

Running v2 therefore means pinning `langfuse<3` on the client side, and that SDK
is a different library:

| Needed by our code | v2 SDK | v4 SDK |
|---|---|---|
| `from langfuse import observe` | ✗ (`langfuse.decorators`) | ✓ |
| `propagate_attributes()` | ✗ (`langfuse_context.update_current_trace`) | ✓ |
| `start_as_current_observation()` | ✗ | ✓ |
| `run_experiment()` / `Evaluation` | ✗ | ✓ |
| `environment` as a filter | ✗ | ✓ |
| `RETRIEVER` / `EVALUATOR` / `AGENT` / `GUARDRAIL` types | ✗ | ✓ |

This is exactly the deprecation table at the top of
[`langfuse_example.py`](langfuse_example.py) — that file is kept precisely
*because* it is v2-era and no longer runs. Moving the server back to v2 would
mean rewriting both working scripts against a dead API, and losing sections
[3](#3-scores--645-of-them-across-all-four-data-types) and
[5](#5-datasets-and-experiments--what-traces-cannot-tell-you) of the inventory
outright — no experiments, no environments, half the observation types.

**Session 4 is about observability practice, not about saving 2 GB.** The whole
point of the session is that the tooling should tell you things you did not
already know; a backend that cannot store an evaluator score cannot do that.

### If memory really is the constraint

Two options that keep v3, in order of preference:

**1. Raise Docker's memory ceiling.** This machine has 18 GB of RAM and Docker
Desktop is only allowed 7.6 GB of it. *Docker Desktop → Settings → Resources →
Memory* — 10–12 GB leaves plenty of headroom and costs nothing.

**2. Cap ClickHouse.** Most of that 1.87 GB is cache it will give back under
pressure. Adding a limit to the `langfuse-clickhouse` service in
`docker-compose.yaml` constrains it:

```yaml
  langfuse-clickhouse:
    # ...
    mem_limit: 1g
```

**3. Stop the stack when you are not using it.** Traces are on disk, not in RAM:

```bash
docker compose --profile langfuse stop     # keeps all data, frees ~3.8 GB
docker compose --profile langfuse start    # back, with everything still there
```

Note `stop`/`start`, not `down`/`up` — and never `down -v`, which deletes the
volumes. If you only need the drift material and not the LLM tracing, plain
`docker compose up -d` brings up Prometheus, Grafana and node-exporter alone
(3 containers, ~270 MB) and never touches Langfuse at all. That is why Langfuse sits behind a
compose **profile** in the first place.


---

## 9. The incident engine — find it from the evidence

Everything above this point is instrumentation: metrics, drift reports, traces.
This section is the part that turns instrumentation into a skill. Every failure
below can be injected with one command, and the exercise is to find it using
only the dashboard, the drift report and the traces — the terminal deliberately
tells you nothing but the symptom.

```bash
make bootstrap          # once — fit and freeze the model + drift reference
make up                 # Prometheus + Grafana
make seed               # 14 simulated days, so no panel starts empty
make api                # terminal 2 — the instrumented API on :8001
make traffic            # terminal 3 — ~20 rps of rides

make list-incidents     # what can be injected
make incident-01        # inject one. It prints a symptom and nothing else.
make heal               # clear every active incident
```

Start at **[RUNBOOK.md](RUNBOOK.md)** — the six-step diagnostic path, with this
repo's real URLs, real SQL and real rollback commands. Every Prometheus alert
links to the relevant section of it.

### The incidents

Instructor view: `make reveal-01` prints the cause, the signals and the lesson.
Do not run it during the lab.

| # | Component | Budget | What it teaches |
|---|---|---|---|
| 01 | upstream feed | 30 min | A unit change passes every schema validator. Only a distribution catches it. |
| 02 | model API | 120 min | Train/serve skew is invisible to *every* drift monitor — only replay finds it. |
| 04 | model API | 10 min | Cardinality is a per-label-**value** cost; `sample_limit` makes it loud and local. |
| 05 | upstream feed | 180 min | A healthy average is how a real segment failure stays invisible. |
| 06 | upstream feed | 24 h | Real, sustained, well-labelled drift can still be the wrong reason to retrain. |
| 07 | RAG prompt | 60 min | A prompt is a deployable artifact. Unversioned means unobservable. |
| 08 | cost | — | Not a break. Budgets are set by the peak week, never the average. |
| 09 | RAG retrieval | 240 min | A global retrieval metric hides a single-language collapse. |
| 10 | model API | 45 min | Two versions behind one endpoint make every aggregate a blend. |
| 13 | host | 240 min | The only alert that fires *before* the outage: extrapolate, don't threshold. |
| 15 | feedback | 360 min | Monitor the monitoring. A frozen metric looks exactly like a healthy one. |

Incidents **11, 12 and 14 are deliberately unimplemented** — implementing the
injector *and* the alert that detects it is homework, submitted as a PR.

### Incident 03 — GPU thermal throttling (instructor aside, not a lab)

Not injectable here, and the omission is the honest one: the ride-duration model
is a RandomForest on CPU, so a GPU panel on this dashboard would be a lie.

The **Ollama container in this same stack is GPU-resident**, though. On a 24 GB
card serving an 8B model, thermal and memory pressure during a long evaluation
run are real. If you have DCGM available, watch `DCGM_FI_DEV_SM_CLOCK` while a
60-item eval runs and narrate the clock dropping. Two minutes, on the
instructor's machine only.

### What the engine is made of

| Path | Role |
|---|---|
| [`incidents/inject.py`](incidents/inject.py) | one entry point; prints symptoms, never causes |
| [`incidents/catalog.py`](incidents/catalog.py) | the failures, each with its symptom, cause and signals |
| [`services/model_api.py`](services/model_api.py) | `P_G_monitoring.py` made runnable, same metric names |
| [`tools/traffic.py`](tools/traffic.py) | the ride feed — three incidents live here, not in the API; `--bad-rate` adds malformed requests |
| [`jobs/daily_drift.py`](jobs/daily_drift.py) | Evidently against a **frozen** reference, results **stored** |
| [`jobs/retrain_gate.py`](jobs/retrain_gate.py) | the five gates that answer "should session 2 run again?" |
| [`tools/replay.py`](tools/replay.py) | re-score logged serving vectors offline — finds incident 02 |
| [`jobs/concept_drift.py`](jobs/concept_drift.py) | `hinkley_adwin.py`'s Page-Hinkley + ADWIN, run over the real late-label error stream |
| [`tools/seed.py`](tools/seed.py) | 14 days of history, without which every trend lesson is invisible |
| [`RUNBOOK.md`](RUNBOOK.md) | the six-step path every alert links into |
| [`incidents/postmortems/`](incidents/postmortems/) | the five-sentence format, plus a worked example |

> **Why the incidents flip a file instead of restarting a container.** The API
> runs on the *host* — Prometheus scrapes `host.docker.internal:8001` — and the
> stack deliberately stays at eight containers. So an incident writes
> `incidents/state.json`, which the API and the traffic generator re-read within
> a second. `make incident-01` therefore works no matter which terminal started
> the API, and healing is instant.

---

## 10. Evaluating the LLM — and evaluating the evaluator

Sections 1–4 measure a regressor against labels. This one measures a generator
against a rubric, which is a harder problem: the ruler is itself a model, and it
can be wrong in ways that look exactly like the system being measured.

### The bilingual knowledge base

[`evals/knowledge_base.py`](evals/knowledge_base.py) holds 44 notes on this
session's own material, each written in **English and Arabic**. That one file is
the entire cost of the bilingual chapter — no vector database, no second
embedding model, no external corpus.

Two things in it are load-bearing:

- **Tokenization.** The original retriever matched `[a-z0-9]+`, which finds
  nothing at all in Arabic script. It is Unicode-aware now, so Arabic retrieval
  works *before* incident 09 breaks it on purpose.
- **Normalisation.** Arabic writes the same word several ways — alef with or
  without hamza, teh marbuta versus heh, optional diacritics, the definite
  article attached to the noun. Search folds these away, and the index and the
  query must fold them **identically**. Incident 09 breaks exactly that
  invariant, and the measured result is the lesson:

  | | zero-hit rate (en) | zero-hit rate (ar) |
  |---|---|---|
  | healthy | 0% | 10% |
  | `make incident-09` | **0%** | **80%** |

  English is byte-identical because every folding step is a no-op on Latin
  script. A global retrieval metric would show this as a mild dip.

### `evals/judge.py` — the judge, with its mechanism showing

RAGAS is a judge too, and a better-engineered one. It also hides what it does
behind a metric name. This file exists so "what is faithfulness, actually?" has
an answer you can read on screen. Grown from `judge_answer()` in
[`ollama_langfuse_rag.py`](ollama_langfuse_rag.py), which is the same idea in
twenty lines, plus the four things that make its numbers safe to quote: a model
fingerprint, counted parse failures, a refusal flag, and per-language reporting.

```bash
make judge          # the two shipped samples, one Arabic and one English
```

The English demo answer contains a claim that is **true in reality and absent
from the context** ("PSI is the standard metric used across the credit risk
industry"). A working groundedness rubric must score it below 1.0 — that gap is
the whole difference between *correct* and *grounded*. It scores 0.667 and names
the offending sentence.

> **A bug worth keeping in the file.** `language_match` originally asked the model
> for a `matches` boolean, and llama3.1:8b returned
> `{"matches": false, "answer_language": "English"}` for an English answer to an
> English question — contradicting itself inside one object, because constrained
> decoding commits to the first field before "thinking about" the second. The fix
> is one line of Python: ask for the two languages, compare them yourself. Order
> your schema so observations precede conclusions, and never ask a model for a
> conclusion you can compute from its own output.

### `evals/calibrate_judge.py` — do we believe it?

A judge produces numbers whether or not it is any good, and the numbers look the
same either way. Raw agreement is not enough: on a set that is 80% grounded, a
judge that always answers "grounded" scores 80% agreement while measuring
nothing. Cohen's kappa subtracts chance agreement, so that judge scores ~0.

```bash
make calibrate      # 50 labelled items, 25 Arabic — takes about 5 minutes
```

Measured here, on `llama3.1:8b`, with zero parse failures:

| language | n | raw agreement | Cohen's κ | verdict |
|---|---|---|---|---|
| English | 25 | 0.920 | **0.841** | usable |
| Arabic | 25 | 0.880 | **0.757** | **do not trust** |

The 0.80 trust floor lands *between the two languages*. Same judge, same rubric,
same run — and one language is certifiable while the other is not. The
disagreements point the same way: on English the judge is too **harsh**; on
Arabic it is too **lenient**, missing planted unsupported claims.

This is slide 53's warning, reproduced on a student laptop as a number. **Do not
"fix" it by swapping in a bigger model** — run it again on a larger judge and
show the gap narrowing instead. The takeaway is not "Arabic is hard", it is that
an Arabic score and an English score from this judge are **not on the same
scale**, so they must never be averaged together or compared to one threshold.

The labels are known by *construction*, not by one person's opinion: a grounded
item's answer is assembled only from its own context, and an ungrounded one is
the same answer with a plausible distractor spliced in. That is auditable, and
it makes every disagreement unambiguously the judge's.

---

### `evals/run_ragas.py` — the library judge, and the gate

RAGAS is a judge too, and a better-engineered one; it also hides the mechanism.
Run both on the same ten samples: where they agree the number is probably real,
where they diverge is the calibration lesson.

```bash
make ragas                    # 10 stratified items from the frozen testset
make ragas-gate               # the CI gate — exits non-zero
python -m evals.run_ragas --source langfuse --hours 24 --sample 0.03 --push
python -m evals.run_ragas --source dataset --compare baseline
```

Three things it refuses to do, each because doing them is how eval harnesses end
up reporting numbers nobody should act on:

- **Print a bare mean.** A measured run here: faithfulness mean 0.573 — but the
  distribution is two items at 0.0 and two at 0.9–1.0. A bimodal 0.85 and a
  uniform 0.85 are different systems and the mean cannot tell them apart.
- **Ignore NaNs.** A parse failure makes RAGAS retry and then drop the sample.
  The dropped samples are the *hard* ones, so the surviving mean flatters you.
  Above 10% the run fails.
- **Gate on the global mean.** Any single language more than 0.10 below the
  threshold fails the run even when the overall number passes — a healthy
  average with one collapsed language is exactly how incident 09 hid.

Refusal items are excluded from the gate and counted separately. Faithfulness
over a refusal is meaningless (there are no claims to ground), and leaving them
in means a testset that deliberately contains unanswerable questions can never
pass its own gate. The bucket is a signal in its own right — a real run reported
*"4 unanswerable items excluded — 1 scored >0.5, i.e. the model answered
anyway"*, which is a confabulation nothing else in the stack would have caught.

### `evals/build_testset.py` — 60 items, frozen

30 English, 30 Arabic. Synthetic items from the knowledge base, plus a
hand-written adversarial block: 8 questions with **no answer in the corpus**, 8
needing **two notes**, 6 deliberately **ambiguous**. The no-answer items are the
only way to find out whether the model refuses or confabulates.

The file is frozen with a content hash and the loader refuses to run against a
modified one. A gate compares runs over time; if the testset changes between
runs, a score movement could be the model or could be the test, and nothing
downstream can tell you which.

### `tools/cost_report.py` — Incident 08

Langfuse prices a generation by matching its model name against a price list,
and it has no entry for `llama3.1:8b` — so `calculatedTotalCost` really is
`0` on every generation in this stack, and a cost dashboard says inference is
free. It is not. The report imports the amortised GPU rate from
`langfuse_workload.py` rather than re-deriving it, and reports the **min and max
week, never the average**, because a budget is set by the week that would have
overrun it.

```bash
make cost                                     # self-hosted, amortised
python -m tools.cost_report --frontier 60     # "what would migrating cost?"
```

> **A version trap worth knowing.** The SDK's `client.api.observations.get_many()`
> is the v2 API and answers 404 — *"only available in a Langfuse v4 write mode"* —
> against the `langfuse/langfuse:3.x` server this stack pins. The SDK major and
> the server major move independently. `cost_report.py` calls the v1 public
> endpoint that a v3 server actually serves.

### Which Langfuse feature is demonstrated where

[`docs/langfuse_capability_map.md`](docs/langfuse_capability_map.md) maps every
capability the session teaches to the file and line range that already
demonstrates it, and lists what to project during Chapter 4, in order. There is
deliberately **no separate tour script**: a demo that exists only to be demoed
rots quietly, because nothing fails when it stops being true.

---

## 11. Running these

The session has two halves, and they run differently.

### The eight snippets still do not run — deliberately

`data_drift_evidently.py`, `label_predicion_drift.py`, `automated_drift_report.py`,
`hinkley_adwin.py`, `embedding_drift.py`, `guardrails_example.py`,
`ragas_example.py` and `P_G_monitoring.py` are **teaching snippets**.
`pip install -e .` makes every import resolve, but each has undefined names you
must supply — deliberately, so it is obvious which part belongs to *your*
pipeline:

| Script | Undefined / missing |
|--------|---------------------|
| `data_drift_evidently.py` | `today`; the parquet files under `data/` |
| `label_predicion_drift.py` | `import pandas as pd`; `today`; `ref_with_labels`, `curr_with_labels` |
| `hinkley_adwin.py` | `trigger_retraining()`; a caller for `monitor_live_predictions(model, stream)` — the file only *defines* it |
| `P_G_monitoring.py` | `app` (the FastAPI instance), `model`, `PredictRequest`; a `start_http_server(8001)` call |
| `embedding_drift.py` | `load_queries()`; builds a SentenceTransformer at import |
| `ragas_example.py` | `trace_id`; the `...` placeholders in the dataset |

`today` is a bare name because in production it is an Airflow template —
`{{ ds }}` passed into the task — not something the script computes. Same for
`trigger_retraining()`: in a real DAG that is a `TriggerDagRunOperator`, not a
local function.

**Each one has a runnable descendant**, so the snippet is what you read and the
module is what you run:

| Snippet | Grown into |
|---|---|
| `P_G_monitoring.py` | [`services/model_api.py`](services/model_api.py) — same metric names, a real service |
| `data_drift_evidently.py` | [`jobs/daily_drift.py`](jobs/daily_drift.py) — same legacy API, frozen reference, results stored |
| `hinkley_adwin.py` | [`jobs/concept_drift.py`](jobs/concept_drift.py) — the same detectors over real late labels |
| `ragas_example.py` | [`evals/run_ragas.py`](evals/run_ragas.py) — three run modes and a gate |
| `embedding_drift.py` | its MMD restated in `jobs/concept_drift.py` (the original is not importable) |
| `guardrails_example.py` | `mask_pii()` in [`ollama_langfuse_rag.py`](ollama_langfuse_rag.py), installed at the SDK boundary |

### Everything in sections 9 and 10 runs

```bash
make bootstrap        # once — fit and freeze the model + the drift reference
make up               # Prometheus + Grafana + node-exporter
make seed             # 14 simulated days, so no panel starts empty
make api              # terminal 2 — the instrumented API on :8001
make traffic          # terminal 3 — ~20 rps of rides
make doctor           # 26 checks; run this BEFORE the session, not during it
```

Then the labs: `make incident-01`, `make drift`, `make gate`, `make replay`,
`make heal`. For the LLM half, `make up-langfuse` and `make seed` with `--llm`,
then `make judge`, `make calibrate`, `make ragas`, `make cost`.

`make help` lists all 25 targets.

### Two failure modes worth pre-empting

> **"Langfuse rejects my keys."** `.env` is probably missing the SDK-side
> variables. The server reads `LANGFUSE_INIT_PROJECT_PUBLIC_KEY` /
> `..._SECRET_KEY`; the SDK reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` /
> `LANGFUSE_BASE_URL`. Both sets must be present **and match**. An `.env` copied
> from an older `.env.example` has only the first set, and `auth_check()` fails
> with "initialized without public_key".

> **"`make seed --llm` dies on `No module named numpy`."** You have two
> virtualenvs: one with the metrics stack and one with the LLM stack. The lab
> needs them in **one** interpreter — `pip install -e ".[metrics,llm,evals]"`.
> `make doctor` detects this and says so.

---

## Where this fits

- **Session 2** — train, track, register the model
- **Session 3** — orchestrate and deploy it
- **Session 4** — watch it, and decide when session 2 needs to run again

That loop is closed here in code, not in prose:
[`jobs/retrain_gate.py`](jobs/retrain_gate.py) is the "decide" step —
drift detected → **gate consulted** → retraining triggered → new model
registered → redeployed → monitored.

The gate is worth the emphasis because most of its job is to say **no**. Drift
being real is not sufficient grounds to retrain: it also has to be sustained,
the pipeline has to be healthy, enough labels have to have arrived, the cooldown
has to have passed, and it must not be Ramadan — because that drift reverts in
three weeks and a model fitted to it will be wrong for the other eleven months.
Every decision is written to `retrain_events`, **including the refusals**, since
a gate that logs only its yeses cannot be audited after it promotes something it
should not have.

That is the honest end of the story sessions 1–4 tell about one small model:
you do not finish by automating the retrain. You finish by being able to explain,
from evidence, why you did or did not run it.
