# Runbook — the ride-duration model is behaving badly

You are on call. Something is wrong with predictions. Work down this page in
order and **do not skip step 1**, because roughly half the incidents in this repo
are not model problems at all.

Every command here is real and runs against this repo. Every dashboard link uses
the ports in [`.env.example`](.env.example) — Prometheus is on **9095**, not 9090.

| | Where |
|---|---|
| Grafana — ML Overview | <http://localhost:3000/d/session4-model-monitoring> |
| Prometheus — graph | <http://localhost:9095/graph> |
| Prometheus — targets | <http://localhost:9095/targets> |
| Prometheus — alerts | <http://localhost:9095/alerts> |
| Langfuse (LLM traces) | <http://localhost:3001> |
| Metrics store | `sqlite3 session_4/metrics_store.db` |

**The five-sentence rule.** When you are done, write the postmortem in
[`incidents/postmortems/TEMPLATE.md`](incidents/postmortems/TEMPLATE.md) before
you close the ticket. Five sentences. If you cannot write them, you have not
finished diagnosing — you have only stopped the bleeding.

---

## Step 1 — Is it the model at all?

Most "the model is broken" pages are infrastructure. Rule that out first; it
takes ninety seconds.

```bash
curl -s localhost:8001/health                  # is it even alive?
curl -s localhost:8001/metrics | grep -c .     # is it exporting anything?
```

```promql
up{job="model-api"}                                    # 0 = Prometheus can't scrape it
rate(api_request_latency_seconds_count[5m])            # is traffic arriving at all?
histogram_quantile(0.95, sum by (le) (rate(api_request_latency_seconds_bucket[5m])))
```

**If `up == 0` but `curl` works**, the service is fine and the *scrape* is
broken. Open <http://localhost:9095/targets> and read the error. `sample limit
exceeded` means a cardinality explosion — go to
[HighDriftedFeatures](#highdriftedfeatures)'s neighbour, [the cardinality
section](#the-scrape-is-failing-sample-limit-exceeded).

**If traffic is zero**, nobody is calling you. That is an upstream or routing
problem, and no amount of model investigation will help.

---

## Step 2 — Is the data pipeline healthy?

The model can only be as good as what arrives. Check the inputs *before* the
outputs.

```bash
python -m jobs.daily_drift            # one window against the frozen reference
```

Read the PSI column, not the p-value. With 20 rps every p-value is significant;
PSI is an effect size and does not inflate with volume.

| PSI | Meaning |
|---|---|
| < 0.10 | stable |
| 0.10 – 0.25 | watch |
| > 0.25 | act |

Then sanity-check the ranges. A unit change (km → miles, seconds → milliseconds)
passes every schema validator ever written, because the values are still valid
numbers of the right type:

```bash
sqlite3 -header -column metrics_store.db \
  "SELECT feature, value FROM drift_metrics
    WHERE metric='psi' AND segment='all'
      AND window_end = (SELECT MAX(window_end) FROM drift_metrics)
    ORDER BY value DESC;"
```

**If exactly one input feature moved hard and the rest are flat**, suspect that
feature's producer, not the model.

---

## Step 3 — What changed, and when?

Almost every incident is a change. Find the change before you theorise.

```bash
sqlite3 -header -column metrics_store.db \
  "SELECT ts, component, version, note FROM deploys ORDER BY ts DESC LIMIT 10;"
```

On the Grafana dashboard, the red **Deploys** annotation marks these on every
panel. Line the symptom's start time up against the nearest annotation.

The annotation is driven by `deploy_events_total`, not by `deploy_info`.
`deploy_info` is cleared and rewritten on every scrape so it only ever holds the
LATEST deploy — its value changes but the series never appears, and Grafana
draws a Prometheus annotation where a series increases. The counter is that
increase. Blue **Retrains** lines come the same way from `retrain_events_total`.

```promql
deploy_info                                            # value = deploy unix ts
increase(deploy_events_total[5m]) > 0                  # what the annotation draws
count(count by (version) (model_version_info))         # > 1 means two models are live
```

**A prompt change leaves no deploy row.** If the affected system is the RAG
service, the change may be a Langfuse prompt label move with no code deploy at
all — check the prompt version on recent traces in Langfuse before concluding
"nothing changed".

---

## Step 4 — Localize it

This is the step people skip, and it is the one that finds the incidents that
matter. **Split every number you have.**

```bash
# Error by city — a new region at 8% of volume cannot move the global average
sqlite3 -header -column metrics_store.db \
  "SELECT city, COUNT(*) n, ROUND(AVG(ABS(y_true-y_pred)),2) mae
     FROM labels WHERE ts >= datetime('now','-1 day') GROUP BY city ORDER BY mae DESC;"
```

```promql
segment_psi_score                     # PSI per feature PER CITY
model_mae_minutes                     # MAE per city
sum by (version) (rate(api_request_latency_seconds_count[5m]))
```

Compare `feature_psi_score` (global) against `segment_psi_score` (per city). A
global 0.03 sitting next to a segment 4.8 is not a contradiction — it is the
answer.

Split by: **city, model version, language, prompt version, time of day.** If a
metric looks healthy, you have not split it finely enough yet.

---

## Step 5 — Is quality actually down?

Distribution shift is not the same as being wrong. Confirm with ground truth
before you retrain anything.

```bash
python -m tools.replay --limit 2000    # serving path vs offline path, same inputs
```

If the two paths disagree on identical inputs, stop: this is **train/serve skew**,
not drift, and retraining will not fix it. Nothing about the request looks wrong
because nothing about the request *is* wrong.

```bash
sqlite3 -header -column metrics_store.db \
  "SELECT date(ts) d, COUNT(*) n, ROUND(AVG(ABS(y_true-y_pred)),2) mae
     FROM labels GROUP BY d ORDER BY d DESC LIMIT 14;"
```

**If MAE is flat and labels stopped arriving, the metric is not healthy — it is
dead.** Check `rate(feedback_total[6h])` before you trust any quality number.

---

## Step 6 — Decide

```bash
python -m jobs.retrain_gate            # prints the decision and every reason
```

The gate refuses for reasons that are all worth reading aloud. In particular it
will refuse inside a **seasonal window** even when drift is real, sustained and
well-labelled — because a challenger validated on a random split of an anomalous
window is validated against the very regime it overfitted.

Rollback options, cheapest first:

| Situation | Action |
|---|---|
| Injected lab incident | `make heal` |
| Bad prompt version | Move the `production` label back in Langfuse. No deploy. |
| Two versions live | Finish or revert the canary — session 3 §2.4: set the `upstream` weights back to 100/0 and `nginx -s reload` |
| Bad model version | Re-point the MLflow registry stage at the previous version (session 2), redeploy |
| Bad upstream feed | Fix the producer. Do **not** retrain to absorb it. |

---

# Alert index

### ModelAPIDown

`up{job="model-api"} == 0` for 2m. **Page.** Users are getting errors right now.

1. `curl -s localhost:8001/health` — if this works, it is the scrape, not the app.
2. <http://localhost:9095/targets> — read `lastError` verbatim.
3. If `lastError` is `sample limit exceeded`, see [below](#the-scrape-is-failing-sample-limit-exceeded).

### NoPredictionTraffic

`rate(model_prediction_duration_min_count[15m]) == 0` for 15m. **Page.** The API
is reachable and serving nothing, which no health check will ever catch.

Check the caller and the routing layer before touching the model.

### PredictLatencyP95High

p95 above 500 ms for 10m. Check whether it is one version or all of them:
`histogram_quantile(0.95, sum by (le, version) (rate(api_request_latency_seconds_bucket[5m])))`.

### FeatureDriftHigh

`feature_psi_score > 0.25` for 30m. **Warn** — nobody fixes drift at 3am.

1. Which feature? Run step 2's SQL.
2. Is it one feature or several? Several at once means the feed changed, not the world.
3. **Split it by city** (step 4). A global number is an average and averages hide segments.
4. Do not retrain until `python -m jobs.retrain_gate` agrees.

### HighDriftedFeatures

`count(feature_psi_score > 0.25) > 2` for 1h. Three features drifting together is
a pipeline problem, not three feature problems. Start at the producer.

### FeedbackSignalDead

`rate(feedback_total[6h]) == 0` for 30m. The quietest serious alert here.

No labels means MAE, the drift-vs-accuracy comparison and the retrain gate's
`enough_new_labels()` are all **frozen at their last good value**. Every quality
dashboard will look fine and mean nothing.

```bash
sqlite3 metrics_store.db "SELECT MAX(ts) FROM labels;"
```

### MultipleVersionsServing

`count(count by (version) (model_version_info)) > 1` for 10m. An unfinished
canary. Until it is resolved every aggregate panel is a blend of two models — so
split by version before reading anything else. Rollback: session 3 §2.4.

### DiskFillPredicted

`predict_linear(disk_free_bytes[6h], 4*3600) < 0` for 30m. The only alert here
that fires *before* the outage. A disk at 85% tells you nothing; a disk losing
4 GB/hour tells you when it dies. Rotate or expand now.

### The scrape is failing: `sample limit exceeded`

A metric acquired an unbounded label (a ride id, a request id, a user id) and is
minting one time series per request. The scrape is rejected **whole**, so the
target goes down and the dashboards go blank rather than noisy.

```promql
prometheus_tsdb_head_series
topk(5, count by (__name__)({__name__=~"model.*"}))
```

Removing the label from the code is not enough on a running process: the series
already registered keep being exported. The collector has to be dropped — which
in production means the restart everyone forgets they need.

Prevention lives in [`tests/test_cardinality_guard.py`](tests/test_cardinality_guard.py),
which fails CI when a metric declares a denylisted label.
