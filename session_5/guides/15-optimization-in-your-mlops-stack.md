# 15 — Optimization in your MLOps stack

A faster model that nobody can reproduce, verify or roll back is a demo. Guides 00
to 13 produced faster artifacts. This guide makes those gains *stay*. A CI gate
fails the build when a PR brings the slowness back. A benchmark card travels with
each artifact into the Session 2 registry. A pinned container builds engines the
serving GPU can load. A shadow comparison and the Session 4 monitoring stack catch
the INT8 model getting worse at night, before a customer does. This is what makes
Session 5 an Ops session and not a kernels session.

---

## 1. The problem it solves

Optimization gains decay in three distinct ways, at three distinct times:

| Failure | When it happens | What catches it | Plugs into |
|---|---|---|---|
| a PR puts Python NMS back in the hot path, or swaps the runtime, and p95 creeps up | **merge time** | the CI perf gate: `ci/perf_gate.py` + `.github/workflows/session5-perf-gate.yml` | Session 2's CI |
| someone deploys `model_int8.tflite` on the strength of the FP32 model's numbers, or ships a TensorRT engine built for another GPU | **release time** | benchmark cards per artifact + pinned build container | Session 2's MLflow registry; Session 3's Triton |
| after cutover, the traffic mix shifts (more night, more rain) and the INT8 model's per-condition weakness starts to dominate | **run time** | shadow disagreement per condition + drift on input statistics | Session 3's shadow release; Session 4's Prometheus, Grafana, Evidently |

A unit test answers "does it run?". Nothing in Sessions 1–4 answers "is it still
fast enough, still accurate enough, and still the artifact we measured?". That is
the gap this guide closes.

---

## 2. Mental model

Every journey-table row carries five metrics plus a cost. In this guide those
numbers become **contracts**, checked at three points, all read from one file:

```mermaid
flowchart LR
  CFG["src/config.py<br/>SLA dataclass"] --> GATE
  PR["pull request"] --> GATE["cpu-gate<br/>relative p95 + accuracy drop"]
  GATE --> MERGE["merge"]
  MERGE --> GPU["gpu-sla-gate on the RTX 3090<br/>absolute p95 <= SLA, pinned container"]
  CFG --> GPU
  GPU --> CARDS["benchmark cards<br/>one per artifact per hardware"]
  CARDS --> REG["MLflow registry<br/>model version + artifacts + cards"]
  REG --> SHADOW["shadow: INT8 next to FP32<br/>per-condition disagreement"]
  SHADOW --> PROM["textfile collector -> Prometheus<br/>alert + Grafana panel"]
  PROM --> DECIDE{"cut over, hold,<br/>or roll back"}
```

> **Why every threshold lives in `src/config.py`.** The `SLA` dataclass holds the
> latency budget (`p95_ms`), the batch size and the allowed accuracy drops
> (`max_map50_drop`, `max_ocr_em_drop`). `ci/perf_gate.py` reads `config.SLA_TARGET`
> and never keeps its own copy. `tools/journey_table.py` and
> `tools/benchmark_card.py` read the same object for the `≤ SLA` column and the SLA
> verdict. Copy the numbers into a workflow YAML and the day someone tightens the
> SLA, the table says one thing and CI enforces another. Both will look
> authoritative.

> **Why relative *and* absolute checks.** A GitHub-hosted runner has no GPU and
> shares its CPU with other workloads, so its absolute p95 means nothing about the
> roadside box. A *ratio* between two rows measured on the same runner in the same
> job does mean something: "this PR made the ONNX Runtime row slower than it was
> against the same reference". The absolute question, "p95 ≤ SLA", only has an
> answer on the target hardware. So it runs only there, behind `--enforce-sla`.

---

## 3. Runnable walkthrough

### 3a. The CI perf gate

The check itself is short: `snippet:perf-gate` in `ci/perf_gate.py`. It runs three
checks and prints every failure with both numbers:

| Check | Rule | Threshold comes from | Runs where |
|---|---|---|---|
| accuracy | baseline (`--baseline`, default `s01_baseline:eager-fp32`) mAP@0.5 minus candidate mAP@0.5, and the same for exact match, may not exceed the allowed drop | `SLA.max_map50_drop`, `SLA.max_ocr_em_drop` | everywhere |
| regression | with `--remeasure`, the candidate is measured again now; its new p95 may not exceed its **own recorded** p95 × `--max-p95-ratio` | the CLI flag (its default is in `ci/perf_gate.py`) | everywhere |
| absolute | candidate p95 ≤ `SLA.p95_ms` (30 ms) | `SLA.p95_ms` | only with `--enforce-sla`, on the target hardware |

It looks rows up by `(row id, hardware_id, profile)`. `hardware_id`
(`src/environment.py`) hashes the CPU, the GPU **and** `ANPR_THREADS`, so the gate
can only compare rows measured on the same machine at the same thread count.
That restriction is intentional.

> **Why latency is compared with the row's own record, not with another row.** An
> early version gated the ONNX Runtime row against the eager PyTorch baseline. On the
> laptop that produced the committed results, the ONNX Runtime row is the *slower*
> one (compare the two s04/s01 rows in the journey table), so that gate failed for a
> hardware reason and would pass on a machine where the order flips. A code change
> that slows ONNX Runtime down, on the other hand, slows it down on every machine —
> which is exactly what re-measuring a row against its own record detects.

**Where the workflow lives.** It is `.github/workflows/session5-perf-gate.yml` at
the *repository root*
([view](../../.github/workflows/session5-perf-gate.yml)), not under `session_5/ci/`,
because GitHub only runs workflows from the root `.github/workflows/` directory.
Session 2's CI workflow sits in the same place for the same reason. The gate
*logic* stays in `session_5/ci/perf_gate.py`, so it runs identically from `make`
and from Actions.

| Job | Runner | Profile | What it does |
|---|---|---|---|
| `cpu-gate` | `ubuntu-latest`, GitHub-hosted | `ci`: tiny data, a few epochs | installs CPU wheels, generates data, trains, runs s01 and s04, then gates `s04_onnx_export:ort-cpu-fp32`: accuracy against `s01_baseline:eager-fp32`, and a re-measured p95 against its own record. Uploads `results/` as a build artifact |
| `gpu-sla-gate` | `[self-hosted, linux, gpu]`, the RTX 3090 box | `quick` | builds `ci/docker/Dockerfile.gpu`, runs `make profile s04 s07 s08` inside it, then gates `s08_tensorrt:student-fp16` with `--enforce-sla`. Enabled only when the repo variable `SESSION5_GPU_RUNNER` is `true` |

Triggers: pull requests touching `session_5/src/**`, `session_5/ci/**`,
`session_5/tools/**`, `session_5/Makefile`, `session_5/requirements*.txt` or the
workflow file, plus `workflow_dispatch` with an `inject_latency_ms` input.

Run it locally, from `session_5/`, after the rows exist (`make s01 s04`):

```bash
make gate          # accuracy vs the eager baseline + re-measured p95 vs the ONNX row's own record
make gate-demo     # the same with an injected detector delay: watch it fail

# on the target GPU, the absolute check as well
python -m ci.perf_gate --candidate s08_tensorrt:student-fp16 --remeasure --enforce-sla
```

**How to see it fail.** `make gate-demo` sets `ANPR_INJECT_LATENCY_MS`. With that
variable set, `src/benchmark.py` wraps the backend's `detect` call in a sleep, so
every frame pays a deliberate regression. `--remeasure` re-runs the candidate
through the harness with `run(spec, record=False)`. The slow measurement is
compared and printed, and it **never overwrites** the stage's real row in
`results/results.json`. In Actions: *Actions → session5-perf-gate → Run workflow*,
set `inject_latency_ms`. That step compares the re-measured ONNX row against its
own clean measurement, so the only difference between the two is the injected
delay. The captured output of a failing run is in
[`ci/README.md`](../ci/README.md#demo-the-gate-failing).

> **Why a regression you inject yourself.** A gate you have never seen fail is a
> gate you do not know works. A mistyped row id, a threshold read as zero, or a step
> whose exit code is swallowed all look exactly like a passing build.

### 3b. Benchmark cards in the model registry

A model version in the registry is not one file. The same trained weights become
several deployable artifacts, and each one has its own accuracy, latency and memory
**on its own hardware**:

| Artifact | Built by | Runs on | Its card |
|---|---|---|---|
| `detector_student.onnx` + `ocr_student.onnx` | s07 | ONNX Runtime on any CPU; Triton (s11) | `s07_distillation__student-distilled-onnx.md` |
| TensorRT engines FP16 / INT8 | s08, inside the pinned container | the GPU model *and* TensorRT version they were built with | `s08_tensorrt__student-fp16.md`, … |
| OpenVINO IR (`.xml` + `.bin`), NNCF INT8 | s09 | x86 CPU, Intel GPU | `s09_openvino__student-int8-nncf.md`, … |
| `.tflite` full-integer INT8 | s10 | ARM CPU via XNNPACK; phone delegates | `s10_tflite_edge__tflite-int8-full.md`, … |

`tools/benchmark_card.py` renders `ci/benchmark_card.md.tmpl` once per measured row
into `results/cards/<hardware_id>_<profile>/<stage>__<variant>.md`. `make table`
runs it together with the journey table. A card records:

- lineage: the parent row it was built from
- runtime · device · precision, and the SLA verdict
- profile and validation-set sha256; git commit and timestamp
- accuracy per condition, and latency per phase
- the throughput curve, artifact size, peak RSS and peak VRAM
- the parity report against its parent, and the full environment

> **Why one card per artifact per hardware, and not one per model version.** The
> INT8 TFLite file and the FP16 TensorRT engine come from the same training run and
> can differ in exactly the ways that matter. Night exact match is one example; peak
> memory on a Raspberry Pi is another. A registry entry that holds one set of
> metrics for "the model" invites someone to deploy the INT8 file on the FP32
> file's numbers. The template's own line applies: a number without this card is a
> rumour.

**Logging them with MLflow.**
[Session 2](../../session_2/README.md#1-mlflow--experiment-tracking--model-registry)
registered one model version per training run and annotated it with tags
([Step 6](../../session_2/README.md#step-6--register-only-the-winner)). Keep that. Then
attach the per-target artifacts as nested runs, one per artifact × hardware, each
holding the artifact, its card and the tags you will search on:

```python
import mlflow
from mlflow import MlflowClient

from src import config, results

MODEL, VERSION = "anpr-plate-reader", "7"          # the version Session 2's Step 6 registered
# The files each deployable row ships. Your paths; an OpenVINO IR is .xml + .bin, log both.
DEPLOYABLE = {
    "s07_distillation:student-distilled-onnx": ["detector_student.onnx", "ocr_student.onnx"],
    "s09_openvino:student-int8-nncf": ["detector_student_int8.xml", "detector_student_int8.bin",
                                       "ocr_student_int8.xml", "ocr_student_int8.bin"],
}

client = MlflowClient()
with mlflow.start_run(run_name=f"{MODEL}-v{VERSION}-deployables"):
    for row in results.load():
        if row["status"] != "ok" or row["id"] not in DEPLOYABLE:
            continue
        card = (config.RESULTS_DIR / "cards" / f"{row['hardware_id']}_{row['profile']}"
                / f"{row['stage']}__{row['variant']}.md")
        with mlflow.start_run(run_name=f"{row['id']}@{row['hardware_id']}", nested=True) as child:
            mlflow.set_tags({"hardware_id": row["hardware_id"], "backend": row["backend"],
                             "precision": row["precision"], "device": row["device"],
                             "profile": row["profile"], "built_from": row["parent"] or "(root)",
                             "git_sha": row["git_sha"], "model_version": VERSION})
            metrics = {"p95_ms": row["latency"]["total_ms"]["p95"], "map50": row["accuracy"]["map50"],
                       "ocr_exact_match": row["accuracy"]["ocr_exact_match"],
                       "peak_rss_mb": row["peak_memory_mb"]["rss"]}
            mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            for name in DEPLOYABLE[row["id"]]:
                mlflow.log_artifact(str(config.artifacts_dir() / name), artifact_path="artifact")
            mlflow.log_artifact(str(card), artifact_path="card")
        # the registry version points at each deployable run: from "v7" you can reach every artifact + card
        client.set_model_version_tag(MODEL, VERSION, f"deployable.{row['stage']}.{row['variant']}.{row['hardware_id']}",
                                     child.info.run_id)
```

At deploy time, the question is "which artifact of version 7 was measured on *this*
box, at *this* precision". That is a search over tags, not a filename convention:

```python
mlflow.search_runs(search_all_experiments=True,
                   filter_string="tags.model_version = '7' and tags.hardware_id = '<target id>' and tags.precision = 'int8'")
```

Session 2 used one alias, `@champion`. With several targets, give each target its
own alias (`client.set_registered_model_alias(MODEL, "edge_x86", VERSION)`), and
move it only when *that target's* card passes. A version can be ready for the GPU
server and not yet for the Raspberry Pi.

### 3c. Pinned build containers

A TensorRT engine is not a portable file. It is a plan compiled for **one GPU model
and one TensorRT version**. Build it with a different TensorRT than the serving
Triton loads and deserialization fails. Build it on a different GPU and it will
not run, or will run with kernels tuned for the wrong card. TensorRT has opt-in
hardware- and version-compatibility build modes that trade some performance for
portability; this session does not use them. The rule the course follows is
simpler: **build where you serve, inside a pinned container**.

`ci/docker/Dockerfile.gpu` pins every layer of that matrix:

| Layer | Pinned to | What breaks if it drifts |
|---|---|---|
| base image | `nvcr.io/nvidia/tensorrt:26.05-py3` | the TensorRT inside it changes with every NGC release |
| TensorRT | 10.16.1.11, the same as `nvcr.io/nvidia/tritonserver:26.05-py3` | an engine built here must deserialize in s11's Triton |
| CUDA | 13.2.1 (from the image) | `onnxruntime-gpu` and torch wheels must match its major version |
| Python wheels | `torch==2.13.0+cu130`, `onnxruntime-gpu==1.30.0`, `nvidia-ml-py` | the same pins as `requirements-gpu.txt` |
| host driver | R580 or newer | CUDA 13.x minor-version compatibility needs it |
| GPU | the RTX 3090 (Ampere, SM 8.6) the self-hosted runner sits on | engines are per GPU model |

Why 26.05 in particular: NGC 26.05 is the last release with TensorRT 10.x in both the TensorRT and Triton containers (https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-05.html).
TensorRT 11 removed implicit INT8 calibration (https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html),
and s08 teaches implicit calibration through `snippet:int8-calibrator`. An
unpinned `:latest` tag would eventually remove the INT8 rows from the journey table
for reasons that have nothing to do with the model.

```bash
docker build -f ci/docker/Dockerfile.gpu -t anpr-gpu:26.05 .
docker run --rm --gpus all -v "$PWD":/workspace/session_5 anpr-gpu:26.05 make all
```

> **Why `--system-site-packages` in the Dockerfile's venv.** The NGC image already
> ships TensorRT's Python bindings. The venv can see them, so pip never installs the
> `tensorrt-cu13` wheel over them. Two TensorRTs on one path is the version
> mismatch above, packed into a single container.

The `gpu-sla-gate` job is this rule applied in CI. It builds the image *on* the
3090 runner, builds the engines inside it, and enforces the SLA on the same box.

### 3d. Shadow-deploying INT8 against FP32 before cutover

[Session 3](../../session_3/README.md#shadow-mode--zero-risk-eval) introduced shadow
mode as a release strategy: the new model sees a copy of every request, and only the
old one answers. `ci/shadow_compare.py` applies it to a precision change
(`snippet:shadow-compare`). The FP32 and INT8 pipelines see identical frames. For
every plate FP32 reads, the INT8 pipeline counts as disagreeing if it finds no box
with IoU above 0.5, or reads different text. The ratio is kept **per condition**,
because INT8 damage is rarely uniform.

```bash
# quantization only: the same baseline weights, FP32 vs s06a's stratified static INT8, both on ONNX Runtime
python -m ci.shadow_compare --fp32 detector_baseline.onnx,ocr_baseline.onnx \
    --candidate detector_int8_stratified.onnx,ocr_int8_stratified.onnx --candidate-backend ort

# the edge candidate: the distilled student FP32 on ONNX Runtime vs its NNCF INT8 IR on OpenVINO
python -m ci.shadow_compare --fp32 detector_student.onnx,ocr_student.onnx \
    --candidate detector_student_int8.xml,ocr_student_int8.xml --candidate-backend openvino
```

Always pass both pairs explicitly, and make them share a parent. The script's
defaults put the student FP32 files next to INT8 files quantized from the
*baseline*, which would measure distillation and quantization at once. The script
does three things:

- writes `results/shadow/shadow_<profile>.prom` in Prometheus text format, one
  `anpr_shadow_plate_disagreement_ratio{condition=…,candidate=…}` gauge per
  condition
- prints the same lines
- exits non-zero when the worst condition exceeds `--max-disagreement`, so the same
  script can be a release gate

> **Why disagreement needs no labels.** Accuracy needs ground truth, and in the field
> plate labels arrive late or never. Disagreement between two models on the same
> frame is available immediately. It is the same move as Session 4's prediction
> drift: a proxy you watch while waiting for the truth. A high disagreement ratio
> doesn't tell you which model is wrong, only that promoting the candidate changes
> the answers users get.

**Where shadow runs when the camera has no uplink.** The roadside box cannot run
FP32 and INT8 side by side. That would double the work inside a 30 ms budget.
Shadow the candidate where there is spare compute: on frames recorded on the box
and replayed on a bench or a central server, or on cameras that do have backhaul.
The code is the same; only the source of the frames changes.

### 3e. Monitoring quantization-induced drift with the Session 4 stack

**Get the gauge into Prometheus.** Session 4's node-exporter
(`session_4/docker-compose.yaml`) does not enable the textfile collector. Add a flag
and a read-only mount pointing at the directory the shadow job writes to:

```yaml
  node-exporter:
    volumes:
      - ../session_5/results/shadow:/textfile:ro
    command:
      - --collector.textfile.directory=/textfile
```

In production, write the file under a temporary name and rename it into place. The
collector reads whatever is in the directory at scrape time, including a
half-written file.

**Alert on it.** A sketch in the style of `session_4/monitoring/prometheus/alert_rules.yml`:

```yaml
  - name: anpr-precision
    rules:
      - alert: ShadowInt8DisagreementHigh
        expr: max by (condition, candidate) (anpr_shadow_plate_disagreement_ratio) > 0.05   # match --max-disagreement
        for: 1h
        labels: {severity: warning}
        annotations:
          summary: "INT8 candidate reads {{ $labels.condition }} plates differently from FP32"
      - alert: ShadowCompareStale        # a gauge nobody refreshes looks exactly like a healthy one
        expr: time() - node_textfile_mtime_seconds{file=~".*shadow_.*\\.prom"} > 86400
        for: 1h
        labels: {severity: warning}
```

**Catch "night traffic grew and INT8 is worse at night".** Each signal alone misses
this failure. The per-condition disagreement ratios can stay flat while the fleet
gets worse, because the *mix* moved toward the condition the INT8 model handles
worst. Two pieces are needed:

1. **Evidently on per-frame input statistics.** In the field, nobody labels a frame
   "night". Log cheap proxies per frame instead: mean luminance, contrast, a blur
   score (variance of the Laplacian), detected plate height, hour of day. Compare
   them with Evidently, using the legacy API that
   [Session 4 Guide C](../../session_4/README.md#guide-c--evidently) pins
   (`evidently<0.7`). The reference should be **the calibration set's frames**:
   that is the distribution the INT8 scales were fit on, so drift away from it is
   drift away from what INT8 was calibrated for.

   ```python
   from evidently.report import Report
   from evidently.metric_preset import DataDriftPreset

   report = Report(metrics=[DataDriftPreset()])
   report.run(reference_data=calibration_frame_stats, current_data=last_24h_frame_stats)
   report.save_html("reports/int8_input_drift.html")
   ```

   Publish the per-condition *share* of recent traffic as a gauge (for example
   `anpr_condition_share{condition=…}`), through the same path Session 4 uses for
   its PSI gauges.

2. **Combine share and damage in PromQL.** Expected disagreement for the traffic you
   actually have:

   ```promql
   sum(anpr_condition_share * on (condition) group_left() max by (condition) (anpr_shadow_plate_disagreement_ratio))
   ```

   The per-condition gauges only move when the model changes. This one also moves
   when the world changes.

**The Grafana panels**, built the way
[Session 4 Guide B](../../session_4/README.md#guide-b--grafana) builds rows:

| Panel | Query | Viz |
|---|---|---|
| Disagreement by condition | `max by (condition) (anpr_shadow_plate_disagreement_ratio)` | Time series, threshold line at the gate value |
| Worst condition now | `topk(1, max by (condition) (anpr_shadow_plate_disagreement_ratio))` | Stat |
| Traffic mix | `anpr_condition_share` | Time series, stacked |
| Expected disagreement for current traffic | the combined query above | Time series, same threshold |
| Shadow data age | `time() - node_textfile_mtime_seconds{file=~".*shadow_.*"}` | Stat, unit seconds |

When the alert fires, work through
[Session 4's RUNBOOK](../../session_4/RUNBOOK.md) as you would for any alert. Step 3,
"what changed", is either a new candidate artifact (read its card) or a new traffic
mix (read the Evidently report).

---

## 4. Measured result

<!-- results:journey -->
#### Hardware `3cc807d0` · profile `quick`

**Measured on:** Apple M3 Pro · 18.0 GB RAM · GPU: none · Darwin 25.5.0 arm64 · Python 3.12.12
**Threads:** ANPR_THREADS=4, OMP_NUM_THREADS=4, ORT intra_op_num_threads=4, inter_op=1
**Latency batch size:** 1 · **SLA:** p95 <= 30.0 ms per frame · **Profile:** `quick` · **Validation set sha256:** `aceb33513379`
**Libraries (as loaded by the rows below):** torch 2.13.0, onnxruntime 1.30.0, openvino 2026.3.1, nncf 3.3.0, ai-edge-litert 2.2.0

| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s00_profile:fast-resize` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.904 | 0.938 | 52.2 | 54.3 | no | 19.2 (b1) | 37.25 | 1417 | — | ok |
| `s00_profile:gpu-decode` | `s01_baseline:eager-fp32` | torch · cuda · fp32 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: nvJPEG decode needs an NVIDIA GPU |
| `s00_profile:nms-in-graph` | `s01_baseline:eager-fp32` | torch · cpu · fp32 · NMS in graph | 0.905 | 0.933 | 54.1 | 56.4 | no | 18.5 (b1) | 37.25 | 1395 | — | ok |
| `s01_baseline:eager-fp32` | (root) | torch · cpu · fp32 | 0.905 | 0.933 | 54.1 | 58.3 | no | 18.5 (b1) | 37.25 | 1316 | — | ok |
| `s02_torch_compile:inductor-fp32` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.905 | 0.933 | 50.1 | 53.0 | no | 20.1 (b1) | 37.25 | 1221 | — | ok |
| `s03_torchscript:jit-trace-fp32` | `s01_baseline:eager-fp32` | torchscript · cpu · fp32 | 0.905 | 0.933 | 54.3 | 61.0 | no | 18.3 (b1) | 37.49 | 1323 | — | ok |
| `s04_onnx_export:ort-cpu-fp32` | `s01_baseline:eager-fp32` | ort · cpu · fp32 | 0.905 | 0.933 | 105.8 | 133.8 | no | 10.2 (b8) | 37.46 | 578 | — | ok |
| `s04_onnx_export:ort-cpu-fp32-nms-in-graph` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · fp32 · NMS in graph | 0.905 | 0.933 | 100.5 | 124.7 | no | 10.4 (b8) | 37.50 | 621 | — | ok |
| `s04_onnx_export:ort-cuda-fp32` | `s01_baseline:eager-fp32` | ort · cuda · fp32 | — | — | — | — | — | — | — | — | — | not_run: CUDAExecutionProvider not available (needs onnxruntime-gpu + NVIDIA GPU) |
| `s04_onnx_export:ort-openvino-ep-cpu` | `s01_baseline:eager-fp32` | ort · cpu · fp32 | — | — | — | — | — | — | — | — | — | not_run: OpenVINOExecutionProvider not available (install onnxruntime-openvino in its own venv; it  |
| `s04_onnx_export:ort-tensorrt-ep-fp16` | `s01_baseline:eager-fp32` | ort · cuda · fp16 | — | — | — | — | — | — | — | — | — | not_run: TensorrtExecutionProvider not available (onnxruntime-gpu + TensorRT 10 libs) |
| `s05_pruning:masked-structured-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.897 | 0.934 | 57.3 | 69.8 | no | 18.3 (b1) | 37.25 | 1227 | — | ok |
| `s05_pruning:masked-structured-50-onnx` | `s05_pruning:masked-structured-50` | ort · cpu · fp32 | 0.897 | 0.934 | 100.7 | 110.2 | no | 10.5 (b8) | 37.46 | 716 | — | ok |
| `s05_pruning:sliced-iterative-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.913 | 0.933 | 30.4 | 32.8 | no | 32.6 (b4) | 13.10 | 1023 | — | ok |
| `s05_pruning:sliced-iterative-50-onnx` | `s05_pruning:sliced-iterative-50` | ort · cpu · fp32 | 0.913 | 0.933 | 37.6 | 41.8 | no | 29.9 (b8) | 13.29 | 494 | — | ok |
| `s05_pruning:sliced-oneshot-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.919 | 0.930 | 29.9 | 43.7 | no | 36.0 (b1) | 13.10 | 980 | — | ok |
| `s05_pruning:sliced-oneshot-50-onnx` | `s05_pruning:sliced-oneshot-50` | ort · cpu · fp32 | 0.919 | 0.930 | 35.8 | 40.3 | no | 31.0 (b4) | 13.29 | 494 | — | ok |
| `s05_pruning:sliced-student-budget` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.859 | 0.902 | 17.1 | 18.8 | yes | 71.8 (b4) | 6.43 | 982 | 4.645 (b1) | ok |
| `s05_pruning:sliced-student-budget-onnx` | `s05_pruning:sliced-student-budget` | ort · cpu · fp32 | 0.859 | 0.902 | 15.8 | 20.1 | yes | 80.2 (b8) | 6.64 | 343 | 4.448 (b1) | ok |
| `s06a_ptq:ort-dynamic-int8` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-dyn | 0.905 | 0.936 | 45.6 | 48.3 | no | 22.5 (b4) | 11.68 | 768 | — | ok |
| `s06a_ptq:ort-static-int8-daytime` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.899 | 0.871 | 34.3 | 44.1 | no | 31.8 (b8) | 11.79 | 308 | — | ok |
| `s06a_ptq:ort-static-int8-detector-only` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-det | 0.892 | 0.853 | 33.1 | 37.3 | no | 32.5 (b4) | 13.33 | 383 | — | ok |
| `s06a_ptq:ort-static-int8-fp32-decode` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.904 | 0.931 | 33.9 | 43.8 | no | 33.0 (b8) | 11.81 | 326 | — | ok |
| `s06a_ptq:ort-static-int8-fp32-decode-daytime` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.905 | 0.930 | 32.7 | 35.8 | no | 32.7 (b4) | 11.81 | 319 | — | ok |
| `s06a_ptq:ort-static-int8-mixed` | `s06a_ptq:ort-static-int8-stratified` | ort · cpu · int8-mixed | 0.906 | 0.934 | 39.2 | 43.0 | no | 27.7 (b8) | 12.14 | 684 | — | ok |
| `s06a_ptq:ort-static-int8-ocr-only` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-ocr | 0.905 | 0.933 | 95.6 | 99.1 | no | 11.1 (b4) | 35.93 | 679 | — | ok |
| `s06a_ptq:ort-static-int8-per-tensor` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.893 | 0.855 | 32.4 | 35.4 | no | 33.9 (b8) | 11.72 | 314 | — | ok |
| `s06a_ptq:ort-static-int8-stratified` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.892 | 0.854 | 32.6 | 35.9 | no | 33.6 (b8) | 11.79 | 366 | — | ok |
| `s06a_ptq:torch-dynamic-int8-ocr` | `s01_baseline:eager-fp32` | torch · cpu · int8-dyn | 0.905 | 0.932 | 55.1 | 57.1 | no | 18.1 (b1) | 35.19 | 1285 | — | ok |
| `s06b_qat:ort-qat-int8-ocr` | `s06a_ptq:ort-static-int8-stratified` | ort · cpu · int8-qat | — | — | — | — | — | — | — | — | — | failed: libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive |
| `s06b_qat:ort-qat-int8-ocr-only` | `s06a_ptq:ort-static-int8-ocr-only` | ort · cpu · int8-qat | 0.905 | 0.930 | 97.2 | 101.6 | no | 11.0 (b8) | 37.37 | 679 | — | ok |
| `s06b_qat:torch-qat-converted-ocr` | `s03_torchscript:jit-trace-fp32` | torchscript · cpu · int8-qat | 0.905 | 0.930 | 55.6 | 68.1 | no | 18.5 (b1) | 35.93 | 1204 | — | ok |
| `s07_distillation:student-distilled` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.898 | 0.947 | 28.8 | 30.7 | no | 53.6 (b8) | 2.46 | 625 | 8.026 (b1) | ok |
| `s07_distillation:student-distilled-onnx` | `s07_distillation:student-distilled` | ort · cpu · fp32 | 0.898 | 0.947 | 16.4 | 19.3 | yes | 69.2 (b4) | 2.68 | 467 | 4.464 (b1) | ok |
| `s07_distillation:student-scratch` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.900 | 0.932 | 28.8 | 31.0 | no | 54.1 (b8) | 2.46 | 766 | 8.038 (b1) | ok |
| `s08_tensorrt:baseline-fp16` | `s04_onnx_export:ort-cpu-fp32` | tensorrt · cuda · fp16 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-fp16` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · fp16 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-fp32` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · fp32 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-int8-daytime` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · int8 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-int8-stratified` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · int8 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s09_openvino:baseline-fp32-latency` | `s04_onnx_export:ort-cpu-fp32` | openvino · cpu · fp32 | 0.905 | 0.933 | 49.1 | 54.6 | no | 22.3 (b4) | 37.27 | 2956 | — | ok |
| `s09_openvino:edge-candidate` | `s09_openvino:student-int8-nncf-fp32-head` | openvino · cpu · int8 · NMS in graph | 0.804 | 0.868 | 8.2 | 9.9 | yes | 131.2 (b8) | 1.06 | 598 | 2.278 (b1) | ok |
| `s09_openvino:intel-gpu` | `s09_openvino:student-fp32-latency` | openvino · gpu · fp32 | — | — | — | — | — | — | — | — | — | not_run: no OpenVINO GPU device (available: ['CPU']); the GPU plugin targets Intel GPUs |
| `s09_openvino:student-device-default-precision` | `s09_openvino:student-fp32-latency` | openvino · cpu · device-default | 0.898 | 0.942 | 8.8 | 10.4 | yes | 119.9 (b4) | 2.53 | 539 | 2.459 (b1) | ok |
| `s09_openvino:student-fp32-latency` | `s07_distillation:student-distilled-onnx` | openvino · cpu · fp32 | 0.898 | 0.947 | 10.1 | 12.1 | yes | 102.8 (b4) | 2.53 | 772 | — | ok |
| `s09_openvino:student-fp32-throughput` | `s09_openvino:student-fp32-latency` | openvino · cpu · fp32 | 0.898 | 0.947 | 21.2 | 23.4 | yes | 77.2 (b8) | 2.53 | 558 | 5.876 (b1) | ok |
| `s09_openvino:student-int8-nncf` | `s09_openvino:student-fp32-latency` | openvino · cpu · int8 | 0.893 | 0.732 | 10.6 | 12.3 | yes | 101.4 (b4) | 1.01 | 603 | 3.016 (b1) | ok |
| `s09_openvino:student-int8-nncf-fp32-decode` | `s09_openvino:student-int8-nncf` | openvino · cpu · int8 | 0.895 | 0.739 | 10.6 | 12.2 | yes | 101.3 (b4) | 1.02 | 605 | 2.959 (b1) | ok |
| `s09_openvino:student-int8-nncf-fp32-head` | `s09_openvino:student-int8-nncf-fp32-decode` | openvino · cpu · int8 | 0.895 | 0.934 | 10.7 | 12.3 | yes | 100.4 (b4) | 1.02 | 605 | 2.925 (b1) | ok |
| `s10_tflite_edge:raspberry-pi-5` | `s10_tflite_edge:tflite-int8-full` | tflite · cpu · int8 | — | — | — | — | — | — | — | — | — | not_run: not measured on a Raspberry Pi: on the device, run `make setup-edge data && make s10` — th |
| `s10_tflite_edge:tflite-fp16` | `s07_distillation:student-distilled-onnx` | tflite · cpu · fp16 | — | — | — | — | — | — | — | — | — | failed: LiteRT cannot load the converted file: {'detector': 'tflite/kernels/conv.cc:360 input_type |
| `s10_tflite_edge:tflite-fp32` | `s07_distillation:student-distilled-onnx` | tflite · cpu · fp32 | 0.898 | 0.947 | 14.5 | 16.2 | yes | 68.7 (b4) | 2.42 | 228 | 4.073 (b1) | ok |
| `s10_tflite_edge:tflite-int8-full` | `s07_distillation:student-distilled-onnx` | tflite · cpu · int8 | — | — | — | — | — | — | — | — | — | failed: onnx2tf did not produce the file: {'detector': 'StrictFullIntegerQuantizationError: Unsupp |
| `s10_tflite_edge:tflite-int8-full-nms` | `s07_distillation:student-distilled-onnx` | tflite · cpu · int8 · NMS in graph | — | — | — | — | — | — | — | — | — | failed: onnx2tf did not produce the file: {'detector': 'StrictFullIntegerQuantizationError: Unsupp |
| `s10_tflite_edge:tflite-int8-ocr-fp32-detector` | `s07_distillation:student-distilled-onnx` | tflite · cpu · int8-ocr | 0.898 | 0.947 | 14.3 | 16.0 | yes | 69.2 (b4) | 1.94 | 230 | 4.105 (b1) | ok |

<details><summary>Lineage: which artifact each row was built from</summary>

```mermaid
flowchart LR
  n0["s00_profile:fast-resize"]
  n1["s00_profile:gpu-decode"]
  n2["s00_profile:nms-in-graph"]
  n3["s01_baseline:eager-fp32"]
  n4["s02_torch_compile:inductor-fp32"]
  n5["s03_torchscript:jit-trace-fp32"]
  n6["s04_onnx_export:ort-cpu-fp32"]
  n7["s04_onnx_export:ort-cpu-fp32-nms-in-graph"]
  n8["s04_onnx_export:ort-cuda-fp32"]
  n9["s04_onnx_export:ort-openvino-ep-cpu"]
  n10["s04_onnx_export:ort-tensorrt-ep-fp16"]
  n11["s05_pruning:masked-structured-50"]
  n12["s05_pruning:masked-structured-50-onnx"]
  n13["s05_pruning:sliced-iterative-50"]
  n14["s05_pruning:sliced-iterative-50-onnx"]
  n15["s05_pruning:sliced-oneshot-50"]
  n16["s05_pruning:sliced-oneshot-50-onnx"]
  n17["s05_pruning:sliced-student-budget"]
  n18["s05_pruning:sliced-student-budget-onnx"]
  n19["s06a_ptq:ort-dynamic-int8"]
  n20["s06a_ptq:ort-static-int8-daytime"]
  n21["s06a_ptq:ort-static-int8-detector-only"]
  n22["s06a_ptq:ort-static-int8-fp32-decode"]
  n23["s06a_ptq:ort-static-int8-fp32-decode-daytime"]
  n24["s06a_ptq:ort-static-int8-mixed"]
  n25["s06a_ptq:ort-static-int8-ocr-only"]
  n26["s06a_ptq:ort-static-int8-per-tensor"]
  n27["s06a_ptq:ort-static-int8-stratified"]
  n28["s06a_ptq:torch-dynamic-int8-ocr"]
  n29["s06b_qat:ort-qat-int8-ocr"]
  n30["s06b_qat:ort-qat-int8-ocr-only"]
  n31["s06b_qat:torch-qat-converted-ocr"]
  n32["s07_distillation:student-distilled"]
  n33["s07_distillation:student-distilled-onnx"]
  n34["s07_distillation:student-scratch"]
  n35["s08_tensorrt:baseline-fp16"]
  n36["s08_tensorrt:student-fp16"]
  n37["s08_tensorrt:student-fp32"]
  n38["s08_tensorrt:student-int8-daytime"]
  n39["s08_tensorrt:student-int8-stratified"]
  n40["s09_openvino:baseline-fp32-latency"]
  n41["s09_openvino:edge-candidate"]
  n42["s09_openvino:intel-gpu"]
  n43["s09_openvino:student-device-default-precision"]
  n44["s09_openvino:student-fp32-latency"]
  n45["s09_openvino:student-fp32-throughput"]
  n46["s09_openvino:student-int8-nncf"]
  n47["s09_openvino:student-int8-nncf-fp32-decode"]
  n48["s09_openvino:student-int8-nncf-fp32-head"]
  n49["s10_tflite_edge:raspberry-pi-5"]
  n50["s10_tflite_edge:tflite-fp16"]
  n51["s10_tflite_edge:tflite-fp32"]
  n52["s10_tflite_edge:tflite-int8-full"]
  n53["s10_tflite_edge:tflite-int8-full-nms"]
  n54["s10_tflite_edge:tflite-int8-ocr-fp32-detector"]
  n3 --> n0
  n3 --> n1
  n3 --> n2
  n3 --> n4
  n3 --> n5
  n3 --> n6
  n6 --> n7
  n3 --> n8
  n3 --> n9
  n3 --> n10
  n3 --> n11
  n11 --> n12
  n3 --> n13
  n13 --> n14
  n3 --> n15
  n15 --> n16
  n3 --> n17
  n17 --> n18
  n6 --> n19
  n6 --> n20
  n6 --> n21
  n6 --> n22
  n6 --> n23
  n27 --> n24
  n6 --> n25
  n6 --> n26
  n6 --> n27
  n3 --> n28
  n27 --> n29
  n25 --> n30
  n5 --> n31
  n3 --> n32
  n32 --> n33
  n3 --> n34
  n6 --> n35
  n33 --> n36
  n33 --> n37
  n33 --> n38
  n33 --> n39
  n6 --> n40
  n48 --> n41
  n44 --> n42
  n44 --> n43
  n33 --> n44
  n44 --> n45
  n44 --> n46
  n46 --> n47
  n47 --> n48
  n52 --> n49
  n33 --> n50
  n33 --> n51
  n33 --> n52
  n33 --> n53
  n33 --> n54
```

</details>
<!-- /results -->

How to read this for the purposes of this guide:

- **Read the hardware header before any row.** Each table is one `(hardware_id,
  profile)` pair, and its header names the CPU, GPU, driver, thread settings and
  library versions. A p95 from one table cannot be compared with a p95 from another.
  The perf gate enforces this by looking rows up by `hardware_id`.
- **The CI gate's two rows are in here.** `s04_onnx_export:ort-cpu-fp32` is the
  candidate and `s01_baseline:eager-fp32` is the reference. Their p95 ratio and
  their accuracy difference are exactly what `make gate` checks. On the GPU table,
  `s08_tensorrt:student-fp16` is the row `gpu-sla-gate` holds to the absolute SLA.
- **Read the `≤ SLA` column only on target hardware.** On a laptop CPU it tells you
  about the laptop.
- **Open the lineage details.** The stages are a tree. The shadow pair you promote
  must share a parent there, or the disagreement ratio mixes two changes.
- **`not_run` rows carry their reason.** A TensorRT row that is `not_run` on the
  M3 Pro says "no CUDA + TensorRT 10 here". It says nothing about TensorRT. The
  same row on the RTX 3090 table is the one that counts.

---

## 5. Gotchas

1. **Thresholds copied into the workflow file.**
   **Symptom:** the SLA was tightened in `src/config.py`, the journey table's `≤ SLA`
   column changed, and CI still passes PRs against the old number.
   Every threshold is read from `config.SLA_TARGET`. The only number the workflow
   passes is the row ids. Keep it that way.

2. **An absolute SLA check on a GitHub-hosted runner.**
   **Symptom:** red builds on PRs that touched documentation, and green builds on
   PRs that doubled the detector's work, depending on what else the shared runner
   was doing.
   `cpu-gate` runs without `--enforce-sla` for this reason. The absolute check
   belongs to `gpu-sla-gate` on the self-hosted RTX 3090.

3. **Changing `ANPR_THREADS` and expecting the gate to find the rows.**
   **Symptom:** `gate: no measured row 's01_baseline:eager-fp32' for this hardware
   (…) and profile — run its stage first`, on a machine where you just ran it.
   `hardware_id` hashes the thread count together with the CPU and GPU. A row
   measured at another thread count belongs to a different machine as far as the
   gate is concerned. Re-run the stages with the same environment the gate runs with.

4. **Re-measuring with `record=True` in a home-made gate.**
   **Symptom:** after a regression demo, `make table` shows the slow numbers as the
   stage's real row, and every later gate compares against the regression.
   `ci/perf_gate.py` re-measures with `run(spec, record=False)`. Any variant you
   write must do the same.

5. **The workflow's path filter does not cover every file that changes performance.**
   **Symptom:** a PR that edits `requirements-gpu.txt`, the `Makefile` or `tools/`
   merges without the gate running at all.
   The trigger lists `session_5/src/**`, `session_5/ci/**`,
   `session_5/requirements.txt` and the workflow file. Extend it if your changes to
   other files can move latency.

6. **An engine built outside the pinned container.**
   **Symptom:** Triton's log reports the TensorRT model failed to load, with an
   engine deserialization or version error, while the same engine ran fine in the
   notebook that built it.
   The notebook's TensorRT is not the server's. Build inside the container whose
   TensorRT matches the serving Triton, on the serving GPU.

7. **The CI profile's accuracy gate read as a model-quality gate.**
   **Symptom:** accuracy passes in `cpu-gate`, and the `quick` or `full` profile
   later shows an INT8 row dropping past the allowed limit.
   `ANPR_PROFILE=ci` trains on tiny data for a few epochs. The workflow comment says
   so: that gate checks the machinery, not model quality. Quality is judged on the
   profile whose card you ship.

8. **A shadow pair that does not share a parent.**
   **Symptom:** high disagreement on every condition, including clear daytime, for
   an INT8 model whose own journey row shows a small accuracy drop.
   You are comparing two architectures, not two precisions. Pass `--fp32` and
   `--candidate` explicitly from the same lineage (section 3d).

9. **Averaging disagreement across conditions.**
   **Symptom:** a flat global disagreement line on the dashboard while night
   disagreement climbs.
   The global number is weighted by volume, so a condition with a small share cannot
   move it: Session 4's incident 05 with lighting in place of city. Keep the
   `condition` label all the way to the panel and the alert.

10. **A card logged without its hardware tag.**
    **Symptom:** a search for the version's INT8 artifact returns three runs, and
    nobody can say which box they were measured on.
    Tag `hardware_id`, `backend`, `precision` on every nested run, and keep the card
    next to the artifact in the same run.

---

## 6. AV comparison callout

> **Context.** Automotive teams build the same three checkpoints at larger scale.
> Every perception model is evaluated per scenario slice (night, rain, glare,
> tunnels) rather than on one aggregate score, for the reason section 3e keys on
> `condition`. The compiler toolchain for each in-vehicle SoC is pinned per vehicle
> program, because an engine is compiled for that chip and that toolchain version.
> New models commonly run in shadow on the fleet, with their outputs logged and
> compared, before they control anything. The ANPR fleet is a smaller version of
> the same shape: many identical edge boxes, no uplink during operation, and a
> release process that has to prove an artifact on the hardware it ships to.

---

## 7. When NOT to use this

- **An absolute latency gate on shared CI.** Use the relative check there, or
  nothing. A gate that fails at random trains people to re-run until green, and that
  is worse than no gate.
- **Benchmark cards for artifacts nobody deploys.** Card the formats that reach a
  target. The rest of the journey table is a lab notebook. It belongs in
  `results.json`, not in the registry.
- **Shadowing on the constrained box itself.** If the target has no headroom for a
  second model, shadow on replayed frames elsewhere (section 3d). A shadow that
  pushes production past its budget has become the incident.
- **Per-condition drift monitoring when you have no condition proxy.** If nothing in
  the frame statistics separates the slices INT8 handles differently, the combined
  query is noise with a label on it. First find out *where* INT8 is weak
  (`make int8-debug`), then decide what to log.
- **A pinned GPU container for a CPU-only fleet.** The Dockerfile pins the
  TensorRT/CUDA/driver matrix. A fleet running OpenVINO or LiteRT still needs pinned
  versions, but in its own image, built for its own target.

---

Previous: [14 — LLM serving: the same four levers](14-llm-serving-same-levers.md) · Next: [16 — Decision guide](16-decision-guide.md)
