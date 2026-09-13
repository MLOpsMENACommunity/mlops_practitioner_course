# ci/ — the performance gate, the shadow check, and the card template

A unit test answers "does it run?". The files here answer "is it still fast enough,
still accurate enough, and still the artifact we measured?", and make the build
fail when the answer is no. The full explanation, and how each piece plugs into
Sessions 2–4, is in
[guide 15 — Optimization in your MLOps stack](../guides/15-optimization-in-your-mlops-stack.md).

## What is in here

| File | What it does |
|---|---|
| `perf_gate.py` | fails with exit code 1 on an accuracy regression or a p95 regression between two rows of `results/results.json` (`snippet:perf-gate`) |
| `shadow_compare.py` | runs an INT8 candidate next to the FP32 pipeline on the same frames, reports plate disagreement **per condition** in Prometheus text format, and exits non-zero above `--max-disagreement` (`snippet:shadow-compare`) |
| `benchmark_card.md.tmpl` | the template `tools/benchmark_card.py` fills, once per measured row, into `results/cards/<hardware_id>_<profile>/<stage>__<variant>.md` |
| `docker/Dockerfile.gpu` | the pinned GPU build and benchmark image: NGC 26.05, with the same TensorRT as the 26.05 Triton container, so an engine built inside it loads in s11's Triton |

Every threshold comes from the `SLA` dataclass in `src/config.py`. Nothing in this
folder keeps its own copy.

## Where the workflow lives, and why

The workflow is `.github/workflows/session5-perf-gate.yml` at the **repository
root** ([view](../../.github/workflows/session5-perf-gate.yml)), not in this folder.
GitHub only runs workflows from the root `.github/workflows/` directory. The YAML
only wires things together: install, measure, call `python -m ci.perf_gate`. All
the logic stays here, so `make gate` locally and the Actions job run the same code.

It triggers on pull requests touching `session_5/src/**`, `session_5/ci/**`,
`session_5/tools/**`, `session_5/Makefile`, `session_5/requirements*.txt` or the workflow
file, and on `workflow_dispatch` with an `inject_latency_ms` input.

## The two gate modes

| Mode | Checks | Where | Why there |
|---|---|---|---|
| **regression** (default) | candidate accuracy drop vs `--baseline` ≤ `SLA.max_map50_drop` / `SLA.max_ocr_em_drop`; with `--remeasure`, the candidate's new p95 ≤ its own recorded p95 × `--max-p95-ratio` | job `cpu-gate` on GitHub-hosted `ubuntu-latest`, `ANPR_PROFILE=ci` | a shared runner has no GPU and noisy neighbours. Its absolute latency means nothing, and comparing two *different* runtimes is a hardware question — but the same row re-measured in the same job against its own record catches a code change that made it slower |
| **absolute** (`--enforce-sla`) | everything above, plus candidate p95 ≤ `SLA.p95_ms` (30 ms) | job `gpu-sla-gate` on a self-hosted runner labelled `[self-hosted, linux, gpu]`: the RTX 3090, inside `docker/Dockerfile.gpu` | the SLA describes the target hardware and only means something there |

`gpu-sla-gate` runs only when the repository variable `SESSION5_GPU_RUNNER` is set
to `true`. Without a registered GPU runner, the job is skipped rather than left
queued forever.

The gate looks rows up by `(row id, hardware_id, profile)`. `hardware_id` hashes the
CPU, the GPU and `ANPR_THREADS`, so both rows must have been measured on this machine
at this thread count.

## Run it locally

From `session_5/`, after the rows exist (`make s01 s04`):

```bash
make gate         # s04_onnx_export:ort-cpu-fp32: accuracy vs the baseline + re-measured p95 vs its own record
make gate-demo    # the same with ANPR_INJECT_LATENCY_MS set: watch it fail

# any row; --baseline defaults to s01_baseline:eager-fp32
python -m ci.perf_gate --candidate <stage:variant> [--baseline <stage:variant>] [--remeasure] [--enforce-sla]
```

`make gate-demo` sets `ANPR_INJECT_LATENCY_MS`, which makes `src/benchmark.py` sleep
inside every detector call, and re-measures with `run(spec, record=False)`. The
regression is printed and gated, and the real row in `results/results.json` is
never overwritten. Its reference is the eager baseline, so whether the injected
delay trips the relative check depends on the gap between those two rows on your
machine. Read both p95 values the gate prints. The Actions input `inject_latency_ms`
does the same, but compares the re-measured ONNX row against its own clean
measurement, so the injected delay is the only difference.

Shadow check, with both pairs from the same lineage (see guide 15, section 3d):

```bash
python -m ci.shadow_compare --fp32 detector_baseline.onnx,ocr_baseline.onnx \
    --candidate detector_int8_stratified.onnx,ocr_int8_stratified.onnx --candidate-backend ort
```

Pinned GPU image:

```bash
docker build -f ci/docker/Dockerfile.gpu -t anpr-gpu:26.05 .
docker run --rm --gpus all -v "$PWD":/workspace/session_5 anpr-gpu:26.05 make all
```

## Demo: the gate failing

<!-- ci-demo-output -->
