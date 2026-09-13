# 02 — Benchmarking properly

Every decision in this session compares measurements: a row against its parent, a p95 against 30 ms. This guide walks through `src/benchmark.py`, the only file in the repo that times anything. It covers why each rule exists, what goes wrong without it, and the parity gate that proves a converted model is correct before it gets timed.

## 1. The problem it solves

Measurement errors in this pipeline aren't random noise. Each one pushes you toward shipping the wrong thing to the roadside box:

- A TensorRT call timed without a sync measures submission, so a GPU engine looks like it fits 30 ms when it doesn't.
- ONNX Runtime at its default thread count uses every core of the M3 Pro, while the box has four. A row that fits on the laptop misses on the pole.
- Accuracy read from the timing loop comes from a few dozen frames, not the validation set, so an INT8 regression at night never shows up.
- If two stages are scored on different validation frames, the journey table compares two different exams.

The harness makes sure every row in `results/results.json` was measured the same way, and records how.

## 2. Mental model

One call, `run(RunSpec(...))`, produces one row:

```
RunSpec -> fresh worker process -> accuracy pass -> latency pass -> throughput or concurrent pass -> row
```

| Rule | Where | Why |
|---|---|---|
| Fresh subprocess per measurement | `run` spawns `python -m src.benchmark '<spec>'` | peak RSS belongs to this backend, not to whatever the stage trained or imported; `OMP_NUM_THREADS` is read when the library initializes, so it has to be set *before* import |
| Accuracy and timing are separate passes | `_worker`: `accuracy_pass`, then `latency_pass` | accuracy covers every validation frame with no clock running; the timing loop discards predictions |
| At least 20 warm-up calls | `config.WARMUP_RUNS` | first calls pay for lazy allocation, kernel selection, `torch.compile` compilation and cold caches |
| Sync before stopping the clock | `timed_call :: lap` calls `backend.sync()` | CUDA and TensorRT calls return as soon as work is queued |
| `time.perf_counter`, never `time.time` | `timed_call` | monotonic and high-resolution; wall-clock time can jump and is coarse |
| Same hash-verified validation split | `load_split("val", verify=True)`; `val_sha256` in every row | every stage takes the same exam |
| Same batch size | `SLA.batch_size` for every latency pass | batch size changes both latency and throughput |
| Fixed, recorded thread counts | `ANPR_THREADS`, `OMP_NUM_THREADS`, ORT `intra_op_num_threads` | thread counts alone can swing CPU results several-fold |
| Full environment capture | `src/environment.py :: capture` | CPU, GPU, driver, OS, Python, library versions, every thread setting |

**Sync is the rule that hides best.** From `src/backends.py :: _TrtEngine.__call__`:

```python
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        # execute_async_v3 returns as soon as the work is QUEUED. Stop a timer here and
        # you have measured submission, not inference.
        stream.synchronize()
```

**Percentiles, not means, and percentiles don't average.** The SLA is a p95 because a camera that misses one frame in twenty misses real plates, whatever the average says. Two traps follow. The mean of several runs' p95 values is not a p95: pool the calls and take the percentile once. And per-phase p95 values don't add up to the total p95, because different calls are slow in different phases. That's why guide 00's tail breakdown averages phases over the slowest calls. Throughput is a rate, so `throughput_pass` does use the mean: `fps = batch / mean latency`.

**Throughput versus concurrent clients.** `throughput_pass` sends client-side batches of growing size. That's right for an in-process runtime and wrong for a server. Triton's dynamic batcher merges *separate* requests that arrive close together, and a single request carrying eight frames leaves it nothing to merge. `concurrent_pass` instead runs N single-frame clients on threads, and the curve's `batch` field then counts clients. s11 rows set `concurrent=True`.

**Peak memory.** `PeakMemory` (`snippet:peak-memory`) samples process RSS on a background thread, plus per-process GPU memory through NVML on CUDA. It doesn't use `torch.cuda.max_memory_allocated`, which sees only PyTorch's caching allocator and misses TensorRT's and ONNX Runtime's device memory. A `vram` of `null` means unmeasured, not zero.

**Rows and lineage.** `src/results.py :: record` keys each row by `(stage:variant, hardware_id, profile)`. `hardware_id` hashes CPU, GPU and `ANPR_THREADS`, so laptop and RTX 3090 rows share one file but never a table. Every row names its `parent`, because the stages form a tree, not a chain, and the journey table draws that tree.

**The parity gate.** Before a converted artifact is timed, `src/parity.py :: check` (`snippet:parity-gate`) compares it with its parent at three levels, because each level alone has a blind spot:

1. **tensors:** `np.testing.assert_allclose` with `rtol=1e-3, atol=1e-5` on raw detector outputs and OCR logits;
2. **boxes:** equal counts after NMS, and every matched box above a minimum IoU;
3. **strings:** decoded plate text equal, character for character.

Logits can match within tolerance while NMS keeps a different box, or a near-tie in the CTC argmax flips a character. Both recognizers get the *same* crops, cut from the reference detector's boxes, so an OCR mismatch can't be a detector difference in disguise. The gate is **strict** for conversions that preserve precision: `torch.compile`, FP32 ONNX Runtime providers, and FP32 TensorRT and OpenVINO. A strict failure records a `failed` row and stops the stage (`src/stages/common.py :: gate`). FP16, INT8, pruned and distilled rows are *expected* to move tensors, so their gate only reports, and the accuracy pass judges them.

## 3. Runnable walkthrough

```bash
make s01             # the baseline row: eager PyTorch FP32, NMS in Python, bilinear resize
make gate            # ci/perf_gate.py, candidate s04 ONNX row against this baseline
make gate-demo       # the same gate failing on an injected detector delay
```

`src/stages/s01_baseline.py` builds one `RunSpec` and calls `run`. On a CUDA machine it also measures `eager-fp32-cpu`, the same pipeline on that machine's CPU, as the no-accelerator reference.

Read `src/benchmark.py` in this order:

1. `run`: the subprocess and its environment, and how a crash becomes a `failed` row.
2. `_worker`: frames loaded into memory (disk I/O isn't part of the pipeline), then the passes inside `PeakMemory`.
3. The four `*_pass` functions, then `timed_call` (`snippet:timing-loop`).

Then read `src/environment.py :: capture`, `src/datasets/splits.py :: load_split` and `src/parity.py :: check`. `ci/perf_gate.py :: remeasure` reuses the harness with `record=False`, so CI never overwrites a stage's row.

## 4. Measured result

<!-- results:stage:s01_baseline -->
<!-- /results -->

A CPU machine shows one row; a CUDA machine shows two. The header lines are part of the result, because without them nobody can reproduce the row.

The row in `results/results.json` holds more than the table: p99 and mean, per-phase statistics, `p95_tail_phases_ms`, `process_first_call_ms` and `latency.cold_first_call_ms`, the throughput curve, per-condition accuracy, VRAM, library versions and `git_sha`.

How to read the outcome:

- **p95 close to p50:** latency is steady, and the tail is the pipeline's own work.
- **p95 far above p50:** rule out the machine first: thermal throttling, another process, or a thread count that oversubscribes the cores. Re-run on a quiet machine before calling it a property of the pipeline. If it persists, guide 00's tail breakdown shows which phase owns it.
- **`fps (batch)` peaks at batch 1:** larger batches didn't raise throughput on this device, so batching isn't a capacity lever here.
- **`fps (batch)` peaks at the largest batch measured:** the curve hadn't saturated. The `full` profile measures larger batches.

## 5. Gotchas

1. **Timing a queued GPU call.** **Symptom:** on the 3090, `detect` for a TensorRT row is implausibly small, and the time shows up in the next phase that copies to the host. **Fix:** synchronize before every clock read. A backend's `sync()` is part of its contract.
2. **Leaving thread counts to the library.** **Symptom:** a colleague's re-run of the same row differs several-fold, and their `environment.threads` shows `OMP_NUM_THREADS: unset`. **Fix:** set `ANPR_THREADS`. In your own scripts, set `OMP_NUM_THREADS` before importing torch or ONNX Runtime; setting `os.environ` after the import has no effect.
3. **Changing threads and "losing" the row.** **Symptom:** after `ANPR_THREADS=8 make s04`, the journey table gains a second section for the same laptop. **Fix:** intended. Thread count is part of `hardware_id`, so compare within one section.
4. **A p95 nobody can reproduce.** **Symptom:** a slide quotes a p95 that isn't in `results.json`, because it was the mean of several runs' p95 values. **Fix:** percentiles come from pooled calls in one pass. Quote the row.
5. **Reading VRAM from the PyTorch allocator.** **Symptom:** a TensorRT row reports a tiny VRAM figure while `nvidia-smi` shows the process holding far more. **Fix:** use per-process NVML, as `PeakMemory` does.
6. **Reading the wrong first-call column.** **Symptom:** the `torch.compile` row's `latency.cold_first_call_ms` is barely above its p50, even though the stage visibly stalls while it compiles. **Fix:** that field is the first call *of the timing loop*, which runs after the accuracy pass has already sent every validation frame through the pipeline. The process's very first call — compilation, lazy initialisation, a TensorRT EP engine build — is `process_first_call_ms`, timed in `src/benchmark.py :: _worker` before anything else touches the pipeline. Backend construction (loading sessions, deserializing engines) happens before either clock; time process start to first plate if you need that too.
7. **Validation data changing under the table.** **Symptom:** `DatasetChanged: val split differs from manifest.json — regenerate with: make data`. **Fix:** run `make data`, then re-run every stage. Rows with a different `val_sha256` took a different exam.
8. **Sending client batches to a server.** **Symptom:** Triton's metrics endpoint reports an average batch size of one while the benchmark was "batching". **Fix:** measure servers with concurrent single-frame clients (`concurrent=True`).

## 6. AV comparison callout

> **Context only — not part of the lab.** Benchmarking a YOLO-class vehicle and pedestrian detector or a lane segmenter for a vehicle follows the same rules with a different emphasis. The deadline is per sensor frame, so the tail matters more than the median, as it does here. It has to run with every other perception model loaded, because they compete for one accelerator. An isolated benchmark is optimistic by design. Input variance drives each model's tail differently. A detector's post-processing grows with scene density, like our plates per frame. A segmentation head does fixed work per frame, so its tail comes mostly from contention. Cold start matters less for a vehicle that boots once per drive than for a roadside box that reboots after every power cut.

## 7. When NOT to use this

- **For attribution.** The harness says how long each phase takes, not which operator or line is responsible. Use the profilers in guide 00.
- **For a one-off kernel or op comparison.** Runtime tools such as `trtexec` or OpenVINO's `benchmark_app` are quicker for that. Bring the winner back through the harness before deciding.
- **As an absolute SLA verdict on shared CI runners.** Their CPUs aren't the box. CI compares relative p95, and the absolute check runs on target hardware.
- **With the strict parity gate on precision-changing conversions.** FP16 and INT8 are supposed to move tensors, so a strict gate would reject every one of them.

Previous: [01 — Why optimize](01-why-optimize.md) · Next: [03 — Formats, runtimes, backends](03-formats-runtimes-backends.md)
