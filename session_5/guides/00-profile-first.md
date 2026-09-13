# 00 — Profile first

Before any stage touches the detector or the recognizer, this guide answers one question with measurements: where do a frame's milliseconds go? Only two of the pipeline's six phases are neural networks. JPEG decode, letterbox, NMS, cropping and CTC decoding share the box's CPU cores with the models, and they alone can decide whether the 30 ms budget is reachable. The rule for the whole session: **you do not touch the model until you have attributed the latency.**

## 1. The problem it solves

The roadside camera hands over a 1280x720 JPEG. Before a plate string exists, the box decodes it, letterboxes it to 384x640, runs the detector and NMS, cuts each plate from the full-resolution frame, resizes it to 32x128 grayscale, runs the recognizer and collapses the CTC output. Once a server is involved, serialization joins the list. `src/config.py :: SLA` requires p95 of that whole chain, at batch 1 on the box, to stay under 30 ms. There's no uplink to hand work off to.

The instinct is to quantize, prune or distil. Those stages cost days of calibration, fine-tuning and accuracy risk. If the frames that miss the budget are slow because of a Python NMS loop or a Pillow resize, none of that work moves the p95. A profile costs one `make` target, so it goes first.

## 2. Mental model

**Measurement and attribution are different jobs, done by different tools.**

| Tool | Question it answers | Quote its numbers? |
|---|---|---|
| The harness, `src/benchmark.py` | how long each phase takes, p50/p95, on the same code path that produces accuracy | yes, and it is the only source |
| `torch.profiler` | which operators, at which shapes, called from which Python line | no: overhead on every op |
| py-spy | which Python functions hold the CPU, including numpy and Pillow code | no |
| Nsight Systems (RTX 3090) | where the CPU and GPU wait for each other | no |

`src/pipeline.py` has no timing code. `src/benchmark.py :: timed_call` (`snippet:timing-loop`) calls the phases one at a time and owns the clock:

```python
    def lap(phase: str) -> None:
        nonlocal last
        pipe.backend.sync()  # GPU calls return when work is queued; stop the clock only when it is done
        now = time.perf_counter()
        marks[phase], last = (now - last) * 1000, now
```

**The tail, not the mean.** The SLA is a p95, so the frames that matter are the slowest 5%. `latency_pass` averages each phase over only the calls at or above the p95:

```python
    tail = [c for c in calls if c["total"] >= total["p95"]] or calls
    tail_phases = {p: round(float(np.mean([c[p] for c in tail])), 3) for p in pipe.PHASES}
```

That is `p95_tail_phases_ms`. The slowest frames can be slow for different reasons than the typical frame. Decode, letterbox and the detector do the same work on every frame. NMS scales with the number of candidates above the score threshold, and crop and OCR scale with plates per frame. The mean breakdown describes the frame you don't need to fix. The tail breakdown describes the frames that break the contract. The tail's phase means also add up to its mean total, which per-phase p95 values don't.

> **Why attribution goes wrong.** Profiling `model(x)` on a random tensor leaves out decode, resize, NMS and crop, so it always blames the model. Profiler numbers carry the overhead of `record_shapes`, `with_stack` and `profile_memory`; the header of `results/profile/torch_ops_*.txt` says so. And without a sync, a GPU kernel's time gets charged to the next phase that blocks on it, usually the `.cpu()` copy.

**The gate.** `src/stages/s00_profile.py :: model_share` (`snippet:profile-gate`) computes detect + ocr as a fraction of the tail. At 50% or above, the models dominate, and model-level stages come next. Below that, fixes start in preprocessing, NMS and I/O. The threshold is a heuristic, but the reasoning behind it is Amdahl's law: if the non-model phases of the tail already exceed 30 ms on their own, no model optimization can meet the SLA.

## 3. Runnable walkthrough

```bash
make profile         # data -> train -> s00, in one command
make s00             # s00 only, once checkpoints exist
make pyspy           # py-spy flame graph of the same workload (sudo on macOS)
```

`src/stages/s00_profile.py :: main`, in order:

1. `baseline_row()` finds `s01_baseline:eager-fp32` for this machine, or measures it. s00 has no baseline row of its own; it *is* the s01 row.
2. `draw_breakdown()` writes `results/profile/phases_<hardware_id>_<profile>.png`, with bars for all frames and for the slowest 5%, and the 30 ms budget drawn as a dashed line.
3. `torch_op_profile()` (`snippet:torch-profiler`) runs five warm-up frames, then wraps every phase in `record_function` using the harness's phase names. It writes `results/profile/torch_ops_<hw>_<profile>.txt` (by op and input shape, then by Python stack) and `artifacts/<profile>/s00_trace.json` for `chrome://tracing` or Perfetto.
4. It writes the gate report to `results/profile/gate_<hw>_<profile>.md`.
5. `fixes()` measures three branch rows off the baseline:

| Row | What changes | Code |
|---|---|---|
| `fast-resize` | `Image.reduce(2)`, an exact-factor box average, instead of Pillow's bilinear resample | `snippet:letterbox` |
| `nms-in-graph` | top-K + Fast NMS as tensor ops inside the detector; fixed `[B, TOPK, 5]` output | `snippet:nms-in-graph` vs `snippet:greedy-nms` |
| `gpu-decode` | batched nvJPEG decode; letterbox as `interpolate` + `pad` on the GPU | `src/pipeline_gpu.py :: GpuPipeline` |

`gpu-decode` needs CUDA, so on the laptop it records `not_run`. Fast NMS differs from the greedy loop in one respect. Greedy NMS lets only a *kept* box suppress others, while Fast NMS lets any higher-scoring candidate suppress, even one that was itself suppressed. So it can drop a box greedy NMS would keep, for example a plate on a second vehicle queued close behind. The accuracy columns show whether that matters here.

**py-spy.** `make pyspy` samples `python -m src.stages.s00_profile --workload 300`, a bare pipeline loop with no harness and no profiler, and writes `results/profile/pyspy_<profile>.svg`. It breaks down the Python stacks that `torch.profiler` reports only as range totals: Pillow's resize, the `while` loop in `greedy_nms`, `ctc_decode`. `py-spy record --gil` keeps only samples from threads holding the GIL, which confirms whether Python itself is the bottleneck.

**Nsight Systems, on the RTX 3090:**

```bash
nsys profile -t cuda,nvtx,osrt python -m src.stages.s00_profile --workload 300
```

- **CPU-GPU gaps.** An empty GPU row while the Python thread is busy means frames are waiting on decode, letterbox, NMS or crop. A faster GPU changes nothing there.
- **Sync points.** Every `.cpu().numpy()` in `src/backends.py :: TorchBackend` blocks until queued kernels finish. They show up as device-to-host copies in the CUDA API row.
- **Kernel launch overhead.** A busy CUDA API row above a sparse GPU row means eager dispatch costs more than compute. That's the problem `torch.compile` and TensorRT address.

NVTX ranges are Nsight's equivalent of `record_function`. The workload loop has none. Wrap phases in `torch.cuda.nvtx.range("detect")` in a scratch copy, or wrap the loop in `torch.autograd.profiler.emit_nvtx()` to annotate every operator.

## 4. Measured result

<!-- results:gate -->
<!-- /results -->

Each phase row gives mean milliseconds and share, first over all frames, then over the slowest 5%. Read the second pair. The last line gives both model shares and the verdict. If the two shares fall on opposite sides of 50%, go with the tail.

<!-- results:stage:s00_profile -->
<!-- /results -->

Read each branch against the baseline row printed above it:

- **`fast-resize`:** compare p95, then both accuracy columns. The detector was trained on bilinear letterboxes (`src/datasets/torch_data.py` calls `letterbox` without `fast`), so this row is a train/serve mismatch. If accuracy stays within `SLA.max_map50_drop` and `SLA.max_ocr_em_drop`, the fix is free. If not, reject it or retrain with the same resize.
- **`nms-in-graph`:** compare the **total**. The `nms` phase shrinks to a threshold filter and the work moves into `detect`, so a smaller `nms` with the same p95 means NMS was never the cost. A change in mAP comes from Fast NMS suppressing differently.
- **`gpu-decode`:** `not_run` on CPU. On the 3090, compare it with the CUDA baseline. Crops still copy back to the CPU, and that copy counts in the crop phase.

Each row also stores `process_first_call_ms` — the first pipeline call in a fresh process, which is what a camera pays after a reboot — and `latency.cold_first_call_ms`, the first call of the timing loop. Gotcha 6 in [guide 02](02-benchmarking-properly.md) explains the difference.

## 5. Gotchas

1. **Benchmarking the model alone.** **Symptom:** a loop over `model(x)` fits the budget, but the harness row says `≤ SLA: no`. **Fix:** measure the six-phase pipeline through the harness. The gap between the two is the pre- and post-processing you never timed.
2. **Quoting `torch.profiler` totals as latency.** **Symptom:** the op table's CPU total is higher than the harness p50, and someone reports a regression that isn't real. **Fix:** profiler files are for attribution; only `results/results.json` holds measurements.
3. **No sync on the GPU.** **Symptom:** on the 3090, `detect` looks almost free while `nms` or `crop` looks expensive. **Fix:** stop the clock after `backend.sync()`, as `timed_call` does. In your own scripts, call `torch.cuda.synchronize()` first.
4. **Profiling cold.** **Symptom:** allocator and kernel-selection events fill the first frames of the trace, and the op table ranks ops you never see again. **Fix:** warm up outside the profiling window, as `torch_op_profile` does.
5. **Acting on the mean breakdown.** **Symptom:** the gate line's all-frames share differs from its slowest-5% share, and a fix chosen from the first doesn't move the p95. **Fix:** choose fixes from `p95_tail_phases_ms`.
6. **fast-resize silently falling back.** **Symptom:** after a camera with a different resolution is installed, `preprocess` is back at its baseline value even though the row says `fast_resize: true`. **Fix:** `letterbox` uses `reduce` only when width and height are exact multiples of the scaled size, and falls back to bilinear otherwise. Pick a camera resolution that divides exactly.
7. **py-spy refusing to attach.** **Symptom:** `make pyspy` exits with a permission error before writing an SVG. **Fix:** on macOS, run it with sudo. In a Linux container, grant `SYS_PTRACE`.

## 6. AV comparison callout

> **Context only — not part of the lab.** For a YOLO-class vehicle and pedestrian detector on an autonomous vehicle, the rule holds but the answer usually differs. Frames arrive raw over the sensor link rather than as JPEG, and resizing typically runs on the ISP or GPU, so decode is rarely on the CPU path. NMS still scales with scene density: a crowded intersection produces far more candidates than an empty road, so the tail depends on the scene, the way ours depends on plates per frame. Lane segmentation has no NMS, and fitting curves to a fixed-size mask is fixed work, so its tail tends to come from contention with other models. Because several models share one accelerator, the profile has to be taken under full multi-model load.

## 7. When NOT to use this

- **Don't re-profile every iteration.** Once the gate says the models dominate on the target hardware, the per-phase breakdown in every harness row is enough.
- **Don't carry a laptop verdict to the box.** An M3 Pro core decodes JPEG differently from a Raspberry Pi core, and a Jetson can decode on its GPU. That's why the gate report is keyed by `hardware_id`.
- **Don't apply the pre/post fixes when the models own the tail.** They trim a small share of the frame, and each carries its own accuracy risk.
- **Never put profiler, py-spy or Nsight numbers in the journey table or the CI gate.**

Previous: none, start here · Next: [01 — Why optimize](01-why-optimize.md)
