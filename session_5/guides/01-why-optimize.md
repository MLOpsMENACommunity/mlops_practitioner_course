# 01 — Why optimize

The prototype works. On a server GPU, a ResNet-34 plate detector and a BiLSTM CRNN recognizer in eager PyTorch FP32 read plates correctly. That isn't the product. The product is the same pipeline in a sealed roadside box, finishing every frame within 30 ms at p95, with no cloud behind it. This guide sets out that constraint, the five numbers every stage is judged on, the levers that move them, and the map of the session.

## 1. The problem it solves

**The box** is a Jetson-class or Raspberry Pi-class device. It has a few ARM cores (`ANPR_THREADS` defaults to 4 to match), a few GB of memory shared by the OS, the runtime and both models, and maybe a small GPU.

**The contract** is `src/config.py :: SLA`, which the CI gate reads directly:

| Field | Meaning |
|---|---|
| `p95_ms` | the 30 ms per-frame budget, end to end, on the box |
| `batch_size` | 1, because a camera delivers one frame at a time |
| `max_map50_drop` | absolute mAP@0.5 drop allowed versus the parent stage |
| `max_ocr_em_drop` | absolute plate exact-match drop allowed versus the parent stage |

**No cloud,** for two reasons, and either one would be enough. Privacy law where the cameras are deployed forbids sending images of passing drivers off the device; only the event (plate string and timestamp) may leave. And many sites are rural, with no uplink that could carry a 1280x720 stream from each camera.

**The fleet.** With N cameras there are two options. Either every camera gets its own box, and the per-frame SLA decides whether that works, or several cameras share a site box, and throughput decides how many it can take. Both come down to cost, and `src/cost.py` computes it from measurements:

```
cost_per_million = usd_per_hour / (usable_fps * 3600) * 1_000_000
```

`usable_fps` is the best throughput at any batch size whose p95 still meets the SLA (`src/cost.py :: usable_point`). Capacity that breaks the budget can't be used. For an edge box, `usd_per_hour` is amortized hardware cost plus power.

> **Why the column says "@ $1/h".** The journey table prices every instance at one dollar per hour. The formula is linear in price, so multiply by your real hourly figure. A price typed into a document goes stale; a measured fps doesn't.

```bash
python -m src.cost --row s09_openvino:student-int8-nncf --usd-per-hour <price> --cameras <N> --camera-fps <fps>
```

## 2. Mental model

**Five metrics per stage, always reported together.**

| Metric | Harness field | Decides |
|---|---|---|
| Accuracy | `accuracy.map50` **and** `accuracy.ocr_exact_match`, overall and `per_condition` | whether the output is still correct |
| Latency | `latency.total_ms` p50/p95, `phases_ms`, `p95_tail_phases_ms` | whether one camera meets the SLA |
| Throughput | `throughput.fps` at the saturating batch, plus the `curve` | cameras per box, and cost |
| Size | `size_mb`, detector + recognizer files | storage, over-the-air updates |
| Peak memory | `peak_memory_mb.rss` and `.vram` | whether the pipeline fits at all |

> **Why three metrics aren't enough.** People remember accuracy, latency and file size. Peak memory is what decides whether it fits: runtimes allocate arenas, workspaces and caches that no file on disk reflects, and importing a framework costs memory before any op runs. That's why `src/preprocess.py` doesn't import torch. Throughput decides capacity and cost, and latency at batch 1 says nothing about how many cameras a site box can take.

**Why accuracy is two numbers.** mAP@0.5 judges boxes, and exact match judges strings. INT8 can leave the boxes intact while characters break. A box survives a small perturbation: it still overlaps ground truth at IoU 0.5 and still clears the score threshold. A string doesn't. The recognizer takes an argmax over 37 classes at each of 32 time steps, and one flipped step (`8` read as `B`, or a character read as the CTC blank) fails the whole plate.

**The levers.**

| Lever | What changes | Stages | Risks |
|---|---|---|---|
| Pre/post-processing | decode, resize, NMS, crop | s00 | train/serve mismatch |
| Graph and runtime | fused kernels, planned memory, no Python dispatch; same math | s02, s04, s08–s10 | conversion bugs (hence the strict parity gate) |
| Precision | FP16, INT8 | s06a, s06b, s08–s10 | accuracy, uneven across conditions |
| Architecture size | fewer channels, or a smaller network | s05, s07 | accuracy; needs training |
| Serving and batching | more frames per instance, not a faster frame | s09 throughput hint, s11 | queue delay spent from the budget |

Session 3 covered the graph and precision levers on ResNet-50, including a first ONNX Runtime and INT8 pass ([Session 3, Level 3](../../session_3/serving_levels/README.md#level-3--optimize-the-runtime)). It ended by finding that preprocessing can cost as much as the model. This session starts from there.

**The part people get wrong: levers compose, but not freely.** The stages form a tree, not a chain. An s08 engine is built from the distilled student's ONNX, not from the pruned detector printed above it in the table. An INT8 calibration is only valid for the network it was calibrated on. Every row names its `parent`, and that row is the only valid comparison.

## 3. Runnable walkthrough

```bash
make doctor          # what this machine can run; GPU-only stages will record not_run
make profile         # data, training, s00
make all             # every stage in order
make table           # journey table, benchmark cards, guide result blocks
```

`ANPR_PROFILE=ci|quick|full` sets data size and epochs. `ANPR_THREADS` sets the thread count for every runtime. To see the numbers behind this guide, read `src/config.py :: SLA`, `src/cost.py :: cost_per_million` (`snippet:cost-per-million`) and `tools/journey_table.py :: cost_cell`.

**Map of the session.** `Branch` is `RunSpec.branch` in each stage file. Only the baseline and `torch.compile` are on `main`.

| Stage | Make | Guide | Built from | Branch |
|---|---|---|---|---|
| s00_profile | `make s00` | [00](00-profile-first.md) | s01 baseline | `pre` |
| s01_baseline | `make s01` | [02](02-benchmarking-properly.md) | root | `main` |
| s02_torch_compile | `make s02` | [04](04-torch-compile-and-torchscript.md) | s01 baseline | `main` |
| s03_torchscript | `make s03` | [04](04-torch-compile-and-torchscript.md) | s01 baseline | `export` |
| s04_onnx_export | `make s04` | [03](03-formats-runtimes-backends.md), [05](05-onnx-and-execution-providers.md) | s01 baseline | `export` |
| s05_pruning | `make s05` | [06](06-pruning.md) | s01 baseline | `pruning` |
| s06a_ptq | `make s06a`, `make int8-debug` | [07](07-quantization-ptq-qat.md), [08](08-when-int8-breaks.md) | s04 `ort-cpu-fp32` | `quantization` |
| s06b_qat | `make s06b` | [07](07-quantization-ptq-qat.md) | s06a `ort-static-int8-stratified` | `quantization` |
| s07_distillation | `make s07` | [09](09-knowledge-distillation.md) | s01 baseline | `distillation` |
| s08_tensorrt | `make s08` | [10](10-tensorrt.md) | s07 `student-distilled-onnx` | `tensorrt` |
| s09_openvino | `make s09` | [12 — OpenVINO](12-openvino.md) | s07 `student-distilled-onnx` | `openvino` |
| s10_tflite_edge | `make s10` | [13 — TFLite and the edge](13-tflite-and-edge.md) | s07 `student-distilled-onnx` | `edge` |
| s11_triton | `make s11` | [11 — Triton](11-triton-inference-server.md) | s07 `student-distilled-onnx` | `serving` |

"Built from" names each stage's main parent. Some rows have a different one: the NMS-in-graph row in s04, the `-onnx` rows in s05, the torch rows in s06a and s06b, `student-distilled-onnx` in s07, `baseline-fp16` in s08, and the baseline IR in s09. The lineage diagram under each journey table shows every edge. The distilled student's ONNX files are the trunk that every deployment stage builds from.

## 4. Measured result

<!-- results:journey -->
<!-- /results -->

- **Read the header lines first.** They give CPU, GPU, OS, every thread setting, and the validation set's hash. Different headers mean different experiments, so never compare numbers across sections.
- **`parent`, then both accuracy columns,** compared against the parent row, not the row printed above it.
- **`p95 ms` and `≤ SLA`:** `yes` means within budget *on the machine named in the header*. A laptop `yes` isn't a roadside `yes`.
- **`fps (batch)`** is the best point on the throughput curve, whatever its latency. **`$/1M frames @ $1/h`** uses the best batch that meets the SLA (`bN`). That batch can be smaller, or the column can be a dash if none meets it.
- **`size MB`** is both model files. **`peak RSS MB`** is the whole worker process. VRAM, per-condition accuracy and phase timings are in `results/results.json`.
- **`status`:** `not_run` and `failed` rows carry a reason. A GPU stage showing `not_run` on the laptop is a result, not a missing row.

## 5. Gotchas

1. **Comparing with the row above instead of the parent.** **Symptom:** an s08 row looks like a regression next to the s07 rows printed above it. **Fix:** read the `parent` column and the lineage diagram.
2. **A price in a slide.** **Symptom:** last term's cost figure no longer matches `make table` output for the same row. **Fix:** keep prices out of documents, and run `python -m src.cost --row <id> --usd-per-hour <price>`.
3. **Counting capacity the SLA can't use.** **Symptom:** `fps (batch)` names one batch size while the cost column shows a smaller `bN`, or a dash. **Fix:** that's intended. Plan capacity from the cost column's operating point.
4. **Judging accuracy by mAP alone.** **Symptom:** mAP is within the allowed drop, but exact match falls, and field reports say plates are found but misread. **Fix:** gate on both, and open `accuracy.per_condition` to see whether one condition accounts for the drop.
5. **Judging fit by file size.** **Symptom:** a pipeline with small model files gets killed or starts swapping on the box. **Fix:** compare `peak RSS MB` with the box's free memory. On a Jetson the GPU shares that memory, so add VRAM.
6. **Mixing profiles.** **Symptom:** mAP differs between the laptop's `quick` section and the 3090's `full` section by more than any stage moves it. **Fix:** `quick` trains less, so compare stages only within one section.
7. **Treating the laptop's `≤ SLA` as the deployment verdict.** **Symptom:** CI passes and the box drops frames. **Fix:** CI checks p95 relative to a reference (`snippet:perf-gate`). The absolute check, `--enforce-sla`, belongs on the target hardware.

## 6. AV comparison callout

> **Context only — not part of the lab.** An autonomous-driving perception stack has the same kind of constraint at a larger scale: several cameras, several models per camera (a YOLO-class vehicle and pedestrian detector, lane segmentation, often more), and a per-frame budget from sensor to planner. A late result there means a stale view of the road. What changes is which metric binds first. The models share one accelerator, so memory and throughput are budgeted per model, not per box. The missing uplink is a latency constraint as much as a privacy one. And accuracy has two layers there too: box mAP, plus a downstream metric (lane geometry, track stability) that can degrade while mAP holds, just as our plate strings can break while the boxes hold.

## 7. When NOT to use this

- **When the prototype already meets the contract on the deployment hardware.** Every stage adds a conversion, a parity gate and accuracy risk that someone has to maintain.
- **Before the baseline is accurate enough.** No lever here adds accuracy, and most spend some.
- **Pulling every lever by default.** Stop at the first row that meets all five constraints on the target box.
- **Using the cost column for an edge box without converting its price.** Amortize hardware and power into an hourly figure first.

Previous: [00 — Profile first](00-profile-first.md) · Next: [02 — Benchmarking properly](02-benchmarking-properly.md)
