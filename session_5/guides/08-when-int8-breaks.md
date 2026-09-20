# 08 — When INT8 breaks

Your INT8 model lost accuracy. Now what? Most material stops at "try QAT", which is the most expensive option and often not the one you need. This guide walks through the debugging loop in `src/quant_debug.py`:

1. **Damage.** Measure the loss per condition.
2. **Sensitivity.** Find which layer groups cause it.
3. **Outliers.** Find which activations make those groups hard to quantize.
4. **Mixed precision.** Keep exactly those layers in FP32 and measure the result as a row.

After the loop comes a ladder of fixes ordered by cost, and a decision rule for when QAT really is the answer.

## 1. The problem it solves

The ANPR box has to hit 30 ms per frame with no uplink. Suppose a static INT8 row from guide 07 is the only variant that fits the latency budget, but its accuracy delta exceeds `SLA.max_ocr_em_drop` or `SLA.max_map50_drop` in `src/config.py`. The obvious options are all bad: ship a camera that misreads plates, fall back to FP32 and miss latency, or start a QAT run without knowing what broke. The loop turns "INT8 is bad" into "these groups are bad, on these conditions, because of this range", which is specific enough to fix cheaply.

## 2. Mental model

### Where the error enters and where it lands

Quantization error enters as rounding or clipping, at both weights and activations, and it compounds through depth. Two kinds of layer are special:

- **The first layer** sees raw pixels, where a low-contrast plate's characters differ from the background by only a few tens of gray levels.
- **The output layers** (`detector.obj`, `detector.box`, the CTC projection `fc`) are decoded directly. A small perturbation there moves a score across the threshold or flips the argmax between two characters, and no later layer absorbs it.
- **The decode arithmetic after them** (`centre ± softplus(box) × stride`). It contains no conv, but the quantizer still inserts Q/DQ pairs around its `Add` and `Concat` — with one scale spanning the whole 640-pixel input, so box corners snap to a grid a few pixels wide. `snippet:fp32-decode` finds these nodes and keeps them in FP32.

That is why `snippet:mixed-precision` keeps both kinds in FP32. It is also why `OUTPUT_LAYERS` in `src/models/` shields the output layers from pruning (`snippet:never-prune-output-layers`).

### Why one wide channel hurts its neighbours

Activations get **one scale per tensor**, even when the weights are per-channel. If one channel runs much wider than the rest, the scale has to fit that channel, and every other channel gets a fraction of the codes. `snippet:outlier-channels` measures this as **spread = max / median** of the per-channel maximum activation over calibration data. With spread `k`, the median channel uses roughly `1/k` of the integer range. It is coarse even though nothing clips.

### The hypothesis this loop tests

For this pipeline, the working hypothesis is that **INT8 can leave detector mAP@0.5 intact while plate exact match falls**. The mechanism is plausible:

- **mAP@0.5 is forgiving.** IoU >= 0.5 is enough, and AP depends on how scores rank, not on their values (except near the pipeline's score threshold).
- **Exact match is not.** Every character must be right (`src/metrics.py :: exact_matches`), and each one is an argmax over 37 classes. Confusable pairs (8/B, 0/O/D, 5/S) sit close together in logit space, and errors compound across a plate.
- **Low contrast is the stress case.** It has the smallest signal relative to ranges set by the rest of the calibration data.

The measured columns decide it:

| What the columns show | Meaning | Next step |
|---|---|---|
| mAP flat; OCR down; `low_contrast` (maybe `night`) far worse than `day` | A range and resolution problem on weak signals | Calibration coverage, then Percentile/Entropy, then mixed precision on the top OCR groups |
| mAP flat; OCR down evenly across conditions | One or two layers are sensitive regardless of input | The sensitivity table. If `fc` or the first conv leads, mixed precision. |
| mAP flat; OCR down; `ort-static-int8-ocr-only` loses nothing but `-detector-only` loses it all | The detector's boxes moved a few pixels — inside IoU 0.5, outside what a crop tolerates | The `decode` group in the sensitivity table; `ort-static-int8-fp32-decode`. QAT on the recognizer cannot help. |
| mAP down too | Detector damage; end-to-end OCR falls with misplaced crops | Check the recognizer-only (GT crop) sensitivity, then fix the detector first |
| Daytime row worse than stratified in the non-day columns only | The calibration set was the bug | Stratify and stop |
| Per-tensor row much worse than per-channel | Weight ranges differ strongly across filters | Keep per-channel; watch runtimes that force per-tensor (guide 13) |
| Nothing moves beyond the SLA | INT8 is fine for accuracy | Latency (guide 07, gotcha 4) |

### The part people get wrong

- **Jumping to QAT.** It is the last rung, not the first.
- **Reading only the overall number.** A drop confined to one condition averages away across five.
- **Treating mAP as the health check.** The boxes can be right while the strings are wrong.
- **Assuming sensitivity is additive.** Two groups that are harmless alone can hurt together, because each one's error becomes the other's input. The probes only rank candidates. The mixed row is the real test.

## 3. Runnable walkthrough

```bash
make int8-debug      # runs s06a first if needed, then src/quant_debug.py
```

It writes `results/int8_debug/report_<hardware>_<profile>.md` and `.json`, and adds the row `s06a_ptq:ort-static-int8-mixed`.

**Step 1: damage (`damage()`).** For every measured s06a row on this hardware and profile, it subtracts the FP32 ONNX row (`s04_onnx_export:ort-cpu-fp32`). The output columns are `map50_delta`, `ocr_em_delta`, and `ocr_em_delta_<condition>` for all five conditions. Negative is worse.

**Step 2: sensitivity (`snippet:layer-sensitivity`).** The dynamo exporter stores each ONNX node's PyTorch module path in the `pkg.torch.onnx.name_scopes` metadata. `layer_groups()` reads it back, which is how `node_Conv_497` becomes `detector.c3.4` (ResNet `layer1`). Groups are cut at module depth 1 for the recognizer and depth 2 for the detector, then quantized one at a time:

```python
for group, nodes in layer_groups("ocr_baseline.onnx", depth=1).items():
    quantize("ocr_baseline.onnx", "ocr_probe.onnx", "crops", ocr_cal, include=nodes)
    report["ocr"][group] = round(ocr_fp32 - ocr_accuracy("ocr_probe.onnx")["all"], 4)
```

`nodes_to_quantize=include` keeps everything else in FP32. Recognizer sensitivity is scored on **ground-truth val crops**, which keeps the detector out of the loop. Detector sensitivity is scored on 200 val frames read by the FP32 recognizer, and groups are ranked by **end-to-end exact match**, with mAP@0.5 lost reported beside it. Ranked by mAP, a group that shifts every box by a few pixels looks harmless. The detector's decode arithmetic is one extra group, `decode`, because it holds no conv and would otherwise never be probed.

**Step 3: outliers (`snippet:outlier-channels`).** Forward hooks on every `Conv2d` of both FP32 models keep a running per-channel `abs().amax` over the stratified calibration batches. Layers are ranked by spread.

**Step 4: mixed precision (`snippet:mixed-precision`).** The first group, the output layers and the `keep_worst` most sensitive groups of each model go to `quantize_static(..., nodes_to_exclude=...)`. The result is measured like any other row, with `ort-static-int8-stratified` as its parent.

### The options ladder, cheapest first

| Rung | Change | Fixes | Cost |
|---|---|---|---|
| 1. Better calibration data | Stratify across conditions; own split; enough frames | Ranges that never saw night or low contrast | A calibration pass |
| 2. Per-channel weights | `per_channel=True` | Filters with very different ranges (BN-folded, depthwise) | None, where the runtime supports it |
| 3. Different `calibrate_method` | `CalibrationMethod.Entropy` or `Percentile` | One outlier stretching an activation range | Slower calibration (histograms) |
| 4. Mixed precision | `nodes_to_exclude` for first, output and top sensitive groups | Loss concentrated in a few groups | Some latency back |
| 5. Equalization (SmoothQuant-style) | Per-channel rescaling between activation and next weights | Activation outlier channels | Graph surgery; not applied here |
| 6. QAT | Fine-tune through fake quantization (guide 07) | Loss spread across many layers | A training run |

> **Why equalization works, and why it matters less here.** SmoothQuant (https://arxiv.org/abs/2211.10438) divides each activation channel by `s_j` and multiplies the next layer's matching weight column by `s_j`. The FP32 output is unchanged, but the outlier moves out of the per-tensor activation and into per-channel weights that can absorb it. It targets transformer `Linear`/`MatMul` layers, where a few channels are huge on every token.
> Our convs are followed by BatchNorm and ReLU, which renormalize channels layer by layer, so we expect fewer systematic outliers. The outlier table is the evidence either way. The conv-network analogue is cross-layer weight equalization (https://arxiv.org/abs/1906.04721). ORT's `SmoothQuant` option in `extra_options` needs the extra `neural_compressor` package. **Not used here.**

### Decision rule for QAT

Choose QAT only when **all** of these hold:

1. The damage is measured per condition on a stratified, per-channel artifact (rungs 1 and 2 are done).
2. Mixed precision on the top sensitive groups still misses `SLA.max_map50_drop` or `SLA.max_ocr_em_drop` against the parent, or keeping enough groups in FP32 to meet it breaks the 30 ms budget.
3. You have training data for all five conditions and can afford a fine-tuning run.

A sharp head in the sensitivity table argues against QAT. A flat spread of small losses argues for it.

## 4. Measured result

<!-- results:int8-debug -->
#### INT8 debugging report (generated by src/quant_debug.py)

##### 1. Damage vs the FP32 ONNX row (absolute deltas; negative = worse)

| variant | mAP all | mAP day | mAP night | mAP rain | mAP motion_blur | mAP low_contrast | EM all | EM day | EM night | EM rain | EM motion_blur | EM low_contrast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s06a_ptq:ort-dynamic-int8 | 0.0004 | -0.0002 | 0.0002 | 0.0 | 0.0001 | 0.0013 | 0.0021 | 0.0029 | 0.0138 | -0.0057 | 0.0 | 0.0 |
| s06a_ptq:ort-static-int8-daytime | -0.0062 | -0.0054 | -0.0021 | -0.0083 | 0.0005 | -0.0068 | -0.0623 | -0.0461 | -0.0689 | -0.0795 | -0.0565 | -0.0774 |
| s06a_ptq:ort-static-int8-detector-only | -0.0124 | -0.0074 | -0.0179 | -0.0178 | 0.0015 | -0.0187 | -0.0803 | -0.049 | -0.0827 | -0.1079 | -0.0968 | -0.1032 |
| s06a_ptq:ort-static-int8-fp32-decode | -0.0007 | 0.0 | -0.0003 | -0.0005 | 0.0015 | -0.0008 | -0.0021 | 0.0 | 0.0 | -0.0113 | 0.0 | 0.0 |
| s06a_ptq:ort-static-int8-fp32-decode-daytime | -0.0002 | 0.0015 | 0.0059 | -0.0028 | 0.0005 | 0.0018 | -0.0032 | 0.0086 | -0.0069 | -0.017 | -0.0162 | 0.0 |
| s06a_ptq:ort-static-int8-mixed | 0.0012 | 0.0026 | 0.0004 | -0.0001 | 0.0004 | 0.0016 | 0.001 | 0.0086 | 0.0138 | 0.0 | -0.0081 | -0.0193 |
| s06a_ptq:ort-static-int8-ocr-only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0069 | -0.0057 | 0.0 | 0.0 |
| s06a_ptq:ort-static-int8-per-tensor | -0.0122 | -0.0125 | -0.0156 | -0.0193 | 0.0026 | -0.0139 | -0.0782 | -0.0577 | -0.0689 | -0.1079 | -0.0807 | -0.0968 |
| s06a_ptq:ort-static-int8-stratified | -0.0124 | -0.0074 | -0.0179 | -0.0178 | 0.0015 | -0.0187 | -0.0792 | -0.049 | -0.0758 | -0.1079 | -0.0968 | -0.1032 |
| s06a_ptq:torch-dynamic-int8-ocr | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0011 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0064 |
| s06b_qat:ort-qat-int8-ocr | -0.0124 | -0.0074 | -0.0179 | -0.0178 | 0.0015 | -0.0187 | -0.0729 | -0.0461 | -0.0482 | -0.0909 | -0.0887 | -0.1226 |
| s06b_qat:ort-qat-int8-ocr-only | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0032 | 0.0029 | 0.0069 | -0.0113 | -0.0081 | -0.0129 |
| s06b_qat:torch-qat-converted-ocr | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0032 | 0.0029 | 0.0069 | -0.0113 | -0.0081 | -0.0129 |

##### 2. Sensitivity — accuracy lost when ONLY this group is INT8

Recognizer FP32 exact match on ground-truth crops: 0.9588

| OCR group | exact match lost |
|---|---|
| cnn.2 | 0.001 |
| cnn.4 | 0.001 |
| cnn.5 | 0.001 |
| cnn.10 | 0.001 |
| cnn.0 | 0.0 |
| cnn.7 | 0.0 |
| fc | 0.0 |

Detector FP32 on 200 val frames, FP32 recognizer: mAP@0.5 0.9084, exact match 0.9421

| detector group | exact match lost | mAP@0.5 lost |
|---|---|---|
| decode | 0.1189 | 0.0326 |
| detector.c3.1 | 0.0096 | -0.0007 |
| detector.obj | 0.0032 | 0.0004 |
| detector.c3.4 | 0.0 | -0.0004 |
| detector.c4.0 | 0.0 | -0.0001 |
| detector.c4.1 | 0.0 | 0.0001 |
| detector.c4.2 | 0.0 | 0.0003 |
| detector.c4.3 | 0.0 | 0.0 |
| detector.c4.4 | 0.0 | 0.0 |
| detector.c4.5 | 0.0 | 0.0001 |
| detector.lat4 | 0.0 | -0.0004 |
| detector.tower.0 | 0.0 | -0.0003 |
| detector.box | 0.0 | 0.0001 |
| detector.c3.5 | -0.0032 | -0.0002 |
| detector.lat3 | -0.0032 | -0.0003 |
| detector.neck.1 | -0.0032 | -0.0002 |
| detector.tower.1 | -0.0032 | 0.0 |

##### 3. Outlier channels (max / median per-channel activation, after BatchNorm)

| layer | max | median | spread |
|---|---|---|---|
| detector_baseline.c3.1 | 20.38 | 4.463 | 4.6x |
| detector_baseline.tower.1.1 | 13.25 | 3.632 | 3.6x |
| detector_baseline.c3.4.0.bn2 | 15.928 | 6.424 | 2.5x |
| detector_baseline.tower.0.1 | 9.785 | 4.192 | 2.3x |
| detector_baseline.c4.0.bn1 | 11.834 | 5.264 | 2.2x |
| detector_baseline.neck.1 | 11.073 | 5.005 | 2.2x |
| detector_baseline.c4.0.bn2 | 10.548 | 5.143 | 2.1x |
| ocr_baseline.cnn.0.1 | 12.401 | 5.99 | 2.1x |

##### 4. Mixed precision

Kept in FP32: `{'ocr_kept_fp32': ['cnn.0', 'cnn.2', 'cnn.4', 'fc'], 'detector_kept_fp32': ['decode', 'detector.box', 'detector.c3.1', 'detector.obj']}` — measured as row `s06a_ptq:ort-static-int8-mixed`.

##### 5. Calibration ranges — daytime-only scale / stratified scale, per activation tensor

A ratio near 1: daytime frames already held the extremes MinMax keeps. Below 1: daytime ranges clip other conditions.

| model | tensors | min | median | max | share ≥10% narrower |
|---|---|---|---|---|---|
| detector | 64 | 0.854 | 1.023 | 1.115 | 5% |
| ocr | 68 | 1.0 | 1.0 | 1.09 | 0% |
<!-- /results -->

<!-- results:stage:s06a_ptq -->
_Hardware `3cc807d0` · profile `quick`_

**Measured on:** Apple M3 Pro · 18.0 GB RAM · GPU: none · Darwin 25.5.0 arm64 · Python 3.12.12
**Threads:** ANPR_THREADS=4, OMP_NUM_THREADS=4, ORT intra_op_num_threads=4, inter_op=1
**Latency batch size:** 1 · **SLA:** p95 <= 30.0 ms per frame · **Profile:** `quick` · **Validation set sha256:** `aceb33513379`
**Libraries (as loaded by the rows below):** torch 2.13.0, onnxruntime 1.30.0, openvino 2026.3.1, nncf 3.3.0, ai-edge-litert 2.2.0

| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s01_baseline:eager-fp32` | (root) | torch · cpu · fp32 | 0.905 | 0.933 | 54.1 | 58.3 | no | 18.5 (b1) | 37.25 | 1316 | — | ok |
| `s04_onnx_export:ort-cpu-fp32` | `s01_baseline:eager-fp32` | ort · cpu · fp32 | 0.905 | 0.933 | 105.8 | 133.8 | no | 10.2 (b8) | 37.46 | 578 | — | ok |
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

<!-- /results -->

- **Damage.** Start at `ocr_em_delta_low_contrast`, then `night`, and compare against `ocr_em_delta_day`. The daytime line minus the stratified line is calibration coverage and nothing else. `map50_delta` beside `ocr_em_delta` tells you which row of the outcome table in section 2 you are in.
- **Sensitivity.** Rows are sorted by accuracy lost. A sharp head points to mixed precision, and a flat tail points to QAT. Values at or below zero are not measurable at this sample size.
- **Outliers.** If the high-spread layers are also the sensitive groups, rungs 3 and 5 are the targeted fixes. If they aren't, the loss is coming from somewhere else.
- **Mixed precision.** Compare `ort-static-int8-mixed` with `ort-static-int8-stratified` for the accuracy recovered, **and** check p95, size MB and "≤ SLA" for what it cost.

## 5. Gotchas

1. **Sensitivity on a small sample.** **Symptom:** a group shows negative sensitivity (quantizing it "improved" accuracy), or the ranking reshuffles between runs. **Fix:** treat small values as noise, rank on a larger slice before trusting close calls, and let the measured mixed row decide.
2. **Choosing FP32 groups on the split you report.** **Symptom:** the mixed row looks better on val than it holds up elsewhere. **Fix:** ranking and scoring both use val, so the selection has seen the test set. That is fine for teaching. On a real project, rank on a separate dev split.
3. **Group names that don't map back.** **Symptom:** every group is named `node_Conv_*`, or one group swallows the model. **Fix:** only the dynamo exporter writes `pkg.torch.onnx.name_scopes`, so a TorchScript export (like guide 07's QAT model) falls back to node names. Tune `depth` if groups are too coarse or too fine.
4. **Outlier hooks see pre-BN, pre-ReLU outputs.** **Symptom:** the outlier table flags a layer that sensitivity says is harmless. **Fix:** the hooks sit on PyTorch `Conv2d` outputs, and after BN folding the ONNX graph quantizes a different tensor. Use spread to find candidates and sensitivity to confirm them. LSTM and `Conv1d` layers are not hooked.
5. **Two quantize calls that are not identical.** **Symptom:** probe or mixed results disagree with what the s06a rows suggest. **Fix:** `quant_debug.quantize` skips the `quant_pre_process` step that `s06a_ptq.static_int8` runs. Use the probes as a ranking, and remember that the mixed row and its stratified parent also differ in preprocessing.
6. **The damage table lags the mixed row.** **Symptom:** `ort-static-int8-mixed` is missing from the damage section, or shows a previous run's values. **Fix:** `damage()` runs before the mixed row is measured, so read the fresh row in the stage table.
7. **FP32 islands that cost latency.** **Symptom:** the mixed row recovers accuracy, but its p95 is worse than both the INT8 and FP32 rows. **Fix:** every island adds Q/DQ boundaries. Exclude contiguous blocks rather than scattered convs, and re-measure latency.
8. **No damage rows.** **Symptom:** the damage section is empty. **Fix:** it only reads s06a rows that match this machine's hardware id and `ANPR_PROFILE`. Run `make s06a` here under the same profile.

## 6. AV comparison callout

> **Context, not measured here.** Think about the same loop on AV perception.
> A YOLO-class detector's version of "mAP holds, OCR falls" is overall mAP holding while small, distant or rare
> objects degrade: a far pedestrian, a traffic light at night. Those are per-slice columns (distance, class,
> weather), and AV teams look there first, as we look at `low_contrast`. Per-layer precision control on in-vehicle
> GPUs does the job `nodes_to_exclude` does here. A traffic-sign classifier's head tends to be sensitive for the
> same reason our `fc` is: its logits are the decision. Lane segmentation's full-resolution decoder plays
> the role of our first conv. The loop is the same, and only the slices change.

## 7. When NOT to use this

- **No s06a row breaks the accuracy budget.** The scan costs one quantization and one evaluation per group. Spend that time on latency instead.
- **The target runtime quantizes on its own terms.** TensorRT (guide 10), NNCF (guide 11) and onnx2tf (guide 13) have their own calibrators. The ORT ranking is a hypothesis there, not a result, so re-measure on the runtime you ship.
- **The loss is spread across many layers.** A flat sensitivity table means mixed precision would keep most of the model in FP32. Go to QAT (guide 07) or a model designed for INT8 (guide 09).
- **You haven't fixed calibration yet.** Sensitivity on a daytime-calibrated artifact ranks layers by how badly bad data hurts them.

---

[← 07 — Quantization: PTQ and QAT](07-quantization-ptq-qat.md) · [09 — Knowledge distillation →](09-knowledge-distillation.md)
