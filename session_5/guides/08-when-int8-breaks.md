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
<!-- /results -->

<!-- results:stage:s06a_ptq -->
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
