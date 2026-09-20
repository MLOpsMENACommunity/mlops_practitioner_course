# 07 — Quantization: PTQ and QAT

Quantization stores weights and computes activations as 8-bit integers instead of 32-bit floats. It is the most direct way to shrink the two models the ANPR camera runs on every frame. This guide covers the three ways to get there: dynamic post-training quantization (PTQ), static PTQ with a calibration set, and quantization-aware training (QAT). It also covers the part most tutorials skip, which is how the calibration set is built. Guide 08 picks up when the INT8 model has lost accuracy.

## 1. The problem it solves

The roadside box has a p95 budget of 30 ms per frame and no uplink. If the edge model misreads a plate, no server re-reads it. So INT8 has to deliver two things together:

- **Cost.** INT8 weights are 4x smaller than FP32, and integer kernels (VNNI on x86, NEON on ARM, TensorRT on the RTX 3090) *can* be faster. Whether they are on this box is a measurement.
- **Accuracy per condition.** An INT8 model that reads daytime plates but fails on hazy low-contrast ones is a regression, even when the overall number looks fine.

| Method | What becomes INT8 | When ranges are fixed | Cost to produce | Rows |
|---|---|---|---|---|
| Dynamic PTQ | weights; activations quantized on the fly | per call, at runtime | no data | `torch-dynamic-int8-ocr`, `ort-dynamic-int8` |
| Static PTQ | weights and activations | once, from a calibration set | a calibration pass | `ort-static-int8-stratified`, `-daytime`, `-per-tensor` |
| QAT | both, trained to survive rounding | during fine-tuning | a training run | `ort-qat-int8-ocr`, `torch-qat-converted-ocr` |

## 2. Mental model

### Scale and zero-point

```text
q     = clamp(round(x / s) + z, qmin, qmax)      # float -> int8
x_hat = s * (q - z)                              # what the next layer effectively sees
```

The error comes from **rounding** (up to `s/2` per value) and **clamping** (anything outside `[qmin, qmax]` is lost). A wide range is coarse but rarely clips. A narrow range is fine-grained but clips more. Every quantization method is a policy for choosing `s`.

### Symmetric vs asymmetric

- **Symmetric** (`z = 0`, `s = max|x| / 127`): for weights, which sit roughly centred on zero. It keeps integer kernels simple.
- **Asymmetric** (`z != 0`): for post-ReLU activations, which are never negative. A symmetric range would waste half its codes on values that never occur.

`QAT_QCONFIG` in s06b uses exactly this split: `quint8` activations over `0..255`, and `per_channel_symmetric` `qint8` weights. ONNX Runtime's static quantizer defaults the same way (`WeightSymmetric` on, `ActivationSymmetric` off). Confirm that in the docstring of your installed version.

> **Why per-channel weights matter for convs.** A per-tensor scale fits the filter with the largest weights, and filters in one conv can differ in magnitude a lot. Folding BatchNorm into the conv multiplies each filter by its own `gamma / sqrt(var)`, which spreads them further. With one shared scale, the small filters get only a handful of integer levels. Per-channel gives each output filter its own scale. Depthwise convs have tiny filters with widely varying ranges, so they suffer most. The MobileNetV3 student and the `conv_ctc` recognizer are full of them (guide 09).

### Calibration picks the range, and MinMax trusts every outlier

Static PTQ runs FP32 inference on a calibration set and records each activation's range. `CalibrationMethod.MinMax`, which s06a uses, takes the observed extremes, so **one outlier sets the range for everything**. A single specular highlight on a wet road stretches a feature map's max, and every other value in that tensor loses resolution. Entropy and Percentile calibration clip the tails on purpose (guide 08).

In our pipeline, a low-contrast plate's characters sit only a few tens of gray levels from the plate background (`src/datasets/synthetic.py`, the `low_contrast` branch). The feature difference that separates "8" from "B" can be a small slice of an activation range calibrated on bright day frames. A slice only a few integer steps wide can round away. That is a mechanism to test, not a result.

### The calibration set is part of the model

The calibration frames are seen once, and their ranges are baked into the artifact (`src/datasets/calibration.py`, `snippet:calibration-set`):

```python
per = n // len(config.CONDITIONS)
return [s for c in config.CONDITIONS for s in [p for p in pool if p.condition == c][:per]]
```

| Decision | This repo | Why |
|---|---|---|
| Size | `n_calib` from the profile in `src/config.py` | Too few frames and MinMax ranges depend on which frames you drew |
| Sampling | `stratified`: equal frames per condition | Night, rain, blur and low contrast have their own ranges |
| The tempting shortcut | `daytime`: same count, day only | Collecting day frames is easy. Whether it costs anything is a measurement, not a rule — see the note below. |
| Source split | `load_split("calib")`, never `val` | Calibrating on the frames you score leaks the test set |
| Preprocessing | the pipeline's own `letterbox` + `to_nchw` | Ranges must describe the input production actually sends |

### QDQ vs QOperator

`QuantFormat.QDQ` keeps the FP32 ops and wraps them in `QuantizeLinear`/`DequantizeLinear` pairs. The runtime fuses `DQ -> Conv -> Q` into an INT8 kernel where it has one. QOperator swaps in ORT-specific ops such as `QLinearConv`. **QDQ is what TensorRT and OpenVINO also consume**, so the scales travel with the artifact to guides 10 and 11. The flip side is that a QDQ pattern the runtime cannot fuse runs as dequantize, FP32 op, quantize, which is slower than plain FP32.

### The part people get wrong

- **"INT8 is a flag."** It is a new model. Its accuracy has to be measured per condition and its speed on the target runtime.
- **"Calibration is a formality."** The ranges *are* the model. A daytime-calibrated artifact is a different artifact.
- **"The error is spread evenly."** It concentrates in specific layers (first conv, output projections, outlier channels) and specific inputs (low contrast).
- **"PyTorch INT8 runs on my GPU."** PyTorch-native INT8 kernels are CPU-only: `x86` (fbgemm) on Intel/AMD, `qnnpack` on ARM (`src/backends.py :: select_quantized_engine`). On an NVIDIA GPU, INT8 means TensorRT (guide 10).
- **"Dynamic quantization quantizes the model."** PyTorch's `quantize_dynamic` swaps only `nn.LSTM` and `nn.Linear` (`snippet:torch-dynamic-quant`). The detector's convs are untouched and only the recognizer changes. ORT's dynamic integer-ops registry also covers `Conv`, so `ort-dynamic-int8` is a different experiment.

## 3. Runnable walkthrough

```bash
make s06a      # dynamic + static PTQ, both calibration strategies, per-tensor
make s06b      # QAT on the recognizer (reuses s06a's stratified INT8 detector)
```

**`src/stages/s06a_ptq.py`:**

1. `torch_dynamic()`: `select_quantized_engine()`, then one `quantize_dynamic` call. On torch 2.13, `torch.ao.quantization` prints a migration notice: eager mode moves to torchao's `quantize_`, FX/pt2e to torchao's `prepare_pt2e`/`convert_pt2e` (https://github.com/pytorch/ao/issues/2259). It still works, but new code belongs in torchao.
2. `static_int8()` (`snippet:ort-static-quant`). `quant_pre_process` runs first, as ORT recommends, with `skip_symbolic_shape=True` (gotcha 10 explains why). Every argument is passed by keyword:

```python
quantize_static(
    model_input=str(pre),
    model_output=str(art(dst)),
    calibration_data_reader=calibration.OrtReader(input_name, batches),
    quant_format=QuantFormat.QDQ,
    per_channel=per_channel,
    calibrate_method=CalibrationMethod.MinMax,
```

3. `ort_rows()`: stratified and daytime artifacts for both models, then stratified again with `per_channel=False`. The parity gate runs with `strict=False`. INT8 is expected to move tensors, so the drift is recorded and the benchmark judges accuracy.

**`src/stages/s06b_qat.py`.** QAT inserts FakeQuantize modules that round and clamp in the forward pass and pass gradients straight through in the backward pass. The weights learn values that survive rounding.

1. `QAT_QCONFIG` (`snippet:qat-qconfig`): **non-fused** `FakeQuantize`, per-channel symmetric weights.
2. `qat_finetune()`: `prepare_qat_fx`, fine-tuning at a low learning rate (nudge the weights, don't retrain them), then `apply(disable_observer)` so the scales freeze. `set_object_type(nn.LSTM, None)` keeps the LSTM in FP32 because FX has no QAT LSTM, which makes the QAT recognizer mixed precision by construction.
3. `export_qdq()` (`snippet:qat-export`): `dynamo=False` with `dynamic_axes` produces Q/DQ pairs that ONNX Runtime runs.
4. `convert_fx(prepared)`: FakeQuantize becomes real INT8 kernels, CPU only.

### When QAT is worth the training cost

1. Run static PTQ with stratified calibration. Read it **per condition**.
2. If the loss sits in a few sensitive layers, keep those layers in FP32 (mixed precision, guide 08). That costs one more calibration pass.
3. Choose QAT only if PTQ plus mixed precision still misses `SLA.max_map50_drop` or `SLA.max_ocr_em_drop` in `src/config.py` (absolute, against the parent row), **and** you can afford a fine-tuning run on data covering all five conditions. QAT fine-tuned on daytime data inherits the daytime calibration bug.

## 4. Measured result

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

<!-- results:stage:s06b_qat -->
_Hardware `3cc807d0` · profile `quick`_

**Measured on:** Apple M3 Pro · 18.0 GB RAM · GPU: none · Darwin 25.5.0 arm64 · Python 3.12.12
**Threads:** ANPR_THREADS=4, OMP_NUM_THREADS=4, ORT intra_op_num_threads=4, inter_op=1
**Latency batch size:** 1 · **SLA:** p95 <= 30.0 ms per frame · **Profile:** `quick` · **Validation set sha256:** `aceb33513379`
**Libraries (as loaded by the rows below):** torch 2.13.0, onnxruntime 1.30.0, openvino 2026.3.1, nncf 3.3.0, ai-edge-litert 2.2.0

| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s03_torchscript:jit-trace-fp32` | `s01_baseline:eager-fp32` | torchscript · cpu · fp32 | 0.905 | 0.933 | 54.3 | 61.0 | no | 18.3 (b1) | 37.49 | 1323 | — | ok |
| `s06a_ptq:ort-static-int8-ocr-only` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-ocr | 0.905 | 0.933 | 95.6 | 99.1 | no | 11.1 (b4) | 35.93 | 679 | — | ok |
| `s06a_ptq:ort-static-int8-stratified` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.892 | 0.854 | 32.6 | 35.9 | no | 33.6 (b8) | 11.79 | 366 | — | ok |
| `s06b_qat:ort-qat-int8-ocr` | `s06a_ptq:ort-static-int8-stratified` | ort · cpu · int8-qat | — | — | — | — | — | — | — | — | — | failed: libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive |
| `s06b_qat:ort-qat-int8-ocr-only` | `s06a_ptq:ort-static-int8-ocr-only` | ort · cpu · int8-qat | 0.905 | 0.930 | 97.2 | 101.6 | no | 11.0 (b8) | 37.37 | 679 | — | ok |
| `s06b_qat:torch-qat-converted-ocr` | `s03_torchscript:jit-trace-fp32` | torchscript · cpu · int8-qat | 0.905 | 0.930 | 55.6 | 68.1 | no | 18.5 (b1) | 35.93 | 1204 | — | ok |

<!-- /results -->

Read the tables in pairs. Each pair changes one thing:

| Row | Compare against | Question |
|---|---|---|
| `ort-static-int8-stratified` | `s04_onnx_export:ort-cpu-fp32` | What does static INT8 cost in accuracy, and buy in p95 and size? |
| `ort-static-int8-daytime` | `ort-static-int8-stratified` | What does calibration coverage alone change? |
| `ort-static-int8-fp32-decode` | `ort-static-int8-stratified` | What does quantizing the box/score decode arithmetic cost? (guide 08) |
| `ort-static-int8-fp32-decode-daytime` | `ort-static-int8-fp32-decode` | Calibration coverage again, with the decode damage out of both rows |
| `ort-static-int8-per-tensor` | `ort-static-int8-stratified` | What do per-channel weight scales buy? |
| `ort-dynamic-int8` | `ort-static-int8-stratified` | Runtime activation scales vs calibrated ones, in speed and accuracy |
| `torch-dynamic-int8-ocr` | `s01_baseline:eager-fp32` | Only the LSTM + Linear change; the detector is identical |
| `ort-qat-int8-ocr` | `ort-static-int8-stratified` | Same INT8 detector; did QAT buy recognizer accuracy over PTQ? |
| `torch-qat-converted-ocr` | `s01_baseline:eager-fp32` | Real PyTorch INT8 kernels on CPU for the QAT recognizer |

- **mAP@0.5 and OCR exact together.** OCR exact match is end to end (box IoU >= 0.5 **and** every character right, `src/metrics.py :: exact_matches`), so detector damage shows up there too.
- **Per-condition numbers are not in this table.** They are in `results/results.json` (`accuracy.per_condition`) and in the damage section of guide 08's report.
- **p50/p95 and "≤ SLA"** say whether INT8 was faster *on this runtime*. If it wasn't, read gotcha 4 before concluding anything.
- **size MB** is the artifact on disk. The torch dynamic row shrinks only the recognizer.
- **Hardware line.** Laptop rows and RTX 3090 rows never share a table.

## 5. Gotchas

1. **The copied `quantize_static` call.** **Symptom:** `TypeError` on an argument name, or a positional argument landing in the wrong slot. **Fix:** pass every argument by keyword and check `inspect.signature(quantize_static)` on your installed onnxruntime. `snippet:ort-static-quant` was checked against 1.30.
2. **Calibrating on the validation split.** **Symptom:** INT8 accuracy on val looks indistinguishable from FP32, while field accuracy is worse. **Fix:** calibrate from `load_split("calib")` only.
3. **Daytime-only calibration.** **Symptom:** the `day` column holds while `night` or `low_contrast` falls, and nothing errors. **Fix:** stratify across all five conditions, and compare daytime against stratified before trusting any artifact. **But check which way the measurement actually goes.** MinMax keeps only the extremes, and on this pipeline the daytime frames already contain them: section 5 of the guide-08 report divides the two sets' activation scales tensor by tensor, and they agree within a few percent. A calibration set costs you accuracy when the conditions you left out push activations *outside* the range you measured — not merely because they are missing. Where nothing clips, the narrower, more homogeneous set can even quantize slightly finer. Read the ratios and the per-condition deltas before repeating the rule.
4. **QDQ that never fuses.** **Symptom:** the INT8 row is no faster than its FP32 parent. **Fix:** check `OrtBackend.active_providers`, and open the model in Netron to see which ops kept Q/DQ pairs. ORT's docs cover S8S8 vs U8S8 and `reduce_range` on x86 with and without VNNI (https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). ORT 1.30's QDQ registry has no `LSTM` entry, so an ONNX `LSTM` op stays FP32 in every static row.
5. **Wrong or missing quantized engine.** **Symptom:** PyTorch INT8 errors about the quantized engine, or silently runs slow reference kernels. **Fix:** call `select_quantized_engine()` before quantizing *and* before running. Don't expect PyTorch INT8 on CUDA at all.
6. **The three QAT export traps.** **Symptom (a):** `Exporting the operator 'aten::fused_moving_avg_obs_fake_quant' to ONNX opset version 18 is not supported`. The default QAT qconfig is fused, so use the non-fused one. **Symptom (b):** the dynamo exporter fails on FakeQuantize's data-dependent `if self.observer_enabled[0] == 1`. Call `disable_observer`, then export with `dynamo=False`. **Symptom (c):** the LSTM was never fake-quantized. FX has no QAT LSTM, so read the row as mixed precision.
7. **Observers left on.** **Symptom:** the exported QDQ scales differ from the ones the fake-quant accuracy was measured with. **Fix:** `prepared.apply(disable_observer)` before evaluation and export.
8. **QAT on Apple Silicon.** **Symptom:** fake-quantize ops fail on `mps`. **Fix:** s06b trains on CPU when there is no CUDA. Plan for the extra time.
9. **Per-tensor weights on depthwise convs.** **Symptom:** a MobileNet-style model loses far more than a ResNet under the same settings. **Fix:** per-channel wherever the runtime allows. Guide 13's converter forces per-tensor, so that row needs its own accuracy check.

10. **ORT pre-processing on a dynamo-exported graph.** **Symptom:** `quant_pre_process` raises `Exception: Incomplete symbolic shape inference` before any quantization happens. **Fix:** `quant_pre_process(src, dst, skip_symbolic_shape=True)` — ONNX shape inference and graph cleanup still run, and every node keeps its name and its `pkg.torch.onnx.name_scopes` metadata (guide 08's sensitivity scan depends on that). Hit while building `s06a`.
11. **Saving the converted QAT model with `torch.save`.** **Symptom:** `AttributeError: 'str' object has no attribute '__name__'`, raised from `torch/fx/graph_module.py` `__reduce__` while pickling the result of `convert_fx`. **Fix:** deploy it as TorchScript — `torch.jit.trace(converted, example)` then `torch.jit.save` (`snippet:qat-convert-deploy`); the `s06b` row `torch-qat-converted-ocr` runs exactly that file. Hit while building `s06b`.

## 6. AV comparison callout

> **Context, not measured here.** Think about the same technique on an AV perception stack.
> A YOLO-class detector on an in-vehicle GPU is also calibrated, and its calibration set must cover the
> operational design domain: night, rain, tunnels, glare. That is our stratification argument with many more axes.
> A traffic-sign classifier ends in one softmax, so logit noise matters only when it flips top-1.
> Our recognizer makes 32 such decisions per crop and needs every character right, so the same noise has more
> chances to change the output. Lane segmentation decides per pixel, and boundary pixels flip first, the
> analogue of our low-contrast character edges. The workflow carries over directly: per-slice metrics first, then per-layer precision.

## 7. When NOT to use this

- **The FP32 row already meets 30 ms with headroom** and the accuracy budget has no slack. INT8 adds a calibration artifact and a per-condition validation burden for a win you don't need.
- **The target runtime has no INT8 kernels for your ops.** Unfused QDQ is slower than FP32. Measure first.
- **You can't build a representative calibration set.** Without night, rain and low-contrast frames you are shipping ranges for a camera you don't have. Stay FP32 (or FP16 on the GPU, guide 10) until you have the data.
- **QAT first.** It costs a training run and ties you to an API that is migrating to torchao. It pays off only after PTQ and mixed precision measurably miss the budget.

---

[← 06 — Pruning](06-pruning.md) · [08 — When INT8 breaks →](08-when-int8-breaks.md)
