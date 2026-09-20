# 09 — Knowledge distillation

Pruning (guide 06) makes a big network smaller. Distillation trains a network that is small by design, with the big network as its teacher. Here both ANPR models get distilled: a ResNet-34 plate detector into a MobileNetV3-Small student, and a BiLSTM CRNN recognizer into a conv-only CTC student. This is also where the most commonly copied KD snippet silently does nothing, because a temperature on a regression loss cancels out exactly. The guide explains why, what to use for each head instead, and how to judge honestly whether the teacher helped.

## 1. The problem it solves

The roadside box has 30 ms per frame, no uplink, and a handful of CPU cores. The ResNet-34 detector and the BiLSTM recognizer are server models. INT8 (guides 07, 08) shrinks their numbers but keeps their architecture. Pruning keeps their topology and thins it. Neither gives you an architecture designed for the hardware.

A small dense student does: MobileNetV3's depthwise-separable convs, and a recognizer with no recurrence. Trained on labels alone, it may under-fit. A one-hot label says "this step is `8`". The teacher says "`8`, rather like `B`, nothing like `X`". Distillation passes on that structure.

There is a second reason. **The conv-only recognizer is an edge-deployability choice, not only a size choice.** Every runtime and quantizer handles convs, and LSTMs are where converters and INT8 struggle. FX graph mode has no QAT LSTM (guide 07), ONNX Runtime 1.30's QDQ quantizer has no LSTM entry, and guide 13 takes the TFLite path.

## 2. Mental model

### Temperature only means something where there is a softmax

Hinton et al. (https://arxiv.org/abs/1503.02531) soften both distributions with `softmax(z / T)`. At `T > 1`, the small probabilities (the "rather like `B`" information) grow large enough to carry gradient. Dividing logits by `T` shrinks the gradients by roughly `1/T^2`, so the loss is multiplied by `T^2` to restore their scale. That is all `T` does, and all of it depends on the softmax.

Now put the same pattern on a regression head:

```text
mse(s/T, t/T) * T^2  =  (1/N) * sum( ((s_i - t_i) / T)^2 ) * T^2  =  (1/N) * sum( (s_i - t_i)^2 )  =  mse(s, t)
```

**The temperature cancels exactly.** The loss value, the gradient and the trained model are identical for every `T`. A KD snippet copied onto box regression with `/T` and `*T**2` is a no-op with an explanatory comment attached. `tests/test_kd_and_cost.py :: test_temperature_cancels_exactly_for_mse` asserts it for several temperatures. Its neighbour, `test_temperature_changes_the_kd_target_when_there_is_a_softmax`, asserts the opposite for the OCR loss.

### What each head gets

| Head | Output | Distillation signal | Temperature? |
|---|---|---|---|
| Recognizer `fc` | `[B, 32, 37]` per-step character logits incl. CTC blank | KL between softened per-step distributions (`snippet:ocr-kd-loss`) **when the two models place characters in the same steps**, otherwise CTC on the teacher's decoded reading (`snippet:sequence-kd`). `snippet:kd-alignment-check` decides, by measuring | **Yes**, for the per-step form. A real classification; `T^2` restores the gradient scale. |
| Detector neck | stride-8 feature map | Hint loss through a 1x1 adapter, FitNets (https://arxiv.org/abs/1412.6550) | No. It is feature regression. |
| Detector `obj` | objectness logit per cell | BCE against the teacher's softened sigmoid | **Yes.** `sigmoid(z) = softmax([z, 0])[0]` is a two-class softmax. |
| Detector `box` | raw box maps | Squared error weighted by teacher confidence | **No.** It would cancel. |

`class DetectorKD` (`snippet:detector-kd-loss`) puts the three detector terms side by side:

```python
hint = F.mse_loss(student.kd_adapter(feats), t_feats)
soft_obj = F.binary_cross_entropy_with_logits(obj / T, torch.sigmoid(t_obj / T)) * T * T
weight = torch.sigmoid(t_obj)  # only where the teacher sees a plate do its boxes carry information
box_match = (weight * (box - t_box).pow(2)).sum() / weight.sum().clamp(min=1.0)
```

> **Why weight the box term by teacher confidence.** Most of the 48 x 80 cells contain no plate. The teacher's box maps there are arbitrary numbers nobody decodes, and matching them would spend the student's capacity copying noise. `sigmoid(t_obj)` focuses the term on the cells whose boxes actually get used.

### Teacher and student must align where you compare them

- **Detector.** ResNet-34 and MobileNetV3-Small both feed the **same stride-8 neck** (`PlateDetector(width=96)`), so `features()` returns maps of identical spatial size and width. The 1x1 adapter is still needed, because channel `i` of the student has no reason to encode what channel `i` of the teacher does. The adapter learns that mapping. It is attached as `student.kd_adapter` **before** `fit_detector` builds its optimizer, so it gets trained. It is deleted (`del det.kd_adapter`) before `io.save` and export, so the deployed student carries no trace of its teacher.
- **Recognizer.** CRNN (https://arxiv.org/abs/1507.05717) and `ConvCTC` both emit `OCR_SEQ_LEN` time steps over the same crop — but the same step *count* is not the same *alignment*, and only alignment makes a per-step KL meaningful. CTC never says where a character belongs; each model picks its own placement. The BiLSTM reads the whole width at once and spreads a plate from the first step to the last; the conv student, whose receptive field is local, emits each character where its ink is. Both read the plate correctly and their per-step distributions still disagree, so `snippet:kd-alignment-check` measures how often the two agree on which steps carry a character. Below `KD_ALIGNMENT_MIN`, `s07` distils at sequence level instead, and the row's notes say which form it used and what it measured. Per-step KD across that gap does not merely fail to help: it pulls the student towards an alignment its receptive field cannot produce, and the student learns neither.

> **Why misaligned sequence lengths break CTC distillation.** CTC has no fixed character positions. It spreads a string over time steps with blanks and repeats, then collapses them. If the student emitted half as many steps, its step `i` would span two teacher steps, and there would be no correct pairing. Resampling doesn't rescue it: downsampling can drop the blank between a doubled character, so "OO" collapses to "O", and upsampling invents repeats. Either design both models to the same length, as `src/models/ocr.py` does, or distill at sequence level by training the student with CTC on the teacher's decoded strings.

CTC outputs are also "peaky": the teacher puts most steps almost entirely on blank. At `T = 1` the KL would mostly teach blank timing. Softening exposes which characters compete at the non-blank steps.

### The part people get wrong

- **Pasting the classification KD loss onto every head.** On regression it is a mathematical no-op.
- **Believing distillation shrinks a model.** The architecture shrinks it. Distillation tries to recover the accuracy the small architecture loses.
- **Crediting the teacher without a control.** The student's accuracy may come from the architecture and schedule alone. That is why `student-scratch` exists.
- **Reading equal step counts as equal alignment.** Two CTC models of the same length can put the same string in different steps. Measure it before comparing them step by step.
- **Distilling from a teacher that is not better.** A teacher only has something to pass on where it is right and the student is not. On a small training set the bigger model can be the *worse* one on held-out data — the `quick` profile's recognizer teacher was, until its training set grew. Compare teacher and `student-scratch` on validation before believing any distillation result.
- **Comparing by parameter count alone.** A conv's cost scales with the resolution it runs at, so equal parameters do not mean equal compute or latency.

## 3. Runnable walkthrough

```bash
make s07       # student-scratch, student-distilled, student-distilled-onnx
make s05       # for the comparison: sliced-student-budget
```

Read `src/stages/s07_distillation.py` in this order:

1. **`ocr_kd_loss`** (`snippet:ocr-kd-loss`). Logits are reshaped to `[B*32, 37]`. `log_softmax(student / T)` and `softmax(teacher / T)` go into `kl_div(..., reduction="batchmean") * t * t`.
2. **`DetectorKD`** (`snippet:detector-kd-loss`). The teacher runs in `eval()` under `torch.no_grad()`: its BatchNorm statistics are frozen and no gradient reaches it. The three terms are summed, with the box term down-weighted.
3. **`train_students(distill)`.** Both runs use the same seed, `det_epochs` and `ocr_epochs`. KD enters through `extra_loss` in `src/train.py`, the same hook pruning and QAT use. It is *added* to the hard losses (focal + GIoU, CTC), not substituted for them.
4. **`main()`.** Two torch rows, then `export_all(..., suffix="student")` and a **strict** parity gate on `student-distilled-onnx`, because an export must not change the student. These ONNX files are what guides 10–13 deploy.

For the pruning side, `src/stages/s05_pruning.py :: ratio_for_budget` searches for the channel ratio at which the sliced ResNet-34 detector matches the student detector's parameter count. `slice_channels` (`snippet:physical-slicing`) then removes channels for real and fine-tunes.

## 4. Measured result

<!-- results:stage:s07_distillation -->
_Hardware `3cc807d0` · profile `quick`_

**Measured on:** Apple M3 Pro · 18.0 GB RAM · GPU: none · Darwin 25.5.0 arm64 · Python 3.12.12
**Threads:** ANPR_THREADS=4, OMP_NUM_THREADS=4, ORT intra_op_num_threads=4, inter_op=1
**Latency batch size:** 1 · **SLA:** p95 <= 30.0 ms per frame · **Profile:** `quick` · **Validation set sha256:** `aceb33513379`
**Libraries (as loaded by the rows below):** torch 2.13.0, onnxruntime 1.30.0, openvino 2026.3.1, nncf 3.3.0, ai-edge-litert 2.2.0

| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s01_baseline:eager-fp32` | (root) | torch · cpu · fp32 | 0.905 | 0.933 | 54.1 | 58.3 | no | 18.5 (b1) | 37.25 | 1316 | — | ok |
| `s07_distillation:student-distilled` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.898 | 0.947 | 28.8 | 30.7 | no | 53.6 (b8) | 2.46 | 625 | 8.026 (b1) | ok |
| `s07_distillation:student-distilled-onnx` | `s07_distillation:student-distilled` | ort · cpu · fp32 | 0.898 | 0.947 | 16.4 | 19.3 | yes | 69.2 (b4) | 2.68 | 467 | 4.464 (b1) | ok |
| `s07_distillation:student-scratch` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.900 | 0.932 | 28.8 | 31.0 | no | 54.1 (b8) | 2.46 | 766 | 8.038 (b1) | ok |

<!-- /results -->

<!-- results:stage:s05_pruning -->
_Hardware `3cc807d0` · profile `quick`_

**Measured on:** Apple M3 Pro · 18.0 GB RAM · GPU: none · Darwin 25.5.0 arm64 · Python 3.12.12
**Threads:** ANPR_THREADS=4, OMP_NUM_THREADS=4, ORT intra_op_num_threads=4, inter_op=1
**Latency batch size:** 1 · **SLA:** p95 <= 30.0 ms per frame · **Profile:** `quick` · **Validation set sha256:** `aceb33513379`
**Libraries (as loaded by the rows below):** torch 2.13.0, onnxruntime 1.30.0, openvino 2026.3.1, nncf 3.3.0, ai-edge-litert 2.2.0

| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s01_baseline:eager-fp32` | (root) | torch · cpu · fp32 | 0.905 | 0.933 | 54.1 | 58.3 | no | 18.5 (b1) | 37.25 | 1316 | — | ok |
| `s05_pruning:masked-structured-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.897 | 0.934 | 57.3 | 69.8 | no | 18.3 (b1) | 37.25 | 1227 | — | ok |
| `s05_pruning:masked-structured-50-onnx` | `s05_pruning:masked-structured-50` | ort · cpu · fp32 | 0.897 | 0.934 | 100.7 | 110.2 | no | 10.5 (b8) | 37.46 | 716 | — | ok |
| `s05_pruning:sliced-iterative-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.913 | 0.933 | 30.4 | 32.8 | no | 32.6 (b4) | 13.10 | 1023 | — | ok |
| `s05_pruning:sliced-iterative-50-onnx` | `s05_pruning:sliced-iterative-50` | ort · cpu · fp32 | 0.913 | 0.933 | 37.6 | 41.8 | no | 29.9 (b8) | 13.29 | 494 | — | ok |
| `s05_pruning:sliced-oneshot-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.919 | 0.930 | 29.9 | 43.7 | no | 36.0 (b1) | 13.10 | 980 | — | ok |
| `s05_pruning:sliced-oneshot-50-onnx` | `s05_pruning:sliced-oneshot-50` | ort · cpu · fp32 | 0.919 | 0.930 | 35.8 | 40.3 | no | 31.0 (b4) | 13.29 | 494 | — | ok |
| `s05_pruning:sliced-student-budget` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.859 | 0.902 | 17.1 | 18.8 | yes | 71.8 (b4) | 6.43 | 982 | 4.645 (b1) | ok |
| `s05_pruning:sliced-student-budget-onnx` | `s05_pruning:sliced-student-budget` | ort · cpu · fp32 | 0.859 | 0.902 | 15.8 | 20.1 | yes | 80.2 (b8) | 6.64 | 343 | 4.448 (b1) | ok |

<!-- /results -->

| Compare | Same | Different | Question |
|---|---|---|---|
| `student-distilled` vs `student-scratch` | architecture, seed, epochs | teacher signal | Did distillation help? |
| `student-distilled` vs `s05_pruning:sliced-student-budget` | detector parameter count | designed small vs sliced large | Distillation or pruning at the same size? |
| `student-distilled-onnx` vs `student-distilled` | weights | runtime | Export parity (strict gate) |

**Keep the confounds in view:**

- **mAP@0.5 is the fair column for the detector question.** s05 rows change only the detector and keep the CRNN recognizer, while s07 swaps both models. OCR exact match and latency in s07 include the recognizer change.
- **Training budgets differ.** The sliced detector fine-tunes from trained ResNet-34 weights. The student trains from scratch for `det_epochs`.
- **Same parameters, different compute.** Compare p50/p95 and size MB directly.

**Interpret either outcome honestly:**

| If you see | It means | It does not mean |
|---|---|---|
| distilled beats sliced-student-budget on mAP, and is faster | At this budget, a model designed small beat a large one cut down | That distillation always beats pruning |
| sliced-student-budget matches or beats distilled on mAP | Inheriting trained weights beat training small from scratch, at this schedule | That the dense student is a bad design. Check latency. |
| distilled ≈ scratch | The teacher added little **on this data and schedule**. Synthetic plates have exact labels, which leaves KD least to add. | That KD "doesn't work". Its value tends to grow with noisy or scarce labels. |
| distilled beats scratch | The teacher's soft targets and features carried information the labels didn't | That the gain survives INT8. Re-measure in guides 10–13. |

## 5. Gotchas

1. **Temperature on regression.** **Symptom:** sweeping `T` on the box loss changes nothing, not the curve and not the metrics. **Fix:** remove it, because it cancels exactly. Keep temperature for the softmax and sigmoid terms only.
2. **Forgetting `T^2`.** **Symptom:** raising `T` makes the student behave more and more like `student-scratch`, because the KD gradients shrink toward nothing and `T` silently rescales the KD weight. **Fix:** multiply softened-distribution losses by `T * T`, as `ocr_kd_loss` and `soft_obj` both do.
3. **An adapter the optimizer never sees.** **Symptom:** the hint loss plateaus high while the hard loss falls, because the student is chasing a fixed random projection. **Fix:** register the adapter on the student before `fit_detector` builds `AdamW(model.parameters())`, as `DetectorKD.__init__` does.
4. **An adapter that ships.** **Symptom:** unexpected `kd_adapter.*` keys in the state dict, loading into a clean `PlateDetector` fails, or the export carries a layer nothing calls. **Fix:** `del det.kd_adapter` before `io.save` and export. The strict parity gate then confirms the ONNX file matches the saved student.
5. **Sequence lengths that differ.** **Symptom:** `kl_div` fails on a shape mismatch after `reshape(-1, c)`. Or, if someone "fixed" it with interpolation, the KD loss stays high and doubled characters go missing. **Fix:** design teacher and student to emit the same `OCR_SEQ_LEN`, or distill on decoded strings.
6. **A teacher that is still learning.** **Symptom:** the teacher's own accuracy changes after a distillation run, or its soft targets jitter between steps. **Fix:** `io.load` returns `eval()` mode, `DetectorKD` calls `.eval()` again, and the forward pass runs under `torch.no_grad()`. Keep all three when refactoring.
7. **An unfair control.** **Symptom:** "distilled beats scratch" disappears on a re-run with different settings. **Fix:** same seed, epochs and data for both runs, which `train_students` enforces. Change one and the comparison is void.
8. **Training time.** **Symptom:** a distilled epoch is much slower than a scratch epoch. **Fix:** this is expected. The teacher runs forward on every augmented batch, so its outputs can't be cached. Budget for it.
9. **Depthwise convs meet INT8.** **Symptom:** the student loses more under INT8 than the ResNet did with the same settings. **Fix:** depthwise layers are the most per-tensor-sensitive (guide 07). Keep per-channel weights where the runtime allows, and run guide 08's loop on the student before deploying it.

## 6. AV comparison callout

> **Context, not measured here.** Think about the same technique on AV perception models.
> A YOLO-class multi-class detector distills features at several pyramid levels, each needing its own alignment,
> where we have one stride-8 level. Its class scores are softmax or per-class sigmoid, so temperature applies as on our
> objectness. Detectors whose box head predicts a discretized distribution over offsets have a softmax there,
> so temperature KD on boxes becomes meaningful. Ours regresses raw values, so it doesn't.
> A traffic-sign classifier is the textbook Hinton case: one softmax, soft targets, `T^2`.
> Lane segmentation distills a per-pixel softmax plus feature hints, and needs matching output resolutions,
> the same constraint as our equal CTC sequence length.

## 7. When NOT to use this

- **You can't afford a training run plus a teacher forward pass per batch.** PTQ (guide 07) costs one calibration pass.
- **A small architecture trained from scratch already meets the budget.** If `student-scratch` is within `SLA.max_map50_drop` and `SLA.max_ocr_em_drop`, the teacher is overhead.
- **The teacher is weak where it matters.** A teacher that misreads low-contrast plates teaches the student to misread them too. Check the teacher's per-condition accuracy first.
- **The student doesn't fit the target runtime.** Distillation can't fix an op the converter rejects. Choose the student for the hardware first (conv-only here, for exactly that reason), then distill.
- **All you need is a smaller file.** INT8 weights are 4x smaller for the price of a calibration pass. Change the architecture when latency or runtime compatibility demands it.

---

[← 08 — When INT8 breaks](08-when-int8-breaks.md) · [10 — TensorRT →](10-tensorrt.md)
