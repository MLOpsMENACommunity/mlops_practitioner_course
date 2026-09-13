# 06 — Pruning

Most pruning tutorials end with a model that is exactly as large and exactly as slow as the one they started with. The usual recipe is `prune.ln_structured` followed by `prune.remove`, and it only writes zeros into weight tensors that keep their shape. This guide runs that recipe anyway, benchmarks it, and exports it. Then it removes channels for real with torch-pruning and benchmarks again. The contrast between those rows is the lesson. Along the way: the mask lifecycle, output layers, and pruning versus a distilled student at the same parameter budget.

## 1. The problem it solves

The baseline detector has a ResNet-34 backbone: a server-sized network sitting in front of a 30 ms per-frame budget on a four-core box with no uplink. If the s00 profile shows the `detect` phase dominating the frame, the network has to get cheaper. The options: remove parts of the trained network (this guide), use fewer bits (INT8, next guides), or train a small imitator (distillation, guide 09). Pruning keeps the trained weights and architecture, and needs a fine-tune, not a new model. Stage s05 changes **only the detector**. The CRNN recognizer is untouched, so any change in OCR exact match comes from different detections producing different crops.

## 2. Mental model

### Three different things called "pruning"

| What | What changes in the tensors | What a dense kernel does | Smaller file? |
|---|---|---|---|
| **Unstructured**: individual weights set to zero | Values only | The same multiply-adds, some of them by zero | No |
| **Masked structured**: whole filters set to zero (`prune.ln_structured`) | Values only; shapes unchanged | The same multiply-adds | No |
| **Physical structured**: channels removed (`torch-pruning`) | Shapes shrink, and every dependent layer shrinks with them | Less work, because the tensors are smaller | Yes |

The tutorial claim is that `prune.ln_structured` plus `prune.remove` gives you a smaller, faster model. **That is false.** A `Conv2d` with 128 output channels runs the same number of multiply-adds whether some filters are zero or not. The ONNX file stores every weight as a full float, and a zero is just as large as any other value. The masked detector's exported ONNX file therefore has the same tensor shapes and the same compute as the baseline.

A quieter issue: most convs here feed a `BatchNorm2d` that keeps its shift and running mean for the zeroed channel, so the channel outputs a **constant**, not zero. Removing it changes what the network computes, which is part of why slicing needs a fine-tune.

Unstructured sparsity pays off only on a runtime with sparse-aware kernels. Semi-structured 2:4 sparsity exists on Ampere-class and newer GPUs, with its own kernels and tooling. On the dense kernels this session uses, **only structured channel removal is something the hardware can use.**

### The mask lifecycle (`snippet:pruning-mask-lifecycle`)

`prune.ln_structured` does not change the weight. It renames the parameter to `weight_orig`, adds a `weight_mask` buffer, and registers a hook that computes `weight = weight_orig * weight_mask` before every forward pass. The gradient reaching a masked position is multiplied by zero, so fine-tuning cannot bring that filter back. `prune.remove` bakes `weight_orig * mask` into a plain `weight` parameter and deletes the mask.

The order is therefore:

**prune → fine-tune with masks attached → `prune.remove()` → export**

Swap the middle two steps and the zeros become ordinary trainable values that the optimizer refills, with no error or warning. s05 runs both orders:

```python
    if remove_first:  # the silent failure: the mask is gone, so Adam refills the zeros
        for conv in convs:
            prune.remove(conv, "weight")
    fit_detector(model, epochs, lr=5e-4, tag="masked-finetune")  # 2. weight = weight_orig * mask, every step
```

It then writes both zero fractions into the masked row's notes ("prune.remove ran FIRST" versus "masks stayed attached").

### Never prune the output layers (`snippet:never-prune-output-layers`)

A naive `for m in model.modules(): if isinstance(m, nn.Conv2d)` loop also selects `obj` (1 channel: plate/no plate) and `box` (4 channels: left, top, right, bottom). Pruning those filters does not delete redundancy. It deletes **outputs**, such as a box side. On the recognizer, the same loop over `nn.Linear` would hit `fc`, the CTC projection: one output per character plus the blank. `prunable_convs` excludes `OUTPUT_LAYERS` (and depthwise convs, `groups != 1`). torch-pruning gets `ignored_layers=[model.obj, model.box]` for the same reason. `tests/test_parity_and_pruning.py :: test_pruning_never_selects_output_layers` enforces it.

### Physical slicing and DepGraph (`snippet:physical-slicing`)

You cannot remove output channels from one conv in isolation. In a ResNet block, the residual add requires the block's last conv and its shortcut to have identical channel counts. In `PlateDetector.features`, `lat3(c3)` is added to the upsampled `lat4` output. Removing a channel in one of those layers forces removal in every layer coupled to it, plus the following BatchNorm and the next layer's input channels. torch-pruning's dependency graph (DepGraph, https://arxiv.org/abs/2301.12900) finds these groups and scores each group's importance as a whole. The library is VainF/Torch-Pruning, MIT licensed; this session installs v1.6.1.

| Argument | Why |
|---|---|
| `MagnitudeImportance(p=1)` | Rank channels by L1 norm, the same criterion as the masked row, so the two rows differ only in masking versus slicing |
| `pruning_ratio`, `iterative_steps` | The ratio is reached after `iterative_steps` calls to `pruner.step()`, with fine-tuning in between |
| `ignored_layers=[model.obj, model.box]` | Output layers, as above |
| `round_to=8` | Channel counts in multiples of 8, which dense kernels are tuned for (https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) |

`pruning_ratio` counts **channels**, not parameters. A conv loses input and output channels together, so parameters drop faster than the ratio suggests; the library's README gives roughly 1-(1-p)^2 (https://github.com/VainF/Torch-Pruning). The image input channels and the ignored layers don't shrink at all. So s05 never reports the ratio as a result: `sparsity()` counts real `params`, `zero_params` and `zero_fraction` into the notes.

A sliced network no longer matches `PlateDetector("resnet34")`: its layer widths changed. Its weights cannot be loaded back into that class. `src/models/io.py` saves it as a whole module (`whole=True`) and loads it with `weights_only=False`. That is safe only because we wrote the file ourselves.

## 3. Runnable walkthrough

```bash
make s05     # depends on s04: every pruned detector is also exported to ONNX
```

`src/stages/s05_pruning.py :: main`, in order:

1. **`prune_and_finetune(..., remove_first=True)`**, then **`prune_and_finetune(...)`**: wrong order, then right, each masking 50% of filters per prunable conv. Both zero fractions go into the notes; only the correct model is measured (`masked-structured-50`).
2. **`slice_channels(ratio=0.5, steps=1, epochs=3 × finetune_epochs)`** gives `sliced-oneshot-50`, and **`slice_channels(0.5, steps=3, finetune_epochs each)`** gives `sliced-iterative-50`. Both get the **same total fine-tuning budget**, so the comparison is about the schedule, not about training time.
3. **`ratio_for_budget`**: a bisection over dry-run slices (`slice_channels_dry`, no training) to find the ratio at which the detector has the MobileNetV3-Small student's parameter count. The result, `sliced-student-budget`, exists to be compared with s07's `student-distilled`.
4. **`measure`**: saves the checkpoint, exports ONNX with guide 05's `export`, runs a **report-only** parity gate (pruning is supposed to change outputs), then benchmarks a torch row and an `-onnx` row. The ONNX row's parent is `s04_onnx_export:ort-cpu-fp32`, and it carries the torch row's parity report. The ONNX export itself is not gated again.

Print the notes for every s05 row from `results/results.json`. They hold the sparsity dictionaries, the ratio chosen for the budget, and the mask-order comparison.

## 4. Measured result

<!-- results:stage:s05_pruning -->
<!-- /results -->

What to compare:

| Question | Rows | Columns |
|---|---|---|
| Does masking shrink the artifact? | `masked-structured-50-onnx` vs `s04_onnx_export:ort-cpu-fp32` | size MB |
| Does masking make anything faster? | `masked-structured-50` vs `s01_baseline:eager-fp32`; `masked-structured-50-onnx` vs `ort-cpu-fp32` | p50, p95, fps |
| Does slicing? | each `sliced-*` row and its `-onnx` twin vs the same parents | size MB, p50, p95, fps, peak RSS MB |
| What did it cost? | every row vs its parent | mAP@0.5, OCR exact, and `accuracy.per_condition` in `results.json` |
| Did the mask order matter? | notes of `masked-structured-50` | the two zero fractions |
| One-shot or iterative? | `sliced-oneshot-50` vs `sliced-iterative-50` | mAP@0.5, OCR exact |

How to interpret each outcome:

- **Masked rows.** Same shapes, so the same size as the parent. If latency differs, compare the gap with the p50–p95 spread before believing it: zeros do not change dense kernel work.
- **Sliced rows faster.** The detector was a real share of the frame, and the thinner layers are still well sized for the kernels.
- **Sliced rows smaller but not faster.** Check `latency.phases_ms`: if `detect` was not the dominant phase, a cheaper detector cannot move the total. Also check which layers were cut: high-resolution early layers dominate compute but may have kept most channels.
- **Iterative beats one-shot.** The rounds in between let the network redistribute information before the next cut, which is the usual claim.
- **One-shot matches or beats iterative.** With the same total budget, the iterative schedule spends epochs training intermediate architectures that get cut again. On the `quick` profile, a small validation set may also make the gap smaller than its noise. Rerun on `full` before drawing a conclusion.

For the budget comparison, read `sliced-student-budget` against the distillation rows:

<!-- results:stage:s07_distillation -->
<!-- /results -->

`student-distilled` swaps **both** models: the MobileNetV3-Small detector *and* the conv-only recognizer. `sliced-student-budget` keeps the baseline CRNN. To compare the detectors alone, use `size_mb.detector` and `latency.phases_ms.detect` in `results.json`, not the totals. Guide 09 covers the student side.

## 5. Gotchas

1. **"Pruned" but same size, same speed.** **Symptom:** the masked row's size MB equals its parent's, and p50/p95 are within noise. `sparsity()` reports millions of zero parameters. **Fix:** masks only make the network *sparse*. To make it *smaller*, physically slice channels (`snippet:physical-slicing`) and export again.

2. **`prune.remove()` before fine-tuning.** **Symptom:** after fine-tuning, the zero fraction has fallen back toward the unpruned model's, and the notes' "prune.remove ran FIRST" figure is far below "masks stayed attached." Accuracy looks great, because nothing was pruned. **Fix:** prune, fine-tune with masks attached, remove, export. Assert the zero fraction right before export.

3. **Pruning the output layers.** **Symptom:** with masking, training runs normally but some box sides collapse to a constant width, or scores stay flat, and mAP@0.5 falls apart. With slicing, `decode` or CTC decoding fails with a shape mismatch, or the recognizer never predicts some characters. **Fix:** exclude `OUTPUT_LAYERS` (`obj`, `box`, `fc`) in both the masking loop and `ignored_layers`, and keep the test.

4. **Reporting the ratio or the mask count as the result.** **Symptom:** a slide says "50% pruned," while the parameter count, file size and latency tell another story. **Fix:** report `params`, `zero_params` and `zero_fraction` from `sparsity()`, plus the measured size, and state whether the zeros are masks or removed channels.

5. **Loading a sliced checkpoint into the original class.** **Symptom:** `load_state_dict` raises `RuntimeError: size mismatch for ...`. Loading the whole module without `weights_only=False` fails with a weights-only unpickling error. **Fix:** save sliced models as whole modules (`io.save(..., whole=True)`) and load with `weights_only=False`, **only for files you produced**. The pickle also refers to `src.models.detector.PlateDetector` by module path, so renaming that module breaks every sliced checkpoint.

6. **Channel counts the kernels don't like.** **Symptom:** the sliced model has fewer parameters but no latency gain on GPU, or even a regression, while a slightly *less* pruned model is faster. **Fix:** `round_to=8`. Avoid `global_pruning=True` without checking per-layer widths, because a global ranking can cut one layer almost to nothing.

## 6. AV comparison callout

> **Context only, not part of this lab.** Pruning a YOLO-class driving detector is the same idea with more places to go wrong. Every pyramid level has its own class, box and objectness convs, and all of them belong in `ignored_layers`. Its neck concatenates and upsamples features across scales, so a single channel couples through many layers. That is the dependency problem DepGraph (https://arxiv.org/abs/2301.12900) solves, and lane segmentation decoders have the same coupling through skip connections. The bigger difference is validation. A driving detector must be re-checked per class, distance band and weather condition, because the average mAP can hold while a rare class loses recall. Our detector has one class, so the analogue is `accuracy.per_condition` (day, night, rain, motion blur, low contrast). Check whether any condition moved more than the average did.

## 7. When NOT to use this

- **You can train a student.** If a small dense architecture trained with distillation meets the budget (guide 09), it is usually simpler to deploy and reason about than a sliced large network. Compare `sliced-student-budget` with `student-distilled` on your own hardware before choosing.
- **The network is not the bottleneck.** If s00 shows decode, letterbox or crop dominating the frame, a thinner detector cannot fix the 30 ms budget.
- **The model is already compact and depthwise.** MobileNet-style blocks have coupled depthwise channels (`prunable_convs` skips `groups != 1`), and there is little redundancy left to cut.
- **You cannot fine-tune.** Without the training data, or a pipeline to retrain on, structured slicing leaves accuracy wherever the cut left it.
- **Quantization alone already meets the SLA.** INT8 on the unpruned model is one conversion and one gate; pruning adds fine-tuning and a new checkpoint format.

Previous: [05 — ONNX and execution providers](05-onnx-and-execution-providers.md) · Next: guide 07
