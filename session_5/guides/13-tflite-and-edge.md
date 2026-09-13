# 13 — TFLite (LiteRT) and the edge

The session's last runtime targets the smallest box, an ARM board with no GPU. We take the s07 student ONNX files, convert them to TFLite flatbuffers with onnx2tf, and quantize them to full-integer INT8 using our own representative frames. Then we run them with the LiteRT interpreter and its XNNPACK CPU kernels. This conversion is the most fragile one in the session. The gotchas below are real failures from building `src/stages/s10_tflite_edge.py`, quoted with their exact messages. The guide ends with the question the session has been building toward: did any row meet 30 ms on the target?

## 1. The problem it solves

The job hasn't changed: a 1280x720 JPEG comes in, plate strings go out, 30 ms per frame at p95, no uplink. What changes is the box. It's now a Raspberry Pi 5-class board: four ARM cores, memory shared by everything, sealed in a passive enclosure on a pole, with no CUDA and no Intel GPU. The runtime has to install small, with no TensorFlow or torch at inference time. It needs INT8 kernels tuned for ARM, memory planned for fixed shapes, and support for **full-integer** models with int8 inputs and outputs, which is what board accelerators typically expect. LiteRT, the new name for TFLite, does all of that. The `tflite-runtime` package stopped at 2.14.0, so install `ai-edge-litert` and use `from ai_edge_litert.interpreter import Interpreter` (https://developers.google.com/edge/litert/migration).

| s10 row | what it is |
|---|---|
| `tflite-fp32` | float32 flatbuffer on XNNPACK; strict parity against the ONNX student |
| `tflite-int8-full` | INT8 weights and activations, int8 input/output tensors, per-tensor scales |
| `tflite-int8-full-nms` | the same, with top-K + Fast NMS in the graph and fast resize |
| `tflite-fp16` | recorded as `failed` with LiteRT's reason if the file won't load (it didn't while this stage was built) |
| `raspberry-pi-5` | a `not_run` placeholder; section 3.7 explains how to measure on the device |

All rows share the parent `s07_distillation:student-distilled-onnx`.

## 2. Mental model

```
student ONNX  (NCHW, dynamic batch)
   |  onnx2tf: onnxsim -> NCHW-to-NHWC rewrite -> static batch 1 -> flatbuffer_direct writer
   v
tflite_<name>/<name>_float32.tflite
             /<name>_float16.tflite
             /<name>_full_integer_quant.tflite       calibrated on <name>.calib.npy
   |  Interpreter(model_path, num_threads=4) -> allocate_tensors -> set_tensor / invoke / get_tensor
   v
XNNPACK delegate on the CPU
```

Pick the conversion tool by which direction it converts:

| tool | direction | status |
|---|---|---|
| tf2onnx | TensorFlow to ONNX | the wrong direction for us |
| onnx-tf | ONNX to TensorFlow | "not actively maintained": https://github.com/onnx/onnx-tensorflow |
| onnx2tf 2.6.8 | ONNX to TFLite | MIT; the default `flatbuffer_direct` backend writes `.tflite` with no TensorFlow: https://pypi.org/pypi/onnx2tf/json |
| litert-torch (formerly ai-edge-torch) | PyTorch to TFLite | requires torch<2.14: https://pypi.org/pypi/litert-torch/json |

onnx2tf pins onnx 1.20.1, onnxruntime 1.26.0 and ai-edge-litert 2.1.2 exactly (https://pypi.org/pypi/onnx2tf/json). The main stack runs onnx 1.22.0 and onnxruntime 1.30.0, so the two can't resolve in one environment. That's why s10 has its own `.venv-edge`.

**What people get wrong:**

- **"Full-integer means floats are gone."** The letterboxed frame is still float. Something has to quantize it at the input and dequantize the outputs. In our adapter that's Python code, and it runs inside the `detect` phase clock.
- **"The float16 file is the fast CPU option."** Float16 mainly makes the file smaller. Ours also kept float16 input tensors, and LiteRT's CPU convolution kernel rejects those.
- **"The throughput curve shows batching."** The converted models have a fixed batch of 1. The adapter loops over frames and crops, so `fps (batch)` measures a loop.
- **"Per-channel INT8 is always an option."** onnx2tf's validator rejected it here (gotcha 2).
- **"A laptop row under 30 ms answers the edge question."** It doesn't. See section 4.

## 3. Runnable walkthrough

### 3.1 A separate environment

```bash
make setup-edge   # python3.12 -m venv .venv-edge; onnx2tf 2.6.8, onnx 1.20.1, onnxruntime 1.26.0, ai-edge-litert 2.1.2
make s10          # runs the stage with .venv-edge/bin/python
```

If you run the stage from the main `.venv`, every row is recorded as `not_run` with the reason `onnx2tf / ai-edge-litert not in this interpreter`.

### 3.2 Convert (`snippet:onnx2tf-convert`)

```python
os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"  # onnx2tf calls `onnxsim`
np.save(out.with_suffix(".calib.npy"), calib.transpose(0, 2, 3, 1))  # representative data, NHWC like the model
onnx2tf.convert(
    input_onnx_file_path=str(art(onnx_name)), output_folder_path=str(out),
    batch_size=1,  # TFLite wants static shapes; the dynamic ONNX batch becomes 1
    copy_onnx_input_output_names_to_tflite=True,
    output_integer_quantized_tflite=True, quant_type="per-tensor",
    custom_input_op_name_np_data_path=[[input_name, str(out.with_suffix(".calib.npy")),
                                        [[[[0.0] * channels]]], [[[[1.0] * channels]]]]],
```

- **`PATH`**: onnx2tf runs the `onnxsim` CLI, which lives in the venv's `bin/`.
- **`batch_size=1`**: TFLite plans memory for static shapes.
- **`copy_onnx_input_output_names_to_tflite`** keeps the names `images`, `scores`, `boxes`, `detections`, `crops` and `logits`. The adapter looks outputs up by name (gotcha 7).
- **`output_integer_quantized_tflite` + `quant_type="per-tensor"`** writes the full-integer file next to the float ones.
- **`custom_input_op_name_np_data_path`** supplies the representative data: an NHWC `.npy`, a per-channel mean and a per-channel std. Our calibration arrays are already preprocessed the way the pipeline does it (`src/datasets/calibration.py`, `snippet:calibration-set`, stratified, from the `calib` split). So the mean is 0 and std is 1, which leaves them unchanged. Passing real normalization constants here would normalize the data twice, and every range would be calibrated on inputs the model never sees.

### 3.3 Full-integer quantization needs representative data

Weight-only quantization can be done from the weights alone. Full-integer quantization also fixes a scale and zero point for every **activation**, and those ranges only come from running real inputs through the float model. TensorFlow's own converter states the same requirement: full-integer quantization needs a `representative_dataset`, and asking for `TFLITE_BUILTINS_INT8` without one fails (https://ai.google.dev/edge/litert/models/post_training_integer_quant). onnx2tf takes the same data as the `.npy` above. Guide 07's rule still holds: calibrate on stratified frames, never on `val`.

### 3.4 Verify every file loads

onnx2tf can raise after writing files, and a written file isn't necessarily loadable. So s10 passes every file through `loads`, which constructs an `Interpreter` and calls `allocate_tensors`. A failure is recorded with `failed(...)` and LiteRT's error text, never silently skipped. Files that pass go to the parity gate: strict for `tflite-fp32`, report-only for the INT8 rows.

### 3.5 INT8 at the boundary: `src/backends.py :: TfliteBackend._invoke`

```python
inp = interp.get_input_details()[0]
scale, zero = inp["quantization"]
if inp["dtype"] == np.int8:  # full-integer model: quantize at the boundary
    x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
interp.set_tensor(inp["index"], x)
interp.invoke()
for det in sorted(interp.get_output_details(), key=lambda d: d["name"]):
    y = interp.get_tensor(det["index"])
    if det["dtype"] == np.int8:
        s, z = det["quantization"]
        y = (y.astype(np.float32) - z) * s
```

Each int8 tensor carries its own `(scale, zero_point)` in the interpreter's details. Quantize on the way in with `round(x / scale + zero)` and clip to the int8 range. Dequantize on the way out with `(q - zero) * scale`. `detect` also transposes every frame from NCHW to NHWC and loops frame by frame (`f[None].transpose(0, 2, 3, 1)`). `ocr` loops crop by crop.

### 3.6 XNNPACK and hardware delegates

LiteRT creates the XNNPACK delegate by default on CPU. The conversion log shows it: `INFO: Created TensorFlow Lite XNNPACK delegate for CPU.` The adapter passes `num_threads=config.THREADS` (4), which matches a Pi 5's core count and every other runtime's row. Two notes on accelerator paths:

- **NNAPI** (Android's accelerator API) is deprecated as of Android 15 (https://developer.android.com/ndk/guides/neuralnetworks/migration-guide). Don't start a new design on it.
- **Coral Edge TPU**: the runtime repository was archived on 19 April 2026 (https://github.com/google-coral/edgetpu). Treat it as a legacy target before you build a product on it.

### 3.7 Measuring on the device

This machine can't measure a Raspberry Pi, so the `raspberry-pi-5` row says so. The Pi needs the same commit, the same profile's data (the splits are hash-verified), and the student ONNX artifacts, because s10 converts them and gates against them:

```bash
# on the Pi 5: 64-bit OS, Python 3.12, a checkout of the same commit (including results/results.json)
make setup-edge
rsync -a laptop:session_5/data/synthetic/quick/ data/synthetic/quick/
rsync -a laptop:session_5/artifacts/quick/      artifacts/quick/
ANPR_PROFILE=quick .venv-edge/bin/python -m src.stages.s10_tflite_edge
```

Call the module directly. `make s10` depends on the phony `s07` target and would retrain distillation on the Pi first. The `tflite-*` rows then land under **the Pi's own hardware heading**, because the hardware id is built from CPU, GPU and thread count. The `raspberry-pi-5` placeholder row is written `not_run` on every machine, the Pi included, so read the Pi's heading and ignore that row. `results.record` replaces only rows with the same id, machine and profile. Copy `results/results.json` back and run `make table` to render both machines side by side.

## 4. Measured result

<!-- results:stage:s10_tflite_edge -->
<!-- /results -->

**Reading the rows:**

- **`tflite-fp32`** passed strict parity, or it wasn't benchmarked. It's the conversion-only reference, so compare it with its ONNX parent to see what LiteRT and XNNPACK change on the same CPU.
- **`tflite-int8-full` vs `tflite-fp32`.** Check accuracy first, per condition (`accuracy.per_condition`), and remember these are **per-tensor** scales. Guide 07's per-tensor vs per-channel rows (ORT, s06a) show the accuracy cost of that choice on this data. Then look at `detect` in `phases_ms`, which includes the Python quantize/dequantize and the NHWC transposes.
- **`tflite-int8-full-nms`** adds NMS in the graph and fast resize, the same fixes guide 12 combined into its edge candidate. Read its phase breakdown to see what's left.
- **`tflite-fp16`**: if its status is `failed`, the reason is gotcha 4.
- **`fps (batch)`** for every TFLite row is a per-frame loop, not a batched kernel.

### The closing question: did any row meet 30 ms on the target?

The `≤ SLA` column compares p95 against 30 ms **on the machine that produced the row**, and the header above each table names that machine. The first results for this repo come from an Apple M3 Pro laptop CPU. That isn't a Raspberry Pi 5: the cores, memory bandwidth and thermal envelope all differ, and the laptop isn't sealed in an enclosure in the sun. The RTX 3090 isn't a Jetson either. So read the table like this:

| what you see | what it means | next step |
|---|---|---|
| `yes` under the laptop or 3090 heading | a hypothesis about the target | measure on the target (3.7) |
| `no` under the laptop heading | a warning worth taking seriously | fix the dominant phase before buying hardware |
| `yes` under the target's heading, accuracy within budget of the parent | an answer | put that row behind the CI perf gate; repeat under sustained load in the enclosure |
| `yes` on the target, INT8 accuracy outside budget | per-tensor cost too high | per-channel via ORT QDQ on the device, QAT (s06b), or mixed precision (guide 08) |
| `no` on the target, `detect` dominates | the model is too big for this box | smaller student, lower input resolution (retrain), or an accelerator |
| `no` on the target, `decode`/`preprocess`/`crop` dominate | the model isn't the problem | s00's fixes, JPEG decode on the camera, a lower capture resolution |
| no row under the target's heading | the question isn't answered yet | the session isn't finished |

Check p95, not p50, because a camera that drops one frame in twenty still misses plates. Also check `process_first_call_ms` for reboots and `peak_memory_mb.rss` against the box's RAM.

## 5. Gotchas

1. **onnxsim isn't found.**
   **Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'onnxsim'`, then `Failed to optimize the onnx file.`, and conversion continues without simplification.
   **Fix:** put the venv's `bin/` on `PATH` before calling onnx2tf, as `convert` does.

2. **Per-channel INT8 fails strict validation.**
   **Symptom:** `quantized_dimension must be in range [0, 1). Was 3`.
   **Fix:** set `quant_type="per-tensor"`. It gives coarser scales, so read the accuracy cost in section 4. If per-tensor can't meet the accuracy budget, per-channel INT8 through ORT QDQ is the fallback route.

3. **The optional INT8-weight/INT16-activation variant fails after the INT8 files exist.**
   **Symptom:** `tflite/kernels/depthwise_conv.cc:159 bias->type != kTfLiteInt64 (INT32 != INT64)`, raised as `flatbuffer_direct fast path failed.`
   **Fix:** `convert` catches the `RuntimeError`, keeps the files already written, and `loads` checks each one. Never assume a converter exception means nothing usable was written, and never assume a written file loads.

4. **The float16 file won't load.**
   **Symptom:** `tflite/kernels/conv.cc:360 input_type == kTfLiteFloat32 || ... was not true. Node number 0 (CONV_2D) failed to prepare.`
   **Fix:** the float16 file kept float16 input tensors. Record the row as failed with that reason, and deploy float32 or full-integer instead.

5. **NHWC and batch 1.**
   **Symptom:** `set_tensor` raises a `ValueError` about a dimension mismatch when you feed NCHW data or more than one frame.
   **Fix:** transpose each frame to NHWC and loop. Treat the TFLite `fps (batch)` as a loop, not batching.

6. **Floats fed straight into an int8 input.**
   **Symptom:** a dtype error from `set_tensor`. Or, if you cast with `astype(np.int8)` yourself, no error at all and no detections.
   **Fix:** quantize with the input's `(scale, zero_point)` and clip, then dequantize outputs with their own pair (3.5).

7. **Outputs come back in a different order.**
   **Symptom:** `boxes` and `scores` swapped, which surfaces as a broadcasting error or nonsense boxes.
   **Fix:** convert with `copy_onnx_input_output_names_to_tflite=True` and select outputs by name. The adapter sorts by name, so `boxes` comes before `scores`. Never rely on output index order.

## 6. AV comparison callout

> **Context: full-integer INT8 on an AV-class network.**
> A YOLO-class multi-class detector or a lane-segmentation network needs full-integer conversion when the target accelerator only accepts int8 tensors.
> Per-tensor fallback costs more there than for our single-class plate detector. With more classes and wider-ranging heads, a single scale per tensor squeezes rare classes the hardest.
> Segmentation emits a dense full-resolution map. Dequantizing it at the boundary is real work every frame, while for our small detector output it's a small share of `detect`.
> A fixed batch of 1 also doesn't fit a multi-camera rig. Our one-camera roadside box can live with a per-frame loop; an AV stack usually can't.

## 7. When NOT to use this

- **The box has an NVIDIA or Intel accelerator.** TensorRT (guide 10) or OpenVINO (guide 12) compile for that silicon.
- **You need real batching** across several cameras. The converted model has batch 1. Use a runtime with dynamic shapes, or a server (guide 11).
- **The converter fights the model.** If per-channel is required for accuracy, or ops won't convert cleanly, stop patching the conversion and deploy the ONNX graph through a runtime that reads it directly.
- **Laptop rows are all you have.** Conversion is only worth its fragility once you can measure on the device.

### Alternatives, briefly

- **ONNX Runtime Mobile / ORT on ARM.** Keep the ONNX graph: no NHWC rewrite, no static batch, and s06a's QDQ INT8 (per-channel included) runs as exported. ORT publishes aarch64 wheels, so the Pi 5 runs it too. It's the first thing to try when gotchas 2 or 5 become deal-breakers.
- **Core ML** for Apple devices. coremltools 6.0 removed the ONNX converter ("Remove ONNX support.", https://github.com/apple/coremltools/releases/tag/6.0), so convert from TorchScript or `torch.export`, not from our ONNX files.

---

[← 12 — OpenVINO](12-openvino.md) · [14 — LLM serving: the same four levers →](14-llm-serving-same-levers.md)
