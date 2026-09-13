# 12 — OpenVINO: IR, NNCF INT8 and performance hints

Not every roadside box has an NVIDIA GPU. OpenVINO compiles the s07 student for x86 and ARM CPUs and for Intel GPUs. This guide converts ONNX to IR with the current API and quantizes it with NNCF on our stratified calibration set. It then compares the LATENCY and THROUGHPUT hints and assembles the **edge-candidate** row, which runs every fix from the session that composes in one pipeline. It ends with a fair way to compare a CPU box and a GPU box in money. The code is in `src/stages/s09_openvino.py` and `src/backends.py :: OpenVinoBackend`.

## 1. The problem it solves

The budget is still 30 ms per frame with no uplink. This time the hardware is cheaper, runs cooler and is easier to buy than a GPU module: an industrial PC in the roadside cabinet, or an ARM board. ONNX Runtime's CPU provider is a sound generic baseline there. OpenVINO's CPU plugin goes further on that silicon. It fuses the graph, picks kernels for the instruction set it detects at load time, runs INT8 where the CPU has integer kernels, and schedules inference across cores according to a goal you declare.

Two facts shape this stage:

- The OpenVINO GPU plugin targets **Intel** GPUs. On the RTX 3090 box, OpenVINO runs on the host CPU, and s09 records `intel-gpu` as `not_run` along with the list of devices it found.
- OpenVINO publishes aarch64 wheels, so the same IR runs on a Raspberry Pi 5's CPU (https://pypi.org/project/openvino).

| s09 row | parent | what it answers |
|---|---|---|
| `baseline-fp32-latency` | `s04_onnx_export:ort-cpu-fp32` | the big model on OpenVINO vs ORT, same CPU (strict parity) |
| `student-fp32-latency` | `s07_distillation:student-distilled-onnx` | the student with the LATENCY hint (strict parity) |
| `student-fp32-throughput` | `s09_openvino:student-fp32-latency` | same IR, THROUGHPUT hint plus `AsyncInferQueue` (strict parity) |
| `student-int8-nncf` | `s09_openvino:student-fp32-latency` | NNCF post-training INT8 (parity report only) |
| `edge-candidate` | `s09_openvino:student-int8-nncf` | INT8, NMS in the graph, and fast resize together |
| `intel-gpu` | `s09_openvino:student-fp32-latency` | runs only if OpenVINO sees an Intel GPU |

## 2. Mental model

```
ONNX --ov.convert_model--> ov.Model --ov.save_model--> IR (.xml graph + .bin weights)
                                                          |  nncf.quantize(IR, calibration frames)
                                                          v
                                               INT8 IR (FakeQuantize nodes)
                                                          |  core.compile_model(ir, "CPU", {PERFORMANCE_HINT, INFERENCE_NUM_THREADS})
                                                          v
                                  CompiledModel --> InferRequest (sync) / AsyncInferQueue (parallel)
```

The IR is portable across OpenVINO devices. The compiled model is built when the process loads it, for whatever device it finds. That is the opposite trade from TensorRT: you don't need a build matrix per device, but the plugin repeats its graph work every time a process starts. `compile_model` runs before any clock starts, and whatever it defers lands in `process_first_call_ms`.

Most OpenVINO tutorials online use tooling that no longer exists:

| old | current |
|---|---|
| Model Optimizer `mo`, `openvino-dev` | `ov.convert_model` (Python) or `ovc` (CLI): https://pypi.org/project/openvino |
| POT (last release 2023.3) | `nncf.quantize`: https://github.com/openvinotoolkit/nncf/releases |
| `from openvino.runtime import Core` | `import openvino as ov`; the namespace was removed in 2026.0: https://github.com/openvinotoolkit/openvino/releases/tag/2025.2.0 |
| NNCF TF backend, `create_compressed_model` | removed in NNCF 3.0: https://github.com/openvinotoolkit/nncf/releases |

The two performance hints are described at https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/cpu-device/performance-hint-and-thread-scheduling.html

| | `LATENCY` | `THROUGHPUT` |
|---|---|---|
| goal | the shortest time for one inference | the most inferences per second |
| streams | as few as possible; all threads work on one request | several; each stream is an executor with its share of the threads |
| how you drive it | one synchronous request at a time | many requests in flight (`AsyncInferQueue`) |
| ANPR fit | one camera, one frame at a time | a backlog, or several cameras on one box |

**What people get wrong:**

- **"An FP32 IR has FP32 weights."** `ov.save_model` defaults to `compress_to_fp16=True` and stores the weights as FP16 without telling you. s09 passes `False` for every IR it saves.
- **"An INT8 IR runs in INT8."** The IR contains FakeQuantize nodes. The plugin decides at compile time which of them execute as integer kernels, and that depends on the CPU. `compiled.get_runtime_model()` shows what actually ran.
- **"THROUGHPUT makes each frame faster."** It gives up single-request latency in exchange for parallel requests. With one camera, it only helps if you have parallel work to give it.
- **"The GPU device means any GPU."** It means Intel GPUs.

## 3. Runnable walkthrough

### 3.1 Install and run

```bash
make setup      # requirements.txt + requirements-openvino.txt: openvino 2026.3.1, nncf 3.3.0
make s09
```

On x86, this OpenVINO release requires a CPU with AVX2.

### 3.2 Convert to IR (`snippet:ov-convert`)

```python
def to_ir(onnx_name: str, xml_name: str) -> None:
    model = ov.convert_model(art(onnx_name))  # the Python API; `ovc` is the CLI. `mo` no longer exists.
    ov.save_model(model, art(xml_name), compress_to_fp16=False)  # the default (True) silently stores FP16 weights
```

s09 converts five graphs: the baseline detector and OCR, the student detector with and without in-graph NMS, and the student OCR. The equivalent CLI command is `ovc artifacts/quick/detector_student.onnx --output_model artifacts/quick/detector_student.xml --compress_to_fp16=False`. The dynamic batch axis from the ONNX export carries over into the IR. If you deploy a single fixed shape, `model.reshape([1, 3, 384, 640])` before compiling lets the plugin plan memory for exactly that shape. Measure the effect before you adopt it.

### 3.3 INT8 with NNCF (`snippet:nncf-quantize`)

```python
quantized = nncf.quantize(
    model,
    nncf.Dataset(batches),  # items are already model inputs, so no transform function is needed
    preset=nncf.QuantizationPreset.PERFORMANCE,  # symmetric weights + activations: fastest on CPU
    subset_size=len(batches),
)
ov.save_model(quantized, art(out_name), compress_to_fp16=False)
```

The nncf 3.3.0 signature, checked locally, is: `quantize(model, calibration_dataset, *, mode=None, preset=None, target_device=ANY, subset_size=300, fast_bias_correction=True, model_type=None, ignored_scope=None, advanced_parameters=None)`.

- **`subset_size` defaults to 300** and silently caps calibration. The `full` profile calibrates on 512 frames and 1024 crops (`n_calib`, `ocr_batches`). Without `subset_size=len(batches)`, most of that data would never be seen, and nothing would warn you.
- **The data** comes from the same `calib` split and stratified strategy as guide 10 (`snippet:calibration-set`): equal frames for day, night, rain, motion blur and low contrast. It never comes from `val`.
- **`nncf.Dataset(items, transform_fn)`**: our items are already model inputs. If yours are dataloader tuples, pass a `transform_fn`.
- **`preset`**: `PERFORMANCE` uses symmetric activations. `MIXED` uses asymmetric activations, which suits activations that aren't one-sided, such as the hard-swish in MobileNetV3 blocks. It's the first knob to try if the INT8 row loses accuracy.
- **`ignored_scope`** keeps named nodes in floating point. Guide 08's sensitivity tooling (`snippet:layer-sensitivity`) tells you which nodes to list.

### 3.4 Compile with a hint

```python
props = {"PERFORMANCE_HINT": hint, "INFERENCE_NUM_THREADS": config.THREADS}
self.det = core.compile_model(str(artifact(detector)), ov_device, props)
self.rec = core.compile_model(str(artifact(ocr)), ov_device, props)
```

Threads are pinned to `ANPR_THREADS` (4), which matches the ORT rows and the core count of the edge boards. The hint then decides how to use those four threads. To see what it chose, read `compiled.get_property("NUM_STREAMS")` and `compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")`.

### 3.5 `AsyncInferQueue` (`snippet:ov-async-queue`)

```python
if self.queue and len(x) > 1:  # one request per frame, run concurrently on the plugin's streams
    self._pending = [None] * len(x)
    for i, frame in enumerate(x):
        self.queue.start_async({0: frame[None]}, userdata=i)
    self.queue.wait_all()
    outs = [np.concatenate(o) for o in zip(*self._pending)]
```

The queue owns a pool of infer requests (four here, via `async_jobs`). Each frame of a batch becomes its own request and runs on one of the plugin's streams. The callback, `_collect`, stores each result under its `userdata` index, so the output order matches the input order no matter which request finishes first. The callback **copies** the output (`.data.copy()`) because a request reuses its output buffer on its next inference.

This path runs only when the batch has more than one frame, which happens in the throughput sweep. The latency pass runs at batch 1 and uses the synchronous request, even for the `student-fp32-throughput` row. So that row's p50/p95 shows what the THROUGHPUT hint does to one frame, and its `fps (batch)` column shows the async fan-out.

### 3.6 The edge candidate: every fix that composes

| fix | from | phase it changes |
|---|---|---|
| distilled student (MobileNetV3-Small detector, conv-only OCR) | s07, guide 09 | `detect`, `ocr` |
| NNCF INT8 | this guide | `detect`, `ocr` |
| top-K + Fast NMS in the graph (`snippet:nms-in-graph`) | s00 | `nms` becomes a threshold filter over `[B,50,5]` |
| `Image.reduce(2)` resize (`snippet:letterbox`) | s00 | `preprocess` |

These fixes stack because each one changes a different phase. The exception is the NMS subgraph, which is quantized along with the rest of the detector. The row's parity report and accuracy pass show whether its IoU arithmetic survived INT8. If it didn't, add those nodes to `ignored_scope`. The THROUGHPUT hint is deliberately left out: one camera needs LATENCY.

## 4. Measured result

<!-- results:stage:s09_openvino -->
<!-- /results -->

**Reading the rows:**

- **`baseline-fp32-latency` against its s04 parent.** Same model, same CPU, same threads, different runtime. This is the cleanest runtime-only comparison in the stage.
- **LATENCY vs THROUGHPUT.** Compare the latency columns (one frame under each compile config) separately from the `fps (batch)` column (fan-out across streams). For a one-camera site, deploy the row whose p95 meets the SLA. A higher fps number doesn't matter for a single camera.
- **INT8 vs FP32.** Start with accuracy: open `accuracy.per_condition` and compare the night and low-contrast slices, not only the overall mAP. Then compare the `detect` phase in `latency.phases_ms`. If INT8 isn't faster on this CPU, check which precision actually executed (`get_runtime_model()`) before drawing conclusions. The Apple M3 Pro producing this repo's first results is an ARM CPU, so results on an x86 box with integer-accelerating instruction sets can differ.
- **`edge-candidate`.** Check which phase dominates `phases_ms` now. That phase is where the remaining budget goes, and it tells you which guide to reopen.
- **`intel-gpu`.** The reason field lists the devices OpenVINO found.

### Reading it as money: a CPU box vs a GPU box

`src/cost.py` (`snippet:cost-per-million`) turns a measured curve into dollars. It never uses a price of its own:

```
cost_per_million = usd_per_hour / (usable_fps * 3600) * 1_000_000
usd_per_hour     = box_price / (lifetime_years * 8760)  +  (watts / 1000) * usd_per_kWh     # an owned roadside box
```

`usable_fps` is the highest-throughput point whose p95 still meets 30 ms (`usable_point`). A point that violates the SLA returns no cost, not a cheap one. The journey table's `$/1M frames @ $1/h` column uses a price of $1 per hour, so multiply it by your own hourly figure.

```bash
python -m src.cost --fps <fps> --p95-ms <p95> --batch 1 --usd-per-hour <your CPU box $/h>
python -m src.cost --fps <fps> --p95-ms <p95> --batch 1 --usd-per-hour <your GPU box $/h> --cameras 4 --camera-fps 5
```

The comparison is only fair if you follow these rules:

1. **Compare rows measured on the boxes you would buy.** The course RTX 3090 is not a roadside GPU module, and a laptop CPU is not the cabinet PC. Rows from the wrong hardware produce a clean-looking number for the wrong question.
2. **Read fps and p95 from one hardware heading.** `--row <id>` takes the last matching row in `results.json` regardless of machine. Once the file holds several machines, pass `--fps` and `--p95-ms` by hand from the table you mean.
3. **At the edge, the unit is the box.** The question is whether one box keeps p95 within 30 ms for the number of cameras at that site. `--cameras` and `--camera-fps` show how many boxes a site needs.
4. **Use your own prices,** amortized over your planned lifetime, with power at the site's tariff, plus cooling and spares.

## 5. Gotchas

1. **FP16 weights in an "FP32" IR.**
   **Symptom:** the `.bin` is noticeably smaller than the ONNX weights, and strict parity reports tensor differences on a row that should be exact.
   **Fix:** pass `compress_to_fp16=False` to `ov.save_model`, or to `ovc` on the command line.

2. **Tutorial code from the old API.**
   **Symptom:** `ModuleNotFoundError: No module named 'openvino.runtime'`, `mo: command not found`, or `ImportError` for POT.
   **Fix:** use `import openvino as ov`, `ov.convert_model` or `ovc`, and `nncf.quantize`. Don't install `openvino-dev`.

3. **Calibration silently capped.**
   **Symptom:** the NNCF progress bar stops at 300 samples when you passed more.
   **Fix:** set `subset_size=len(batches)`, as s09 does.

4. **Every frame in a batch returns the same detections.**
   **Symptom:** the async throughput path produces duplicated or shifted results. The strict gate on `student-fp32-throughput` fails, but only at batch sizes above 1.
   **Fix:** copy the output tensors inside the callback, and key results by `userdata`, not by completion order.

5. **THROUGHPUT chosen for a single camera.**
   **Symptom:** the throughput row has the better `fps (batch)`, but its batch-1 p95 is worse than the LATENCY row's.
   **Fix:** you're reading the trade the hint was designed to make. One camera means LATENCY. Use THROUGHPUT when there is parallel work to feed it.

6. **Asking for `GPU` on an NVIDIA box.**
   **Symptom:** `compile_model(..., "GPU")` fails, and `ov.Core().available_devices` lists only `CPU`.
   **Fix:** OpenVINO's GPU plugin is for Intel GPUs. On NVIDIA hardware, use TensorRT (guide 10) and let OpenVINO cover the CPU.

7. **INT8 loses accuracy on hard conditions.**
   **Symptom:** `student-int8-nncf` holds overall mAP but drops on `night` or `low_contrast` in `accuracy.per_condition`.
   **Fix:** check the calibration strategy first. Then try `preset=MIXED`, then move sensitive layers into `ignored_scope`. Output heads are the usual suspects (guide 06's `snippet:never-prune-output-layers` explains why).

8. **An "FP32" IR that runs in FP16.** **Symptom:** the strict parity gate stops s09 on its first FP32 row with `Not equal to tolerance rtol=0.001, atol=1e-05` and `Mismatched elements: 12718 / 15360 (82.8%)` — small, systematic differences in every detector score, although nothing was quantized. **Cause:** the CPU plugin chooses its own inference precision per device; on the ARM laptop that produced the committed results, `core.get_property("CPU", "INFERENCE_PRECISION_HINT")` returned `float16`. **Fix:** pass `INFERENCE_PRECISION_HINT: "f32"` for any FP32 comparison (`snippet:ov-precision-hint`). s09 keeps a `student-device-default-precision` row, and every OpenVINO row records the precision that actually ran in `inference_precision`. Hit while building s09.

## 6. AV comparison callout

> **Context: the same workflow on an AV-class network.**
> A YOLO-class multi-class detector or a lane-segmentation network goes through the same steps: convert, calibrate with NNCF on scenario-stratified frames, and compile with a hint. These models mostly run on x86 compute, either for offline log processing or on in-vehicle Intel platforms.
> Segmentation heads end in large upsampling and argmax layers. Those are cheap to keep in floating point and expensive to get wrong in INT8, so they go straight into `ignored_scope`.
> A multi-camera rig has real parallel work, so THROUGHPUT streams plus an async queue per camera are a natural fit.
> Our roadside unit has one camera and one frame at a time, so it uses the LATENCY hint and a synchronous request. The same hint would be the wrong choice for the rig.

## 7. When NOT to use this

- **The box has an NVIDIA GPU.** TensorRT compiles for that silicon, and OpenVINO can only use the host CPU.
- **The target is an accelerator OpenVINO doesn't drive,** such as a vendor NPU or a DSP. Use that vendor's toolchain, or a LiteRT delegate (guide 13).
- **You want one runtime across very different devices, with minimal code.** ONNX Runtime with the OpenVINO execution provider (s04's `ort-openvino-ep-cpu` row) keeps the ORT API and gets most of the plugin's work done for you.
- **The CPU phases dominate.** If `decode` and `crop` own the frame, a faster model barely moves p95. Fix those phases first (s00).

---

[← 11 — Triton Inference Server](11-triton-inference-server.md) · [13 — TFLite (LiteRT) and the edge →](13-tflite-and-edge.md)
