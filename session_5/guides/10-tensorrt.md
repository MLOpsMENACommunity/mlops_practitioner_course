# 10 — TensorRT: compiling the student into a GPU engine

In guide 05, ONNX Runtime's TensorRT execution provider compiled the ONNX graph for you. This guide does the same compile by hand. We parse the s07 student, declare the shapes and precisions we want, calibrate INT8 on our own frames, and then open the engine to see what TensorRT fused. The stage lives in `src/stages/s08_tensorrt.py` and the runtime adapter is `src/backends.py :: TensorRTBackend`. It needs an NVIDIA GPU and TensorRT 10.x. On any other machine every row is recorded `not_run` with the reason, and the guide reads correctly either way.

## 1. The problem it solves

This is the roadside ANPR camera. A 1280x720 JPEG is letterboxed to 384x640. A single-class anchor-free detector finds plates, NMS removes duplicates, and each plate is cropped from the full-resolution frame. The crop is resized to 32x128 grayscale and a CRNN/CTC recognizer reads it. Everything from decode to plate string has to fit in **30 ms per frame at p95**, on a Jetson-class box with no uplink.

On an NVIDIA GPU, ONNX Runtime's CUDA provider runs the student graph node by node with general-purpose kernels. TensorRT instead compiles the graph for **one GPU**:

- it **fuses** layers, so there are fewer kernel launches and fewer intermediate tensors;
- it **times candidate kernels** (tactics) on the real device and keeps the fastest;
- it **picks a precision per layer** from the ones you allow;
- it **plans memory** for the whole graph up front.

The output is a `.plan` engine that runs only on that GPU model with that TensorRT version. This guide is about that trade.

| s08 row | parent | what it answers |
|---|---|---|
| `baseline-fp16` | `s04_onnx_export:ort-cpu-fp32` | what TensorRT does for the big ResNet-34/BiLSTM pair |
| `student-fp32` | `s07_distillation:student-distilled-onnx` | does the engine compute what ONNX computes? (strict gate) |
| `student-fp16` | same | what FP16 kernels buy on this GPU |
| `student-int8-stratified` | same | implicit INT8, calibrated on equal frames per condition |
| `student-int8-daytime` | same | the same calibrator, fed daytime frames only |

FP8 is out of scope. It needs an Ada (SM 8.9) or newer GPU (TensorRT 10.3 added FP8 convolution on Ada: https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html). The course RTX 3090 is Ampere (SM 8.6), so it gets FP16 and INT8 but not FP8.

## 2. Mental model

Think of the ONNX file as source code and the engine as the binary compiled from it for one machine.

| | ONNX graph | TensorRT engine |
|---|---|---|
| contains | ops and weights | fused layers, chosen kernels, per-layer precision, memory plan |
| portable across | runtimes, OSes, devices | nothing: one GPU model + one TensorRT version + its shape profiles |
| produced by | `torch.onnx.export` (s04, s07) | `builder.build_serialized_network` |
| shape freedom | the axes exported as dynamic | only inside an optimization profile's `[min, max]` |

**What people get wrong:**

- **"Copy the engine to the Jetson."** It won't deserialize on another GPU or TensorRT version. Rebuild it on the box that serves.
- **"The FP16 flag makes every layer FP16."** In TensorRT 10 the FP16 and INT8 flags *permit* those precisions. The builder still chooses per layer by measured speed.
- **"`execute_async_v3` returned, so inference is done."** It returns once the work is queued. A timer stopped at that point measures submission, not inference.
- **"Calibration trains the model."** No weights change. Calibration only records activation ranges on the frames you give it.

> **Why the EP is the low-code path.** The TensorRT EP in s04 (`snippet:trt-execution-provider`) partitions the graph and hands the supported subgraphs to TensorRT. It builds and caches engines on first use (`trt_engine_cache_enable`) and runs the rest on CUDA or CPU. It handles parsing, shapes, caching and fallback for you. What it hides is which layers fused, which precision each layer got, and what data calibrated INT8. If the EP row meets the SLA and the accuracy budget, stop there. Build by hand when you need to control those decisions or see them.

> **TensorRT 10 vs 11: why this course pins 10.16.1.11.**
> NGC 26.05 is the last monthly release that ships TensorRT 10.x. Both its tensorrt and tritonserver images carry 10.16.1.11 on CUDA 13.2.1: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-05.html
> TensorRT 11.0 removed implicit quantization: `IInt8Calibrator` and its subclasses, `setDynamicRange`, and the weak-typing FP16/INT8/FP8 builder flags: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html
> trtexec 11 removed `--fp16 --int8 --best --calib`: https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x-trtexec.html
> These still work in 10.x (deprecated), so s08 can teach the calibrator. On TensorRT 11, INT8 needs an explicit Q/DQ graph instead. You can produce one with s06a's QDQ export (`snippet:ort-static-quant`) run on the student, or with ModelOpt using `--calibration_data_path` (requires TensorRT >= 10.0): https://nvidia.github.io/Model-Optimizer/guides/_onnx_quantization.html

## 3. Runnable walkthrough

### 3.1 Get the pinned stack

```bash
make doctor      # CUDA device, compute capability, driver R580+, TensorRT 10.16.x
make setup-gpu   # Linux x86_64: onnxruntime-gpu 1.30.0, tensorrt-cu13==10.16.1.11, torch cu130
make s08

# or the pinned container (same TensorRT as Triton 26.05)
docker build -f ci/docker/Dockerfile.gpu -t anpr-gpu:26.05 .
docker run --rm --gpus all -v "$PWD":/workspace/session_5 anpr-gpu:26.05 make s08
```

CUDA 13.x needs host driver R580+ (https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). torch 2.13.0 has cu130 wheels (https://download.pytorch.org/whl/cu130/torch/). onnxruntime-gpu 1.30.0's TensorRT EP links `libnvinfer.so.10` (https://pypi.org/pypi/onnxruntime-gpu/json), so the whole stack stays on 10.x.

### 3.2 Build: `src/stages/s08_tensorrt.py :: build_engine` (`snippet:trt-build`)

```python
network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
parser = trt.OnnxParser(network, logger)
if not parser.parse(onnx_path.read_bytes()):
    raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)  # scratch space tactics may use
cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # keep layer names for the inspector
profile = builder.create_optimization_profile()
profile.set_shape(input_name, *shapes)  # kernels tuned for `opt`, valid anywhere in [min, max]
```

- **Parser errors.** Print *all* of them, because the first is often a consequence of a later unsupported op. The exports use opset 18 (`src/config.py :: ONNX_OPSET`), which every runtime in this session accepts.
- **Workspace** (4 GB here) is scratch memory a tactic may use. Tactics that need more are skipped. If the limit is too small, a slower tactic wins silently, or nothing fits and the build fails. On a Jetson the GPU shares RAM with the camera process, so workspace competes with it.
- **`DETAILED` profiling verbosity** keeps layer information in the engine, so you can inspect it later (3.5).
- **`build_serialized_network`** is the current builder call; the old `build_engine` is gone. It returns `None` on failure, and the stage raises with the file name and precision.

```python
if precision in ("fp16", "int8"):
    cfg.set_flag(trt.BuilderFlag.FP16)  # INT8 builds keep FP16 for layers with no INT8 kernel
if precision == "int8":
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = calibrator
    cfg.set_calibration_profile(profile)
```

INT8 builds also set FP16. Without it, a layer that has no INT8 kernel falls back to FP32 instead of FP16.

### 3.3 Optimization profiles

| engine | min | opt | max |
|---|---|---|---|
| detector `images` | `1x3x384x640` | `1x3x384x640` | `max_b x3x384x640` |
| OCR `crops` | `1x1x32x128` | `4x1x32x128` (`MAX_PLATES`) | `max_b*4 x1x32x128` |

`max_b` is the largest of the profile's `throughput_batches`. Kernels are **timed at `opt`**. Any shape between min and max still runs, but on kernels chosen for `opt`. The detector's `opt` is 1 because a camera delivers one frame at a time, which is the SLA row. OCR's `opt` is 4 because a frame yields zero to four crops. If a second shape matters as much, add a second profile and select it per context with `set_optimization_profile_async`.

### 3.4 INT8 calibration on our frames (`snippet:int8-calibrator`)

```python
def get_batch(self, names: list[str]) -> list[int] | None:
    if self.i == len(self.batches):
        return None  # tells TensorRT calibration is finished
    self.buffer.copy_(torch.from_numpy(self.batches[self.i]))
    self.i += 1
    return [int(self.buffer.data_ptr())]

def read_calibration_cache(self) -> bytes | None:
    # A cache from a different calibration set is silently reused. Delete it when the data changes.
    return self.cache.read_bytes() if self.cache.exists() else None
```

TensorRT runs the network over every batch and histograms each tensor. `IInt8EntropyCalibrator2` then picks the per-tensor range that minimizes KL divergence. `get_batch` returns **device pointers**, so each batch is copied into a preallocated torch CUDA tensor. Frames come from the `calib` split, never `val` (`snippet:calibration-set`). They are grouped into batches of 8 frames and 16 crops, and those batch sizes must fit inside the calibration profile.

The two INT8 rows differ only in data. `stratified` takes equal frames from each of day, night, rain, motion_blur and low_contrast. `daytime` takes day frames only. Each strategy writes its own cache (`artifacts/<profile>/calib_{det,ocr}_<strategy>.cache`).

> **Why a daytime row.** Daytime frames are the easiest to collect, and a daytime-calibrated engine can look fine on overall mAP. This row makes that mistake visible. The damage shows up in the night and low-contrast slices, which nobody checks until the night shift complains.

### 3.5 Look inside the engine

```python
info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
names = [layer if isinstance(layer, str) else layer.get("Name", "") for layer in info["Layers"]]
return {"layers": len(names), "fused": [n for n in names if "+" in n][:12]}
```

`inspect_layers` records each detector engine's layer count and its first fused names in the row's `notes`. TensorRT joins fused layer names with `+`. Compare the layer count against the ONNX node count to see how much was fused, and compare the INT8 notes against FP16 to see whether quantization changed what could fuse. As a sanity check outside Python, TensorRT 10.16's trtexec builds the same engine:

```bash
trtexec --onnx=artifacts/quick/detector_student.onnx \
  --minShapes=images:1x3x384x640 --optShapes=images:1x3x384x640 --maxShapes=images:8x3x384x640 \
  --fp16 --saveEngine=/tmp/det.plan --dumpLayerInfo --profilingVerbosity=detailed
```

If trtexec builds and s08 doesn't, look at your Python config, not the graph. `trtexec --int8` without a calibration cache is a speed probe only, so never read accuracy from it. trtexec timings never become `results.json` rows either.

### 3.6 Run it: `src/backends.py :: _TrtEngine.__call__`

```python
self.ctx.set_input_shape(self.inputs[0], tuple(x.shape))
self.ctx.set_tensor_address(self.inputs[0], x.data_ptr())
for name in self.outputs:
    dtype = self.dtypes[self.engine.get_tensor_dtype(name)]
    outs[name] = torch.empty(tuple(self.ctx.get_tensor_shape(name)), dtype=dtype, device="cuda")
    self.ctx.set_tensor_address(name, outs[name].data_ptr())
stream = torch.cuda.current_stream()
self.ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
```

This is TensorRT 10's tensor-address API: set the input shape, which must lie inside the profile; bind every I/O tensor by name; enqueue on a CUDA stream. Torch CUDA tensors serve as the buffers, so there's no pycuda dependency. `stream.synchronize()` makes the timing honest, and the harness also calls `backend.sync()` at every phase lap (`snippet:timing-loop`). Both the host-to-device and device-to-host copies happen inside `detect`.

### 3.7 The parity gate

`common.gate` compares each student engine against the ONNX Runtime CPU student (`snippet:parity-gate`). It checks raw tensors, post-NMS boxes and decoded strings. `student-fp32` is **strict**: tensors within `rtol=1e-3, atol=1e-5`, box IoU of at least 0.99, and identical strings. Otherwise the row is recorded `failed` and the stage stops. FP16 and INT8 are report-only because they are expected to move tensors, and the benchmark judges their accuracy. `baseline-fp16` has no student reference and is not gated.

## 4. Measured result

<!-- results:stage:s08_tensorrt -->
<!-- /results -->

**If the rows say `not_run`,** the reason column says why: no CUDA device, `tensorrt` not installed, or a TensorRT that isn't 10.x. That is a correct result for that machine. Run `make s08` on the RTX 3090 and then `make table`, and a second table appears under the 3090's hardware heading. Machines never share a table.

**Once the rows are measured:**

- **`student-fp32`: check parity before speed.** Its parent is an ORT *CPU* row, so their latency gap mixes device and runtime. For runtime alone on the same GPU, compare `baseline-fp16` against s04's `ort-tensorrt-ep-fp16` and `ort-cuda-fp32` under the same heading.
- **FP32 to FP16.** Look at `detect` in `latency.phases_ms`, not only the total. If `detect` shrinks but p95 barely moves, the frame is spending its time in CPU phases, and the next fix is in s00.
- **Stratified vs daytime.** The table shows only overall accuracy. Open `accuracy.per_condition` for both rows and compare `night`, `rain` and `low_contrast`. If the rows agree, calibration coverage didn't matter for this model and data. If daytime loses on the hard slices, that is range clipping.
- **`notes`** hold the fusion evidence. **`peak_memory_mb.vram`** is per-process NVML (`null` when unmeasured). **`process_first_call_ms`** is what a rebooting camera pays for its first frame (engine deserialization itself happens before that clock starts).

## 5. Gotchas

1. **An engine built elsewhere won't load.**
   **Symptom:** `engine failed to deserialize — built with another TensorRT version or GPU?`
   **Fix:** rebuild on the serving box. JetPack 7.2 for Jetson Orin ships TensorRT 10.16.2 on CUDA 13.2.1 (https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2), and JetPack 6.2.2 ships TensorRT 10.3. Never copy the 3090's `.plan` to a Jetson. Triton must also ship the same TensorRT (26.05 for this repo).

2. **The clock stopped before the GPU finished.**
   **Symptom:** an implausibly fast `detect` phase, while a later phase absorbs the missing time.
   **Fix:** synchronize the stream before reading the clock, in any timing code you write.

3. **The calibration cache is silently reused.**
   **Symptom:** you changed the calibration data, and the INT8 build is suspiciously quick with accuracy identical to the last run.
   **Fix:** `read_calibration_cache` returns any cache it finds, and TensorRT trusts it. Delete `artifacts/<profile>/calib_*.cache` whenever the data changes.

4. **A shape falls outside the profile.**
   **Symptom:** `set_input_shape` rejects a batch size you never declared, or an INT8 build fails because a calibration batch exceeds the profile max.
   **Fix:** derive `max` from the batch sizes you actually run, and keep the calibration batches (8 frames, 16 crops) at or below it. Check this first if you shrink `throughput_batches`.

5. **Strict parity fails on the FP32 engine.**
   **Symptom:** `PARITY GATE FAILED for s08_tensorrt:student-fp32` with small tensor differences.
   **Fix:** read the report before you loosen any tolerance. If only tensors drift, check `cfg.get_flag(trt.BuilderFlag.TF32)`; clearing it rules out TF32 math on Ampere. If boxes or strings differ, suspect the graph and bisect it with trtexec.

6. **TensorRT 11, or two TensorRTs on the path.**
   **Symptom:** `not_run` with `TensorRT 11.x: this stage targets 10.x`, or `AttributeError` on `IInt8EntropyCalibrator2`.
   **Fix:** use the pin in `requirements-gpu.txt`. Inside `ci/docker/Dockerfile.gpu`, the venv reuses the image's TensorRT (`--system-site-packages`), so never install `tensorrt-cu13` on top. If you need to stay on 11, switch to a Q/DQ graph.

## 6. AV comparison callout

> **Context: the same build for an autonomous-vehicle perception stack.**
> A YOLO-class multi-class detector or a lane-segmentation network goes through the same steps: parse, profile, calibrate, inspect. What differs is the workload you declare and what calibration has to cover.
> A vehicle has several cameras sharing one accelerator, so `opt` is the camera count, not 1. Segmentation produces dense full-resolution outputs, which moves workspace and memory-bandwidth pressure to the head of the network.
> Calibration must span the whole operating domain and be checked per class, because a clipped range can erase a rare class without moving overall mAP.
> Our plate detector is one class, one camera, one frame at a time. Two lessons carry over unchanged: stratified calibration, and rebuilding on the target.

## 7. When NOT to use this

- **The target has no NVIDIA GPU.** Use OpenVINO (guide 12) or LiteRT (guide 13).
- **You can't build on every target device type.** A per-GPU, per-TensorRT engine matrix is a real burden. The ORT TensorRT EP with its engine cache, or plain ORT CUDA, is easier to maintain.
- **The model isn't the bottleneck.** If decode, letterbox and crop dominate the frame, a faster engine barely moves p95. Fix s00 first.
- **The EP row already meets the SLA and accuracy budget.** A hand-built engine adds code, a calibration cache and a rebuild step to your release.
- **You need INT8 on TensorRT 11.** The calibrator taught here doesn't exist there.

---

[← 09 — Knowledge distillation](09-knowledge-distillation.md) · [11 — Triton Inference Server →](11-triton-inference-server.md)
