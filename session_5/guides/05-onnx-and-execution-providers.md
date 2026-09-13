# 05 — ONNX and execution providers

ONNX is a file **format**. ONNX Runtime is a **runtime** that executes it. An execution provider is the **backend** the runtime hands the work to: CPU, CUDA, TensorRT or OpenVINO. Keep those three words separate and most ONNX confusion goes away. This guide exports the baseline detector and recognizer once, checks and gates the graphs, then runs the *same* `.onnx` files through every provider the machine has. Every later stage (static INT8, TensorRT, OpenVINO, LiteRT, Triton) starts from files exported by this code.

## 1. The problem it solves

The roadside box will not run your training environment. A Raspberry Pi-class box runs ONNX Runtime or LiteRT on four cores. A Jetson-class box runs TensorRT. Neither has room for a full PyTorch install, a Python model class, or your training repo. Both still have to meet 30 ms per frame with no uplink to fall back on.

ONNX separates *where the model was trained* from *where it runs*. You export once. Every runtime on the deployment matrix reads that file, and the parity gate proves the file still computes what PyTorch computed. That last part only holds if the model and everything around it (letterbox, NMS, crop, CTC decode) are split the same way on both sides.

## 2. Mental model

### Export traces a program; it does not copy a model

Since PyTorch 2.9, `torch.onnx.export` uses the dynamo exporter by default. It runs `torch.export` to capture the forward pass as a graph, then translates that graph into ONNX operators (https://docs.pytorch.org/docs/2.14/onnx_export.html). Capture uses an example input, which has two consequences:

- **Dimensions are constants unless you declare them dynamic.** Leave out the batch dimension and the file only accepts the example's batch size.
- **An example value of 1 gets specialized.** torch.export treats a size-1 dimension as a constant even when you declared it dynamic. That is why `export_all` uses an example batch of **2**.

Here is the spelling this session uses (`snippet:onnx-export`):

```python
        dynamic_shapes={"x": {0: batch}},  # the successor of dynamic_axes; omit it and batch is frozen at 2
```

Session 1's toy export ([`session_1/pytorch_to_onnx.py`](../../session_1/pytorch_to_onnx.py)) used the older spelling:

```python
    dynamic_axes={  # allow variable batch size at inference
        "features": {0: "batch_size"},
        "duration": {0: "batch_size"},
    },
```

`dynamic_axes` is deprecated in favour of `dynamic_shapes`, and `dynamo=False` prints a legacy-exporter warning (https://docs.pytorch.org/docs/2.14/onnx_export.html). Note the change of key. `dynamic_axes` was keyed by ONNX `input_names`. `dynamic_shapes` is keyed by the **Python argument name of `forward`**: that is `x` in both `DetectorExport.forward(self, x)` and `CRNN.forward(self, x)`, even though the ONNX inputs are named `images` and `crops`. Session 1 also exported with a batch-1 dummy input. If you rerun it on this torch, open the result and check that the batch dimension is a symbol, not a number.

`onnx.checker.check_model(..., full_check=True)` validates the graph structure and runs shape inference. It says nothing about whether the numbers are right. That is the parity gate's job.

### The file is the model, not the pipeline

This is the part people get wrong. Two perfect exports still disagree if the code *around* them differs.

| Step | In the ONNX graph? | Where it lives |
|---|---|---|
| JPEG decode, letterbox to 384x640 | **No** | `src/preprocess.py` (`snippet:letterbox`) |
| Scale to [0,1], HWC to NCHW | **No** | `src/preprocess.py :: to_nchw` |
| Backbone, neck, `obj`/`box` heads | Yes | `PlateDetector` |
| Decode: sigmoid, softplus, grid centres | Yes | `DetectorExport` |
| Top-K + NMS | **Optional** | `detector_baseline_nms.onnx` uses in-graph Fast NMS (`snippet:nms-in-graph`); `detector_baseline.onnx` leaves greedy NMS to Python (`snippet:greedy-nms`) |
| Crop from the full-resolution frame, 32x128 grey | **No** | `src/preprocess.py :: crop_plates` |
| CRNN | Yes | `ocr_baseline.onnx` |
| CTC greedy decode | **No** | `snippet:ctc-greedy-decode` |

> **Why the NMS-in-graph row has its own parent.** Fast NMS runs as plain tensor ops over a fixed top-K, and an already-suppressed box can still suppress others, so it is not greedy NMS. The row is gated against PyTorch running the *same* in-graph NMS (`TorchBackend(..., nms_in_graph=True)`), its parent is `ort-cpu-fp32`, and any accuracy cost of the swap is judged by the benchmark, not by parity.

### Opset is a compatibility choice

`config.ONNX_OPSET = 18` is accepted by every consumer in this session's matrix: ORT 1.24 inside Triton 26.05, TensorRT 10.16's parser, OpenVINO 2026, onnx2tf. A newer opset buys nothing for Conv, LSTM and TopK, and loses you whichever converter lags.

### Execution providers: a preference list, not a guarantee

`providers=[TensorRT, CUDA, CPU]` means "try these in order." ORT assigns each graph node to the first provider that can run it. If a provider fails to load entirely, ORT falls back to the next one. When the fallback is CPU, the only sign of it is a log line.

| Row (`s04_onnx_export:`) | Providers | Needs | Gate |
|---|---|---|---|
| `ort-cpu-fp32` | CPU | `onnxruntime` | strict |
| `ort-cpu-fp32-nms-in-graph` | CPU | same | strict |
| `ort-cuda-fp32` | CUDA, CPU | `onnxruntime-gpu` + NVIDIA GPU | strict |
| `ort-tensorrt-ep-fp16` | TensorRT (FP16, engine cache), CUDA, CPU | `onnxruntime-gpu` + TensorRT 10 libs | report only (precision changes) |
| `ort-openvino-ep-cpu` | OpenVINO (`device_type` CPU) | `onnxruntime-openvino`, own venv | strict |

The `onnxruntime-gpu` 1.30.0 wheel includes the CUDA and TensorRT providers, and its TensorRT provider links `libnvinfer.so.10`, so it needs TensorRT 10 (https://pypi.org/pypi/onnxruntime-gpu/json). `onnxruntime-openvino` is a separate package (latest 1.24.1, bundling OpenVINO 2025.4.1) that *replaces* `onnxruntime` (https://pypi.org/pypi/onnxruntime-openvino/json).

> **Why most teams ship through providers.** `snippet:trt-execution-provider` is a few lines of options on a session you already have. Guide 10 builds TensorRT engines by hand and guide 12 converts to OpenVINO IR. Read those as "what the provider is doing for you": parsing, partitioning, engine building and caching. The raw APIs earn their extra code when you need control the provider doesn't expose, such as custom INT8 calibration, explicit optimization profiles, or async request queues.

The TensorRT provider **builds an engine** the first time a session runs, which can take a long time. With `trt_engine_cache_enable`, the engine is stored and reused. The cache is tied to the model, precision, profile shapes and GPU. An input shape outside the cached profile triggers a rebuild (https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

## 3. Runnable walkthrough

```bash
make s04                                   # CPU rows; others record not_run with a reason
make setup-gpu && make s04                 # on the RTX 3090: adds CUDA and TensorRT EP rows
```

Read `src/stages/s04_onnx_export.py` in this order:

1. **`export`** (`snippet:onnx-export`): `dynamo=True`, `dynamic_shapes` with `torch.export.Dim("batch", min=1, max=64)`, `opset_version=config.ONNX_OPSET`, `external_data=False` (one self-contained file), then the full checker.
2. **`export_all`**: the batch-2 comment, and the three files it writes. Later stages call this same function with a different `suffix`.
3. **`provider_rows`**: each `RunSpec` with the reason it records as `not_run` if its provider is missing.
4. **`main`**: for each available row, `common.gate(..., strict=spec.precision == "fp32")`, then `run(spec)`.
5. **`src/backends.py :: OrtBackend`**: pinned thread counts, and `active_providers`, recorded because of silent fallback.

## 4. Measured result

<!-- results:stage:s04_onnx_export -->
<!-- /results -->

How to read it:

- **`ort-cpu-fp32` against `s01_baseline:eager-fp32`.** Same weights and precision, strict gate passed. mAP@0.5 and OCR exact should match the parent exactly. If they don't, look at preprocessing before blaming the export. Then compare p50/p95 and fps: the difference is runtime plus kernels, since the math is the same.
- **`ort-cpu-fp32-nms-in-graph` against `ort-cpu-fp32`.** In `results/results.json`, compare `latency.phases_ms`. The work moves from the `nms` phase into `detect`. Whether the total goes down depends on how the Python NMS compared with a fixed top-K tensor operation on this machine. Any accuracy difference comes from the change of algorithm (Fast NMS versus greedy), not from the export.
- **size MB.** The `.onnx` files, plus any `.onnx.data` sidecar (`Backend.size_mb` counts it).
- **peak RSS MB.** An ORT process does not load torch, so compare it with the eager row to see what the runtime itself costs.
- **GPU rows.** Look at `active_providers` in `results/results.json` before reading a single latency. If `TensorrtExecutionProvider` is missing from a TensorRT row, you measured CUDA or CPU under a TensorRT label. `active_providers` is recorded from the *detector* session and lists only the registered providers. It does not show which nodes each provider ran. For that, set `SessionOptions.log_severity_level` to verbose and read how nodes were assigned.
- **TensorRT cold start.** In the worker, the untimed accuracy pass runs before latency is measured, so engine building never shows up in p50/p95. To see it, time `make s04` once with an empty `artifacts/<profile>/trt_ep_cache/` and once with it populated.
- **`not_run` rows.** The status text is the reason. A CPU laptop will show the CUDA, TensorRT and OpenVINO provider rows here. That is a true record of this machine, not a failure.

## 5. Gotchas

1. **Batch frozen at export.** **Symptom:** single-frame tests pass. Then production batching, or the throughput pass, fails with an input shape error naming the dimension, and the model's input in Netron shows a number where the batch should be. **Fix:** declare `dynamic_shapes` for every dimension that varies, and gate at a batch size different from the export example.

2. **Example batch of 1.** **Symptom:** you *did* declare a dynamic batch, but the exported input still has a fixed batch of 1, or the exporter reports that the dimension was specialized. **Fix:** use an example batch of 2 or more, as `export_all` does.

3. **Pre- or post-processing differs between pipelines.** **Symptom:** the ONNX Runtime and PyTorch tensors match in isolation, but the deployed service finds different plates. Boxes are shifted by the padding offset, or strings drift at night. **Fix:** share one preprocessing function between training, gating and serving. `Pipeline` does this, and the parity gate feeds both backends the same letterboxed batch and the same crops. Only move steps into the graph deliberately, the way `DetectorExport` does with NMS.

4. **`onnxruntime` and `onnxruntime-gpu` in one environment.** **Symptom:** `get_available_providers()` shows only CPU on a GPU box, or imports break after an upgrade, because both wheels install the same `onnxruntime` package. **Fix:** uninstall one before installing the other (`make setup-gpu` does this). The same rule applies to `onnxruntime-openvino`: give it its own venv.

5. **Provider silently falls back to CPU.** **Symptom:** a "TensorRT" or "CUDA" row whose latency looks like the CPU row, a warning about failing to load `libnvinfer` or a CUDA library at session creation, and `active_providers` without the provider you asked for. **Fix:** always check `active_providers`. In production, fail at startup if the provider you require is not in `session.get_providers()`.

6. **Opset too new for a downstream converter.** **Symptom:** ORT runs the file fine, then the TensorRT parser, OpenVINO or onnx2tf rejects it with an unsupported-opset or unsupported-op error. **Fix:** pick the opset from the *oldest* consumer on your matrix (`config.ONNX_OPSET`), and re-verify the whole matrix whenever you change it.

7. **External data left behind.** **Symptom:** a model too large for a single protobuf file loads on the build machine, then fails on the target with an error about a missing external data file. Only `model.onnx` was copied, and its weights live in `model.onnx.data` next to it. **Fix:** treat the `.onnx` and its `.data` file as one artifact when you copy, hash or upload. Our models are small, so `external_data=False` keeps them in one file.

8. **float64 constants in the graph.** **Symptom:** Netron shows `DOUBLE` initializers or `Cast` nodes to float64, and a downstream parser or INT8 tool rejects or mishandles them. **Fix:** create constants and buffers as float32. `PlateDetector` builds its grid centres with `.float()` for this reason. Python-scalar math on integer tensors is the usual source.

## 6. AV comparison callout

> **Context only, not part of this lab.** A YOLO-class driving detector goes through the same export, but the in-graph line moves. With many classes and per-class thresholds, NMS is often handed to the deployment runtime, commonly as a TensorRT NMS plugin. That ties the file to that runtime, a trade our CPU-first matrix avoids with plain-tensor Fast NMS. Lane segmentation has no NMS at all: its output is a dense per-pixel map, and the post-processing that matters (argmax, curve fitting) usually stays outside the graph in C++. Batching differs too. A car's camera rig has a fixed number of cameras, so batch is often exported static with one engine per rig. Our variable number of plate crops makes the OCR batch genuinely dynamic. The preprocessing trap is the same in both worlds: resize and normalization outside the graph must match training exactly.

## 7. When NOT to use this

- **The deployment target runs PyTorch anyway.** On a GPU server that keeps a Python torch process, `torch.compile` (guide 04) may be enough, and an export just adds a parity gate to maintain.
- **The model depends on data-dependent shapes or custom kernels** that the exporter or your downstream converters cannot express. Weigh weeks of rewriting against the win, and consider keeping that part outside the graph.
- **You are still changing the architecture every day.** Export when a candidate is close to shipping, and let CI re-export and re-gate.
- **Your matrix has one target with a mature direct path from PyTorch.** ONNX then adds a conversion and a gate without portability you'll use.

Previous: [04 — torch.compile and TorchScript](04-torch-compile-and-torchscript.md) · Next: [06 — Pruning](06-pruning.md)
