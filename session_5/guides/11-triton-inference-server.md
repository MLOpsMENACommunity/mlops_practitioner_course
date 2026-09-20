# 11 — Triton Inference Server: the ANPR pipeline as a service

Session 3 served one ResNet-50 from one `config.pbtxt` ([level_4_triton](../../session_3/serving_levels/level_4_triton/README.md)) and showed that `request_success / exec_count` gives you the average batch size. This guide goes deeper with the same server. You'll put five models in one repository, run the pipeline server-side with Business Logic Scripting (BLS), and measure dynamic batching with concurrent clients against a twin model that has batching turned off. You'll also hot-reload a new version and read the server's own counters. The product is now "NVIDIA Dynamo-Triton, formerly Triton Inference Server", and the server, ensembles and Python backend are unchanged (https://developer.nvidia.com/dynamo-triton). Code: `src/stages/s11_triton.py`, `triton_serving/`.

## 1. The problem it solves

Guides 10, 12 and 13 run the pipeline inside one process. For one camera with a 30 ms per-frame budget and no uplink, that's the right design. A junction cabinet is different: it holds a camera per approach lane plus ingest, storage and health processes, and they all share one accelerator. That setup raises new questions:

- N cameras send N separate single-frame requests. What batches them without pushing any frame past 30 ms?
- The detector and recognizer should be loaded once, not once per camera process.
- A new recognizer version has to roll out without stopping the cameras.
- There's no uplink, so the box has to expose its state locally for site monitoring to scrape.

Triton answers each of these with configuration: a dynamic batcher per model, instance groups, version directories with explicit load, and Prometheus metrics. Section 7 covers when that's more than you need.

## 2. Mental model

```
camera client --gRPC/HTTP--> frontend --> model queue --> dynamic batcher --> instance --> backend (ORT / Python / TensorRT)
                                                     waits up to max_queue_delay for company
```

| concept | what it is | here |
|---|---|---|
| model repository | one directory per model, one numbered directory per version | `triton_serving/model_repository/` |
| `config.pbtxt` | the serving layer for one model | one per model |
| backend | what executes the model | `onnxruntime_onnx` (four models), `python` (`anpr`) |
| dynamic batcher | merges separate requests into one execution | on `detector`, `detector_nms`, `ocr` |
| instance group | parallel copies and their device | `count: 1`; `anpr`: `count: 2 kind: KIND_CPU` |
| BLS | Python model code that calls other models in the same server | `anpr/1/model.py` |

**What people get wrong:**

- **"A batch of 8 was fast, so dynamic batching works."** A client-side batch is a single request. The batcher merges *separate* requests that arrive close together, so you need concurrent clients to exercise it. `src/benchmark.py :: concurrent_pass` provides them, and every s11 row sets `concurrent=True` (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html#dynamic-batcher).
- **"`max_queue_delay_microseconds` is a timeout."** It's how long the batcher *may hold* a request while it waits for company. Under light load a request can wait that long for nothing, so treat it as latency you choose to spend.
- **"BLS is an ensemble."** An ensemble is a declarative DAG where every step runs once per request. BLS is Python code that decides at run time which models to call.
- **"The average batch size equals `preferred_batch_size`."** That field is a target. The counters tell you what the server actually did.

## 3. Runnable walkthrough

### 3.1 The repository

```
triton_serving/model_repository/
  detector/           config.pbtxt   1/model.onnx     raw scores + boxes; NMS in the client
  detector_nobatch/   config.pbtxt   1/model.onnx     the same file, no dynamic_batching block
  detector_nms/       config.pbtxt   1/model.onnx     top-K + Fast NMS in the graph, [50,5] out
  ocr/                config.pbtxt   1/model.onnx     2/ appears during the hot-reload step
  anpr/               config.pbtxt   1/model.py       the whole pipeline: Python backend, BLS
```

`make s11` copies the s07 student ONNX files into each `1/` and deletes any stale `2/`. Model files are gitignored, while the configs and `model.py` are committed. The folder is `triton_serving/` because a top-level `triton/` would shadow the `triton` package that `torch._dynamo` imports, and imports then fail with "module 'triton' has no attribute 'language'". `tests/test_repo_hygiene.py` enforces the name.

### 3.2 The configs

`detector/config.pbtxt`:

```
max_batch_size: 8                       # must not exceed the batch range the ONNX graph was exported with
input [
  { name: "images" data_type: TYPE_FP32 dims: [ 3, 384, 640 ] }   # batch dimension is implicit
]
dynamic_batching {
  preferred_batch_size: [ 4, 8 ]
  max_queue_delay_microseconds: 2000
}
instance_group [ { count: 1 } ]
parameters { key: "intra_op_thread_count" value: { string_value: "4" } }
```

| field | meaning | why this value |
|---|---|---|
| `max_batch_size: 8` | largest batch the server may form; `dims` leave out the batch axis | the exported graph's batch range |
| `preferred_batch_size: [4, 8]` | sizes the batcher tries to reach before the delay expires | the sizes the throughput sweep measures |
| `max_queue_delay_microseconds: 2000` | the longest a request waits for company | 2 ms out of a 30 ms frame |
| `count: 1`, no `kind` | on the GPU if there is one, otherwise a CPU instance | one config for the 3090 and for Docker on a laptop |
| `intra_op_thread_count: "4"` | ORT threads inside the server | matches `ANPR_THREADS`, so rows stay comparable |

The other three ONNX models differ only in these lines:

```
# detector_nobatch: identical, minus the whole dynamic_batching block. The pair is the experiment.
# detector_nms:
output [ { name: "detections" data_type: TYPE_FP32 dims: [ 50, 5 ] } ]   # x0, y0, x1, y1, score; suppressed rows score 0
# ocr:
max_batch_size: 32
input  [ { name: "crops"  data_type: TYPE_FP32 dims: [ 1, 32, 128 ] } ]
output [ { name: "logits" data_type: TYPE_FP32 dims: [ 32, 37 ] } ]      # 32 time steps x (36 characters + CTC blank)
dynamic_batching { max_queue_delay_microseconds: 1000 }
```

`ocr` has no preferred sizes because each frame brings zero to four crops, so any merged size helps. Its delay is 1 ms. A frame that uses both models queues twice, so budget the two delays together against 30 ms.

`anpr/config.pbtxt`:

```
backend: "python"
max_batch_size: 0                        # one frame per request; the models it calls batch across requests
input [ { name: "frame" data_type: TYPE_UINT8 dims: [ 720, 1280, 3 ] } ]
output [
  { name: "detections" data_type: TYPE_FP32 dims: [ -1, 5 ] },
  { name: "plates" data_type: TYPE_STRING dims: [ -1 ] }
]
instance_group [ { count: 2 kind: KIND_CPU } ]
```

> **Why `KIND_CPU` with count 2.** Letterbox, crop and CTC decode are numpy work. Each Python-backend instance runs in its own stub process, so two frames are processed in parallel without a shared GIL. That count is also a **ceiling**. `model.py` makes synchronous BLS calls, so `detector_nms` never sees more concurrent requests from `anpr` than there are `anpr` instances, and the batcher can't merge requests that never arrive together. To raise the ceiling, add instances or switch to asynchronous BLS (`async def execute`, `await request.async_exec()`): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html#business-logic-scripting

### 3.3 Start the server (`snippet:triton-docker-run`)

```python
gpus = ["--gpus", "all"] if common.has_cuda() else []  # needs the NVIDIA Container Toolkit
subprocess.run(["docker", "run", "-d", "--name", CONTAINER, *gpus,
                "-p", "8000:8000", "-p", "8001:8001", "-p", "8002:8002",  # HTTP, gRPC, Prometheus metrics
                "-v", f"{config.TRITON_REPO}:/models", IMAGE,
                "tritonserver", "--model-repository=/models",
                "--model-control-mode=explicit", "--load-model=*"], check=True)
```

```bash
make triton-up        # populate, start, wait until every model is ready
curl -s localhost:8000/v2/health/ready -o /dev/null -w '%{http_code}\n'
curl -s localhost:8000/v2/models/anpr/config | python -m json.tool
make triton-down
```

`nvcr.io/nvidia/tritonserver:26.05-py3` ships Triton 2.69.0, ORT 1.24.4 and TensorRT 10.16.1.11, the same TensorRT as `requirements-gpu.txt` (https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-05.html). The image has amd64 and arm64 manifests. In **explicit** mode the server loads only what it's told and never polls the repository (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html#model-control-mode-explicit).

### 3.4 Prove the server matches the local pipeline

```bash
python -m triton_serving.client --protocol grpc --frames 40
python -m triton_serving.client --protocol http
```

`prove` sends validation frames to `anpr` and runs the same student models locally in ONNX Runtime. Every plate the local pipeline reads must be found by the server with IoU above 0.9 and the identical string. If agreement falls below `min_agreement`, the script **exits non-zero**. The local run uses fast resize because `model.py` shrinks 1280x720 by exactly a factor of 2 with a box average, which is the same operation as PIL's `Image.reduce(2)` (`snippet:letterbox`).

### 3.5 BLS, and what an ensemble would look like

`anpr/1/model.py :: call`:

```python
request = pb_utils.InferenceRequest(model_name=model, requested_output_names=[output], inputs=[pb_utils.Tensor(name, array)])
response = request.exec()
if response.has_error():
    raise pb_utils.TritonModelException(response.error().message())
return pb_utils.get_output_tensor_by_name(response, output).as_numpy()
```

`execute` letterboxes the frame, calls `detector_nms`, filters the detections and maps them to frame pixels. It calls `ocr` **only when plates were found**, with exactly that many crops. That data-dependent control flow is what BLS is for. The file needs only numpy, so the stock image runs it unmodified.

An ensemble is a fixed DAG instead (sketch, not in the repo; https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html):

```
platform: "ensemble"
ensemble_scheduling {
  step [
    { model_name: "preprocess"   model_version: -1  input_map { key: "frame" value: "frame" }   output_map { key: "images" value: "images" } },
    { model_name: "detector_nms" model_version: -1  input_map { key: "images" value: "images" } output_map { key: "detections" value: "dets" } }
  ]
}
```

Choose an ensemble when the DAG is fixed and each step is a model or a non-Python backend. Choose BLS when control flow depends on the data, as it does here. Turning our pipeline into a fixed DAG would mean always running OCR on a padded `[4,1,32,128]` tensor, which is the same trade static-shape runtimes make.

### 3.6 Hot reload (`snippet:triton-hot-reload`)

```python
shutil.copytree(config.TRITON_REPO / "ocr" / "1", config.TRITON_REPO / "ocr" / "2", dirs_exist_ok=True)
client.load_model("ocr")  # explicit mode: the repository is re-read for this model only
ready = [v for v in ("1", "2") if client.is_model_ready("ocr", v)]
```

The model uses `version_policy: { latest: { num_versions: 1 } }`, so after `load_model("ocr")` version 2 is ready and version 1 is unloaded. There's no restart and the other models aren't touched. s11 writes the ready versions into every row's notes. To roll back, delete `2/` and call `load_model` again.

### 3.7 Metrics: what the server actually batched

```python
before = counters(model)
with ThreadPoolExecutor(clients) as pool:
    list(pool.map(send, range(requests)))  # concurrent single-frame requests, no clock running
after = counters(model)
return round((after[0] - before[0]) / max(after[1] - before[1], 1), 2)
```

**Average batch size = successful requests / executions** (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/metrics.html). A value of 1.0 means nothing was ever merged. `nv_inference_queue_duration_us` is the total time spent queued, which is what you paid for batching. Both counters count from model load, so reading them once after the benchmark would mix in the harness's one-at-a-time accuracy pass and drag the average towards 1. `src/stages/s11_triton.py :: batch_size_under_load` (`snippet:triton-batch-size-from-metrics`) reads them **before and after** a burst of concurrent single-frame requests, for `detector` and for `detector_nobatch`, and writes the delta into every s11 row's notes.

### 3.8 perf_analyzer as the cross-check

```bash
docker run --rm --net=host nvcr.io/nvidia/tritonserver:26.05-py3-sdk \
  perf_analyzer -m detector -i grpc -u localhost:8001 --concurrency-range 1:8
```

perf_analyzer sends synthetic tensors at fixed concurrency and splits server time into queue and compute (https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/perf_analyzer.html). Use it to confirm the shape of the harness curve. It is **not** a source of `results.json` rows: it skips decode, letterbox, NMS and crop, and its random data can't measure accuracy. Only the harness writes results.

## 4. Measured result

<!-- results:stage:s11_triton -->
_No measured rows yet — run the stage's make target._
<!-- /results -->

**If the rows say `not_run`,** the reason names the missing piece: Docker, the daemon, or the image (`docker pull nvcr.io/nvidia/tritonserver:26.05-py3`). Without an NVIDIA GPU the container runs CPU instances, and those rows show Triton's overhead and batching on a CPU, not on a GPU box.

**Reading the rows:**

- **`fps (batch)` here is a client count.** Each curve point in `throughput.curve` stores `"clients": n`.
- **`triton-grpc-client-pipeline` vs `triton-grpc-nobatch`.** Same model file, client and concurrency, with one config block removed. Compare fps at each client count. Where the curves diverge, batching is buying throughput. At that same point check p95, because a point that goes over 30 ms is capacity you can't use (`src/cost.py :: usable_point`). If the curves overlap, read the average batch sizes in the notes. Near 1.0 means requests never met in the queue. Well above 1.0 with no fps gain means the device was already saturated.
- **gRPC vs HTTP.** `tritonclient.http` sends tensors as binary, not JSON numbers. The gap between the rows is protocol and connection overhead, and it's worth reading rather than assuming.
- **`triton-bls-server-pipeline` vs the client-pipeline rows.** Three things change at once: where letterbox and crop run (server numpy vs client PIL), where NMS runs (in the graph vs greedy in the client), and how many round trips each frame costs (one vs two). Its phase breakdown has work only in `decode` and `detect`, and `detect` is the whole server round trip (`src/pipeline_remote.py`). Its `parity` field is the gRPC proof from 3.4.

## 5. Gotchas

1. **The batch axis is inside `dims`, or `max_batch_size` exceeds the export.**
   **Symptom:** the model fails to load with a shape mismatch, or requests fail once the batcher forms a batch larger than the ONNX graph accepts.
   **Fix:** when `max_batch_size > 0`, `dims` excludes the batch axis. Keep `max_batch_size` within the verified export range. With `max_batch_size: 0` (`anpr`), `dims` is the full shape.

2. **You tested the batcher with client-side batches.**
   **Symptom:** `request_success / exec_count` is 1.0 for `detector`, and the batched and nobatch rows look identical.
   **Fix:** send concurrent single-frame requests (`concurrent_pass`, or `perf_analyzer --concurrency-range`) to a freshly started server.

3. **The queue delay eats the frame budget.**
   **Symptom:** at one client, the batched model's p50 sits above `detector_nobatch`'s, and `nv_inference_queue_duration_us` grows with every request.
   **Fix:** under light load, requests wait for company that never comes. Keep the delays small against 30 ms (2 ms and 1 ms here), and tune them at the site's real camera count.

4. **`--gpus all` without the NVIDIA Container Toolkit.**
   **Symptom:** `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`.
   **Fix:** install the toolkit. s11 passes `--gpus` only when torch sees CUDA.

5. **A new version directory isn't served.**
   **Symptom:** `ocr/2/` exists, but only version 1 is ready.
   **Fix:** explicit mode never polls the repository. Call `load_model("ocr")` or `POST /v2/repository/models/ocr/load`.

6. **The Python model imports something the image lacks.**
   **Symptom:** `anpr` fails to load with `ModuleNotFoundError`, and the BLS row fails with it.
   **Fix:** keep `model.py` numpy-only, or package a custom execution environment as the Python backend README describes. A `pip install` inside a running container is lost on restart.

7. **A TensorRT plan built with another TensorRT.**
   **Symptom:** a `tensorrt_plan` model fails to load with a deserialization error.
   **Fix:** build engines with the TensorRT the image ships. For 26.05 that's 10.16.1.11 (guide 10, gotcha 1).

## 6. AV comparison callout

> **Context: the same server in an autonomous-vehicle program.**
> Offline, fleet jobs replay recorded drives through a YOLO-class detector and a lane-segmentation network or auto-label new data. Thousands of clips send requests concurrently and no frame has a deadline, so the batcher is tuned for throughput per GPU-hour with large batches and generous delays.
> On the vehicle, a fixed camera set with a hard per-frame deadline gets its batching from a static batch equal to the camera count, compiled into the perception process, with no queue at all.
> Our roadside cabinet sits between the two: a handful of cameras and a hard 30 ms budget. That's why the delays stay at a couple of milliseconds and the BLS instance count caps concurrency on purpose.

## 7. When NOT to use this

- **One camera, one process.** An in-process runtime (guides 10, 12, 13) has no network hop, no serialization and no container, and there's nothing to batch.
- **The box is short on memory.** The server, one Python stub per `anpr` instance and one ORT session per model all add to peak RSS. Measure that before putting Triton on a Raspberry Pi-class box.
- **Requests never overlap.** Dynamic batching can then only add queue delay.
- **A Python team with one model.** BentoML (Session 3) is the lighter trade.
- **An LLM.** Use a scheduler built for autoregressive decode (Session 3, Level 5; guide 14).

---

[← 10 — TensorRT](10-tensorrt.md) · [12 — OpenVINO →](12-openvino.md)
