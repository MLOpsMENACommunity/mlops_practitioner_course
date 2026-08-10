# Triton for ResNet-50

The same model as every other level, served by a C++ inference server instead of
Python. `config.pbtxt` **is** the serving layer — everything `level_2_batching.py`
wrote by hand is a stanza in a text file.

```
model_repository/
  resnet50/
    config.pbtxt        # the entire serving layer, as configuration
    1/model.onnx        # written by ../level_4_triton_export.py (verified)
```

## Run it

```bash
python ../level_4_triton_export.py     # writes + verifies 1/model.onnx (102 MB)

docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v "$(pwd)/model_repository:/models" \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models
```

Then:

```bash
curl -s localhost:8000/v2/health/ready -o /dev/null -w '%{http_code}\n'   # 200
curl -s localhost:8000/v2/models/resnet50/config | python -m json.tool

# Triton takes tensors, not JPEGs -- preprocessing is the CALLER's job here,
# which is exactly the Level 3 lesson showing up as an architectural choice.
curl -s localhost:8002/metrics | grep -E 'nv_inference_(exec_count|request_success)'
```

That last pair of metrics is the one to watch: `request_success / exec_count` **is**
your average batch size — the same number `level_2_batching.py` had to compute by
hand for its `/metrics` endpoint.

## Read it against `level_2_batching.py`

| Level 2, by hand | `config.pbtxt` |
|---|---|
| `asyncio.Queue` + window + futures (~90 lines) | `dynamic_batching { ... }` |
| `MAX_BATCH_SIZE` | `preferred_batch_size` |
| `BATCH_WINDOW_MS = 8` | `max_queue_delay_microseconds: 8000` |
| `QUEUE_MAXSIZE` + 503 | `default_queue_policy { max_queue_size, REJECT }` |
| `MODEL_VERSION = "v1"` constant | `version_policy` + version directories |
| not possible without a rewrite | `instance_group { count: 2 }` (Level 3, Lever C) |
| a separate export + rebuild | `optimization { ... tensorrt FP16 }` (Lever A + B) |

## Platform caveat — read before pulling 15 GB

**This machine is `linux/aarch64` (Apple Silicon) and cannot usefully run Triton.**
NVIDIA publishes `nvcr.io/nvidia/tritonserver` for `linux/amd64` only; the ARM (SBSA)
builds target NVIDIA ARM server hardware, not an M-series Mac. Options:

| Option | Works? | Notes |
|---|---|---|
| Pull amd64 image, emulate | Technically | ~15 GB, very slow, no GPU. Fine to *see* it boot, useless for benchmarking |
| Linux x86 box or cloud GPU VM | Yes | The real answer |
| Read `config.pbtxt` against Level 2 | Yes | What this directory is mainly for |

The image was deliberately **not** pulled for you — 15 GB is your disk and your
bandwidth. Everything Triton-specific here is correct and ready for a host that can
run it: the `config.pbtxt` tensor names and shapes were cross-checked against the
actual export, and `model.onnx` was verified against PyTorch at batch sizes 1, 8 and
32 (max abs diff 6.9e-06, top-1 agreement at every size).

## Why bother, when BentoML already worked?

`level_4_bentoml_service.py` serves this same model in ~60 lines of Python with no
conversion step. Choose Triton when you need what it uniquely provides:

- **No GIL anywhere on the request path.** The server is C++.
- **Multiple frameworks in one process.** A TensorRT vision model, an ONNX ranker and
  a PyTorch encoder behind one endpoint, one GPU, one deployment.
- **Model ensembles.** Chain preprocess → infer → postprocess as a server-side DAG so
  intermediate tensors never make a network hop. This is the proper fix for the
  preprocessing bottleneck Level 3 uncovered.
- **Concurrent model execution.** `instance_group.count` runs N copies that genuinely
  execute in parallel (Level 3, Lever C).
- **Hot reload.** Drop a `2/` directory in and it loads live — no restart.

The cost is the conversion step, another config format to get right, and a much
heavier runtime. For one model in a Python shop, BentoML is the better trade. For a
GPU fleet serving several models under a latency SLA, Triton is what the trade was
designed for.
