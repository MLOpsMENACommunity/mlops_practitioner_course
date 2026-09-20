# Benchmark card — `s01_baseline:eager-fp32`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `(root)` |
| Runtime · device · precision | torch · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:31:22+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9048** · plate exact match **0.9335**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9184 | 0.9568 | 347 |
| night | 0.8436 | 0.931 | 145 |
| rain | 0.9292 | 0.9318 | 176 |
| motion_blur | 0.917 | 0.9597 | 124 |
| low_contrast | 0.9276 | 0.8645 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **54.121 ms** · p95 **58.337 ms** · p99 63.764 ms · first call of the process 62.742 ms
(includes compilation / engine build) · first call of the timing loop 50.462 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.279 | 3.03 |
| preprocess | 3.337 | 3.485 |
| detect | 45.448 | 48.914 |
| nms | 0.23 | 0.273 |
| crop | 0.124 | 0.176 |
| ocr | 3.567 | 4.16 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 18.46 | 54.302 | 56.04 |
| 4 | 17.27 | 230.988 | 235.157 |
| 8 | 17.05 | 466.052 | 493.926 |

## Size and memory

Artifact size (MB): `{"detector": 32.378, "ocr": 4.871, "total": 37.249}` · peak RSS 1316.3 MB · peak VRAM None MB

## Parity report vs parent

```json
null
```

## Notes



## Environment

```json
{
 "cpu": "Apple M3 Pro",
 "cpu_cores_logical": 11,
 "ram_gb": 18.0,
 "os": "Darwin 25.5.0 arm64",
 "python": "3.12.12",
 "threads": {
  "ANPR_THREADS": 4,
  "OMP_NUM_THREADS": "4",
  "ort_intra_op_num_threads": 4,
  "ort_inter_op_num_threads": 1,
  "torch_num_threads": 4
 },
 "libraries": {
  "torch": "2.13.0",
  "torchvision": "0.28.0",
  "onnx": "1.22.0",
  "onnxruntime": "1.30.0",
  "openvino": "2026.3.1",
  "nncf": "3.3.0",
  "ai-edge-litert": "2.2.0",
  "tritonclient": "2.72.0",
  "numpy": "2.4.6",
  "pillow": "12.3.0"
 },
 "load_avg_1m": 4.27
}
```
