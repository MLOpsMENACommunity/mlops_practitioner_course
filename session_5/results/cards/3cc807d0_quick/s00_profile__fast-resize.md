# Benchmark card — `s00_profile:fast-resize`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s01_baseline:eager-fp32` |
| Runtime · device · precision | torch · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:07:34+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9039** · plate exact match **0.9377**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9174 | 0.9597 | 347 |
| night | 0.8405 | 0.931 | 145 |
| rain | 0.9293 | 0.9318 | 176 |
| motion_blur | 0.9171 | 0.9677 | 124 |
| low_contrast | 0.9267 | 0.8774 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **52.213 ms** · p95 **54.303 ms** · p99 56.861 ms · first call of the process 67.178 ms
(includes compilation / engine build) · first call of the timing loop 48.845 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.278 | 3.112 |
| preprocess | 1.143 | 1.192 |
| detect | 45.722 | 46.967 |
| nms | 0.232 | 0.259 |
| crop | 0.125 | 0.165 |
| ocr | 3.544 | 4.13 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 19.17 | 52.073 | 53.417 |
| 4 | 17.95 | 222.73 | 227.433 |
| 8 | 17.85 | 445.406 | 468.814 |

## Size and memory

Artifact size (MB): `{"detector": 32.378, "ocr": 4.871, "total": 37.249}` · peak RSS 1416.7 MB · peak VRAM None MB

## Parity report vs parent

```json
null
```

## Notes

Image.reduce(2) instead of bilinear resize

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
 "load_avg_1m": 4.81
}
```
