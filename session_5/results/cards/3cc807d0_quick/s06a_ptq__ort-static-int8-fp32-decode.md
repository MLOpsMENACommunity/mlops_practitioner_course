# Benchmark card — `s06a_ptq:ort-static-int8-fp32-decode`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · int8 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:49:48+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9041** · plate exact match **0.9314**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9184 | 0.9568 | 347 |
| night | 0.8433 | 0.931 | 145 |
| rain | 0.9287 | 0.9205 | 176 |
| motion_blur | 0.9185 | 0.9597 | 124 |
| low_contrast | 0.9268 | 0.8645 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **33.92 ms** · p95 **43.847 ms** · p99 56.2 ms · first call of the process 42.857 ms
(includes compilation / engine build) · first call of the timing loop 28.324 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.499 | 3.821 |
| preprocess | 3.552 | 6.148 |
| detect | 24.655 | 30.535 |
| nms | 0.21 | 0.327 |
| crop | 0.138 | 0.216 |
| ocr | 2.619 | 5.882 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 29.7 | 33.591 | 37.084 |
| 4 | 30.45 | 128.354 | 147.022 |
| 8 | 32.96 | 238.043 | 257.384 |

## Size and memory

Artifact size (MB): `{"detector": 8.364, "ocr": 3.449, "total": 11.813}` · peak RSS 326.3 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 6.795347213745117, "box_count_equal": true, "min_matched_box_iou": 0.9663269519805908, "strings_equal": 0.9, "passed": null}
```

## Notes

stratified INT8, except the detector's box/score decode ops stay FP32

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
 "load_avg_1m": 8.47
}
```
