# Benchmark card — `s06a_ptq:ort-static-int8-detector-only`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · int8-det |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:49:01+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8924** · plate exact match **0.8532**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.911 | 0.9078 | 347 |
| night | 0.8257 | 0.8483 | 145 |
| rain | 0.9114 | 0.8239 | 176 |
| motion_blur | 0.9185 | 0.8629 | 124 |
| low_contrast | 0.9089 | 0.7613 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **33.126 ms** · p95 **37.313 ms** · p99 38.378 ms · first call of the process 41.864 ms
(includes compilation / engine build) · first call of the timing loop 28.101 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.318 | 3.247 |
| preprocess | 3.496 | 4.556 |
| detect | 24.403 | 24.99 |
| nms | 0.12 | 0.15 |
| crop | 0.107 | 0.153 |
| ocr | 3.503 | 6.906 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 29.58 | 33.263 | 37.406 |
| 4 | 32.47 | 123.207 | 126.871 |
| 8 | 31.36 | 245.912 | 296.737 |

## Size and memory

Artifact size (MB): `{"detector": 8.346, "ocr": 4.98, "total": 13.326}` · peak RSS 383.4 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 9.026477813720703, "box_count_equal": true, "min_matched_box_iou": 0.6339802742004395, "strings_equal": 1.0, "passed": null}
```

## Notes

detector INT8 (stratified), recognizer FP32

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
 "load_avg_1m": 7.02
}
```
