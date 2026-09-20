# Benchmark card — `s06a_ptq:ort-static-int8-stratified`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · int8 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:55:07+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8924** · plate exact match **0.8543**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.911 | 0.9078 | 347 |
| night | 0.8257 | 0.8552 | 145 |
| rain | 0.9114 | 0.8239 | 176 |
| motion_blur | 0.9185 | 0.8629 | 124 |
| low_contrast | 0.9089 | 0.7613 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **32.562 ms** · p95 **35.923 ms** · p99 36.698 ms · first call of the process 41.862 ms
(includes compilation / engine build) · first call of the timing loop 28.044 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.316 | 3.397 |
| preprocess | 3.493 | 5.082 |
| detect | 24.388 | 25.107 |
| nms | 0.117 | 0.148 |
| crop | 0.107 | 0.151 |
| ocr | 2.566 | 4.479 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 30.74 | 32.528 | 34.819 |
| 4 | 33.36 | 119.16 | 123.428 |
| 8 | 33.58 | 235.709 | 242.914 |

## Size and memory

Artifact size (MB): `{"detector": 8.346, "ocr": 3.449, "total": 11.795}` · peak RSS 366.3 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 9.026477813720703, "box_count_equal": true, "min_matched_box_iou": 0.6339802742004395, "strings_equal": 0.9, "passed": null}
```

## Notes

calibration: stratified, 125 frames, per-channel

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
 "load_avg_1m": 5.6
}
```
