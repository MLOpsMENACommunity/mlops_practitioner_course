# Benchmark card — `s10_tflite_edge:tflite-int8-ocr-fp32-detector`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s07_distillation:student-distilled-onnx` |
| Runtime · device · precision | tflite · cpu · int8-ocr |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:12:30+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8978** · plate exact match **0.9472**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9026 | 0.9856 | 347 |
| night | 0.8533 | 0.9655 | 145 |
| rain | 0.9268 | 0.9545 | 176 |
| motion_blur | 0.9051 | 0.9677 | 124 |
| low_contrast | 0.918 | 0.8194 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **14.323 ms** · p95 **15.955 ms** · p99 16.582 ms · first call of the process 26.733 ms
(includes compilation / engine build) · first call of the timing loop 13.724 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.226 | 3.005 |
| preprocess | 3.315 | 3.384 |
| detect | 8.914 | 9.007 |
| nms | 0.196 | 0.223 |
| crop | 0.163 | 0.291 |
| ocr | 0.391 | 0.612 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 67.66 | 14.701 | 15.447 |
| 4 | 69.2 | 57.273 | 61.363 |
| 8 | 69.08 | 114.974 | 120.197 |

## Size and memory

Artifact size (MB): `{"detector": 1.772, "ocr": 0.17, "total": 1.942}` · peak RSS 230.1 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 2.7340736389160156, "box_count_equal": true, "min_matched_box_iou": 0.9999986290931702, "strings_equal": 1.0, "passed": null}
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
  "onnx": "1.20.1",
  "onnxruntime": "1.26.0",
  "ai-edge-litert": "2.1.2",
  "onnx2tf": "2.6.8",
  "numpy": "2.2.6",
  "pillow": "12.3.0"
 },
 "load_avg_1m": 3.52
}
```
