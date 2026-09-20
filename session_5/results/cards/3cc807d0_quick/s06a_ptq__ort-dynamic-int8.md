# Benchmark card — `s06a_ptq:ort-dynamic-int8`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · int8-dyn |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:47:25+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9052** · plate exact match **0.9356**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9182 | 0.9597 | 347 |
| night | 0.8438 | 0.9448 | 145 |
| rain | 0.9292 | 0.9261 | 176 |
| motion_blur | 0.9171 | 0.9597 | 124 |
| low_contrast | 0.9289 | 0.8645 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **45.566 ms** · p95 **48.317 ms** · p99 49.286 ms · first call of the process 50.625 ms
(includes compilation / engine build) · first call of the timing loop 43.293 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.345 | 3.291 |
| preprocess | 3.506 | 5.039 |
| detect | 37.459 | 38.761 |
| nms | 0.238 | 0.265 |
| crop | 0.121 | 0.167 |
| ocr | 2.345 | 3.498 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 21.95 | 45.314 | 48.149 |
| 4 | 22.49 | 175.497 | 182.518 |
| 8 | 22.33 | 357.547 | 364.07 |

## Size and memory

Artifact size (MB): `{"detector": 8.265, "ocr": 3.418, "total": 11.683}` · peak RSS 767.8 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 28.02362060546875, "box_count_equal": true, "min_matched_box_iou": 0.9847174286842346, "strings_equal": 1.0, "passed": null}
```

## Notes

weights only; activations scaled per call

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
 "load_avg_1m": 6.18
}
```
