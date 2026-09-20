# Benchmark card — `s09_openvino:edge-candidate`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s09_openvino:student-int8-nncf-fp32-head` |
| Runtime · device · precision | openvino · cpu · int8 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:03:42+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8045** · plate exact match **0.868**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.831 | 0.9193 | 347 |
| night | 0.7792 | 0.8345 | 145 |
| rain | 0.8004 | 0.8636 | 176 |
| motion_blur | 0.8601 | 0.8468 | 124 |
| low_contrast | 0.8316 | 0.8065 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **8.175 ms** · p95 **9.851 ms** · p99 10.618 ms · first call of the process 29.679 ms
(includes compilation / engine build) · first call of the timing loop 7.361 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.225 | 2.997 |
| preprocess | 1.117 | 1.161 |
| detect | 4.98 | 5.327 |
| nms | 0.009 | 0.016 |
| crop | 0.092 | 0.146 |
| ocr | 0.623 | 0.897 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 121.92 | 8.071 | 9.28 |
| 4 | 128.84 | 30.014 | 34.754 |
| 8 | 131.17 | 59.883 | 65.769 |

## Size and memory

Artifact size (MB): `{"detector": 0.831, "ocr": 0.227, "total": 1.058}` · peak RSS 598.0 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 588.9495849609375, "box_count_equal": false, "min_matched_box_iou": 0.0, "strings_equal": 1.0, "passed": null}
```

## Notes

distilled student + NNCF INT8 (decode and CTC projection FP32) + NMS in graph + Image.reduce resize

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
 "load_avg_1m": 4.4
}
```
