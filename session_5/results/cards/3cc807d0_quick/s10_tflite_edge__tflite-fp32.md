# Benchmark card — `s10_tflite_edge:tflite-fp32`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s07_distillation:student-distilled-onnx` |
| Runtime · device · precision | tflite · cpu · fp32 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:12:09+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8978** · plate exact match **0.9472**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9026 | 0.9856 | 347 |
| night | 0.8533 | 0.9586 | 145 |
| rain | 0.9268 | 0.9489 | 176 |
| motion_blur | 0.9051 | 0.9758 | 124 |
| low_contrast | 0.918 | 0.8258 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **14.538 ms** · p95 **16.174 ms** · p99 17.289 ms · first call of the process 25.479 ms
(includes compilation / engine build) · first call of the timing loop 13.845 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.246 | 2.99 |
| preprocess | 3.312 | 3.459 |
| detect | 8.894 | 9.088 |
| nms | 0.2 | 0.237 |
| crop | 0.165 | 0.293 |
| ocr | 0.573 | 0.889 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 68.2 | 14.55 | 15.396 |
| 4 | 68.68 | 58.023 | 60.632 |
| 8 | 68.63 | 116.111 | 119.559 |

## Size and memory

Artifact size (MB): `{"detector": 1.772, "ocr": 0.647, "total": 2.419}` · peak RSS 228.0 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": true, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.00048828125, "box_count_equal": true, "min_matched_box_iou": 0.9999986290931702, "strings_equal": 1.0, "passed": true}
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
 "load_avg_1m": 3.71
}
```
