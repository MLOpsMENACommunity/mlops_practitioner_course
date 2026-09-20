# Benchmark card — `s07_distillation:student-distilled`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s01_baseline:eager-fp32` |
| Runtime · device · precision | torch · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:01:20+0300 |

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

p50 **28.814 ms** · p95 **30.66 ms** · p99 31.399 ms · first call of the process 37.665 ms
(includes compilation / engine build) · first call of the timing loop 27.595 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.275 | 3.087 |
| preprocess | 3.324 | 3.497 |
| detect | 22.713 | 23.761 |
| nms | 0.23 | 0.263 |
| crop | 0.121 | 0.165 |
| ocr | 1.003 | 1.267 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 34.61 | 28.795 | 29.996 |
| 4 | 45.93 | 85.052 | 98.625 |
| 8 | 53.64 | 147.773 | 161.007 |

## Size and memory

Artifact size (MB): `{"detector": 1.805, "ocr": 0.66, "total": 2.465}` · peak RSS 625.2 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 1e-05, "max_abs_diff": 99.38619995117188, "box_count_equal": true, "min_matched_box_iou": 0.8622832298278809, "strings_equal": 0.8, "passed": null}
```

## Notes

OCR KD: teacher/student per-step character alignment agrees 66% — sequence-level CTC on the teacher reading (alignment-free)

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
 "load_avg_1m": 4.53
}
```
