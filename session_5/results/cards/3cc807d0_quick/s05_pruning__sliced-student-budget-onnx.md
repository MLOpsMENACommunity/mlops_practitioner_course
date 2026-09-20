# Benchmark card — `s05_pruning:sliced-student-budget-onnx`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s05_pruning:sliced-student-budget` |
| Runtime · device · precision | ort · cpu · fp32 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:46:22+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8593** · plate exact match **0.9018**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.8819 | 0.9395 | 347 |
| night | 0.8301 | 0.931 | 145 |
| rain | 0.8794 | 0.875 | 176 |
| motion_blur | 0.8726 | 0.9597 | 124 |
| low_contrast | 0.8748 | 0.7742 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **15.845 ms** · p95 **20.082 ms** · p99 20.897 ms · first call of the process 22.33 ms
(includes compilation / engine build) · first call of the timing loop 10.393 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.329 | 3.418 |
| preprocess | 3.498 | 4.736 |
| detect | 6.92 | 7.663 |
| nms | 0.206 | 0.24 |
| crop | 0.095 | 0.141 |
| ocr | 3.421 | 6.911 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 62.45 | 15.73 | 19.597 |
| 4 | 77.19 | 51.112 | 55.87 |
| 8 | 80.16 | 98.987 | 104.072 |

## Size and memory

Artifact size (MB): `{"detector": 1.664, "ocr": 4.98, "total": 6.644}` · peak RSS 343.1 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": true, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.000274658203125, "box_count_equal": true, "min_matched_box_iou": 0.9999994039535522, "strings_equal": 1.0, "passed": true}
```

## Notes

ratio 0.753 to reach the student's 451,357 params: {'params': 377789, 'zero_params': 0, 'zero_fraction': 0.0}

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
 "load_avg_1m": 6.24
}
```
