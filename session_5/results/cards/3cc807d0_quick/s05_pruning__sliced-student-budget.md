# Benchmark card — `s05_pruning:sliced-student-budget`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s01_baseline:eager-fp32` |
| Runtime · device · precision | torch · cpu · fp32 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:45:59+0300 |

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

p50 **17.072 ms** · p95 **18.823 ms** · p99 20.069 ms · first call of the process 25.268 ms
(includes compilation / engine build) · first call of the timing loop 13.293 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.263 | 3.062 |
| preprocess | 3.328 | 3.494 |
| detect | 8.568 | 9.232 |
| nms | 0.22 | 0.26 |
| crop | 0.109 | 0.16 |
| ocr | 3.466 | 4.126 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 59.8 | 17.147 | 17.854 |
| 4 | 71.85 | 54.389 | 59.269 |
| 8 | 66.29 | 114.482 | 144.976 |

## Size and memory

Artifact size (MB): `{"detector": 1.558, "ocr": 4.871, "total": 6.429}` · peak RSS 981.9 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 74.3367919921875, "box_count_equal": true, "min_matched_box_iou": 0.7370740175247192, "strings_equal": 1.0, "passed": null}
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
 "load_avg_1m": 5.14
}
```
