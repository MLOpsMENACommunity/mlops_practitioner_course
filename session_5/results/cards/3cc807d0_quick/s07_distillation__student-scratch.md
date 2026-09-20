# Benchmark card — `s07_distillation:student-scratch`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s01_baseline:eager-fp32` |
| Runtime · device · precision | torch · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:02:22+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8999** · plate exact match **0.9324**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9149 | 0.9914 | 347 |
| night | 0.8548 | 0.9034 | 145 |
| rain | 0.9201 | 0.9205 | 176 |
| motion_blur | 0.9065 | 0.9758 | 124 |
| low_contrast | 0.9164 | 0.8065 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **28.788 ms** · p95 **30.952 ms** · p99 32.212 ms · first call of the process 37.526 ms
(includes compilation / engine build) · first call of the timing loop 27.429 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.255 | 3.033 |
| preprocess | 3.319 | 3.413 |
| detect | 22.752 | 23.383 |
| nms | 0.228 | 0.254 |
| crop | 0.121 | 0.164 |
| ocr | 1.008 | 1.239 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 34.56 | 28.858 | 29.825 |
| 4 | 47.19 | 84.461 | 87.211 |
| 8 | 54.15 | 145.255 | 156.879 |

## Size and memory

Artifact size (MB): `{"detector": 1.804, "ocr": 0.66, "total": 2.464}` · peak RSS 765.8 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 1e-05, "max_abs_diff": 103.89801788330078, "box_count_equal": true, "min_matched_box_iou": 0.8701735138893127, "strings_equal": 0.8, "passed": null}
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
  "onnx": "1.22.0",
  "onnxruntime": "1.30.0",
  "openvino": "2026.3.1",
  "nncf": "3.3.0",
  "ai-edge-litert": "2.2.0",
  "tritonclient": "2.72.0",
  "numpy": "2.4.6",
  "pillow": "12.3.0"
 },
 "load_avg_1m": 4.69
}
```
