# Benchmark card — `s05_pruning:sliced-iterative-50`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s01_baseline:eager-fp32` |
| Runtime · device · precision | torch · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:43:12+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9126** · plate exact match **0.9335**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9193 | 0.951 | 347 |
| night | 0.8644 | 0.9379 | 145 |
| rain | 0.9283 | 0.9148 | 176 |
| motion_blur | 0.9273 | 0.9597 | 124 |
| low_contrast | 0.9278 | 0.8903 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **30.368 ms** · p95 **32.82 ms** · p99 33.959 ms · first call of the process 38.012 ms
(includes compilation / engine build) · first call of the timing loop 25.818 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.358 | 3.171 |
| preprocess | 3.486 | 3.616 |
| detect | 21.396 | 22.667 |
| nms | 0.25 | 0.294 |
| crop | 0.138 | 0.19 |
| ocr | 3.644 | 4.412 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 29.33 | 32.734 | 39.868 |
| 4 | 32.64 | 122.453 | 129.766 |
| 8 | 32.57 | 238.535 | 262.874 |

## Size and memory

Artifact size (MB): `{"detector": 8.226, "ocr": 4.871, "total": 13.097}` · peak RSS 1023.3 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 54.40802001953125, "box_count_equal": true, "min_matched_box_iou": 0.9334323406219482, "strings_equal": 1.0, "passed": null}
```

## Notes

{'params': 2121365, 'zero_params': 0, 'zero_fraction': 0.0}

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
 "load_avg_1m": 7.95
}
```
