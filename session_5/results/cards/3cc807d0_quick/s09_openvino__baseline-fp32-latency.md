# Benchmark card — `s09_openvino:baseline-fp32-latency`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | openvino · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:03:29+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9048** · plate exact match **0.9335**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9184 | 0.9568 | 347 |
| night | 0.8436 | 0.931 | 145 |
| rain | 0.9292 | 0.9318 | 176 |
| motion_blur | 0.917 | 0.9597 | 124 |
| low_contrast | 0.9276 | 0.8645 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **49.073 ms** · p95 **54.632 ms** · p99 70.697 ms · first call of the process 130.767 ms
(includes compilation / engine build) · first call of the timing loop 41.282 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.304 | 3.137 |
| preprocess | 3.351 | 3.495 |
| detect | 36.583 | 41.555 |
| nms | 0.237 | 0.266 |
| crop | 0.131 | 0.179 |
| ocr | 7.071 | 9.472 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 20.73 | 48.736 | 51.338 |
| 4 | 22.33 | 179.165 | 183.551 |
| 8 | 21.08 | 365.955 | 456.312 |

## Size and memory

Artifact size (MB): `{"detector": 32.394, "ocr": 4.879, "total": 37.273}` · peak RSS 2955.9 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.00335693359375, "box_count_equal": true, "min_matched_box_iou": 0.9997584223747253, "strings_equal": 1.0, "passed": null}
```

## Notes

parity recorded, not enforced: OpenVINO on arm64

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
 "load_avg_1m": 4.13
}
```
