# Benchmark card — `s06a_ptq:ort-static-int8-fp32-decode-daytime`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · int8 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:50:35+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9046** · plate exact match **0.9303**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9199 | 0.9654 | 347 |
| night | 0.8495 | 0.9241 | 145 |
| rain | 0.9264 | 0.9148 | 176 |
| motion_blur | 0.9175 | 0.9435 | 124 |
| low_contrast | 0.9294 | 0.8645 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **32.661 ms** · p95 **35.828 ms** · p99 37.439 ms · first call of the process 45.936 ms
(includes compilation / engine build) · first call of the timing loop 28.105 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.324 | 3.201 |
| preprocess | 3.502 | 5.855 |
| detect | 24.379 | 25.194 |
| nms | 0.189 | 0.213 |
| crop | 0.117 | 0.16 |
| ocr | 2.577 | 4.455 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 30.84 | 32.487 | 34.832 |
| 4 | 32.67 | 120.223 | 132.778 |
| 8 | 28.61 | 236.525 | 568.204 |

## Size and memory

Artifact size (MB): `{"detector": 8.364, "ocr": 3.449, "total": 11.813}` · peak RSS 319.4 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 12.454208374023438, "box_count_equal": true, "min_matched_box_iou": 0.9666069746017456, "strings_equal": 1.0, "passed": null}
```

## Notes

daytime calibration, decode kept FP32

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
 "load_avg_1m": 7.09
}
```
