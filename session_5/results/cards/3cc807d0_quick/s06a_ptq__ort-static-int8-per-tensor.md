# Benchmark card — `s06a_ptq:ort-static-int8-per-tensor`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · int8 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:54:22+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8926** · plate exact match **0.8553**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9059 | 0.8991 | 347 |
| night | 0.828 | 0.8621 | 145 |
| rain | 0.9099 | 0.8239 | 176 |
| motion_blur | 0.9196 | 0.879 | 124 |
| low_contrast | 0.9137 | 0.7677 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **32.38 ms** · p95 **35.447 ms** · p99 36.641 ms · first call of the process 42.507 ms
(includes compilation / engine build) · first call of the timing loop 28.135 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.327 | 3.479 |
| preprocess | 3.499 | 5.008 |
| detect | 24.25 | 24.942 |
| nms | 0.121 | 0.147 |
| crop | 0.112 | 0.155 |
| ocr | 2.562 | 4.439 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 31.02 | 32.296 | 34.55 |
| 4 | 33.09 | 120.167 | 127.046 |
| 8 | 33.93 | 235.571 | 238.931 |

## Size and memory

Artifact size (MB): `{"detector": 8.278, "ocr": 3.44, "total": 11.718}` · peak RSS 313.6 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 18.549407958984375, "box_count_equal": true, "min_matched_box_iou": 0.5822523236274719, "strings_equal": 1.0, "passed": null}
```

## Notes

calibration: stratified, per-tensor

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
 "load_avg_1m": 5.66
}
```
