# Benchmark card — `s09_openvino:student-device-default-precision`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s09_openvino:student-fp32-latency` |
| Runtime · device · precision | openvino · cpu · device-default |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:03:55+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8981** · plate exact match **0.9419**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9042 | 0.9856 | 347 |
| night | 0.8481 | 0.9448 | 145 |
| rain | 0.9268 | 0.9432 | 176 |
| motion_blur | 0.9045 | 0.9758 | 124 |
| low_contrast | 0.9197 | 0.8129 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **8.806 ms** · p95 **10.446 ms** · p99 11.124 ms · first call of the process 19.412 ms
(includes compilation / engine build) · first call of the timing loop 8.387 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.227 | 2.987 |
| preprocess | 3.309 | 3.393 |
| detect | 3.376 | 3.551 |
| nms | 0.198 | 0.231 |
| crop | 0.095 | 0.142 |
| ocr | 0.486 | 0.692 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 112.97 | 8.735 | 9.831 |
| 4 | 119.93 | 33.042 | 35.139 |
| 8 | 119.86 | 66.282 | 70.854 |

## Size and memory

Artifact size (MB): `{"detector": 1.868, "ocr": 0.666, "total": 2.534}` · peak RSS 538.7 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.91094970703125, "box_count_equal": true, "min_matched_box_iou": 0.9801679253578186, "strings_equal": 1.0, "passed": null}
```

## Notes

no INFERENCE_PRECISION_HINT: the plugin's own choice; the row's inference_precision says which

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
 "load_avg_1m": 3.8
}
```
