# Benchmark card — `s06a_ptq:torch-dynamic-int8-ocr`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s01_baseline:eager-fp32` |
| Runtime · device · precision | torch · cpu · int8-dyn |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:56:26+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9048** · plate exact match **0.9324**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9184 | 0.9568 | 347 |
| night | 0.8436 | 0.931 | 145 |
| rain | 0.9292 | 0.9318 | 176 |
| motion_blur | 0.917 | 0.9597 | 124 |
| low_contrast | 0.9276 | 0.8581 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **55.144 ms** · p95 **57.132 ms** · p99 60.561 ms · first call of the process 77.837 ms
(includes compilation / engine build) · first call of the timing loop 50.624 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.266 | 3.042 |
| preprocess | 3.331 | 3.389 |
| detect | 45.622 | 46.448 |
| nms | 0.232 | 0.257 |
| crop | 0.123 | 0.17 |
| ocr | 4.359 | 5.02 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 18.15 | 55.242 | 57.608 |
| 4 | 17.27 | 230.495 | 239.787 |
| 8 | 16.91 | 466.7 | 550.912 |

## Size and memory

Artifact size (MB): `{"detector": 32.378, "ocr": 2.809, "total": 35.187}` · peak RSS 1285.2 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.5548439025878906, "box_count_equal": true, "min_matched_box_iou": 1.0, "strings_equal": 1.0, "passed": null}
```

## Notes

conv layers untouched: dynamic quantization has no activation ranges to give a conv

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
 "load_avg_1m": 4.3
}
```
