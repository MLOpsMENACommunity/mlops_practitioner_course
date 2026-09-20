# Benchmark card — `s04_onnx_export:ort-cpu-fp32-nms-in-graph`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s04_onnx_export:ort-cpu-fp32` |
| Runtime · device · precision | ort · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:38:45+0300 |

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

p50 **100.476 ms** · p95 **124.749 ms** · p99 142.14 ms · first call of the process 105.535 ms
(includes compilation / engine build) · first call of the timing loop 91.901 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.472 | 3.858 |
| preprocess | 3.624 | 5.871 |
| detect | 89.552 | 111.183 |
| nms | 0.033 | 0.053 |
| crop | 0.162 | 0.294 |
| ocr | 3.6 | 7.2 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 9.52 | 102.274 | 122.532 |
| 4 | 9.83 | 400.881 | 458.657 |
| 8 | 10.38 | 734.192 | 987.579 |

## Size and memory

Artifact size (MB): `{"detector": 32.517, "ocr": 4.98, "total": 37.497}` · peak RSS 620.7 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": true, "rtol": 0.001, "atol": 1e-05, "max_abs_diff": 6.103515625e-05, "box_count_equal": true, "min_matched_box_iou": 0.999998152256012, "strings_equal": 1.0, "passed": true}
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
 "load_avg_1m": 7.26
}
```
