# Benchmark card — `s09_openvino:student-int8-nncf-fp32-decode`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s09_openvino:student-int8-nncf` |
| Runtime · device · precision | openvino · cpu · int8 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:05:09+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8946** · plate exact match **0.7392**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9017 | 0.7954 | 347 |
| night | 0.8417 | 0.6483 | 145 |
| rain | 0.9256 | 0.7216 | 176 |
| motion_blur | 0.906 | 0.7903 | 124 |
| low_contrast | 0.9159 | 0.6774 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **10.628 ms** · p95 **12.169 ms** · p99 13.386 ms · first call of the process 22.387 ms
(includes compilation / engine build) · first call of the timing loop 10.184 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.246 | 3.023 |
| preprocess | 3.327 | 3.44 |
| detect | 4.919 | 5.27 |
| nms | 0.206 | 0.246 |
| crop | 0.102 | 0.147 |
| ocr | 0.671 | 0.959 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 93.87 | 10.534 | 12.113 |
| 4 | 101.35 | 39.277 | 41.275 |
| 8 | 100.83 | 79.06 | 82.636 |

## Size and memory

Artifact size (MB): `{"detector": 0.797, "ocr": 0.22, "total": 1.017}` · peak RSS 605.2 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 101.64420318603516, "box_count_equal": true, "min_matched_box_iou": 0.9554895162582397, "strings_equal": 0.9, "passed": null}
```

## Notes

the fix s06a needed on ONNX Runtime — the detector's decode kept FP32 — and on this runtime it changes almost nothing: here the loss is in the recognizer

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
 "load_avg_1m": 3.36
}
```
