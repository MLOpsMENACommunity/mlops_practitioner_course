# Benchmark card — `s09_openvino:student-int8-nncf`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s09_openvino:student-fp32-latency` |
| Runtime · device · precision | openvino · cpu · int8 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:04:53+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8934** · plate exact match **0.7318**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9006 | 0.7896 | 347 |
| night | 0.843 | 0.669 | 145 |
| rain | 0.9207 | 0.7045 | 176 |
| motion_blur | 0.9029 | 0.7419 | 124 |
| low_contrast | 0.917 | 0.6839 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **10.58 ms** · p95 **12.333 ms** · p99 13.158 ms · first call of the process 22.211 ms
(includes compilation / engine build) · first call of the timing loop 10.041 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.236 | 3.002 |
| preprocess | 3.313 | 3.45 |
| detect | 4.922 | 5.205 |
| nms | 0.204 | 0.236 |
| crop | 0.1 | 0.142 |
| ocr | 0.684 | 0.956 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 92.11 | 10.731 | 12.552 |
| 4 | 101.36 | 39.305 | 41.249 |
| 8 | 100.04 | 79.079 | 84.32 |

## Size and memory

Artifact size (MB): `{"detector": 0.789, "ocr": 0.22, "total": 1.009}` · peak RSS 602.8 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 101.64420318603516, "box_count_equal": true, "min_matched_box_iou": 0.9508662819862366, "strings_equal": 0.9, "passed": null}
```

## Notes

everything quantized, decode included — the same failure s06a measures on ONNX Runtime

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
 "load_avg_1m": 3.18
}
```
