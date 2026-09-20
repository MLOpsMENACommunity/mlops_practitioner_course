# Benchmark card — `s09_openvino:student-fp32-throughput`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s09_openvino:student-fp32-latency` |
| Runtime · device · precision | openvino · cpu · fp32 |
| SLA verdict | **meets p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:04:37+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8978** · plate exact match **0.9472**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9026 | 0.9856 | 347 |
| night | 0.8533 | 0.9586 | 145 |
| rain | 0.9268 | 0.9489 | 176 |
| motion_blur | 0.9051 | 0.9758 | 124 |
| low_contrast | 0.918 | 0.8258 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **21.233 ms** · p95 **23.406 ms** · p99 24.772 ms · first call of the process 35.96 ms
(includes compilation / engine build) · first call of the timing loop 19.317 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.259 | 3.002 |
| preprocess | 3.319 | 3.65 |
| detect | 14.592 | 15.715 |
| nms | 0.212 | 0.271 |
| crop | 0.11 | 0.165 |
| ocr | 1.521 | 2.182 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 47.27 | 21.223 | 22.036 |
| 4 | 71.07 | 56.087 | 58.655 |
| 8 | 77.15 | 102.568 | 106.761 |

## Size and memory

Artifact size (MB): `{"detector": 1.868, "ocr": 0.666, "total": 2.534}` · peak RSS 557.8 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.0031890869140625, "box_count_equal": true, "min_matched_box_iou": 0.999680757522583, "strings_equal": 1.0, "passed": null}
```

## Notes

THROUGHPUT hint: the plugin creates parallel streams; a batch fans out over AsyncInferQueue; parity recorded, not enforced: OpenVINO on arm64

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
 "load_avg_1m": 3.22
}
```
