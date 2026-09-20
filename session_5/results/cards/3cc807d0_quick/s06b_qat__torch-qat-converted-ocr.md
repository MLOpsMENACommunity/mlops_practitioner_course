# Benchmark card — `s06b_qat:torch-qat-converted-ocr`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s03_torchscript:jit-trace-fp32` |
| Runtime · device · precision | torchscript · cpu · int8-qat |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T21:00:42+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9048** · plate exact match **0.9303**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9184 | 0.9597 | 347 |
| night | 0.8436 | 0.9379 | 145 |
| rain | 0.9292 | 0.9205 | 176 |
| motion_blur | 0.917 | 0.9516 | 124 |
| low_contrast | 0.9276 | 0.8516 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **55.62 ms** · p95 **68.056 ms** · p99 82.478 ms · first call of the process 100.085 ms
(includes compilation / engine build) · first call of the timing loop 50.097 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.33 | 3.247 |
| preprocess | 3.436 | 3.784 |
| detect | 46.489 | 58.618 |
| nms | 0.246 | 0.296 |
| crop | 0.134 | 0.189 |
| ocr | 3.679 | 4.904 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 18.46 | 54.461 | 56.4 |
| 4 | 17.29 | 231.253 | 236.739 |
| 8 | 16.03 | 460.835 | 623.757 |

## Size and memory

Artifact size (MB): `{"detector": 32.578, "ocr": 3.352, "total": 35.93}` · peak RSS 1204.1 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 4.408419132232666, "box_count_equal": true, "min_matched_box_iou": 1.0, "strings_equal": 0.9, "passed": null}
```

## Notes

convert_fx -> torch.jit.trace; LSTM stays FP32

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
 "load_avg_1m": 6.83
}
```
