# Benchmark card — `s06a_ptq:ort-static-int8-mixed`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s06a_ptq:ort-static-int8-stratified` |
| Runtime · device · precision | ort · cpu · int8-mixed |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:51:29+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.906** · plate exact match **0.9345**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.921 | 0.9654 | 347 |
| night | 0.844 | 0.9448 | 145 |
| rain | 0.9291 | 0.9318 | 176 |
| motion_blur | 0.9174 | 0.9516 | 124 |
| low_contrast | 0.9292 | 0.8452 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **39.177 ms** · p95 **43.023 ms** · p99 44.146 ms · first call of the process 46.738 ms
(includes compilation / engine build) · first call of the timing loop 33.357 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.328 | 3.35 |
| preprocess | 3.503 | 5.29 |
| detect | 30.303 | 31.318 |
| nms | 0.234 | 0.259 |
| crop | 0.122 | 0.162 |
| ocr | 3.234 | 6.368 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 25.61 | 39.133 | 42.242 |
| 4 | 27.44 | 145.785 | 149.314 |
| 8 | 27.71 | 285.792 | 297.897 |

## Size and memory

Artifact size (MB): `{"detector": 8.401, "ocr": 3.738, "total": 12.139}` · peak RSS 684.4 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": false, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 11.29541015625, "box_count_equal": true, "min_matched_box_iou": 0.9767089486122131, "strings_equal": 1.0, "passed": null}
```

## Notes

FP32 kept: {'ocr_kept_fp32': ['cnn.0', 'cnn.2', 'cnn.4', 'fc'], 'detector_kept_fp32': ['decode', 'detector.box', 'detector.c3.1', 'detector.obj']}

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
 "load_avg_1m": 6.78
}
```
