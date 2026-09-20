# Benchmark card — `s05_pruning:sliced-oneshot-50-onnx`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s05_pruning:sliced-oneshot-50` |
| Runtime · device · precision | ort · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:45:35+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.9188** · plate exact match **0.9303**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.927 | 0.9452 | 347 |
| night | 0.8756 | 0.9448 | 145 |
| rain | 0.928 | 0.9261 | 176 |
| motion_blur | 0.9217 | 0.9597 | 124 |
| low_contrast | 0.944 | 0.8645 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **35.794 ms** · p95 **40.309 ms** · p99 41.102 ms · first call of the process 44.284 ms
(includes compilation / engine build) · first call of the timing loop 30.159 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.322 | 3.299 |
| preprocess | 3.508 | 5.407 |
| detect | 26.833 | 27.648 |
| nms | 0.227 | 0.258 |
| crop | 0.112 | 0.158 |
| ocr | 3.515 | 6.909 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 27.6 | 36.64 | 38.751 |
| 4 | 31.02 | 128.448 | 132.531 |
| 8 | 30.16 | 260.548 | 291.744 |

## Size and memory

Artifact size (MB): `{"detector": 8.31, "ocr": 4.98, "total": 13.29}` · peak RSS 494.5 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": true, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.0004100799560546875, "box_count_equal": true, "min_matched_box_iou": 0.9999985694885254, "strings_equal": 1.0, "passed": true}
```

## Notes

{'params': 2121365, 'zero_params': 0, 'zero_fraction': 0.0}

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
 "load_avg_1m": 6.23
}
```
