# Benchmark card — `s05_pruning:masked-structured-50-onnx`

> Generated from results/results.json by `python -m tools.benchmark_card`. Store it next to
> the artifact in the model registry (Session 2: `mlflow.log_artifact(card_path)`), one card
> per format and per hardware target. A number without this card is a rumour.

| | |
|---|---|
| Built from | `s05_pruning:masked-structured-50` |
| Runtime · device · precision | ort · cpu · fp32 |
| SLA verdict | **MISSES p95 <= 30.0 ms** |
| Profile · validation sha256 | `quick` · `aceb3351337989b1aad220931918af912eda18d7d842d697b52fc7693ada479c` |
| Git commit · measured at | `1e513a6` · 2026-09-20T20:42:28+0300 |

## Accuracy (full validation set, untimed pass)

mAP@0.5 **0.8966** · plate exact match **0.9345**

| condition | mAP@0.5 | exact match | plates |
|---|---|---|---|
| day | 0.9139 | 0.9597 | 347 |
| night | 0.8596 | 0.9586 | 145 |
| rain | 0.9076 | 0.9091 | 176 |
| motion_blur | 0.9106 | 0.9355 | 124 |
| low_contrast | 0.9146 | 0.8839 | 155 |

## Latency (batch 1, 200 runs after 20 warm-up calls)

p50 **100.734 ms** · p95 **110.178 ms** · p99 124.439 ms · first call of the process 111.553 ms
(includes compilation / engine build) · first call of the timing loop 93.706 ms

| phase | p50 ms | p95 ms |
|---|---|---|
| decode | 1.464 | 3.821 |
| preprocess | 3.645 | 5.938 |
| detect | 90.3 | 97.021 |
| nms | 0.261 | 0.317 |
| crop | 0.151 | 0.204 |
| ocr | 3.662 | 7.332 |

## Throughput

| batch / clients | frames/s | p50 ms | p95 ms |
|---|---|---|---|
| 1 | 9.91 | 100.617 | 105.558 |
| 4 | 10.4 | 381.241 | 405.503 |
| 8 | 10.48 | 757.462 | 781.611 |

## Size and memory

Artifact size (MB): `{"detector": 32.484, "ocr": 4.98, "total": 37.464}` · peak RSS 715.6 MB · peak VRAM None MB

## Parity report vs parent

```json
{"frames": 4, "strict": true, "rtol": 0.001, "atol": 0.0001, "max_abs_diff": 0.0003662109375, "box_count_equal": true, "min_matched_box_iou": 0.9999991655349731, "strings_equal": 1.0, "passed": true}
```

## Notes

{'params': 8461093, 'zero_params': 4223072, 'zero_fraction': 0.4991} of 8,461,093; after fine-tuning, zero fraction is 29.5% when prune.remove ran FIRST vs 49.9% when masks stayed attached

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
 "load_avg_1m": 9.56
}
```
