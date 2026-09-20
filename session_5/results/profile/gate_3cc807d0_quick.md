# Profile gate — Apple M3 Pro, profile quick

p50 54.121 ms, p95 58.337 ms, batch 1

| phase | mean ms (all frames) | share | mean ms (slowest 5%) | share |
|---|---|---|---|---|
| decode | 1.52 | 3% | 1.74 | 3% |
| preprocess | 3.38 | 6% | 3.54 | 6% |
| detect | 46.07 | 85% | 53.29 | 84% |
| nms | 0.23 | 0% | 0.28 | 0% |
| crop | 0.12 | 0% | 0.18 | 0% |
| ocr | 3.10 | 6% | 4.66 | 7% |

Model share of the slowest 5%: **91%** (of all frames: 90%) — the MODELS dominate the tail — model-level optimization is the right next step.

## Pre/post-processing fixes

Noise floor — the unchanged baseline measured three times: p50 spread 0.187 ms, p95 spread 2.379 ms.

| fix | targeted phase | phase p50 ms: baseline → fix | end-to-end p50 Δ ms | end-to-end p95 Δ ms | end-to-end change |
|---|---|---|---|---|---|
| `fast-resize` | preprocess | 3.34 → 1.14 | -1.9 | -4.0 | faster, beyond 2x the noise floor |
| `nms-in-graph` | nms | 0.23 → 0.02 | -0.0 | -1.9 | within run-to-run noise: read the phase column, not the total |
| `gpu-decode` | decode | — | — | — | not_run: no CUDA device: nvJPEG decode needs an NVIDIA GPU |
