# 16 — Decision guide: which path do I take?

Thirteen stages produced a table, not a recommendation. This guide turns the table
into a recommendation for **your** deployment target. Choose the target first,
because the right sequence of stages for a Raspberry Pi is not the right sequence
for a Jetson or for a GPU server. For each target it gives the stages to run in
order, the rows to read to decide, the SLA check that settles it, and what a
`not_run` row means for that target. At the end: a table from symptoms in the
journey table to the next lever, the fleet cost arithmetic, and the case for not
optimizing at all.

---

## Three rules before any path

1. **Profile first, on every target.** `make profile` runs s00 and its gate
   (`snippet:profile-gate`). If the model is under half of the p95 tail, the next
   fix is in preprocessing and postprocessing (guide 00: fast resize, NMS in the
   graph, GPU decode), and no runtime or precision lever will help yet.
2. **The SLA is decided on the target, at batch 1.** `SLA.p95_ms` (30 ms) and
   `SLA.batch_size` (one frame at a time) in `src/config.py` describe a camera. A
   row measured anywhere else is evidence about *direction*, never a verdict.
3. **`not_run` is missing evidence, not bad evidence.** Every `not_run` row carries
   its reason: no CUDA, no Intel GPU, not measured on the device. It means "this
   machine could not answer". A path that depends on a `not_run` row is undecided
   until someone runs that stage on the hardware that can.

> **Why the target comes before the lever.** Levers do not compose the same way
> everywhere. INT8 on a CPU without VNNI-class instructions, dynamic batching on a
> compute-saturated CPU, FP8 on an Ampere GPU: each is a great lever somewhere and a
> non-event or a failure somewhere else. Session 3 found the same thing about
> `channels_last`: a flag that helps on one device and hurts on another is the
> normal case.

---

## The measured journey

<!-- results:journey -->
#### Hardware `3cc807d0` · profile `quick`

**Measured on:** Apple M3 Pro · 18.0 GB RAM · GPU: none · Darwin 25.5.0 arm64 · Python 3.12.12
**Threads:** ANPR_THREADS=4, OMP_NUM_THREADS=4, ORT intra_op_num_threads=4, inter_op=1
**Latency batch size:** 1 · **SLA:** p95 <= 30.0 ms per frame · **Profile:** `quick` · **Validation set sha256:** `aceb33513379`
**Libraries (as loaded by the rows below):** torch 2.13.0, onnxruntime 1.30.0, openvino 2026.3.1, nncf 3.3.0, ai-edge-litert 2.2.0

| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `s00_profile:fast-resize` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.904 | 0.938 | 52.2 | 54.3 | no | 19.2 (b1) | 37.25 | 1417 | — | ok |
| `s00_profile:gpu-decode` | `s01_baseline:eager-fp32` | torch · cuda · fp32 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: nvJPEG decode needs an NVIDIA GPU |
| `s00_profile:nms-in-graph` | `s01_baseline:eager-fp32` | torch · cpu · fp32 · NMS in graph | 0.905 | 0.933 | 54.1 | 56.4 | no | 18.5 (b1) | 37.25 | 1395 | — | ok |
| `s01_baseline:eager-fp32` | (root) | torch · cpu · fp32 | 0.905 | 0.933 | 54.1 | 58.3 | no | 18.5 (b1) | 37.25 | 1316 | — | ok |
| `s02_torch_compile:inductor-fp32` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.905 | 0.933 | 50.1 | 53.0 | no | 20.1 (b1) | 37.25 | 1221 | — | ok |
| `s03_torchscript:jit-trace-fp32` | `s01_baseline:eager-fp32` | torchscript · cpu · fp32 | 0.905 | 0.933 | 54.3 | 61.0 | no | 18.3 (b1) | 37.49 | 1323 | — | ok |
| `s04_onnx_export:ort-cpu-fp32` | `s01_baseline:eager-fp32` | ort · cpu · fp32 | 0.905 | 0.933 | 105.8 | 133.8 | no | 10.2 (b8) | 37.46 | 578 | — | ok |
| `s04_onnx_export:ort-cpu-fp32-nms-in-graph` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · fp32 · NMS in graph | 0.905 | 0.933 | 100.5 | 124.7 | no | 10.4 (b8) | 37.50 | 621 | — | ok |
| `s04_onnx_export:ort-cuda-fp32` | `s01_baseline:eager-fp32` | ort · cuda · fp32 | — | — | — | — | — | — | — | — | — | not_run: CUDAExecutionProvider not available (needs onnxruntime-gpu + NVIDIA GPU) |
| `s04_onnx_export:ort-openvino-ep-cpu` | `s01_baseline:eager-fp32` | ort · cpu · fp32 | — | — | — | — | — | — | — | — | — | not_run: OpenVINOExecutionProvider not available (install onnxruntime-openvino in its own venv; it  |
| `s04_onnx_export:ort-tensorrt-ep-fp16` | `s01_baseline:eager-fp32` | ort · cuda · fp16 | — | — | — | — | — | — | — | — | — | not_run: TensorrtExecutionProvider not available (onnxruntime-gpu + TensorRT 10 libs) |
| `s05_pruning:masked-structured-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.897 | 0.934 | 57.3 | 69.8 | no | 18.3 (b1) | 37.25 | 1227 | — | ok |
| `s05_pruning:masked-structured-50-onnx` | `s05_pruning:masked-structured-50` | ort · cpu · fp32 | 0.897 | 0.934 | 100.7 | 110.2 | no | 10.5 (b8) | 37.46 | 716 | — | ok |
| `s05_pruning:sliced-iterative-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.913 | 0.933 | 30.4 | 32.8 | no | 32.6 (b4) | 13.10 | 1023 | — | ok |
| `s05_pruning:sliced-iterative-50-onnx` | `s05_pruning:sliced-iterative-50` | ort · cpu · fp32 | 0.913 | 0.933 | 37.6 | 41.8 | no | 29.9 (b8) | 13.29 | 494 | — | ok |
| `s05_pruning:sliced-oneshot-50` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.919 | 0.930 | 29.9 | 43.7 | no | 36.0 (b1) | 13.10 | 980 | — | ok |
| `s05_pruning:sliced-oneshot-50-onnx` | `s05_pruning:sliced-oneshot-50` | ort · cpu · fp32 | 0.919 | 0.930 | 35.8 | 40.3 | no | 31.0 (b4) | 13.29 | 494 | — | ok |
| `s05_pruning:sliced-student-budget` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.859 | 0.902 | 17.1 | 18.8 | yes | 71.8 (b4) | 6.43 | 982 | 4.645 (b1) | ok |
| `s05_pruning:sliced-student-budget-onnx` | `s05_pruning:sliced-student-budget` | ort · cpu · fp32 | 0.859 | 0.902 | 15.8 | 20.1 | yes | 80.2 (b8) | 6.64 | 343 | 4.448 (b1) | ok |
| `s06a_ptq:ort-dynamic-int8` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-dyn | 0.905 | 0.936 | 45.6 | 48.3 | no | 22.5 (b4) | 11.68 | 768 | — | ok |
| `s06a_ptq:ort-static-int8-daytime` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.899 | 0.871 | 34.3 | 44.1 | no | 31.8 (b8) | 11.79 | 308 | — | ok |
| `s06a_ptq:ort-static-int8-detector-only` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-det | 0.892 | 0.853 | 33.1 | 37.3 | no | 32.5 (b4) | 13.33 | 383 | — | ok |
| `s06a_ptq:ort-static-int8-fp32-decode` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.904 | 0.931 | 33.9 | 43.8 | no | 33.0 (b8) | 11.81 | 326 | — | ok |
| `s06a_ptq:ort-static-int8-fp32-decode-daytime` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.905 | 0.930 | 32.7 | 35.8 | no | 32.7 (b4) | 11.81 | 319 | — | ok |
| `s06a_ptq:ort-static-int8-mixed` | `s06a_ptq:ort-static-int8-stratified` | ort · cpu · int8-mixed | 0.906 | 0.934 | 39.2 | 43.0 | no | 27.7 (b8) | 12.14 | 684 | — | ok |
| `s06a_ptq:ort-static-int8-ocr-only` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8-ocr | 0.905 | 0.933 | 95.6 | 99.1 | no | 11.1 (b4) | 35.93 | 679 | — | ok |
| `s06a_ptq:ort-static-int8-per-tensor` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.893 | 0.855 | 32.4 | 35.4 | no | 33.9 (b8) | 11.72 | 314 | — | ok |
| `s06a_ptq:ort-static-int8-stratified` | `s04_onnx_export:ort-cpu-fp32` | ort · cpu · int8 | 0.892 | 0.854 | 32.6 | 35.9 | no | 33.6 (b8) | 11.79 | 366 | — | ok |
| `s06a_ptq:torch-dynamic-int8-ocr` | `s01_baseline:eager-fp32` | torch · cpu · int8-dyn | 0.905 | 0.932 | 55.1 | 57.1 | no | 18.1 (b1) | 35.19 | 1285 | — | ok |
| `s06b_qat:ort-qat-int8-ocr` | `s06a_ptq:ort-static-int8-stratified` | ort · cpu · int8-qat | — | — | — | — | — | — | — | — | — | failed: libc++abi: terminating due to uncaught exception of type std::__1::system_error: recursive |
| `s06b_qat:ort-qat-int8-ocr-only` | `s06a_ptq:ort-static-int8-ocr-only` | ort · cpu · int8-qat | 0.905 | 0.930 | 97.2 | 101.6 | no | 11.0 (b8) | 37.37 | 679 | — | ok |
| `s06b_qat:torch-qat-converted-ocr` | `s03_torchscript:jit-trace-fp32` | torchscript · cpu · int8-qat | 0.905 | 0.930 | 55.6 | 68.1 | no | 18.5 (b1) | 35.93 | 1204 | — | ok |
| `s07_distillation:student-distilled` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.898 | 0.947 | 28.8 | 30.7 | no | 53.6 (b8) | 2.46 | 625 | 8.026 (b1) | ok |
| `s07_distillation:student-distilled-onnx` | `s07_distillation:student-distilled` | ort · cpu · fp32 | 0.898 | 0.947 | 16.4 | 19.3 | yes | 69.2 (b4) | 2.68 | 467 | 4.464 (b1) | ok |
| `s07_distillation:student-scratch` | `s01_baseline:eager-fp32` | torch · cpu · fp32 | 0.900 | 0.932 | 28.8 | 31.0 | no | 54.1 (b8) | 2.46 | 766 | 8.038 (b1) | ok |
| `s08_tensorrt:baseline-fp16` | `s04_onnx_export:ort-cpu-fp32` | tensorrt · cuda · fp16 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-fp16` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · fp16 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-fp32` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · fp32 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-int8-daytime` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · int8 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s08_tensorrt:student-int8-stratified` | `s07_distillation:student-distilled-onnx` | tensorrt · cuda · int8 | — | — | — | — | — | — | — | — | — | not_run: no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU |
| `s09_openvino:baseline-fp32-latency` | `s04_onnx_export:ort-cpu-fp32` | openvino · cpu · fp32 | 0.905 | 0.933 | 49.1 | 54.6 | no | 22.3 (b4) | 37.27 | 2956 | — | ok |
| `s09_openvino:edge-candidate` | `s09_openvino:student-int8-nncf-fp32-head` | openvino · cpu · int8 · NMS in graph | 0.804 | 0.868 | 8.2 | 9.9 | yes | 131.2 (b8) | 1.06 | 598 | 2.278 (b1) | ok |
| `s09_openvino:intel-gpu` | `s09_openvino:student-fp32-latency` | openvino · gpu · fp32 | — | — | — | — | — | — | — | — | — | not_run: no OpenVINO GPU device (available: ['CPU']); the GPU plugin targets Intel GPUs |
| `s09_openvino:student-device-default-precision` | `s09_openvino:student-fp32-latency` | openvino · cpu · device-default | 0.898 | 0.942 | 8.8 | 10.4 | yes | 119.9 (b4) | 2.53 | 539 | 2.459 (b1) | ok |
| `s09_openvino:student-fp32-latency` | `s07_distillation:student-distilled-onnx` | openvino · cpu · fp32 | 0.898 | 0.947 | 10.1 | 12.1 | yes | 102.8 (b4) | 2.53 | 772 | — | ok |
| `s09_openvino:student-fp32-throughput` | `s09_openvino:student-fp32-latency` | openvino · cpu · fp32 | 0.898 | 0.947 | 21.2 | 23.4 | yes | 77.2 (b8) | 2.53 | 558 | 5.876 (b1) | ok |
| `s09_openvino:student-int8-nncf` | `s09_openvino:student-fp32-latency` | openvino · cpu · int8 | 0.893 | 0.732 | 10.6 | 12.3 | yes | 101.4 (b4) | 1.01 | 603 | 3.016 (b1) | ok |
| `s09_openvino:student-int8-nncf-fp32-decode` | `s09_openvino:student-int8-nncf` | openvino · cpu · int8 | 0.895 | 0.739 | 10.6 | 12.2 | yes | 101.3 (b4) | 1.02 | 605 | 2.959 (b1) | ok |
| `s09_openvino:student-int8-nncf-fp32-head` | `s09_openvino:student-int8-nncf-fp32-decode` | openvino · cpu · int8 | 0.895 | 0.934 | 10.7 | 12.3 | yes | 100.4 (b4) | 1.02 | 605 | 2.925 (b1) | ok |
| `s10_tflite_edge:raspberry-pi-5` | `s10_tflite_edge:tflite-int8-full` | tflite · cpu · int8 | — | — | — | — | — | — | — | — | — | not_run: not measured on a Raspberry Pi: on the device, run `make setup-edge data && make s10` — th |
| `s10_tflite_edge:tflite-fp16` | `s07_distillation:student-distilled-onnx` | tflite · cpu · fp16 | — | — | — | — | — | — | — | — | — | failed: LiteRT cannot load the converted file: {'detector': 'tflite/kernels/conv.cc:360 input_type |
| `s10_tflite_edge:tflite-fp32` | `s07_distillation:student-distilled-onnx` | tflite · cpu · fp32 | 0.898 | 0.947 | 14.5 | 16.2 | yes | 68.7 (b4) | 2.42 | 228 | 4.073 (b1) | ok |
| `s10_tflite_edge:tflite-int8-full` | `s07_distillation:student-distilled-onnx` | tflite · cpu · int8 | — | — | — | — | — | — | — | — | — | failed: onnx2tf did not produce the file: {'detector': 'StrictFullIntegerQuantizationError: Unsupp |
| `s10_tflite_edge:tflite-int8-full-nms` | `s07_distillation:student-distilled-onnx` | tflite · cpu · int8 · NMS in graph | — | — | — | — | — | — | — | — | — | failed: onnx2tf did not produce the file: {'detector': 'StrictFullIntegerQuantizationError: Unsupp |
| `s10_tflite_edge:tflite-int8-ocr-fp32-detector` | `s07_distillation:student-distilled-onnx` | tflite · cpu · int8-ocr | 0.898 | 0.947 | 14.3 | 16.0 | yes | 69.2 (b4) | 1.94 | 230 | 4.105 (b1) | ok |

<details><summary>Lineage: which artifact each row was built from</summary>

```mermaid
flowchart LR
  n0["s00_profile:fast-resize"]
  n1["s00_profile:gpu-decode"]
  n2["s00_profile:nms-in-graph"]
  n3["s01_baseline:eager-fp32"]
  n4["s02_torch_compile:inductor-fp32"]
  n5["s03_torchscript:jit-trace-fp32"]
  n6["s04_onnx_export:ort-cpu-fp32"]
  n7["s04_onnx_export:ort-cpu-fp32-nms-in-graph"]
  n8["s04_onnx_export:ort-cuda-fp32"]
  n9["s04_onnx_export:ort-openvino-ep-cpu"]
  n10["s04_onnx_export:ort-tensorrt-ep-fp16"]
  n11["s05_pruning:masked-structured-50"]
  n12["s05_pruning:masked-structured-50-onnx"]
  n13["s05_pruning:sliced-iterative-50"]
  n14["s05_pruning:sliced-iterative-50-onnx"]
  n15["s05_pruning:sliced-oneshot-50"]
  n16["s05_pruning:sliced-oneshot-50-onnx"]
  n17["s05_pruning:sliced-student-budget"]
  n18["s05_pruning:sliced-student-budget-onnx"]
  n19["s06a_ptq:ort-dynamic-int8"]
  n20["s06a_ptq:ort-static-int8-daytime"]
  n21["s06a_ptq:ort-static-int8-detector-only"]
  n22["s06a_ptq:ort-static-int8-fp32-decode"]
  n23["s06a_ptq:ort-static-int8-fp32-decode-daytime"]
  n24["s06a_ptq:ort-static-int8-mixed"]
  n25["s06a_ptq:ort-static-int8-ocr-only"]
  n26["s06a_ptq:ort-static-int8-per-tensor"]
  n27["s06a_ptq:ort-static-int8-stratified"]
  n28["s06a_ptq:torch-dynamic-int8-ocr"]
  n29["s06b_qat:ort-qat-int8-ocr"]
  n30["s06b_qat:ort-qat-int8-ocr-only"]
  n31["s06b_qat:torch-qat-converted-ocr"]
  n32["s07_distillation:student-distilled"]
  n33["s07_distillation:student-distilled-onnx"]
  n34["s07_distillation:student-scratch"]
  n35["s08_tensorrt:baseline-fp16"]
  n36["s08_tensorrt:student-fp16"]
  n37["s08_tensorrt:student-fp32"]
  n38["s08_tensorrt:student-int8-daytime"]
  n39["s08_tensorrt:student-int8-stratified"]
  n40["s09_openvino:baseline-fp32-latency"]
  n41["s09_openvino:edge-candidate"]
  n42["s09_openvino:intel-gpu"]
  n43["s09_openvino:student-device-default-precision"]
  n44["s09_openvino:student-fp32-latency"]
  n45["s09_openvino:student-fp32-throughput"]
  n46["s09_openvino:student-int8-nncf"]
  n47["s09_openvino:student-int8-nncf-fp32-decode"]
  n48["s09_openvino:student-int8-nncf-fp32-head"]
  n49["s10_tflite_edge:raspberry-pi-5"]
  n50["s10_tflite_edge:tflite-fp16"]
  n51["s10_tflite_edge:tflite-fp32"]
  n52["s10_tflite_edge:tflite-int8-full"]
  n53["s10_tflite_edge:tflite-int8-full-nms"]
  n54["s10_tflite_edge:tflite-int8-ocr-fp32-detector"]
  n3 --> n0
  n3 --> n1
  n3 --> n2
  n3 --> n4
  n3 --> n5
  n3 --> n6
  n6 --> n7
  n3 --> n8
  n3 --> n9
  n3 --> n10
  n3 --> n11
  n11 --> n12
  n3 --> n13
  n13 --> n14
  n3 --> n15
  n15 --> n16
  n3 --> n17
  n17 --> n18
  n6 --> n19
  n6 --> n20
  n6 --> n21
  n6 --> n22
  n6 --> n23
  n27 --> n24
  n6 --> n25
  n6 --> n26
  n6 --> n27
  n3 --> n28
  n27 --> n29
  n25 --> n30
  n5 --> n31
  n3 --> n32
  n32 --> n33
  n3 --> n34
  n6 --> n35
  n33 --> n36
  n33 --> n37
  n33 --> n38
  n33 --> n39
  n6 --> n40
  n48 --> n41
  n44 --> n42
  n44 --> n43
  n33 --> n44
  n44 --> n45
  n44 --> n46
  n46 --> n47
  n47 --> n48
  n52 --> n49
  n33 --> n50
  n33 --> n51
  n33 --> n52
  n33 --> n53
  n33 --> n54
```

</details>
<!-- /results -->

Read it in this order:

1. **Header.** Which CPU, GPU, thread count, profile and library versions. If none
   of the tables were measured on your target, every path below starts with "run
   the stages there".
2. **The `≤ SLA` and `p95 ms` columns** for the rows your path names, on the table
   closest to your target.
3. **Accuracy against the row's parent,** not against the baseline. The allowed drop
   in `src/config.py` is per stage.
4. **Per-condition accuracy** on the row's benchmark card
   (`results/cards/<hardware>_<profile>/<stage>__<variant>.md`). Aggregate mAP hides
   a night-only collapse.
5. **The lineage details.** A TensorRT engine built from the distilled student does
   not contain the pruning rows printed above it.

---

## Decision flowchart

```mermaid
flowchart TD
  A["make profile: s00 gate report"] --> B{"Model under half of the p95 tail?"}
  B -- yes --> C["Guide 00 fixes first: fast resize, NMS in graph, GPU decode"]
  C --> A
  B -- no --> D{"Where does it run?"}
  D -- "ARM CPU, no accelerator" --> P1["s07 student -> s06a static INT8 stratified -> s10 TFLite full-integer + NMS in graph"]
  D -- "x86 CPU box" --> P2["s07 student -> s09 OpenVINO IR, NNCF INT8, LATENCY hint -> edge-candidate"]
  D -- "Jetson Orin" --> P3["s07 student -> s04 ONNX -> s08 TensorRT FP16, then INT8, built on the Orin"]
  D -- "central GPU server" --> P4["s07 student -> s08 TensorRT -> s11 Triton, dynamic batching"]
  D -- "Android phone / NPU" --> P5["s07 student -> s10 full-integer INT8, per-tensor -> LiteRT + NPU delegate on device"]
  D -- "Apple device" --> P6["s07 student -> s03 TorchScript or torch.export -> Core ML on device"]
  P1 & P2 & P3 & P4 & P5 & P6 --> E{"p95 at batch 1 within SLA, measured on the target?"}
  E -- no --> F["Symptom table below -> next lever"]
  F --> E
  E -- yes --> G{"Per-condition accuracy drop within SLA vs parent?"}
  G -- no --> H["make int8-debug: sensitivity -> mixed precision; or s06b QAT"]
  H --> E
  G -- yes --> I["perf gate + benchmark card + shadow -> cutover (guide 15)"]
```

---

## Path by deployment target

### 1. Raspberry Pi 5, or any ARM CPU without an accelerator

The hardest budget in the session: a few ARM cores, shared memory bandwidth, a
thermal envelope. `ANPR_THREADS` defaults to 4 in `src/config.py` because that
matches the core count of the edge boxes this session targets.

| | |
|---|---|
| **Sequence** | s00 profile and its fixes → s07 distillation (reduce the work per frame first) → s06a static INT8 with the stratified calibration set → s10 TFLite full-integer INT8 with NMS in the graph and fast resize |
| **Rows to read** | `s00_profile:fast-resize`, `s00_profile:nms-in-graph` · `s07_distillation:student-distilled` against `s05_pruning:sliced-student-budget` · `s06a_ptq:ort-static-int8-stratified` against `…-daytime` · `s10_tflite_edge:tflite-int8-full`, `tflite-int8-full-nms` · `s10_tflite_edge:raspberry-pi-5` |
| **SLA check** | only `s10_tflite_edge:raspberry-pi-5`, measured on the Pi. The M3 Pro is ARM too, but it is a different core, memory system and thermal design. Its rows rank the candidates; they do not pass the SLA for the Pi |
| **`not_run` means** | `raspberry-pi-5` is `not_run` until someone runs s10 on the device. Until then the path is a ranked shortlist, not a decision. `tflite-fp16` records why the float16 file does not load in LiteRT's CPU kernels, which rules out FP16 as a Pi lever in this pipeline |

> **Why distill before quantizing here.** A distilled student cuts compute by
> architecture; INT8 cuts it by precision. On a CPU with no headroom you usually
> need both. Distilling first also means calibration and per-condition checks run
> once, on the model you ship, not on a baseline you then throw away.

### 2. x86 roadside box, CPU only (Intel or AMD)

| | |
|---|---|
| **Sequence** | s00 → s04 ONNX Runtime CPU → s07 → s09 OpenVINO: IR, NNCF INT8, `PERFORMANCE_HINT=LATENCY` → `edge-candidate` (INT8 student + NMS in graph + fast resize) |
| **Rows to read** | `s04_onnx_export:ort-cpu-fp32`, `ort-openvino-ep-cpu` · `s06a_ptq:ort-static-int8-stratified` · `s09_openvino:student-fp32-latency` against `student-fp32-throughput` · `s09_openvino:student-int8-nncf`, `edge-candidate` · `s09_openvino:intel-gpu` |
| **SLA check** | `edge-candidate` at batch 1 on the box's own CPU model. Whether INT8 beats FP32 depends on whether the CPU has VNNI-class integer instructions, which is exactly why this row has to come from that CPU |
| **`not_run` means** | `intel-gpu` is `not_run` when no Intel GPU is present. If your box has an Intel iGPU, that row is the one to run next, not a detail to skip. On AMD CPUs, measure `s06a_ptq:ort-static-int8-stratified` next to the OpenVINO rows on the box, rather than assuming either runtime wins |

> **Why the LATENCY hint and not THROUGHPUT.** A camera delivers one frame at a
> time. The THROUGHPUT hint plus an `AsyncInferQueue` (`snippet:ov-async-queue`)
> buys frames per second across a batch, which is capacity one camera cannot use,
> and it can cost p95 at batch 1. Read the two `student-fp32-*` rows to see what the
> hint changes on your CPU.

### 3. Jetson Orin

JetPack 7.2 ships TensorRT 10.16.2 and CUDA 13.2.1 (https://developer.nvidia.com/embedded/jetpack/downloads/archive-7.2).

| | |
|---|---|
| **Sequence** | s00 (check `s00_profile:gpu-decode`: preprocessing placement matters on a GPU) → s07 → s04 ONNX export → s08 TensorRT FP16 → s08 INT8 with the stratified calibrator (`snippet:int8-calibrator`), all **built on the Orin** |
| **Rows to read** | on the RTX 3090 table: `s08_tensorrt:student-fp32` (parity), `student-fp16`, `student-int8-entropy` against `student-int8-daytime`, and `s04_onnx_export:ort-tensorrt-ep-fp16`. These show the *method* and the accuracy behaviour of each precision. They are not Orin latencies |
| **SLA check** | the same `s08_tensorrt:*` rows re-measured on the Orin, in a table whose header names the Orin |
| **`not_run` means** | on a CPU machine, every s08 row is `not_run` ("no CUDA + TensorRT 10"). That says nothing about TensorRT on the Orin. Engines from the course's x86 container will not load there: different GPU, different architecture, different TensorRT patch. Treat the 3090 rows as a rehearsal and build again on the device |

> **Why INT8 calibration still works here.** JetPack 7.2 is still on TensorRT 10.x,
> so s08's implicit calibrator is available. On a TensorRT 11 stack, INT8 means a
> Q/DQ ONNX from s06a instead. Check which TensorRT your JetPack ships before
> choosing a quantization route.

### 4. Central GPU server aggregating many cameras (RTX 3090-class or data-center GPU)

This path assumes the cameras *have* backhaul to a server, which the no-uplink
roadside design in this session does not. The decision changes from "cheapest box
that meets the SLA for one camera" to "lowest cost per million frames across the
fleet".

| | |
|---|---|
| **Sequence** | s00 (decide where decode and letterbox run: client or server) → s07 → s08 TensorRT FP16 / INT8 → s11 Triton with dynamic batching (`snippet:triton-docker-run`, `snippet:triton-hot-reload`) |
| **Rows to read** | `s08_tensorrt:baseline-fp16` against `student-fp16` (what the big GPU buys without distillation) · `s11_triton:triton-grpc-client-pipeline` against `triton-grpc-nobatch` (does batching help at your concurrency) · `triton-http-client-pipeline` (transport cost) · `triton-bls-server-pipeline` (whole pipeline server-side) · the `$/1M frames @ $1/h` column |
| **SLA check** | per-frame p95 **including the network**, at the concurrency the fleet produces. The `$/1M` column already uses only batch sizes whose p95 meets the SLA (`usable_point` in `src/cost.py`) |
| **`not_run` means** | s11 rows are `not_run` without Docker, and s08 rows without CUDA + TensorRT 10. On a data-center GPU from the Ada generation or newer, FP8 becomes a candidate lever: TensorRT has supported FP8 convolution on Ada since 10.3 (https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html). This session has no FP8 row, so that choice needs its own measurement |

> **Why the GPU server is the one place batching is the main lever.** A GPU at
> batch 1 has idle compute; many cameras give it concurrent frames to fill it with.
> Read `triton-grpc-nobatch` against `triton-grpc-client-pipeline` before believing
> that, because too little concurrency means the server has nothing to batch.

### 5. Android phone or NPU

| | |
|---|---|
| **Sequence** | s07 → s10: ONNX → onnx2tf → full-integer INT8 with int8 input/output tensors, per-tensor scales, NMS in the graph → on device: LiteRT with the vendor's GPU or NPU delegate, measured as a new row |
| **Rows to read** | `s10_tflite_edge:tflite-int8-full`, `tflite-int8-full-nms` · `s06a_ptq:ort-static-int8-per-tensor` against `ort-static-int8-stratified`, a preview of what per-tensor scales cost in accuracy, since NPU-friendly INT8 and s10's onnx2tf export both use per-tensor · `tflite-fp16` for why the float16 file is refused |
| **SLA check** | on the phone, with the delegate that will ship. Operators the delegate does not support fall back to the CPU, splitting the graph, so NMS in the graph can help or hurt depending on the NPU. Only an on-device row settles it |
| **`not_run` means** | nothing in this session runs on Android. Every row is a desktop proxy for the file you will ship, useful for accuracy (the TFLite file's per-condition accuracy is the phone's) but not for latency |

NNAPI is deprecated as of Android 15; use LiteRT with GPU/NPU delegates instead (https://developer.android.com/ndk/guides/neuralnetworks/migration-guide).

> **Why `MAX_PLATES` is fixed.** NPUs and static-shape runtimes want fixed tensor
> shapes. `src/config.py` fixes the OCR batch at `MAX_PLATES` so the same recognizer
> file works for TFLite and TensorRT profiles without dynamic shapes.

### 6. Apple device (iPhone, iPad, Mac)

coremltools 6.0 removed the ONNX converter: convert from TorchScript or torch.export, not from the s04 ONNX file (https://github.com/apple/coremltools/releases/tag/6.0).

| | |
|---|---|
| **Sequence** | s07 → s03 TorchScript trace (`snippet:jit-trace`) of the student, or `torch.export` → Core ML conversion → an on-device row. Core ML decides per layer whether to use CPU, GPU or Neural Engine |
| **Rows to read** | `s03_torchscript:*`, whose parity against eager confirms the traced graph is the model · `s07_distillation:student-distilled` for accuracy per condition · the M3 Pro CPU tables, which *are* Apple silicon, but through ONNX Runtime / OpenVINO on the CPU cores, not Core ML |
| **SLA check** | on the device, through Core ML. There is no Core ML stage in this session. Add one as a backend in `src/backends.py` (`@register("coreml")`) and the harness records the same five metrics, lineage and environment as every other row |
| **`not_run` means** | there are no Core ML rows to be `not_run`. The absence is the finding: this target's latency is unmeasured |

---

## Symptom in the journey table → next lever

| Symptom you read | Next lever | Where |
|---|---|---|
| s00's gate report: the model is under half of the p95 tail | preprocessing and postprocessing fixes, not the model | guide 00; `snippet:profile-gate`, `snippet:letterbox`, `snippet:nms-in-graph` |
| OCR exact match falls **only on `low_contrast`** after INT8 | mixed precision: keep the sensitive layers in FP32 | guide 08; `make int8-debug`; `snippet:layer-sensitivity`, `snippet:mixed-precision` |
| `…-daytime` INT8 falls at night; `…-stratified` does not | it is the calibration set, not INT8 | `snippet:calibration-set` |
| INT8 falls broadly even with stratified calibration | find outlier channels, then mixed precision or QAT on the model that needs it | `snippet:outlier-channels`; s06b, `snippet:qat-qconfig` |
| `per-tensor` row clearly worse than per-channel | per-channel scales where the runtime allows; QAT where it does not (NPU, s10) | s06a; s06b |
| masked pruning row: same size MB and same p95 as its parent | masks change nothing physical; slice channels or distill | `snippet:pruning-mask-lifecycle`, `snippet:physical-slicing`; s07 |
| `size MB` fine, **`peak RSS MB`** too high for the box | runtime memory: arena allocation, thread count, one session per model; not a smaller model | `snippet:peak-memory`; the header's thread settings |
| `p50` within SLA, `p95` not | tail work: cold first call, recompilation on a new shape, thread contention, Python postprocessing | `process_first_call_ms` and the timing loop's first call on the card; s02 recompiles; `snippet:greedy-nms` |
| `fps (batch)` saturates at batch 1 on a CPU | batching will not help, because the CPU is already compute-saturated; reduce work per frame (distill, INT8) | Session 3 measured this on ResNet-50: [level 2](../../session_3/serving_levels/README.md#level-2--fastapi--a-hand-written-dynamic-batcher) |
| `triton-grpc-nobatch` ≈ `triton-grpc-client-pipeline` | the server had nothing to batch: raise concurrency or queue delay, check Triton's batch metrics | s11 |
| `$/1M frames` shows `—` | no batch size meets the SLA: this is a latency problem, not a cost problem | `src/cost.py` `usable_point` |
| TensorRT FP16 fails its parity gate against FP32 | layers overflowing in FP16: keep them at FP32 in the build | `snippet:parity-gate`, `snippet:trt-build` |
| everything you need is `not_run` | run the stage on hardware that can, and do not decide from another machine's table | the row's reason text |

---

## Fleet cost: from rows to a bill

The journey table prices one instance at $1/hour. The fleet question is how many
instances, and what an hour actually costs. `src/cost.py` has both, as formulas:

```text
cost_per_million = usd_per_hour / (usable_fps * 3600) * 1_000_000   # snippet:cost-per-million
load_fps         = cameras * camera_fps                             # fleet()
instances        = ceil(load_fps / usable_fps)
usd_per_month    = instances * usd_per_hour * 730

# an edge box has no hourly price; amortize it
usd_per_hour     = hardware_price / amortization_hours
                 + (watts / 1000) * usd_per_kwh
                 + maintenance_usd_per_hour
```

```bash
python -m src.cost --row s09_openvino:edge-candidate --usd-per-hour <amortized price> --cameras <N> --camera-fps <fps>
```

`--row` takes the last measured row with that id in `results/results.json`,
whichever machine wrote it. Check that row's card header before trusting the bill.

> **Why edge and central fleets optimize different things.** With no uplink, every
> camera needs its own box: `instances = cameras`, whatever the throughput. Capacity
> above one camera's frame rate is paid for and never used. The decision is **the
> cheapest box whose p95 at batch 1 meets the SLA**. The `$/1M frames` column
> assumes you can fill the instance, which only a central server fed by many cameras
> can. There, `usable_fps` divides the load, and the lowest cost per million frames
> wins.

---

## Common wrong turns

1. **Deciding on the laptop's table.**
   **Symptom:** a candidate passes `≤ SLA` in the course table and misses it on the
   roadside box, and nobody can say which number was wrong.
   Neither was wrong; they describe two machines. Check the header, then re-measure
   on the target (rule 2).

2. **Reading the table as a chain.**
   **Symptom:** expecting the TensorRT INT8 row to include the pruning gain printed
   above it, then treating the difference as a TensorRT problem.
   Rows are a tree. Open the lineage and compare each row with its parent only.

3. **Treating `not_run` as "slow" or "broken".**
   **Symptom:** a team drops TensorRT from a Jetson plan because every s08 row in
   the course table is `not_run`.
   The reason column says "no CUDA + TensorRT 10 here". Missing evidence is a reason
   to run the stage on the device, not a verdict (rule 3).

4. **Picking INT8 on aggregate accuracy.**
   **Symptom:** the INT8 row passes the mAP drop limit, and night plates are
   misread in the field.
   Aggregate accuracy is weighted by the validation mix. Read per-condition accuracy
   on the card, and shadow per condition before cutover (guide 15).

5. **Buying throughput for a single camera.**
   **Symptom:** an edge box configured with the THROUGHPUT hint or a large batch
   posts the best `fps` in the table and misses p95 at batch 1.
   A camera is batch 1. On the edge, throughput above one camera's frame rate is
   capacity nobody uses.

6. **Comparing rows across profiles.**
   **Symptom:** the `full` profile's baseline looks worse than the `quick`
   profile's student, and the student gets chosen for the wrong reason.
   Profiles differ in data size and epochs, and the table never puts them side by
   side on purpose. Compare within one `(hardware, profile)` table.

7. **Converting ONNX to Core ML from an old tutorial.**
   **Symptom:** the ONNX converter is missing from `coremltools`.
   It was removed in coremltools 6.0. Convert from the TorchScript or `torch.export`
   graph (path 6).

8. **Following an NNAPI delegate tutorial for a new Android app.**
   **Symptom:** the delegate path is deprecated on current Android and behaves
   differently across devices.
   Use LiteRT with GPU or NPU delegates (path 5).

---

## AV comparison callout

> **Context.** In automotive programs the flowchart runs backwards. The in-vehicle
> compute platform is chosen years before the models that will run on it, and it is
> fixed for the life of the vehicle. So the question is never "which hardware for
> this model", only "which model fits this hardware and still meets the deadline".
> Roadside ANPR fleets reach the same position once the cameras are installed: the
> box is a given, and every later model has to pass on it. That is why paths 1–3
> start from the target's own table, and why a `not_run` row for the installed
> hardware blocks the decision instead of being skipped.

---

## When NOT to optimize at all

- **The baseline already meets the SLA on the target, with headroom, at the fleet's
  real traffic.** Every stage you add is an artifact to card, gate, shadow and
  rebuild when the toolchain moves. If `s01_baseline:eager-fp32` passes on the
  installed box, the cheapest optimization is none.
- **Accuracy is the problem, not latency.** If exact match is already too low at
  FP32, INT8 and distillation can only spend accuracy you do not have. Go back to
  Session 2 and the data first.
- **The hardware is not decided yet.** Optimizing for a target you might not buy
  produces rows for the wrong table. Decide the target, then run its path.
- **You cannot check per-condition accuracy.** Without a validation set that covers
  night, rain, blur and low contrast, a precision change is untestable where it
  usually breaks. Build the evaluation before the optimization.
- **The model changes every week.** Each retrain re-triggers calibration, engine
  builds, cards and a shadow run. Automate that pipeline (guide 15) before adding
  levers, or keep the model in the one runtime that needs no per-version build.
- **The time is spent outside the model.** If s00 says I/O, decode or network
  dominate, a faster model changes nothing the user sees.

---

Previous: [15 — Optimization in your MLOps stack](15-optimization-in-your-mlops-stack.md) · Next: [Session 5 README](../README.md)
