# 04 — torch.compile and TorchScript

Two tools with similar names that do opposite jobs. `torch.compile` makes the model you already run faster in the Python process you already have. It changes no weights and produces no file. TorchScript changes nothing about speed. It turns the model into a file that runs without your Python class code, which is what a LibTorch C++ service or Triton's PyTorch backend needs. This guide runs both on the unchanged baseline detector and recognizer, gates them strictly against eager PyTorch, and shows where each one's cost is recorded.

## 1. The problem it solves

The baseline row (`s01_baseline:eager-fp32`) runs the ResNet-34 detector and the BiLSTM CRNN in eager PyTorch. In eager mode, every operator is dispatched from the Python interpreter as its own kernel call, and every intermediate tensor is allocated separately. The roadside box has a 30 ms per-frame budget, no uplink and four cores. Some of the eager cost is framework overhead, not convolution math.

Before you touch a single weight, find out how much of that latency is overhead. Pruning (guide 06), INT8 and distillation all put accuracy at risk. `torch.compile` does not: the weights stay the same, so the plates it reads must stay the same too. That makes it the cheapest first experiment.

TorchScript solves a deployment problem instead. Suppose the thing that loads the model is a C++ process, or a Triton server you don't control. Your `PlateDetector` class does not exist there, so you need a self-contained file.

## 2. Mental model

### torch.compile: capture, then generate code

| Step | What happens | Where it can go wrong |
|---|---|---|
| **Dynamo** | Reads the Python bytecode of `forward` and records the tensor ops into an FX graph. Also records *guards*: the shapes, dtypes, devices and Python values the graph assumed. | Code it cannot trace causes a **graph break**. |
| **Inductor** | Fuses the graph's operators and generates kernels: C++/OpenMP on CPU, Triton kernels on CUDA. | Needs a working C++ compiler on CPU. |
| **Call** | If every guard holds, the generated code runs. If a guard fails, Dynamo compiles again. | New shapes cause **recompiles**. |

Here is what people get wrong. `torch.compile(model)` compiles nothing when you call it. It returns a wrapper immediately. Compilation happens on the **first call**, and again whenever a guard fails. By default, Dynamo specializes on the first shape it sees. When a second distinct shape arrives, it recompiles once with that dimension marked dynamic. `dynamic=True` asks for symbolic shapes from the start. That is why the recognizer gets it and the detector does not (`src/backends.py :: TorchBackend.__init__`):

```python
        if compile:
            # dynamic=True on the recognizer: the number of plates per frame varies,
            # and without it every new crop count triggers a recompile mid-benchmark.
            self.det, self.rec = torch.compile(self.det), torch.compile(self.rec, dynamic=True)
```

Every frame reaches the detector as one letterboxed 384x640 tensor. The recognizer gets as many 32x128 crops as there are plates in the frame, which could be none, one or four.

A **graph break** does not raise an error. When Dynamo hits `.item()`, `.numpy()`, `print`, or a Python `if` on a tensor value, it ends the graph, runs that code in the interpreter, and starts a new graph. The model still works. You just get less fusion and more trips back to Python.

> **Why it is the default first move.** One line (`snippet:torch-compile`), no export, no new format, and a strict parity gate proves it changed nothing. If it does not help, you lost minutes, not accuracy.

A compiled model is also **not an artifact**: nothing new is saved, and every new process compiles again. Inductor's on-disk caches can make later starts cheaper; measure that on your box.

### TorchScript: a file, not a speedup

| | `torch.jit.trace` | `torch.jit.script` |
|---|---|---|
| How | Runs the model once on an example input and records the ops that executed | Compiles a subset of Python source, including control flow |
| Control flow | Frozen: the branch taken for the example is the only one in the file | Kept |
| Fails how | Silently, when a later input would have taken another branch | Loudly, at compile time, on unsupported Python |
| Used here | Yes (`snippet:jit-trace`) | No |

Tracing is safe for our models because their branches depend on configuration, not on data. `DetectorExport.forward` checks `if not self.with_nms`, and that value is fixed when the object is built. The traced `detector_baseline.ts` contains only the decode path. It cannot run NMS, and it was never meant to. A branch on a *tensor value* would be frozen the same way, and nothing would warn you at inference time.

A `.ts` file loads with `torch.jit.load` and no import of `src.models`. `TorchScriptBackend` never touches the model classes. That is its whole value: LibTorch C++ services and Triton's PyTorch backend load these files. It is also on its way out. Deprecation warnings were added in PyTorch 2.10, and in 2.14 `torch.jit.script/trace/save/load` raise a visible FutureWarning pointing to torch.compile / torch.export (https://github.com/pytorch/pytorch/releases/tag/v2.14.0). This session pins torch 2.13.0, even though 2.14 is the latest stable release, because the edge stack's litert-torch requires torch<2.14 (https://pypi.org/pypi/litert-torch/json).

> **Why s03 records warnings instead of this guide quoting them.** Which warning appears depends on the exact version installed. `trace()` catches every warning `torch.jit` raises on *your* install, prints them, and writes them into the row's `notes`. Read the notes, not a blog post about a different version. `TracerWarning`s land there too, and those matter more than the deprecation notice.

## 3. Runnable walkthrough

```bash
make s02    # torch.compile (needs: make train)
make s03    # TorchScript trace + save + load
```

**`src/stages/s02_torch_compile.py :: main`**
- `snippet:torch-compile` builds `TorchBackend(..., compile=True)`. The compile branch is in `src/backends.py :: TorchBackend.__init__`.
- `common.gate(spec, TorchBackend(**opts), compiled)` runs eager and compiled pipelines on four validation frames. It is **strict** (`snippet:parity-gate`): raw tensors within `rtol`/`atol`, identical box counts, matched-box IoU, and plate strings equal character for character. Compiling must not change a single plate.
- The `except Exception` branch records `not_run` with the first line of the error instead of crashing. That covers a missing C++ compiler or an unsupported platform. A `not_run` row with a reason is a valid result.

**`src/stages/s03_torchscript.py`**
- `snippet:jit-trace` calls `model.eval()` inside the helper, so no call site can forget it. It records warnings and saves the `.ts` file.
- The example batches are 2 (detector) and 3 (recognizer). The parity gate then runs at **batch 4** (`common.sample_jpegs(4)`), a size neither trace saw. If tracing had frozen a batch dimension, the gate would catch it here.
- The row's `notes` hold the warning list.

Every benchmark then runs in a **fresh worker process** (`src/benchmark.py`, module docstring). The compile done for the gate is not reused; the worker compiles again.

To see what Dynamo is doing, run the stage with its logs on:

```bash
TORCH_LOGS="graph_breaks,recompiles" make s02
```

## 4. Measured result

<!-- results:stage:s02_torch_compile -->
<!-- /results -->

<!-- results:stage:s03_torchscript -->
<!-- /results -->

Both rows have `s01_baseline:eager-fp32` as parent, and that row appears in each block for comparison.

| Column | How to read it for these rows |
|---|---|
| mAP@0.5, OCR exact | Same weights, strict gate passed. If either differs from the parent *at all*, four parity frames missed something. Treat that as a bug, not noise. |
| p50 / p95 | Lower than the parent: framework overhead was a real share of the frame. Level or higher: the time is in kernels that were already optimized library calls, or the graph is breaking and recompiling. Run with `TORCH_LOGS` before concluding anything. |
| fps (batch) | The throughput pass uses the profile's batch sizes. The detector is compiled without `dynamic=True`, so each new batch size can recompile. That happens during warm-up, not in the timed calls. |
| size MB | `s02`: the same `.pt` files as the parent, because there is no new artifact. `s03`: the `.ts` files. |
| peak RSS MB | Compilation pulls the compiler stack and generated code into the process. Compare with the parent before calling a compiled model lighter. |

**Where the compile cost is recorded.** Every row stores two first-call numbers. `process_first_call_ms` is the first pipeline call in the fresh worker process, timed in `src/benchmark.py :: _worker` before the accuracy pass — for the compiled row, that call pays Dynamo tracing and Inductor code generation. `latency.cold_first_call_ms` is the first call of the timing loop, after the accuracy pass has already run every validation frame through the pipeline.
- Compare `process_first_call_ms` between the s02 row and its parent: the difference is what compilation costs a camera that has just rebooted.
- If `cold_first_call_ms` is still well above p50, something was still cold after the accuracy pass — a crop count the recognizer had not seen, or an allocator warming up to a new batch shape.

## 5. Gotchas

1. **Forgetting `model.eval()` before tracing.** **Symptom:** the `.ts` file gives different outputs run to run for the same frame (active dropout), or a plate's detections change depending on which other frames share the batch (BatchNorm using batch statistics). Strict parity then fails on the tensor check. **Fix:** call `eval()` inside the export helper, the way `trace()` does. `io.load` also returns models in eval mode.

2. **Tracing with a batch-1 example.** **Symptom:** a `TracerWarning` about converting a tensor to a Python value appears in the row's notes. The file works at batch 1, then fails with a shape error, or silently returns wrong results, at batch 4. **Fix:** trace with an example batch that is not 1, and gate at yet another size, as s03 does with 2 and 3 for tracing and 4 for the gate.

3. **Graph breaks from host-side code in `forward`.** **Symptom:** the compiled row is no faster than eager, `TORCH_LOGS=graph_breaks` lists `.item()`, `.numpy()` or `print`, and `torch.compile(..., fullgraph=True)` raises. **Fix:** keep NumPy and Python logic out of `forward`. This is why greedy NMS (`snippet:greedy-nms`) and CTC decoding (`snippet:ctc-greedy-decode`) live in the pipeline, not in the model.

4. **Recompilation storms.** **Symptom:** p95 sits far above p50 and spikes line up with changes in plate count. `TORCH_LOGS=recompiles` prints guard failures on a size. Eventually Dynamo hits its recompile limit, warns, and runs that function eagerly for the rest of the process. **Fix:** use `dynamic=True` on dimensions that really vary, or pad to a few fixed sizes. `config.MAX_PLATES` exists for the static-shape runtimes.

5. **First-request latency spike in production.** **Symptom:** after a deploy or reboot, the camera's first frames miss the 30 ms SLA and the health check may time out. The benchmark never showed it because warm-up absorbed it. **Fix:** before reporting ready, warm up with every shape production will send: one frame, plus each crop count up to `MAX_PLATES`. `config.WARMUP_RUNS` exists for the same reason.

6. **No C++ toolchain, or an unsupported device.** **Symptom:** the `torch.compile(...)` line succeeds, then the *first call* raises an Inductor error mentioning the compiler. On Apple-silicon `mps`, support is newer and less complete than on CPU or CUDA, so you may get an error or a run that is not faster. **Fix:** install a compiler (build-essential; Xcode command-line tools on macOS), and benchmark on the device you ship. `common.device()` picks only `cuda` or `cpu`, and s02 records `not_run` rather than crashing.

## 6. AV comparison callout

> **Context only, not part of this lab.** A YOLO-class detector on a car sees a fixed resolution from each camera at a fixed rate. Its input shapes rarely change, so the recompiles that force `dynamic=True` on our recognizer mostly disappear. Static shapes are torch.compile's best case. Lane segmentation, with its dense per-pixel output, is similar. The constraint that dominates there is *where* the model runs. In-vehicle stacks commonly deploy through TensorRT or a C++ runtime, not a Python process. torch.compile's win then stays on the training and evaluation cluster, and the shipped artifact comes from an export path (guide 05). Our plate detector sits in between: a fixed frame shape, but a variable number of crops downstream. That one dimension is what shaped our compile settings.

## 7. When NOT to use this

- **torch.compile when the target is not a Python PyTorch process.** On the Raspberry Pi (ONNX Runtime or LiteRT) or a Jetson (TensorRT), a compiled model has nothing to ship. The deployable file comes from guide 05.
- **torch.compile when cold start matters more than steady state and you cannot warm up**, such as a camera that must answer its first frame on time after power-on.
- **torch.compile when input shapes are unbounded.** You will spend the gain on recompiles.
- **TorchScript for a new project, or as an optimization.** It is a format, not a speedup. Reach for `.ts` only when a consumer you don't control requires it: an existing LibTorch service or a Triton PyTorch-backend deployment.

Previous: guide 03 · Next: [05 — ONNX and execution providers](05-onnx-and-execution-providers.md)
