# 14 — LLM serving: the same four levers

Everything in this session was done to one ANPR pipeline, and nothing here is
specific to plates. The same four levers make a large language model cheaper and
faster: the graph and runtime, precision, architecture size, and serving. What
changes is the workload's shape. One LLM request is not one forward pass. It is one
big pass over the prompt, then one small pass per generated token, and those two
phases hit different hardware limits. So the metrics change completely, and the
lever that matters most depends on which phase you are paying for.

---

## 1. The problem it solves

[Session 3's vLLM level](../../session_3/serving_levels/README.md#level-5--vllm-serving-an-llm)
served an LLM with one command. It ended on a sentence worth taking seriously:
[classic batching does not transfer to LLMs](../../session_3/serving_levels/README.md#why-classic-batching-does-not-transfer-to-llms)
because the fix is a different scheduler, not a bigger batch.

After this session you will be asked to make an LLM endpoint cheaper or faster, and
the tempting move is to reuse the ANPR playbook as it stands: export, quantize,
batch, report frames per second. Each of those steps has an LLM equivalent, but used
without changes they produce the wrong numbers:

| ANPR habit | What goes wrong on an LLM |
|---|---|
| one latency number per request | a user waits for the first token and then for every later token; one number hides both |
| throughput = frames/s | tokens/s rises with concurrency while each user's stream slows down |
| INT8 speeds up the whole forward pass | weight-only INT4 speeds up the phase that reads memory, not the phase that does arithmetic |
| dynamic batching groups requests | requests finish at different lengths; a fixed batch runs at the pace of its longest member |
| accuracy = mAP / exact match | the output is free text; quality is an evaluation score, not a parity check |

This guide is the translation table, so that the reasoning carries over and the
wrong numbers don't.

---

## 2. Mental model

### Two phases, two bottlenecks

**Prefill** processes the whole prompt in one forward pass. Every prompt token goes
through every layer at once as large matrix multiplications. The GPU is busy with
arithmetic, so prefill is **compute-bound**, much like our detector on a
1280×720 frame. Prefill produces the first output token and fills the **KV cache**:
the attention keys and values for every prompt token, kept so they are never
recomputed.

**Decode** then generates one token per forward pass, per sequence. Each step does
very little arithmetic, but it must read *all* the weights plus that sequence's
whole KV cache from memory to produce a single token. The GPU spends most of
the step waiting on memory, so decode is **memory-bandwidth-bound**. At batch 1,
decode tokens/s is roughly capped by memory bandwidth divided by the bytes read per
token.

> **Why this decides which lever matters.** A lever that cuts the *bytes read per
> decode step* speeds up decode: weight-only INT4, a quantized KV cache, a smaller
> model. A lever that speeds up *large matrix multiplications* speeds up prefill:
> FP8 compute kernels, fused attention kernels. Batching many decode sequences
> together puts the idle compute to work, which is the whole reason continuous
> batching exists. That holds until the GPU becomes compute-saturated, and then
> decode behaves like a CPU running our detector.

> **Why our OCR batches cleanly and an LLM does not.** The CRNN recognizer reads
> the whole plate in one pass. CTC collapses all time steps at once
> (`snippet:ctc-greedy-decode` in `src/postprocess.py`), so every plate costs the
> same compute and a batch finishes together. An autoregressive decoder has a loop
> whose length depends on the output. Nothing in the ANPR pipeline has that loop.
> That is the structural difference behind every row below.

### The four levers, mapped

| Lever | In this session (ANPR) | For an LLM | Phase it mainly helps |
|---|---|---|---|
| **Graph / runtime** | `torch.compile` (s02), ONNX Runtime (s04), TensorRT (s08), OpenVINO (s09) | TensorRT-LLM, vLLM and SGLang: fused attention and MLP kernels, CUDA graphs for decode steps | both; fused attention matters most for long prompts |
| **Precision** | FP16 engines (s08), static INT8 with a calibration set (s06a), QAT (s06b), mixed precision (`make int8-debug`) | weight-only INT4 via GPTQ (https://arxiv.org/abs/2210.17323) or AWQ (https://arxiv.org/abs/2306.00978); W8A8 via SmoothQuant (https://arxiv.org/abs/2211.10438); FP8 on Ada/Hopper and newer; KV-cache quantization | weight-only and KV-cache quantization: decode. FP8 / W8A8 compute: prefill and decode |
| **Architecture size** | distilled student (s07), physically sliced pruning (s05) | distillation into a smaller LLM; speculative decoding, where a small draft model proposes tokens and the large model verifies several in one pass | decode |
| **Serving** | Triton dynamic batching (s11), usable batch under the SLA (`src/cost.py`) | continuous (iteration-level) batching; KV cache managed in pages by PagedAttention (https://arxiv.org/abs/2309.06180); prefix caching; chunked prefill | decode throughput, and queueing time inside TTFT |

Three notes on that table:

- **Speculative decoding is a size lever wearing a serving costume.** The draft
  model is cheap to run. The large model checks several proposed tokens in one
  forward pass, which looks like prefill: compute-bound and parallel. So several
  memory-bound decode steps become one compute-bound verification step. How much
  that buys depends on how often the draft's tokens are accepted, and that has to be
  measured on your traffic.
- **FP8 is a hardware question before it is a model question.** TensorRT has
  supported FP8 convolution on Ada (SM 8.9) since TensorRT 10.3, so FP8 is not
  Hopper-only (https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.3.0.html).
  The course's RTX 3090 is Ampere and has no FP8. That is why s08 stops at INT8,
  and it rules FP8 out as an LLM lever on that card.
- **Serving engines are runtimes and schedulers together.** TensorRT-LLM, vLLM and
  SGLang each bundle the kernels (lever 1) with the batching and KV-cache manager
  (lever 4). For an LLM, you usually pick the engine instead of picking a lever
  directly.

### The metrics change entirely

| Metric | Definition | Replaces, from the journey table |
|---|---|---|
| **TTFT**, time to first token | request arrival to first streamed token: queueing + prefill. Measured in milliseconds; report p50 and p95 | `p50 ms` / `p95 ms` for the part of the wait a user feels first |
| **TPOT / ITL**, time per output token / inter-token latency | the gap between consecutive streamed tokens: decode. Measured in milliseconds per token; report p50 and p95 | nothing. ANPR has no second latency |
| **tokens/s** | output tokens per second, per user (the reading speed) *and* for the whole system (the bill) | `fps (batch)` |
| **cost per million tokens** | the same `src/cost.py` formula, with tokens in place of frames | `$/1M frames @ $1/h` |
| **quality** | a task evaluation score: RAGAS, a calibrated judge, exact match on a frozen testset | `mAP@0.5`, `OCR exact` |
| **peak VRAM** | weights **plus** KV cache at the concurrency you serve, which grows with context length and with the number of active sequences | `peak RSS MB`, `size MB` |

The cost formula carries over unchanged. Only the unit moves:

```text
cost_per_million_tokens = usd_per_hour / (usable_tokens_per_s * 3600) * 1_000_000
```

"Usable" means the same thing it means in `src/cost.py` (`snippet:cost-per-million`).
It is the highest throughput at a concurrency whose TTFT p95 **and** TPOT p95 both
still meet the SLA. Tokens produced at a concurrency where every user waits too long
are capacity you cannot sell. The LLM literature calls this *goodput*.

---

## 3. Runnable walkthrough

**No code in this session — the mapping below.** The runnable LLM server is
[Session 3, level 5](../../session_3/serving_levels/README.md#level-5--vllm-serving-an-llm).
It already exposes the precision lever as a flag (`--quantization awq|gptq|fp8`),
ships a load generator (`vllm bench serve`), and publishes running/waiting requests
and KV-cache usage on `/metrics`, the same Prometheus endpoint Session 4 scrapes.
Treat the table below as a key for reading that level with this session's eyes.

| ANPR concept (this session) | LLM equivalent |
|---|---|
| one frame through detector + OCR | one request: a prefill, then N decode steps |
| `SLA.p95_ms` in `src/config.py` | two SLAs: TTFT p95 and TPOT p95 |
| `latency_pass` at batch 1 (`snippet:timing-loop`) | a streaming client timestamping the first token and every gap after it |
| `throughput_pass` curve and `usable_point` | a concurrency sweep with `vllm bench serve`; goodput at the SLA |
| `$/1M frames @ $1/h` | `$/1M tokens @ $1/h`, same formula |
| warm-up runs and `process_first_call_ms` | model load + CUDA-graph capture; the first request after start |
| `profile-gate`: is the model even the bottleneck? | is it prefill (long prompts, TTFT) or decode (long outputs, TPOT)? |
| ONNX export + parity gate (`snippet:parity-gate`) | a checkpoint conversion checked by a task eval, not bitwise parity |
| TensorRT engine tied to GPU + TRT version | TensorRT-LLM engine tied the same way; GGUF files stay portable |
| calibration set, stratified across conditions (`snippet:calibration-set`) | GPTQ/AWQ calibration text, stratified across languages, domains, prompt lengths |
| `ort-static-int8-daytime`, the worked failure | calibrating on one language or domain and serving another |
| per-condition accuracy (day, night, rain, …) | per-slice eval: per language, per task, per prompt length |
| outlier activation channels (`snippet:outlier-channels`) | the activation outliers SmoothQuant migrates into the weights |
| mixed precision: sensitive layers stay FP32 (`snippet:mixed-precision`) | keeping some layers (often embeddings and the output head) in higher precision |
| distilled student (s07) | a smaller LLM distilled from the large one |
| sliced-student-budget vs student-distilled | a pruned large model vs a small model trained dense |
| Triton dynamic batching: wait for a batch, run it, return it | continuous batching: sequences join and leave the running batch at every decode step |
| static OCR batch `MAX_PLATES` for static-shape runtimes | KV-cache blocks allocated on demand in pages (PagedAttention) |
| peak RSS / VRAM from `snippet:peak-memory` | VRAM for weights + KV cache at target concurrency and context length |
| TFLite INT8 on a Raspberry Pi (s10) | a GGUF quantized model under llama.cpp on a CPU |
| perf gate on p95 + accuracy drop (`snippet:perf-gate`) | a gate on TTFT/TPOT p95 + eval-score drop (Session 4 Guide E's RAGAS gate) |
| shadow INT8 against FP32 (`snippet:shadow-compare`) | shadow the INT4 model against FP16 and score both with the same judge |
| Prometheus + Grafana on per-frame latency | Langfuse traces per generation (Session 4 Guide D) plus the engine's `/metrics` |

### llama.cpp and GGUF: the edge path for LLMs

llama.cpp (https://github.com/ggml-org/llama.cpp) plays the role for LLMs that
TFLite plays in s10. It is a portable CPU-first runtime with its own file format
(GGUF) and its own quantized types. Conversion takes two steps, and blog posts get it
wrong often enough to be worth showing the correct commands:

```bash
# 1. Hugging Face checkpoint -> GGUF at F16 (the script lives in the llama.cpp repo)
python convert_hf_to_gguf.py ./models/my-model --outtype f16 --outfile my-model-f16.gguf

# 2. F16 GGUF -> a quantized GGUF, with the llama-quantize binary built from the same repo
./llama-quantize my-model-f16.gguf my-model-Q4_K_M.gguf Q4_K_M
```

The quantize types and options are documented at
https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md. The old
`convert.py` no longer exists, and `convert.py --outtype q4_k_m`, which still appears
in blog posts, does not run. Conversion and quantization are separate steps for the
same reason s06a quantizes an already-exported ONNX graph: you keep a
full-precision artifact to compare every quantized one against.

---

## 4. Measured result

This session measures nothing on an LLM, so `results/results.json` has no LLM rows
and this guide renders no results block. That is deliberate. A tokens/s figure
copied from a vendor page was measured on someone else's GPU, prompt lengths and
concurrency. It is a rumour in the sense of `ci/benchmark_card.md.tmpl`.

What transfers is the row contract. A measured LLM row needs the same five-plus-cost
shape as a journey-table row, keyed per hardware and with a parent:

| Field | How to get it |
|---|---|
| parent (lineage) | the checkpoint it was converted or quantized from: FP16 → AWQ INT4 is a tree edge, like `student-distilled-onnx` → `student-int8-entropy` |
| hardware + engine + precision | GPU model, driver, engine and version, quantization method; the same header `tools/journey_table.py` prints |
| TTFT p50/p95, TPOT p50/p95 | a streaming load generator: `vllm bench serve` from Session 3 level 5 |
| tokens/s at each concurrency | the same sweep; pick the usable point under both SLAs |
| quality | a frozen testset and a gate, as in [Session 4 Guide E (RAGAS)](../../session_4/README.md#guide-e--ragas), reported per slice as in [Session 4 section 10](../../session_4/README.md#10-evaluating-the-llm--and-evaluating-the-evaluator) |
| peak VRAM | weights + KV cache at that concurrency and context length |
| cost per million tokens | the formula in section 2 |

In production, [Session 4 Guide D (Langfuse)](../../session_4/README.md#guide-d--langfuse)
traces each generation with its token usage and timing. That is where TTFT and
per-request token counts come from once real traffic replaces the benchmark.

---

## 5. Gotchas

1. **Reporting tokens/s alone.**
   **Symptom:** a change raises system tokens/s in the benchmark, and users report
   the endpoint feels slower. More concurrency packed more sequences into each
   decode step, and TPOT p95 grew for everyone.
   Report TTFT p95 and TPOT p95 next to throughput, and cost only the usable point.
   `src/cost.py` refuses to cost an operating point that breaks the SLA for exactly
   this reason.

2. **Expecting weight-only INT4 to speed up prefill.**
   **Symptom:** TPOT improves after GPTQ or AWQ, TTFT on long prompts does not, and
   someone concludes the quantization is broken.
   Weight-only quantization shrinks the bytes decode has to read. Prefill is
   compute-bound, and INT4 weights still have to be dequantized for the arithmetic.
   Pick the lever by the phase you are paying for: long prompts and short answers
   need prefill levers; chat with long answers needs decode levers.

3. **Calibrating on the wrong text.**
   **Symptom:** the English eval is unchanged after quantization, and the Arabic
   slice of the same eval drops.
   This is `ort-static-int8-daytime` again, with languages in place of lighting.
   GPTQ and AWQ fit their scales to calibration text, just as s06a fits activation
   ranges to calibration frames. Stratify the calibration set across the slices you
   serve, and read the eval per slice. Session 4 reports its judge per language for
   this reason.

4. **Using bitwise parity as the gate for a quantized LLM.**
   **Symptom:** every quantized checkpoint fails an exact-output comparison, so the
   team disables the gate entirely.
   One flipped token early in a greedy decode changes every token after it, so
   output-level parity is the wrong test. Keep `snippet:parity-gate` for conversions
   that should be numerically identical (FP16 checkpoint → FP16 engine). Gate
   precision changes on a task-eval drop with an allowed tolerance, the way
   `SLA.max_ocr_em_drop` tolerates a bounded change.

5. **Copying a `convert.py --outtype q4_k_m` command from a blog post.**
   **Symptom:** the script is not found in a current llama.cpp checkout, or the
   output type is rejected.
   Convert with `convert_hf_to_gguf.py` to F16, then quantize with `llama-quantize`,
   as in section 3.

6. **Sizing VRAM from the model file.**
   **Symptom:** the quantized weights fit on the GPU with room to spare, and the
   server still runs out of memory, or starts preempting sequences, once many users
   send long contexts.
   The KV cache grows with active sequences × context length, and it is not in the
   checkpoint's file size. Size memory at the target concurrency and context length.
   This is the LLM version of reading `peak RSS MB` instead of `size MB`.

7. **Copying an FP8 recipe from an H100 guide onto the RTX 3090.**
   **Symptom:** the FP8 path is unavailable on the Ampere card, and the "FP8"
   deployment either fails or is not running FP8 kernels.
   FP8 needs Ada or newer. On the 3090 your precision levers are FP16, INT8 and
   weight-only INT4, the same boundary s08 draws for the ANPR engines.

8. **Assuming speculative decoding helps at every load.**
   **Symptom:** a speculative setup that helped at low concurrency stops helping as
   concurrency rises.
   Speculation turns memory-bound decode into compute-bound verification. Once
   batching has already saturated the GPU's compute, there is little idle compute
   left to trade. Measure it across the concurrency sweep, not at one point.

---

## 6. AV comparison callout

> **Context, not a recipe.** Vehicle programs now put language and vision-language
> models on in-car compute, and the same prefill/decode split applies there. But a
> driving control loop cannot wait for a token stream. A model that answers in
> tokens fits tasks where a human-scale wait is fine: cabin assistants, scene
> descriptions, offline triage of fleet data. Hard deadlines stay with
> single-pass networks like our detector. The ANPR camera makes the same split: the
> plate read is single-pass and budgeted at 30 ms; anything conversational would
> live off the frame path. The optimization questions (memory bandwidth on the SoC,
> an engine tied to that SoC's toolchain, calibration data matching the fleet's
> conditions) are the ones this session already asked.

---

## 7. When NOT to use this

- **You call a hosted LLM API.** You control none of the four levers: no runtime,
  no precision, no scheduler. Your levers are prompt length (TTFT), output length
  (TPOT) and model choice. Your cost is the provider's per-token price, not
  `usd_per_hour`.
- **Quality is not yet measured.** Do not quantize weights or the KV cache before a
  frozen eval and a gate exist ([Session 4 Guide E](../../session_4/README.md#guide-e--ragas)).
  Without them you cannot tell a cheaper model from a worse one. That is the LLM
  version of optimizing ANPR without per-condition accuracy.
- **Traffic is low and latency is generous.** One small model under llama.cpp on a
  CPU, or an unmodified vLLM server, may already meet both SLAs. Session 3's
  decision path applies unchanged: stop at the first level that meets the contract.
- **The model is not autoregressive.** Classifiers, embedding models, rerankers and
  our CRNN run one pass per input. TTFT and TPOT mean nothing there. Use the ANPR
  metrics and Triton-style dynamic batching instead.
- **The bottleneck is outside the model.** Retrieval, tool calls and network hops
  often dominate an LLM application's latency. Trace first (Session 4 Guide D), the
  way s00 profiles first. If the generation span is a small share of the trace,
  none of these levers is the fix.

---

Previous: [13 — TFLite and the edge](13-tflite-and-edge.md) · Next: [15 — Optimization in your MLOps stack](15-optimization-in-your-mlops-stack.md)
