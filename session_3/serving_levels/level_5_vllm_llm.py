"""
LEVEL 5 — Serving an LLM with vLLM.

LLMs don't fit the batching story from levels 2-4: each request generates a
different number of tokens, so vLLM uses continuous batching + PagedAttention
instead. You don't configure any of it — it's the default.

    pip install vllm    # needs a CUDA GPU (Linux); no Apple Silicon support

1. Start an OpenAI-compatible server (one command, no code):

    vllm serve Qwen/Qwen2.5-7B-Instruct \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size 1 \      # number of GPUs to shard across
        --max-model-len 8192            # max context window

2-3. Call it with the OpenAI client — zero code change vs the OpenAI API.
4.   Or skip HTTP entirely and use the in-process Python API.
"""

# ── 2. Chat completion — identical to calling OpenAI ─────────────────────────
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Explain PagedAttention in one sentence."}],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)

# ── 3. Streaming — users see tokens as they generate ─────────────────────────
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about MLOps."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# ── 4. Python API (in-process, no HTTP server) ───────────────────────────────
# Batch/offline use: pass ALL prompts in one call and vLLM schedules them
# together — looping one prompt at a time gives up the throughput win.
if False:  # flip to True to run offline inference instead of calling the server
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", dtype="float16")
    params = SamplingParams(temperature=0.8, max_tokens=512)
    outputs = llm.generate(["Tell me about MLOps in 3 bullet points."], params)
    for out in outputs:
        print(out.outputs[0].text)
