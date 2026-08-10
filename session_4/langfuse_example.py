"""Langfuse concepts, as a v2-era sketch. NOT RUNNABLE — see ollama_langfuse_rag.py.

Two separate reasons this file does not execute:

1. It's a snippet — `vector_store`, `build_prompt` and `call_llm` are never defined.
2. Every Langfuse API below was REMOVED in SDK v4, which is what
   `pip install -e ".[llm]"` gives you. The import on the next line raises
   ModuleNotFoundError immediately.

   v2 (below)                              v4 (ollama_langfuse_rag.py)
   ─────────────────────────────────────   ──────────────────────────────────────
   from langfuse.decorators import ...     from langfuse import observe, get_client
   langfuse_context.update_current_trace   propagate_attributes(...)
   langfuse_context.update_current_obs...  langfuse.update_current_span(...)
   langfuse.score(...)                     langfuse.create_score(...) /
                                             score_current_trace(...)
   langfuse.generation(...)                langfuse.start_as_current_observation(
                                             as_type="generation", ...)

Kept as-is because the SHAPE it teaches is still right — a trace per request,
a span per pipeline step, scores attached afterwards. For working code against a
real model, read ollama_langfuse_rag.py instead.
"""

from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()   # reads LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

# ── @observe auto-creates a trace for every function call ─────
@observe()
def rag_pipeline(user_query: str, user_id: str) -> str:
    """Full RAG pipeline — each step becomes a span in Langfuse."""

    # Step 1: Retrieve context
    docs = retrieve_documents(user_query)   # span: retrieve

    # Step 2: Build prompt
    prompt = build_prompt(user_query, docs)

    # Step 3: Generate — langfuse captures tokens, cost, latency
    response = call_llm(prompt)             # span: generation

    # Attach metadata to the trace
    langfuse_context.update_current_trace(
        user_id=user_id,
        tags=["rag", "production"],
        metadata={"n_docs_retrieved": len(docs)},
    )
    return response

@observe(name="retrieve")
def retrieve_documents(query: str) -> list:
    results = vector_store.similarity_search(query, k=5)
    # Log retrieval quality score
    langfuse_context.update_current_observation(
        input=query,
        output=[r.page_content[:200] for r in results],
        metadata={"top_score": results[0].metadata.get("score", 0)},
    )
    return results

# ── Manual scoring — attach human or automated scores ─────────
trace_id = langfuse_context.get_current_trace_id()
langfuse.score(
    trace_id=trace_id,
    name="user_feedback",     # thumbs up/down
    value=1,                  # 1 = positive, 0 = negative
    comment="Response was accurate and helpful",
)

import tiktoken
from langfuse import Langfuse
from prometheus_client import Counter, Gauge

# ── Count tokens before calling the LLM ──────────────────────
enc = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# ── Prometheus metrics for token/cost tracking ────────────────
TOKEN_COUNTER = Counter("llm_tokens_total", "Total tokens used",
                        ["model", "type", "feature"])  # type: prompt|completion
COST_GAUGE    = Gauge("llm_cost_usd_per_hour", "Estimated LLM cost/hour", ["model"])

# ── Cost calculation (update prices as they change) ──────────
COST_PER_1K = {
    "gpt-4o":          {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini":     {"prompt": 0.00015, "completion": 0.0006},
    "claude-sonnet":   {"prompt": 0.003,  "completion": 0.015},
}

def track_llm_call(model, prompt, response, feature="unknown"):
    prompt_tokens     = count_tokens(prompt)
    completion_tokens = count_tokens(response)
    prices = COST_PER_1K.get(model, {"prompt": 0, "completion": 0})
    cost   = (prompt_tokens/1000 * prices["prompt"]
            + completion_tokens/1000 * prices["completion"])

    TOKEN_COUNTER.labels(model=model, type="prompt",     feature=feature).inc(prompt_tokens)
    TOKEN_COUNTER.labels(model=model, type="completion", feature=feature).inc(completion_tokens)

    # Log to Langfuse for per-trace cost visibility
    langfuse.generation(
        model=model, usage={"prompt_tokens": prompt_tokens,
                             "completion_tokens": completion_tokens},
        metadata={"estimated_cost_usd": cost, "feature": feature}
    )
    return cost

# ── Grafana alert: daily cost > $50 → trigger alert ──────────
# increase(llm_tokens_total{type="completion"}[24h]) * 0.000015
