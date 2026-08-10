from ragas import evaluate
from ragas.metrics import (
    faithfulness,          # Is the answer grounded in the retrieved context?
    answer_relevancy,      # Does the answer address the question asked?
    context_recall,        # Did retrieval find all relevant documents?
    context_precision,     # Were retrieved documents actually useful?
)
from datasets import Dataset
from langfuse import Langfuse

# ── Prepare evaluation dataset ────────────────────────────────
data = {
    "question":   ["What is the cancellation policy?", ...],
    "answer":     ["You can cancel within 24 hours...", ...],  # LLM output
    "contexts":   [["Policy doc chunk 1", "Policy doc chunk 2"], ...],  # retrieved
    "ground_truth":["Cancellations are free within 24 hours.", ...],    # optional
}
dataset = Dataset.from_dict(data)

# ── Run evaluation ─────────────────────────────────────────────
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result)
# {'faithfulness': 0.82, 'answer_relevancy': 0.91,
#  'context_recall': 0.75, 'context_precision': 0.88}

# ── Log scores to Langfuse for trend tracking ─────────────────
lf = Langfuse()
for metric_name, score in result.items():
    lf.score(trace_id=trace_id, name=f"ragas_{metric_name}", value=score)

# ── Set thresholds — alert if any metric drops below ──────────
THRESHOLDS = {"faithfulness": 0.8, "answer_relevancy": 0.85,
              "context_recall": 0.7, "context_precision": 0.8}
for metric, threshold in THRESHOLDS.items():
    if result[metric] < threshold:
        raise ValueError(f"RAGAS alert: {metric}={result[metric]:.2f} < {threshold}")
