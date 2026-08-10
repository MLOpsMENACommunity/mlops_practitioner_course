"""Evaluate stage: score the RAG system with RAGAS on a fixed test set.

Classical ML equivalent: evaluating the model on a held-out test split.

For every question in data/eval/test_set.jsonl we retrieve top-k chunks from
the vector index, generate an answer with an LLM, then let RAGAS score
faithfulness, answer relevancy, context recall and context precision.
The scores land in reports/ragas.json (a DVC metrics file) and, optionally,
in MLflow when MLFLOW_TRACKING_URI is set.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import structlog
import yaml

# Reuse the exact embedding logic from the embed stage so queries are
# embedded the same way as the indexed chunks.
from embed import embed_texts

log = structlog.get_logger()

PARAMS_PATH = Path("params.yaml")
INDEX_DIR = Path("data/index")
TEST_SET_PATH = Path("data/eval/test_set.jsonl")
REPORT_PATH = Path("reports/ragas.json")
TOP_K = 4

SYSTEM_PROMPT = (
    "أنت مساعد قانوني. أجب عن السؤال بدقة اعتمادًا فقط على المقاطع المرجعية "
    "المقدمة لك. إذا لم تجد الإجابة في المقاطع فقل ذلك صراحة."
)


def load_manifest(index_dir: Path) -> dict:
    """Read the index manifest written by the embed stage."""
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        log.error("manifest_missing", expected=str(manifest_path), hint="Run the embed stage first.")
        sys.exit(1)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_test_set(path: Path) -> list[dict]:
    """Load the evaluation questions (question / ground_truth / reference_doc)."""
    if not path.is_file():
        log.error("test_set_missing", expected=str(path))
        sys.exit(1)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve_faiss(query_vector: np.ndarray, k: int) -> list[str]:
    """Retrieve top-k chunk texts from the FAISS index."""
    try:
        import faiss
    except ImportError:
        log.error("faiss_not_installed", hint="pip install faiss-cpu")
        sys.exit(1)
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    query = query_vector.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(query)
    _, ids = index.search(query, k)
    metadata: dict[int, str] = {}
    with (INDEX_DIR / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            metadata[record["id"]] = record["text"]
    return [metadata[i] for i in ids[0] if i in metadata]


def retrieve_chroma(query_vector: np.ndarray, k: int) -> list[str]:
    """Retrieve top-k chunk texts from the Chroma collection."""
    try:
        import chromadb
    except ImportError:
        log.error("chromadb_not_installed", hint="pip install chromadb")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    collection = client.get_collection("rag_chunks")
    result = collection.query(query_embeddings=[query_vector.tolist()], n_results=k)
    return result["documents"][0]


def retrieve(query_vector: np.ndarray, index_type: str, k: int = TOP_K) -> list[str]:
    """Dispatch retrieval to the backend recorded in the manifest."""
    if index_type == "faiss":
        return retrieve_faiss(query_vector, k)
    if index_type == "chroma":
        return retrieve_chroma(query_vector, k)
    log.error("unknown_index_type", index_type=index_type)
    sys.exit(1)


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate an answer with an OpenAI-compatible chat completions API.

    Set OPENAI_BASE_URL to point at any OpenAI-compatible server — e.g. a
    vLLM deployment's /v1 endpoint — and the very same code works unchanged;
    only the base URL (and possibly the model name) differs.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai_not_installed", hint="pip install openai")
        sys.exit(1)
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"))  # None -> api.openai.com
    model = os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")
    context_block = "\n\n---\n\n".join(contexts)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"المقاطع المرجعية:\n{context_block}\n\nالسؤال: {question}",
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def run_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Score the collected RAG outputs with RAGAS."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        log.error("ragas_not_installed", hint="pip install ragas datasets")
        sys.exit(1)
    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    scores = result.to_pandas()[
        ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    ].mean()
    return {metric: float(value) for metric, value in scores.items()}


def write_report(scores: dict[str, float], n_questions: int, path: Path) -> dict:
    """Write the metrics file consumed by `dvc metrics show` and gate.py."""
    report = {
        "faithfulness": round(scores["faithfulness"], 4),
        "answer_relevancy": round(scores["answer_relevancy"], 4),
        "context_recall": round(scores["context_recall"], 4),
        "context_precision": round(scores["context_precision"], 4),
        "n_questions": n_questions,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def log_to_mlflow(params: dict, report: dict) -> None:
    """Log params + metrics to MLflow when MLFLOW_TRACKING_URI is set."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        log.info("mlflow_skipped", reason="MLFLOW_TRACKING_URI not set")
        return
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow_not_installed", hint="pip install mlflow")
        return
    mlflow.set_experiment("arabic-rag-experiments")
    with mlflow.start_run():
        mlflow.log_params(
            {
                "embedding_model": params["embedding_model"],
                "index_type": params["index_type"],
                "chunk_size": params["chunk_size"],
                "chunk_overlap": params["chunk_overlap"],
                "top_k": TOP_K,
            }
        )
        mlflow.log_metrics(
            {
                metric: report[metric]
                for metric in (
                    "faithfulness",
                    "answer_relevancy",
                    "context_recall",
                    "context_precision",
                )
            }
        )
    log.info("mlflow_logged", experiment="arabic-rag-experiments")


def main() -> None:
    """Run the evaluate stage."""
    with PARAMS_PATH.open(encoding="utf-8") as f:
        params = yaml.safe_load(f)
    manifest = load_manifest(INDEX_DIR)
    test_set = load_test_set(TEST_SET_PATH)
    log.info(
        "evaluation_start",
        n_questions=len(test_set),
        index_type=manifest["index_type"],
        embedding_model=manifest["embedding_model"],
    )

    questions = [item["question"] for item in test_set]
    ground_truths = [item["ground_truth"] for item in test_set]
    query_vectors = embed_texts(questions, manifest["embedding_model"])

    contexts: list[list[str]] = []
    answers: list[str] = []
    for question, query_vector in zip(questions, query_vectors):
        retrieved = retrieve(query_vector, manifest["index_type"])
        contexts.append(retrieved)
        answers.append(generate_answer(question, retrieved))
        log.debug("question_answered", question=question[:60])

    scores = run_ragas(questions, answers, contexts, ground_truths)
    report = write_report(scores, len(test_set), REPORT_PATH)
    log.info("evaluation_complete", **{k: v for k, v in report.items() if k != "timestamp"})
    log_to_mlflow(params, report)


if __name__ == "__main__":
    main()
