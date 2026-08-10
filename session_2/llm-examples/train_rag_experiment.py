"""Track Arabic-legal RAG configuration experiments with MLflow.

Session 2 teaching example: experiment tracking applied to LLM/RAG systems.

We sweep chunking + embedding configurations over a tiny Arabic legal corpus
(snippets styled after Egyptian civil code articles), evaluate each pipeline
with RAGAS, and use MLflow exactly the way we used it for classical ML.

# ---------------------------------------------------------------------------
# How MLflow concepts from classical ML map onto RAG experimentation
# ---------------------------------------------------------------------------
#   hyperparameters      -> chunking config (chunk_size, overlap, embedding model)
#   training run         -> one RAG pipeline build (chunk -> embed -> index) + evaluation
#   MAE / accuracy       -> faithfulness / answer_relevancy / context_recall / precision
#   model artifact       -> the config + index recipe (YAML + RAGAS results JSON)
#   model registry       -> best RAG config promoted to "production"
#   feature engineering  -> chunking + embedding (how raw text becomes vectors)
#   prompt versioning    -> MLflow Prompt Registry (register / alias / load)
# ---------------------------------------------------------------------------

Run with:
    OPENAI_API_KEY=sk-... python train_rag_experiment.py

Without OPENAI_API_KEY the script degrades gracefully: OpenAI-embedding configs
are skipped, generation and RAGAS (which need an LLM judge) are skipped, and
only retrieval latency + params are logged for the local-embedding configs.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import structlog
import yaml

# ---------------------------------------------------------------------------
# Logging setup (structlog instead of print: structured, greppable, timestamped)
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger()

EXPERIMENT_NAME = "arabic-rag-experiments"
PROMPT_NAME = "arabic-legal-assistant"
TOP_K = 3

# ---------------------------------------------------------------------------
# Corpus: stand-in snippets styled after Egyptian civil code articles.
# In the real project this would be the full legal corpus loaded from disk.
# ---------------------------------------------------------------------------
DOCUMENTS: list[str] = [
    # Contracts: formation by matching offer and acceptance.
    "المادة ٨٩ من القانون المدني: يتم العقد بمجرد أن يتبادل طرفان التعبير عن "
    "إرادتين متطابقتين، مع مراعاة ما يقرره القانون فوق ذلك من أوضاع معينة "
    "لانعقاد العقد. ويجوز أن يكون التعبير عن الإرادة باللفظ أو بالكتابة أو "
    "بالإشارة المتداولة عرفاً، كما يجوز أن يكون باتخاذ موقف لا تدع ظروف الحال "
    "شكاً في دلالته على حقيقة المقصود.",
    # Property: scope of ownership.
    "المادة ٨٠٢ من القانون المدني: لمالك الشيء وحده، في حدود القانون، حق "
    "استعماله واستغلاله والتصرف فيه. ومالك الشيء يملك كل ما يعد من عناصره "
    "الجوهرية بحيث لا يمكن فصله عنه دون أن يهلك أو يتلف أو يتغير، ويشمل ملك "
    "الأرض ما فوقها وما تحتها إلى الحد المفيد في التمتع بها علواً وعمقاً.",
    # Leases: definition of the lease contract.
    "المادة ٥٥٨ من القانون المدني: الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكّن "
    "المستأجر من الانتفاع بشيء معين مدة معينة لقاء أجر معلوم. ويلتزم المؤجر "
    "أن يسلم المستأجر العين المؤجرة وملحقاتها في حالة تصلح معها لأن تفي بما "
    "أعدت له من المنفعة وفقاً لما تم عليه الاتفاق أو لطبيعة العين.",
    # Sale: definition of the sale contract.
    "المادة ٤١٨ من القانون المدني: البيع عقد يلتزم به البائع أن ينقل للمشتري "
    "ملكية شيء أو حقاً مالياً آخر في مقابل ثمن نقدي. ويلتزم البائع بتسليم "
    "المبيع للمشتري في الحالة التي كان عليها وقت البيع، ويشمل التسليم ملحقات "
    "الشيء المبيع وكل ما أعد بصفة دائمة لاستعماله.",
    # Compensation for unlawful acts.
    "المادة ١٦٣ من القانون المدني: كل خطأ سبب ضرراً للغير يلزم من ارتكبه "
    "بالتعويض. ويقدر القاضي مدى التعويض عن الضرر الذي لحق المضرور، ويشمل "
    "التعويض ما لحق الدائن من خسارة وما فاته من كسب، بشرط أن يكون ذلك نتيجة "
    "طبيعية للعمل غير المشروع.",
]

# Five Arabic test questions about the snippets, with ground-truth answers.
TEST_QUESTIONS: list[dict[str, str]] = [
    {
        "question": "متى يتم انعقاد العقد وفقاً للقانون المدني؟",
        "ground_truth": "يتم العقد بمجرد أن يتبادل طرفان التعبير عن إرادتين متطابقتين، "
        "مع مراعاة ما يقرره القانون من أوضاع معينة لانعقاد العقد.",
    },
    {
        "question": "ما هي حقوق مالك الشيء وفقاً للمادة ٨٠٢؟",
        "ground_truth": "لمالك الشيء وحده، في حدود القانون، حق استعماله واستغلاله "
        "والتصرف فيه.",
    },
    {
        "question": "ما هو تعريف عقد الإيجار؟",
        "ground_truth": "الإيجار عقد يلتزم المؤجر بمقتضاه أن يمكّن المستأجر من الانتفاع "
        "بشيء معين مدة معينة لقاء أجر معلوم.",
    },
    {
        "question": "بماذا يلتزم البائع في عقد البيع؟",
        "ground_truth": "يلتزم البائع أن ينقل للمشتري ملكية شيء أو حقاً مالياً آخر في "
        "مقابل ثمن نقدي، وأن يسلم المبيع في الحالة التي كان عليها وقت البيع.",
    },
    {
        "question": "ماذا يشمل التعويض عن العمل غير المشروع؟",
        "ground_truth": "يشمل التعويض ما لحق الدائن من خسارة وما فاته من كسب، بشرط أن "
        "يكون ذلك نتيجة طبيعية للعمل غير المشروع.",
    },
]

# System prompt for the Arabic legal assistant (also versioned in the Prompt Registry).
SYSTEM_PROMPT = (
    "أنت مساعد قانوني متخصص في القانون المدني المصري. "
    "أجب عن سؤال المستخدم بالاعتماد حصرياً على النصوص القانونية المقدمة في السياق. "
    "إذا لم يوجد الجواب في السياق فقل ذلك صراحة ولا تخترع نصوصاً قانونية.\n\n"
    "السياق:\n{{context}}"
)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentConfig:
    """One RAG pipeline configuration — the 'hyperparameters' of this experiment."""

    chunk_size: int
    overlap: int
    embedding_model: str

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dict (for YAML/params logging)."""
        return asdict(self)


def build_configs() -> list[ExperimentConfig]:
    """Build the experiment grid.

    chunk_sizes and overlaps are paired index-wise -- (256, 32), (512, 64),
    (1024, 128) -- then crossed with both embedding models: 3 x 2 = 6 runs.
    A full cartesian product (3 x 3 x 2 = 18) would work too, but pairing
    keeps the sweep small and readable for teaching.
    """
    chunk_sizes = [256, 512, 1024]
    overlaps = [32, 64, 128]
    embedding_models = ["text-embedding-3-small", "nomic-embed-text"]

    return [
        ExperimentConfig(chunk_size=size, overlap=overlap, embedding_model=model)
        for size, overlap in zip(chunk_sizes, overlaps)
        for model in embedding_models
    ]


# ---------------------------------------------------------------------------
# RAG pipeline steps ("feature engineering": raw text -> chunks -> vectors)
# ---------------------------------------------------------------------------
def chunk_documents(documents: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Split documents into overlapping character-based chunks.

    Simple sliding window: each chunk is `chunk_size` characters and
    consecutive chunks share `overlap` characters. Real systems would chunk
    on sentence/article boundaries, but the tracking story is identical.
    """
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for doc in documents:
        for start in range(0, len(doc), step):
            chunk = doc[start : start + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
            if start + chunk_size >= len(doc):
                break
    return chunks


def embed_texts(texts: list[str], embedding_model: str, is_query: bool = False) -> np.ndarray | None:
    """Embed texts with the configured model; return L2-normalized vectors.

    - "text-embedding-3-small" -> OpenAI embeddings API (needs OPENAI_API_KEY)
    - "nomic-embed-text"       -> local sentence-transformers model

    Returns None (with a structlog warning) when the required dependency or
    API key is missing, so callers can skip this config gracefully.
    """
    if embedding_model == "text-embedding-3-small":
        try:
            from openai import OpenAI
        except ImportError:
            log.warning("openai package not installed, skipping", model=embedding_model)
            return None
        if not os.environ.get("OPENAI_API_KEY"):
            log.warning("OPENAI_API_KEY unset, skipping OpenAI embeddings")
            return None
        client = OpenAI()
        response = client.embeddings.create(model=embedding_model, input=texts)
        vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    elif embedding_model == "nomic-embed-text":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            log.warning("sentence-transformers not installed, skipping", model=embedding_model)
            return None
        # nomic-embed-text-v1.5 expects task prefixes on inputs.
        prefix = "search_query: " if is_query else "search_document: "
        model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        vectors = np.asarray(model.encode([prefix + t for t in texts]), dtype=np.float32)
    else:
        log.warning("unknown embedding model", model=embedding_model)
        return None

    # Normalize so inner product == cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def build_index(embeddings: np.ndarray) -> Any | None:
    """Build a FAISS inner-product index over normalized embeddings."""
    try:
        import faiss
    except ImportError:
        log.warning("faiss-cpu not installed, skipping index build")
        return None
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def retrieve(index: Any, query_vector: np.ndarray, chunks: list[str], top_k: int = TOP_K) -> list[str]:
    """Return the top-k most similar chunks for a query vector."""
    _, indices = index.search(query_vector.reshape(1, -1), top_k)
    return [chunks[i] for i in indices[0] if i >= 0]


def generate_answer(question: str, contexts: list[str], system_prompt: str) -> str | None:
    """Generate an answer with an OpenAI chat completion, grounded in contexts.

    Returns None (gracefully) when the API key or package is unavailable.
    """
    try:
        from openai import OpenAI
    except ImportError:
        log.warning("openai package not installed, skipping generation")
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    client = OpenAI()
    context_block = "\n\n---\n\n".join(contexts)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt.replace("{{context}}", context_block)},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Evaluation ("test set metrics": RAGAS instead of MAE/accuracy)
# ---------------------------------------------------------------------------
def evaluate_with_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """Score the pipeline with RAGAS (needs an OpenAI judge under the hood).

    Returns a dict of metric name -> score; empty dict when RAGAS or the API
    key is unavailable.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        log.warning("OPENAI_API_KEY unset, skipping RAGAS (needs an LLM judge)")
        return {}
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        log.warning("ragas not installed, skipping evaluation")
        return {}

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": q,
                "response": a,
                "retrieved_contexts": ctx,
                "reference": gt,
            }
            for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
        ]
    )
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    scores_df = result.to_pandas()
    metrics: dict[str, float] = {}
    for name in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
        if name in scores_df.columns:
            value = float(scores_df[name].mean())
            if not np.isnan(value):
                metrics[name] = value
    return metrics


# ---------------------------------------------------------------------------
# One experiment = one MLflow run ("training run")
# ---------------------------------------------------------------------------
def run_experiment(config: ExperimentConfig) -> dict[str, Any] | None:
    """Build + evaluate one RAG configuration inside an MLflow run.

    Returns a summary dict (config + metrics + run_id) for best-config
    selection, or None when the config could not be executed.
    """
    run_name = f"chunk{config.chunk_size}_overlap{config.overlap}_{config.embedding_model}"
    log.info("starting experiment", run_name=run_name)

    # ---- Chunk + embed + index (done before opening the run so a missing
    # dependency doesn't leave an empty run behind) ----
    chunks = chunk_documents(DOCUMENTS, config.chunk_size, config.overlap)
    chunk_vectors = embed_texts(chunks, config.embedding_model)
    if chunk_vectors is None:
        log.warning("skipping config, embeddings unavailable", run_name=run_name)
        return None
    index = build_index(chunk_vectors)
    if index is None:
        log.warning("skipping config, faiss unavailable", run_name=run_name)
        return None

    # ---- Retrieval + generation over the test questions ----
    questions = [item["question"] for item in TEST_QUESTIONS]
    ground_truths = [item["ground_truth"] for item in TEST_QUESTIONS]
    answers: list[str] = []
    contexts: list[list[str]] = []
    retrieval_latencies_ms: list[float] = []
    generation_latencies_ms: list[float] = []
    generation_available = True

    for question in questions:
        start = time.perf_counter()
        query_vector = embed_texts([question], config.embedding_model, is_query=True)
        retrieved = retrieve(index, query_vector[0], chunks)
        retrieval_latencies_ms.append((time.perf_counter() - start) * 1000)
        contexts.append(retrieved)

        start = time.perf_counter()
        answer = generate_answer(question, retrieved, SYSTEM_PROMPT)
        if answer is None:
            generation_available = False
            answers.append("")
        else:
            generation_latencies_ms.append((time.perf_counter() - start) * 1000)
            answers.append(answer)

    if not generation_available:
        log.warning("generation skipped (no API key), RAGAS will be skipped too", run_name=run_name)

    # ---- Evaluate (only meaningful when we actually generated answers) ----
    ragas_scores: dict[str, float] = {}
    if generation_available:
        ragas_scores = evaluate_with_ragas(questions, answers, contexts, ground_truths)

    # ---- Log everything to MLflow ----
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "chunk_size": config.chunk_size,
                "overlap": config.overlap,
                "embedding_model": config.embedding_model,
                "n_chunks": len(chunks),
                "n_test_questions": len(questions),
            }
        )

        metrics: dict[str, float] = dict(ragas_scores)
        metrics["mean_retrieval_latency_ms"] = float(np.mean(retrieval_latencies_ms))
        if generation_latencies_ms:
            metrics["mean_generation_latency_ms"] = float(np.mean(generation_latencies_ms))
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp_dir:
            results_path = Path(tmp_dir) / "ragas_results.json"
            results_path.write_text(
                json.dumps(
                    {
                        "config": config.to_dict(),
                        "scores": ragas_scores,
                        "questions": questions,
                        "answers": answers,
                        "contexts": contexts,
                        "ground_truths": ground_truths,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            config_path = Path(tmp_dir) / "config.yaml"
            config_path.write_text(
                yaml.safe_dump(config.to_dict(), allow_unicode=True), encoding="utf-8"
            )
            mlflow.log_artifact(str(results_path))
            mlflow.log_artifact(str(config_path))

        log.info("experiment finished", run_name=run_name, **metrics)
        return {"config": config, "metrics": metrics, "run_id": run.info.run_id}


# ---------------------------------------------------------------------------
# Best-config registration ("model registry": promote the winner)
# ---------------------------------------------------------------------------
def register_best_config(results: list[dict[str, Any]]) -> None:
    """Pick the best config by faithfulness and register it as RAGConfig.

    The winning config YAML is logged in a dedicated run tagged
    best_config=true — the RAG analogue of promoting a model to production.
    """
    scored = [r for r in results if "faithfulness" in r["metrics"]]
    if not scored:
        log.warning("no runs produced a faithfulness score, skipping best-config registration")
        return

    best = max(scored, key=lambda r: r["metrics"]["faithfulness"])
    config: ExperimentConfig = best["config"]
    log.info(
        "registering best config",
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        embedding_model=config.embedding_model,
        faithfulness=best["metrics"]["faithfulness"],
    )

    with mlflow.start_run(run_name="best_rag_config"):
        mlflow.set_tag("best_config", "true")
        mlflow.set_tag("selected_by", "faithfulness")
        mlflow.set_tag("source_run_id", best["run_id"])
        mlflow.log_params(config.to_dict())
        mlflow.log_metric("faithfulness", best["metrics"]["faithfulness"])
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "RAGConfig.yaml"
            config_path.write_text(
                yaml.safe_dump(config.to_dict(), allow_unicode=True), encoding="utf-8"
            )
            mlflow.log_artifact(str(config_path), artifact_path="RAGConfig")


# ---------------------------------------------------------------------------
# Prompt Registry ("model registry, but for prompts")
# ---------------------------------------------------------------------------
def demo_prompt_registry() -> None:
    """Version the system prompt in MLflow's Prompt Registry and load it back.

    register -> alias to "production" -> load by alias is exactly how serving
    code should consume prompts: no hardcoded prompt strings in production.
    Wrapped in try/except because the genai prompt APIs need mlflow >= 2.12.
    """
    try:
        prompt = mlflow.genai.register_prompt(
            name=PROMPT_NAME,
            template=SYSTEM_PROMPT,
            commit_message="Arabic legal assistant system prompt, grounded-answers-only",
        )
        log.info("prompt registered", name=PROMPT_NAME, version=prompt.version)

        mlflow.genai.set_prompt_alias(PROMPT_NAME, alias="production", version=prompt.version)
        log.info("prompt promoted", name=PROMPT_NAME, alias="production")

        # This is the line serving code would run at startup:
        production_prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
        log.info(
            "prompt loaded by alias",
            name=PROMPT_NAME,
            template_preview=production_prompt.template[:60],
        )
    except (AttributeError, ImportError) as exc:
        log.warning("prompt registry unavailable, need mlflow >= 2.12", error=str(exc))
    except Exception as exc:  # registry backend may not support prompts
        log.warning("prompt registry call failed", error=str(exc))


# ---------------------------------------------------------------------------
# Entry point: the full sweep
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full experiment sweep, register the winner, demo the registry."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    configs = build_configs()
    log.info("starting sweep", experiment=EXPERIMENT_NAME, n_configs=len(configs))

    if not os.environ.get("OPENAI_API_KEY"):
        log.warning(
            "OPENAI_API_KEY unset: OpenAI-embedding configs, generation, and "
            "RAGAS will be skipped; local configs still log latency + params"
        )

    results: list[dict[str, Any]] = []
    for config in configs:
        if config.embedding_model == "text-embedding-3-small" and not os.environ.get(
            "OPENAI_API_KEY"
        ):
            log.warning("skipping OpenAI-embedding config, no API key", config=config.to_dict())
            continue
        result = run_experiment(config)
        if result is not None:
            results.append(result)

    log.info("sweep finished", n_completed=len(results), n_configs=len(configs))
    register_best_config(results)
    demo_prompt_registry()


if __name__ == "__main__":
    main()
