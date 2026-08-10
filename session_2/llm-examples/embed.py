"""Embed stage: turn text chunks into a searchable vector index.

Classical ML equivalent: the compiled/trained model artifact.

Reads data/chunks/chunks.jsonl, embeds each chunk with the model named in
params.yaml, and builds either a FAISS or a Chroma index under data/index/.
A manifest.json is written alongside so that evaluate_ragas.py knows which
backend and embedding model to use for queries.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import structlog
import yaml

log = structlog.get_logger()

PARAMS_PATH = Path("params.yaml")
CHUNKS_PATH = Path("data/chunks/chunks.jsonl")
INDEX_DIR = Path("data/index")


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(path: Path) -> list[dict]:
    """Read chunk records from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_openai(texts: list[str], model: str) -> np.ndarray:
    """Embed texts with the OpenAI embeddings API."""
    try:
        from openai import OpenAI
    except ImportError:
        log.error("openai_not_installed", hint="pip install openai")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        log.error(
            "openai_api_key_missing",
            hint="export OPENAI_API_KEY=... before running this stage",
        )
        sys.exit(1)
    client = OpenAI()
    vectors: list[list[float]] = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=model, input=texts[i : i + batch_size])
        vectors.extend(item.embedding for item in response.data)
    return np.asarray(vectors, dtype=np.float32)


def embed_sentence_transformers(texts: list[str], model: str) -> np.ndarray:
    """Embed texts locally with sentence-transformers.

    Any non-OpenAI model name (e.g. "nomic-embed-text") maps to the
    nomic-ai/nomic-embed-text-v1.5 checkpoint.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error(
            "sentence_transformers_not_installed",
            hint="pip install sentence-transformers",
        )
        sys.exit(1)
    log.info("loading_local_embedding_model", requested=model)
    encoder = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
    )
    return np.asarray(
        encoder.encode(texts, show_progress_bar=False), dtype=np.float32
    )


def embed_texts(texts: list[str], model: str) -> np.ndarray:
    """Dispatch embedding to OpenAI or a local sentence-transformers model."""
    if model == "text-embedding-3-small":
        return embed_openai(texts, model)
    return embed_sentence_transformers(texts, model)


def build_faiss_index(vectors: np.ndarray, chunks: list[dict], index_dir: Path) -> None:
    """Build an inner-product FAISS index over L2-normalized vectors.

    With normalized vectors, inner product == cosine similarity.
    Also writes metadata.jsonl mapping vector id (line number) -> chunk.
    """
    try:
        import faiss
    except ImportError:
        log.error("faiss_not_installed", hint="pip install faiss-cpu")
        sys.exit(1)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(index_dir / "index.faiss"))
    with (index_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for vector_id, chunk in enumerate(chunks):
            f.write(json.dumps({"id": vector_id, **chunk}, ensure_ascii=False) + "\n")


def build_chroma_index(vectors: np.ndarray, chunks: list[dict], index_dir: Path) -> None:
    """Build a persistent Chroma collection under data/index/chroma/."""
    try:
        import chromadb
    except ImportError:
        log.error("chromadb_not_installed", hint="pip install chromadb")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(index_dir / "chroma"))
    collection = client.get_or_create_collection("rag_chunks")
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=vectors.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source_file": c["source_file"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ],
    )


def write_manifest(
    index_dir: Path, embedding_model: str, index_type: str, embedding_dim: int, n_vectors: int
) -> None:
    """Write manifest.json so downstream stages know how to query the index."""
    manifest = {
        "embedding_model": embedding_model,
        "index_type": index_type,
        "embedding_dim": embedding_dim,
        "n_vectors": n_vectors,
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def dir_size_mb(path: Path) -> float:
    """Total size of all files under a directory, in megabytes."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def main() -> None:
    """Run the embed stage."""
    params = load_params()
    embedding_model: str = params["embedding_model"]
    index_type: str = params["index_type"]

    if not CHUNKS_PATH.is_file():
        log.error("chunks_missing", expected=str(CHUNKS_PATH), hint="Run the ingest stage first.")
        sys.exit(1)
    chunks = load_chunks(CHUNKS_PATH)
    log.info("embedding_chunks", n_chunks=len(chunks), model=embedding_model)

    start = time.perf_counter()
    vectors = embed_texts([c["text"] for c in chunks], embedding_model)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if index_type == "faiss":
        build_faiss_index(vectors, chunks, INDEX_DIR)
    elif index_type == "chroma":
        build_chroma_index(vectors, chunks, INDEX_DIR)
    else:
        log.error("unknown_index_type", index_type=index_type, supported=["faiss", "chroma"])
        sys.exit(1)

    write_manifest(INDEX_DIR, embedding_model, index_type, vectors.shape[1], len(chunks))
    log.info(
        "embed_complete",
        n_vectors=len(chunks),
        embedding_dim=int(vectors.shape[1]),
        index_size_mb=round(dir_size_mb(INDEX_DIR), 2),
        time_taken_s=round(time.perf_counter() - start, 2),
    )


if __name__ == "__main__":
    main()
