"""Ingest stage: chunk raw documents from the knowledge base.

Classical ML equivalent: feature engineering.

Reads every .txt and .pdf file in data/knowledge_base/, splits them into
overlapping chunks, and writes data/chunks/chunks.jsonl (one JSON object
per line). Run from this directory (DVC does this for you via `dvc repro`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger()

# All paths are relative to this folder — DVC runs each stage with
# CWD = the directory containing dvc_rag.yaml.
PARAMS_PATH = Path("params.yaml")
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
OUTPUT_PATH = Path("data/chunks/chunks.jsonl")


def load_params() -> dict:
    """Load pipeline parameters from params.yaml."""
    with PARAMS_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_txt(path: Path) -> str:
    """Read a plain-text document."""
    return path.read_text(encoding="utf-8")


def read_pdf(path: Path) -> str | None:
    """Extract text from a PDF using pypdf; return None if pypdf is missing."""
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf_not_installed", hint="pip install pypdf", skipped=str(path))
        return None
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(kb_dir: Path) -> dict[str, str]:
    """Read all .txt and .pdf files from the knowledge base directory.

    Returns a mapping of file name -> full document text.
    """
    documents: dict[str, str] = {}
    for path in sorted(kb_dir.glob("*")):
        if path.suffix.lower() == ".txt":
            documents[path.name] = read_txt(path)
        elif path.suffix.lower() == ".pdf":
            text = read_pdf(path)
            if text is not None:
                documents[path.name] = text
        else:
            log.debug("skipping_unsupported_file", file=str(path))
    return documents


def simple_recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Minimal fallback splitter used when LangChain is not installed.

    Splits on paragraph/sentence boundaries where possible, then falls back
    to a hard character window with overlap.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]
        # Prefer to cut at the last paragraph, then sentence, then space.
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > chunk_size // 2:
                    end = start + cut + len(sep)
                    window = text[start:end]
                    break
        chunk = window.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one document into chunks, preferring LangChain's splitter."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            log.warning(
                "langchain_not_installed",
                hint="pip install langchain-text-splitters",
                fallback="simple built-in recursive splitter",
            )
            return simple_recursive_split(text, chunk_size, chunk_overlap)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def chunk_documents(
    documents: dict[str, str], chunk_size: int, chunk_overlap: int
) -> list[dict]:
    """Chunk every document and attach metadata to each chunk."""
    records: list[dict] = []
    for source_file, text in documents.items():
        for chunk_index, chunk in enumerate(split_text(text, chunk_size, chunk_overlap)):
            records.append(
                {
                    "text": chunk,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "char_count": len(chunk),
                }
            )
    return records


def write_chunks(records: list[dict], output_path: Path) -> None:
    """Write chunk records as JSONL (one object per line, UTF-8)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the ingest stage."""
    params = load_params()
    chunk_size: int = params["chunk_size"]
    chunk_overlap: int = params["chunk_overlap"]

    if not KNOWLEDGE_BASE_DIR.is_dir():
        log.error(
            "knowledge_base_missing",
            expected_dir=str(KNOWLEDGE_BASE_DIR),
            hint="Create the directory and add .txt or .pdf documents to it.",
        )
        sys.exit(1)

    documents = load_documents(KNOWLEDGE_BASE_DIR)
    if not documents:
        log.error(
            "knowledge_base_empty",
            expected_dir=str(KNOWLEDGE_BASE_DIR),
            hint="Add .txt or .pdf documents so the pipeline has something to index.",
        )
        sys.exit(1)

    records = chunk_documents(documents, chunk_size, chunk_overlap)
    write_chunks(records, OUTPUT_PATH)

    mean_chunk_size = (
        sum(r["char_count"] for r in records) / len(records) if records else 0.0
    )
    log.info(
        "ingest_complete",
        n_documents=len(documents),
        n_chunks=len(records),
        mean_chunk_size=round(mean_chunk_size, 1),
        output=str(OUTPUT_PATH),
    )


if __name__ == "__main__":
    main()
