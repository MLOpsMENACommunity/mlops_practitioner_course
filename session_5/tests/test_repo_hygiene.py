"""Rules the brief sets for the material itself, enforced mechanically.

1. results.json rows carry lineage and environment, and measured rows carry all five metrics.
2. No hand-typed latency outside a generated results block in README.md or guides/.
3. Every `snippet:<name>` a guide cites exists in the code, so slides never cite a dead anchor.
4. No top-level `triton/` folder (it shadows the `triton` package torch imports).
"""

from __future__ import annotations

import json
import re

import pytest

from src import config

DOCS = [p for p in (config.ROOT / "README.md", *sorted((config.ROOT / "guides").glob("*.md"))) if p.exists()]
RESULT_BLOCK = re.compile(r"<!-- results:[^ ]+ -->.*?<!-- /results -->", re.S)
CODE_BLOCK = re.compile(r"```.*?```", re.S)
ALLOWED_MS = {"30 ms", "30ms", "2 ms", "1 ms", "5 ms", "8 ms"}  # the SLA budget and config values the text explains


def rows() -> list[dict]:
    if not config.RESULTS_JSON.exists():
        pytest.skip("no results yet")
    return json.loads(config.RESULTS_JSON.read_text())


def test_every_row_has_lineage_and_environment() -> None:
    for row in rows():
        assert "parent" in row, row["id"]
        assert row["status"] in {"ok", "not_run", "failed"}, row["id"]
        threads = row["environment"]["threads"]
        assert {"ANPR_THREADS", "OMP_NUM_THREADS", "ort_intra_op_num_threads"} <= threads.keys(), row["id"]


def test_measured_rows_have_all_five_metrics_and_skipped_rows_say_why() -> None:
    for row in rows():
        if row["status"] == "ok":
            assert {"map50", "ocr_exact_match"} <= row["accuracy"].keys()
            assert {"p50", "p95"} <= row["latency"]["total_ms"].keys()
            assert row["size_mb"]["total"] >= 0 and row["peak_memory_mb"]["rss"] > 0
            assert row["throughput"] is None or row["throughput"]["curve"]
        else:
            assert row.get("reason"), f"{row['id']} is {row['status']} without a reason"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_no_hand_typed_latency_in_prose(doc) -> None:
    prose = CODE_BLOCK.sub("", RESULT_BLOCK.sub("", doc.read_text()))
    for line in prose.splitlines():
        if "http" in line:  # a cited external figure carries its source link on the same line
            continue
        for match in re.finditer(r"\b\d+(?:\.\d+)?\s?ms\b", line):
            assert match.group(0) in ALLOWED_MS, f"{doc.name}: hand-written measurement {match.group(0)!r} in: {line.strip()}"


def test_cited_snippets_exist() -> None:
    code = "\n".join(p.read_text() for p in (config.ROOT / "src").rglob("*.py"))
    code += "\n".join(p.read_text() for p in (config.ROOT / "ci").rglob("*.py"))
    defined = set(re.findall(r"# --- snippet:([\w-]+) ---", code))
    for doc in DOCS:
        for name in re.findall(r"snippet:([\w-]+)", doc.read_text()):
            assert name in defined, f"{doc.name} cites snippet:{name}, which no source file defines"


def test_no_top_level_triton_folder() -> None:
    assert not (config.ROOT / "triton").exists()
