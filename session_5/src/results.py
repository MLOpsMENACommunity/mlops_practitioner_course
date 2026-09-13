"""results/results.json — written only by code, never by hand.

A row is keyed by (stage:variant, hardware, profile). Re-running a stage replaces
its own row on the same machine and leaves every other machine's rows alone, so
a laptop run and an RTX 3090 run can live in one file without mixing.
"""

from __future__ import annotations

import json
from typing import Any

from src import config


def load() -> list[dict[str, Any]]:
    if not config.RESULTS_JSON.exists():
        return []
    return json.loads(config.RESULTS_JSON.read_text())


def _key(row: dict) -> tuple:
    return row["id"], row["hardware_id"], row["profile"]


def record(row: dict[str, Any]) -> None:
    rows = [r for r in load() if _key(r) != _key(row)]
    rows.append(row)
    rows.sort(key=lambda r: (r["hardware_id"], r["profile"], r["id"]))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON.write_text(json.dumps(rows, indent=1) + "\n")


def find(row_id: str, hardware_id: str, profile: str) -> dict | None:
    for row in load():
        if _key(row) == (row_id, hardware_id, profile):
            return row
    return None
