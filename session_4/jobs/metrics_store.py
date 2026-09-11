"""The metrics store: drift results, late labels, deploys and retrain decisions.

Why SQLite and not Postgres. ``docker-compose.yaml`` runs Prometheus + Grafana,
and six more containers behind the ``langfuse`` profile. A seventh container
just to hold four small tables would break the stack budget for a lesson that is
about *keeping* the numbers, not about which engine keeps them.

Why keep them at all, when Prometheus is right there: Prometheus is a 15-day
ring buffer of *time series*. "Which features drifted against which frozen
reference, in the window ending Tuesday" is a *fact with provenance*, and the
retrain gate (jobs/retrain_gate.py) has to be able to ask it about last month.
That is a database question. This is the smallest thing that answers it.

Read it from the command line during an incident — RUNBOOK.md step 4 uses these:

    sqlite3 -header -column metrics_store.db \\
      "select ts, component, version from deploys order by ts desc limit 5;"
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("METRICS_STORE_PATH", PROJECT_ROOT / "metrics_store.db"))

SCHEMA = """
-- One row per (window, feature, statistic). `reference_id` names the FROZEN
-- baseline the window was compared against — without it a drift number is
-- uninterpretable, because "drifted" is meaningless without "drifted from what".
CREATE TABLE IF NOT EXISTS drift_metrics (
    window_start  TEXT NOT NULL,
    window_end    TEXT NOT NULL,
    feature       TEXT NOT NULL,
    metric        TEXT NOT NULL,        -- psi | ks | p_value | share_drifted
    value         REAL NOT NULL,
    segment       TEXT NOT NULL DEFAULT 'all',   -- 'all' or a city (incident 05)
    model_version TEXT NOT NULL,
    reference_id  TEXT NOT NULL
);

-- Late labels. y_pred is copied in when the label lands so MAE is one query and
-- not a join against a prediction log that Prometheus already aggregated away.
CREATE TABLE IF NOT EXISTS labels (
    request_id TEXT PRIMARY KEY,
    ts         TEXT NOT NULL,
    city       TEXT NOT NULL DEFAULT 'cairo',
    y_true     REAL NOT NULL,
    y_pred     REAL NOT NULL
);

-- What changed and when. This is the table the Grafana annotation and RUNBOOK
-- step 3 ("deploy timeline") read: most "model" incidents are deploy incidents.
CREATE TABLE IF NOT EXISTS deploys (
    ts        TEXT NOT NULL,
    component TEXT NOT NULL,
    version   TEXT NOT NULL,
    git_sha   TEXT NOT NULL DEFAULT '',
    note      TEXT NOT NULL DEFAULT ''
);

-- Every retrain DECISION, including the refusals. A gate that only logs its
-- yeses cannot be audited after it promotes something it should not have.
CREATE TABLE IF NOT EXISTS retrain_events (
    ts               TEXT NOT NULL,
    trigger          TEXT NOT NULL,
    features_drifted TEXT NOT NULL DEFAULT '',
    decision         TEXT NOT NULL,     -- retrain | hold
    reasons          TEXT NOT NULL DEFAULT '',
    promoted         INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS drift_by_feature ON drift_metrics (feature, window_end);
CREATE INDEX IF NOT EXISTS labels_by_ts     ON labels (ts);
CREATE INDEX IF NOT EXISTS deploys_by_ts    ON deploys (ts);
"""


def connect() -> sqlite3.Connection:
    """Open the store, creating and migrating it on first use."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_deploy(
    component: str, version: str, git_sha: str = "", note: str = "", ts: str | None = None
) -> None:
    """Log a change to `component` — incidents/inject.py calls this on every flip.

    `ts` exists so tools/seed.py can back-date a deploy timeline. A history where
    every release happened at the same instant cannot be correlated against
    anything, which defeats the point of having one.
    """
    with connect() as conn:
        conn.execute(
            "INSERT INTO deploys (ts, component, version, git_sha, note) VALUES (?,?,?,?,?)",
            (ts or _now(), component, version, git_sha, note),
        )


def record_drift(rows: Iterable[Sequence[Any]]) -> int:
    """Insert drift rows: (window_start, window_end, feature, metric, value,
    segment, model_version, reference_id). Returns the row count."""
    rows = list(rows)
    with connect() as conn:
        conn.executemany(
            "INSERT INTO drift_metrics (window_start, window_end, feature, metric,"
            " value, segment, model_version, reference_id) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def record_labels(rows: Iterable[Sequence[Any]]) -> int:
    """Insert late labels: (request_id, ts, city, y_true, y_pred). Idempotent."""
    rows = list(rows)
    with connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO labels (request_id, ts, city, y_true, y_pred)"
            " VALUES (?,?,?,?,?)",
            rows,
        )
    return len(rows)


def record_retrain_event(
    trigger: str,
    decision: str,
    reasons: str,
    features_drifted: str = "",
    promoted: bool = False,
    ts: str | None = None,
) -> None:
    """Log a retrain decision — including a refusal, which is the interesting case."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO retrain_events (ts, trigger, features_drifted, decision,"
            " reasons, promoted) VALUES (?,?,?,?,?,?)",
            (ts or _now(), trigger, features_drifted, decision, reasons, int(promoted)),
        )


def drift_history(
    feature: str, days: int = 14, segment: str = "all", metric: str = "psi"
) -> list[sqlite3.Row]:
    """Return `feature`'s drift rows over the last `days`, oldest first.

    Feeds ``retrain_gate.is_sustained()``: one spike is noise, two days is a trend.

    Filtering on `metric` is not optional. Every window stores both an effect size
    (psi) and a p-value, and they live on completely different scales — a p-value
    of 0.03 means "certainly drifted" while a PSI of 0.03 means "perfectly
    stable". Mix them in one series and every threshold you apply is nonsense.
    """
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM drift_metrics WHERE feature=? AND segment=? AND metric=?"
            " AND window_end>=? ORDER BY window_end",
            (feature, segment, metric, since),
        ).fetchall()


def mae_by_city(hours: int = 24) -> list[sqlite3.Row]:
    """MAE on late labels, split by city — the signal that catches incident 05.

    A global MAE stays flat when one new city is 8% of traffic. Splitting it does not.
    """
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    with connect() as conn:
        return conn.execute(
            "SELECT city, COUNT(*) AS n, AVG(ABS(y_true - y_pred)) AS mae"
            " FROM labels WHERE ts >= ? GROUP BY city ORDER BY mae DESC",
            (since,),
        ).fetchall()


def latest_deploys(limit: int = 10) -> list[sqlite3.Row]:
    """Most recent deploys, newest first — RUNBOOK.md step 3."""
    with connect() as conn:
        return conn.execute("SELECT * FROM deploys ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()


def latest_retrain_events(limit: int = 50) -> list[sqlite3.Row]:
    """Most recent retrain DECISIONS, newest first — holds included.

    The dashboard's retrain-history panel reads this. Holds are in the result on
    purpose: a gate that only shows its yeses looks like it never refused, and
    "why didn't we retrain" is the question the panel exists to answer.
    """
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM retrain_events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
