"""Pre-flight check. Run this BEFORE the session, not during it.

    python -m tools.doctor

Every line is a thing that has broken a live class at least once: a missing
artifact, a port someone else took, a scrape target that is up but returning the
wrong metric names, an incident left injected from a rehearsal. Each failure
prints the command that fixes it, so nobody has to read this file.

Exits non-zero if anything required is broken, so CI can run it too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

from incidents import state
from jobs import metrics_store
from services.ride_model import ARTIFACTS, DATA

MODEL_API = os.getenv("MODEL_API_URL", "http://localhost:8001")
LANGFUSE = os.getenv("LANGFUSE_BASE_URL", "http://localhost:3001")
OLLAMA = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
PROM = f"http://localhost:{os.getenv('PROMETHEUS_PORT', '9095')}"
GRAFANA = f"http://localhost:{os.getenv('GRAFANA_PORT', '3000')}"

#: Names the dashboard and the alert rules query. A target that is UP but
#: exporting different names is the failure that looks like everything is fine.
REQUIRED_METRICS = (
    "model_prediction_duration_min",
    "api_request_latency_seconds",
    "feedback_total",
    "model_version_info",
    # The four rows of the dashboard. A long-running `make api` started before
    # these were added answers /health perfectly and scrapes green while half
    # the dashboard sits empty — which is why they are checked by NAME here
    # rather than trusted because the target is up.
    "api_requests_total",
    "api_process_cpu_percent",
    "api_validation_failures_total",
    "deploy_events_total",
)

_results: list[tuple[bool | None, str, str]] = []


def check(ok: bool, label: str, fix: str = "") -> bool:
    """Record one check; `fix` is printed only on failure."""
    _results.append((ok, label, fix))
    return ok


def warn(label: str, fix: str = "") -> None:
    """Record a non-fatal gap.

    Printed in the checklist as WARN so it cannot scroll past unnoticed, but
    excluded from the pass/fail tally and the exit code — some parts of the
    lab are genuinely optional and a red line there would be noise.
    """
    _results.append((None, label, fix))


def _get(url: str, timeout: float = 3.0) -> httpx.Response | None:
    try:
        return httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return None


def check_artifacts() -> None:
    """The frozen model, both scalers, and the frozen drift reference."""
    for name in ("rf_model.pkl", "scaler.pkl", "scaler_stale.pkl"):
        check((ARTIFACTS / name).exists(), f"artifact {name}", "run: make bootstrap")
    check(
        (DATA / "reference.parquet").exists(),
        "frozen drift reference",
        "run: make bootstrap  (drift against a MOVING baseline detects nothing)",
    )


def check_model_api() -> None:
    """The API is up, serving /metrics directly, and exporting the right names."""
    health = _get(f"{MODEL_API}/health")
    if not check(
        health is not None and health.status_code == 200,
        f"model API {MODEL_API}",
        "run: make api   (in its own terminal)",
    ):
        return
    resp = _get(f"{MODEL_API}/metrics")
    body = resp.text if resp else ""
    check(
        resp is not None and resp.status_code == 200,
        "GET /metrics returns 200 without a redirect",
        "a 307 here means /metrics is mounted, not routed",
    )
    for metric in REQUIRED_METRICS:
        check(
            metric in body,
            f"exports {metric}",
            "the dashboard queries this name verbatim — if the name is in the source "
            "but missing here, the running process predates it: restart make api",
        )


def check_prometheus() -> None:
    """Prometheus is up on the port .env.example says, and the target is healthy."""
    resp = _get(f"{PROM}/-/healthy")
    if not check(resp is not None, f"Prometheus {PROM}", "run: make up"):
        return
    targets = _get(f"{PROM}/api/v1/targets")
    if targets is None:
        return
    pools = targets.json().get("data", {}).get("activeTargets", [])
    node = [t for t in pools if t.get("labels", {}).get("job") == "node"]
    check(
        bool(node) and node[0].get("health") == "up",
        "node-exporter scrape target is up",
        "run: docker compose up -d node-exporter   "
        "(without it the Resources row loses its host CPU and memory panels)",
    )
    model = [t for t in pools if t.get("labels", {}).get("job") == "model-api"]
    check(bool(model), "Prometheus knows the model-api job", "check prometheus.yml scrape_configs")
    if model:
        health = model[0].get("health")
        check(
            health == "up",
            f"model-api scrape target is up (is: {health})",
            "on Linux, host.docker.internal needs extra_hosts — already set in compose. "
            "If it says 'sample limit exceeded', an incident is still injected: make heal",
        )


def check_grafana() -> None:
    """Grafana is up and its provisioned Prometheus datasource resolves."""
    resp = _get(f"{GRAFANA}/api/health")
    check(resp is not None, f"Grafana {GRAFANA}", "run: make up")


def check_clean_state() -> None:
    """No incident left injected from a rehearsal — the classic pre-class trap."""
    active = state.read_state().get("active", [])
    check(not active, f"no incident injected (active: {active or 'none'})", "run: make heal")


def check_store() -> None:
    """The metrics store opens and has been seeded."""
    try:
        with metrics_store.connect() as conn:
            rows = conn.execute("SELECT COUNT(*) AS n FROM drift_metrics").fetchone()["n"]
    except Exception as exc:  # noqa: BLE001 — any failure here is a hard fail
        check(False, f"metrics store: {exc}", "delete metrics_store.db and re-run: make seed")
        return
    check(True, f"metrics store opens ({Path(metrics_store.DB_PATH).name})")
    check(
        rows > 0,
        f"drift history present ({rows} rows)",
        "run: make seed   (every trend panel is empty without it)",
    )


def check_llm_stack() -> None:
    """The Langfuse half. Skipped entirely when the llm profile is not running,
    because labs 1 and 2 do not need it and a red line here would be noise."""
    resp = _get(f"{LANGFUSE}/api/public/health")
    if resp is None:
        print("  (langfuse profile not running — skipping LLM checks)")
        return
    check(resp.status_code == 200, f"Langfuse {LANGFUSE}", "run: make up-langfuse")

    tags = _get(f"{OLLAMA}/api/tags")
    if not check(
        tags is not None,
        f"Ollama {OLLAMA}",
        "run: ollama serve   (OLLAMA_HOST=0.0.0.0 if containers must reach it)",
    ):
        return
    names = {m["name"] for m in tags.json().get("models", [])}
    check(
        OLLAMA_MODEL in names,
        f"Ollama has {OLLAMA_MODEL}",
        f"run: ollama pull {OLLAMA_MODEL}   (the tool-calling demo needs a tool-trained model)",
    )


def check_one_environment() -> None:
    """Both halves of the lab should live in ONE interpreter.

    The metrics half needs numpy/sklearn/evidently (Python <3.13); the LLM half
    needs langfuse and ollama. Split them across two venvs and `make seed --llm`
    dies on an import, which is a confusing way to find out you have an
    environment problem. `pip install -e ".[metrics,llm]"` is the fix.

    Labs 1 and 2 need only the metrics half, so a missing LLM half is reported
    but not failed.
    """

    def importable(name: str) -> bool:
        try:
            __import__(name)
            return True
        except ImportError:
            return False

    check(
        importable("numpy") and importable("sklearn") and importable("evidently"),
        "metrics deps importable (numpy, sklearn, evidently)",
        'pip install -e ".[metrics]"',
    )
    if not importable("langfuse"):
        warn(
            "langfuse SDK NOT in this interpreter (labs 3 and 4 cannot trace)",
            'pip install -e ".[llm]"   (the server being up says nothing about the client)',
        )


def main() -> None:
    """Run every check and print the checklist."""
    check_artifacts()
    check_model_api()
    check_prometheus()
    check_grafana()
    check_store()
    check_clean_state()
    check_one_environment()
    check_llm_stack()

    print("\n  Session 4 pre-flight\n")
    failed = warned = 0
    for ok, label, fix in _results:
        print(f"    {'WARN' if ok is None else 'PASS' if ok else 'FAIL'}  {label}")
        if ok is None:
            warned += 1
        elif not ok:
            failed += 1
        if not ok and fix:
            print(f"          -> {fix}")
    total = len(_results) - warned
    tail = f"  ({warned} warning{'s' if warned != 1 else ''})" if warned else ""
    print(f"\n  {total - failed}/{total} checks passed{tail}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
