"""Mechanical acceptance checks for the README exercises.

    python -m tools.check_exercises 3        # check one exercise
    python -m tools.check_exercises all

Each check reads results/results.json (or runs a command) and prints PASS/FAIL with the
evidence. Nobody has to eyeball a table to grade an exercise.
"""

from __future__ import annotations

import os
import subprocess
import sys

from src import config, environment, results


def rows_here() -> list[dict]:
    hw = environment.hardware_id(environment.capture())
    return [r for r in results.load() if r["hardware_id"] == hw and r["profile"] == config.profile().name and r["status"] == "ok"]


def row(row_id: str) -> dict | None:
    return next((r for r in rows_here() if r["id"] == row_id), None)


def ex1() -> tuple[bool, str]:
    """Thread scaling: the ORT CPU row measured at ANPR_THREADS=1 and at 4 (two hardware ids)."""
    found = {r["environment"]["threads"]["ANPR_THREADS"]: r["latency"]["total_ms"]["p95"]
             for r in results.load() if r["id"] == "s04_onnx_export:ort-cpu-fp32" and r["status"] == "ok"}  # fmt: skip
    return {1, 4} <= found.keys(), f"p95 by thread count: {found}"


def ex2() -> tuple[bool, str]:
    """Break calibration on purpose: a night-only calibration row exists and was measured."""
    r = row("s06a_ptq:ort-static-int8-night")
    return r is not None, "row s06a_ptq:ort-static-int8-night " + ("found" if r else "missing — add a 'night' strategy to calibration.frames")


def ex3() -> tuple[bool, str]:
    """Masks vs slices: your 30% sliced ONNX is smaller on disk than the unpruned ONNX."""
    mine, base = row("s05_pruning:sliced-oneshot-30-onnx"), row("s04_onnx_export:ort-cpu-fp32")
    if not (mine and base):
        return False, "need rows s05_pruning:sliced-oneshot-30-onnx and s04_onnx_export:ort-cpu-fp32"
    return mine["size_mb"]["total"] < base["size_mb"]["total"], f"size {mine['size_mb']['total']} MB vs {base['size_mb']['total']} MB"


def ex4() -> tuple[bool, str]:
    """INT8 within budget: mixed precision OR QAT keeps OCR exact match inside the SLA drop."""
    base = row("s04_onnx_export:ort-cpu-fp32")
    allowed = config.SLA_TARGET.max_ocr_em_drop
    for rid in ("s06a_ptq:ort-static-int8-mixed", "s06b_qat:ort-qat-int8-ocr"):
        r = row(rid)
        if r and base and base["accuracy"]["ocr_exact_match"] - r["accuracy"]["ocr_exact_match"] <= allowed:
            return True, f"{rid} is within {allowed} of FP32"
    return False, f"neither INT8 row is within {allowed} exact match of FP32 yet"


def ex5() -> tuple[bool, str]:
    """Watch the CI gate fail: an injected +40 ms regression must make ci.perf_gate exit 1."""
    env = os.environ | {"ANPR_INJECT_LATENCY_MS": "40"}
    cmd = [sys.executable, "-m", "ci.perf_gate", "--candidate", "s07_distillation:student-distilled-onnx", "--remeasure"]
    code = subprocess.run(cmd, env=env, cwd=config.ROOT, capture_output=True).returncode
    return code == 1, f"perf_gate exit code with injection: {code}"


def ex6() -> tuple[bool, str]:
    """Meet the SLA: some row on this machine has p95 <= SLA with accuracy inside the allowed drops."""
    base, sla = row("s01_baseline:eager-fp32"), config.SLA_TARGET
    if not base:
        return False, "no baseline row on this machine"
    good = [r["id"] for r in rows_here() if r["latency"]["total_ms"]["p95"] <= sla.p95_ms
            and base["accuracy"]["map50"] - r["accuracy"]["map50"] <= sla.max_map50_drop
            and base["accuracy"]["ocr_exact_match"] - r["accuracy"]["ocr_exact_match"] <= sla.max_ocr_em_drop]  # fmt: skip
    return bool(good), f"rows meeting the SLA here: {good or 'none'}"


def ex7() -> tuple[bool, str]:
    """Dynamic batching measured: both Triton client-pipeline rows exist with a throughput curve."""
    on, off = row("s11_triton:triton-grpc-client-pipeline"), row("s11_triton:triton-grpc-nobatch")
    ok = bool(on and off and on["throughput"] and off["throughput"])
    return ok, "both rows present" if ok else "run make s11 (Docker) first"


CHECKS = {str(i): f for i, f in enumerate((ex1, ex2, ex3, ex4, ex5, ex6, ex7), start=1)}


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    failed = 0
    for key, fn in CHECKS.items():
        if which not in ("all", key):
            continue
        ok, evidence = fn()
        failed += not ok
        print(f"  exercise {key}: {'PASS' if ok else 'FAIL'}  {fn.__doc__.splitlines()[0]}\n               {evidence}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
