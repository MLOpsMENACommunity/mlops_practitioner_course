"""CI performance gate: fail the build on an SLA breach or a regression.

    python -m ci.perf_gate --candidate s04_onnx_export:ort-cpu-fp32                 # accuracy vs baseline
    python -m ci.perf_gate --candidate s04_onnx_export:ort-cpu-fp32 --remeasure     # + latency vs its own record
    ANPR_INJECT_LATENCY_MS=40 python -m ci.perf_gate --candidate ... --remeasure    # watch it fail

Every threshold comes from src/config.py (SLA_TARGET); the gate never keeps its own copy.
  1. accuracy    candidate mAP@0.5 and exact match may not drop more than the SLA allows vs --baseline
  2. regression  with --remeasure: the candidate is measured again now, and its p95 may not exceed
                 its RECORDED p95 by more than --max-p95-ratio. Comparing a row with its own record
                 on the same machine is what makes this meaningful on any runner: ONNX Runtime vs
                 eager PyTorch differs by hardware, a code change that slows ONNX Runtime does not.
  3. absolute    with --enforce-sla: p95 <= SLA p95 — only meaningful on the target hardware
Exits 1 on any failure and prints which check failed, with both numbers.
"""

from __future__ import annotations

import argparse
import sys

from src import config, environment, results
from src.benchmark import RunSpec, run


def find(row_id: str) -> dict:
    hw = environment.hardware_id(environment.capture())
    row = results.find(row_id, hw, config.profile().name)
    if not row or row["status"] != "ok":
        raise SystemExit(f"gate: no measured row {row_id!r} for this hardware ({hw}) and profile — run its stage first")
    return row


def remeasure(row: dict) -> dict:
    spec = RunSpec(row["stage"], row["variant"], row["parent"], row["backend"], row["options"],
                   fast_resize=row["fast_resize"], pipeline=row.get("pipeline", "cpu"), branch=row["branch"],
                   precision=row["precision"], device=row["device"], throughput=False)  # fmt: skip
    return run(spec, record=False)  # never overwrite the row the stage recorded


# --- snippet:perf-gate ---
def check(candidate: dict, baseline: dict, recorded: dict | None, max_p95_ratio: float, enforce_sla: bool) -> list[str]:
    sla, failures = config.SLA_TARGET, []
    for metric, allowed in (("map50", sla.max_map50_drop), ("ocr_exact_match", sla.max_ocr_em_drop)):
        drop = baseline["accuracy"][metric] - candidate["accuracy"][metric]
        if drop > allowed:
            failures.append(f"{metric} dropped {drop:.4f} vs {baseline['id']} (allowed {allowed})")
    p95 = candidate["latency"]["total_ms"]["p95"]
    if recorded is not None and p95 > recorded["latency"]["total_ms"]["p95"] * max_p95_ratio:
        failures.append(f"p95 {p95:.1f} ms > {max_p95_ratio:.2f} x recorded p95 {recorded['latency']['total_ms']['p95']:.1f} ms")
    if enforce_sla and p95 > sla.p95_ms:
        failures.append(f"p95 {p95:.1f} ms breaks the SLA of {sla.p95_ms} ms")
    return failures


# --- end-snippet ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail on SLA breach, accuracy regression or latency regression.")
    parser.add_argument("--candidate", required=True, help="stage:variant row to gate")
    parser.add_argument("--baseline", default="s01_baseline:eager-fp32", help="accuracy reference row")
    parser.add_argument("--max-p95-ratio", type=float, default=1.15)
    parser.add_argument("--remeasure", action="store_true", help="measure the candidate again and compare with its record")
    parser.add_argument("--enforce-sla", action="store_true", help="also require p95 <= SLA (target hardware only)")
    args = parser.parse_args()

    baseline, recorded = find(args.baseline), find(args.candidate)
    candidate = recorded
    if args.remeasure:
        candidate = remeasure(recorded)
        if candidate["status"] != "ok":
            print(f"gate: re-measurement failed: {candidate.get('reason')}")
            sys.exit(1)
    failures = check(candidate, baseline, recorded if args.remeasure else None, args.max_p95_ratio, args.enforce_sla)
    print(f"gate: {args.candidate} p95 {candidate['latency']['total_ms']['p95']} ms "
          f"(recorded {recorded['latency']['total_ms']['p95']} ms), mAP {candidate['accuracy']['map50']}, "
          f"EM {candidate['accuracy']['ocr_exact_match']} | baseline {args.baseline} mAP {baseline['accuracy']['map50']}, "
          f"EM {baseline['accuracy']['ocr_exact_match']}")  # fmt: skip
    for failure in failures:
        print(f"gate: FAIL  {failure}")
    print("gate: PASS" if not failures else f"gate: {len(failures)} check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
