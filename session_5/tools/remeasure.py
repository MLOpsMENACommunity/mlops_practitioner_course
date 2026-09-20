"""Re-measure rows that were recorded on a busy machine, without retraining anything.

    python -m tools.remeasure --noisy          # list the rows whose measurement ran under load
    python -m tools.remeasure --noisy --run    # measure those rows again and replace them
    python -m tools.remeasure --row s06a_ptq:ort-static-int8-stratified --run

A row keeps everything needed to measure it again: backend, options, pipeline, device. What it
cannot keep is a quiet machine. `load_avg_1m` in the row's environment says what else was running,
and a latency measured next to a browser and a virus scanner is that machine's latency, not the
runtime's. This re-runs the same spec and overwrites the row for THIS hardware and profile only.

Rows whose artifacts have been deleted since (a stage's intermediate exports) are skipped with a
message naming the stage to run instead.
"""

from __future__ import annotations

import argparse

from src import config, environment, results
from src.benchmark import RunSpec, run


def load_of(row: dict) -> float:
    """Peak load while the row was measured; rows recorded before that was sampled fall back to the after-the-fact average."""
    return row.get("load_peak", row["environment"]["load_avg_1m"])


def noisy(rows: list[dict], cores: int) -> list[dict]:
    """Rows measured while more than the benchmark's own threads plus half the cores were runnable."""
    return [r for r in rows if r["status"] == "ok" and load_of(r) - config.THREADS > cores / 2]


def spec_of(row: dict) -> RunSpec:
    return RunSpec(row["stage"], row["variant"], row["parent"], row["backend"], row["options"],
                   fast_resize=row["fast_resize"], pipeline=row.get("pipeline", "cpu"), branch=row["branch"],
                   precision=row["precision"], device=row["device"], concurrent=row.get("concurrent", False),
                   parity=row.get("parity"), notes=row.get("notes", ""))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--row", action="append", default=[], help="stage:variant, repeatable")
    parser.add_argument("--noisy", action="store_true", help="every row of this hardware+profile measured under load")
    parser.add_argument("--all", action="store_true", help="every measured row of this hardware+profile")
    parser.add_argument("--run", action="store_true", help="measure them again (without this, only lists)")
    args = parser.parse_args()

    env = environment.capture()
    hw, profile = environment.hardware_id(env), config.profile().name
    mine = [r for r in results.load() if r["hardware_id"] == hw and r["profile"] == profile]
    chosen = [r for r in mine if r["id"] in args.row] if args.row else []
    if args.all:
        chosen += [r for r in mine if r["status"] == "ok" and r not in chosen]
    if args.noisy:
        chosen += [r for r in noisy(mine, env["cpu_cores_logical"] or 1) if r not in chosen]
    if not chosen:
        print("nothing to re-measure")
        return

    print(f"  {len(chosen)} row(s) on hardware {hw}, profile {profile}; this machine's load average is now {env['load_avg_1m']}")
    for row in chosen:
        print(f"    {row['id']:<48} load peaked at {load_of(row):<7} p95 {row['latency']['total_ms']['p95']} ms")
    if not args.run:
        print("  (re-run with --run once the machine is quiet)")
        return
    for row in chosen:
        run(spec_of(row))


if __name__ == "__main__":
    main()
