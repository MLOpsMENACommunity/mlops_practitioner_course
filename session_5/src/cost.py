"""Cost per million inferences, from measured throughput. The formula, never a figure.

    cost_per_million = usd_per_hour / (usable_fps * 3600) * 1_000_000

"usable" is the whole point: throughput only counts at a batch size whose p95 still
meets the SLA. A batch that triples fps but blows the latency budget is capacity
you cannot use.

    python -m src.cost --fps <measured fps> --p95-ms <measured p95> --usd-per-hour <your price> --batch 4
    python -m src.cost --row s09_openvino:edge-candidate --usd-per-hour <price> --cameras 200 --camera-fps 5
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

from src import config


@dataclass(frozen=True)
class CostInputs:
    rps: float  # frames/sec one instance sustains at this batch size
    p95_ms: float  # per-frame p95 at that operating point
    usd_per_hour: float  # instance price, or amortized hardware + power for an edge box
    batch_size: int
    sla_ms: float = config.SLA_TARGET.p95_ms


# --- snippet:cost-per-million ---
def cost_per_million(i: CostInputs) -> float | None:
    """USD per 1M frames, or None if this operating point violates the SLA."""
    if i.p95_ms > i.sla_ms or i.rps <= 0:
        return None
    return i.usd_per_hour / (i.rps * 3600) * 1_000_000


# --- end-snippet ---


def usable_point(curve: list[dict], sla_ms: float = config.SLA_TARGET.p95_ms) -> dict | None:
    """Highest-throughput point on a measured curve whose p95 meets the SLA."""
    ok = [p for p in curve if p["p95_ms"] <= sla_ms]
    return max(ok, key=lambda p: p["fps"]) if ok else None


def fleet(cameras: int, camera_fps: float, capacity_fps: float, usd_per_hour: float) -> dict:
    """How many instances a fleet of cameras needs, and what that costs per month."""
    load = cameras * camera_fps
    instances = math.ceil(load / capacity_fps)
    return {"load_fps": load, "instances": instances, "usd_per_month": round(instances * usd_per_hour * 730, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost per million inferences from measured throughput.")
    parser.add_argument("--row", help="stage:variant id in results.json (uses its throughput curve)")
    parser.add_argument("--fps", type=float)
    parser.add_argument("--p95-ms", type=float)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--usd-per-hour", type=float, required=True)
    parser.add_argument("--sla-ms", type=float, default=config.SLA_TARGET.p95_ms)
    parser.add_argument("--cameras", type=int)
    parser.add_argument("--camera-fps", type=float, default=5.0)
    args = parser.parse_args()

    if args.row:
        from src import environment, results

        hw = environment.hardware_id(environment.capture())  # THIS machine's row: capacity is hardware-specific
        rows = [r for r in results.load() if r["id"] == args.row and r.get("status") == "ok"
                and r["hardware_id"] == hw and r["profile"] == config.profile().name]  # fmt: skip
        if not rows:
            raise SystemExit(f"no measured row {args.row!r} for this machine ({hw}) and profile in results.json")
        point = usable_point(rows[-1]["throughput"]["curve"], args.sla_ms)
        if point is None:
            raise SystemExit(f"{args.row}: no batch size meets p95 <= {args.sla_ms} ms")
        args.fps, args.p95_ms, args.batch = point["fps"], point["p95_ms"], point["batch"]
    inputs = CostInputs(args.fps, args.p95_ms, args.usd_per_hour, args.batch, args.sla_ms)
    cpm = cost_per_million(inputs)
    print(f"  {inputs}")
    print(f"  cost per 1M frames: {'violates SLA' if cpm is None else f'${cpm:.4f}'}")
    if args.cameras and cpm is not None:
        print(f"  fleet: {fleet(args.cameras, args.camera_fps, args.fps, args.usd_per_hour)}")


if __name__ == "__main__":
    main()
