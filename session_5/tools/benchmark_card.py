"""One benchmark card per measured artifact, rendered from results.json.

A model version in the registry is not one file. It is an ONNX graph, a TensorRT
engine, a TFLite file — each with its own accuracy, latency and memory on its own
hardware. The card travels with the artifact so nobody deploys `model_int8.tflite`
on the strength of the FP32 model's numbers.

    python -m tools.benchmark_card            # -> results/cards/<hardware>_<profile>/<stage>__<variant>.md
"""

from __future__ import annotations

import json
from string import Template

from src import config, results

TEMPLATE = config.ROOT / "ci" / "benchmark_card.md.tmpl"


def card(row: dict) -> str:
    acc, lat = row["accuracy"], row["latency"]
    per_condition = "\n".join(
        f"| {c} | {v['map50']} | {v['ocr_exact_match']} | {v['n_plates']} |" for c, v in acc["per_condition"].items()
    )
    curve = "\n".join(f"| {p['batch']} | {p['fps']} | {p['p50_ms']} | {p['p95_ms']} |" for p in (row.get("throughput") or {}).get("curve", []))
    phases = "\n".join(f"| {p} | {s['p50']} | {s['p95']} |" for p, s in lat["phases_ms"].items())
    sla_ok = lat["total_ms"]["p95"] <= config.SLA_TARGET.p95_ms
    return Template(TEMPLATE.read_text()).substitute(
        id=row["id"], parent=row["parent"] or "(root)", backend=row["backend"], device=row["device"],
        precision=row["precision"], map50=acc["map50"], ocr_em=acc["ocr_exact_match"], per_condition=per_condition,
        p50=lat["total_ms"]["p50"], p95=lat["total_ms"]["p95"], p99=lat["total_ms"]["p99"], cold=lat["cold_first_call_ms"],
        first=row.get("process_first_call_ms", "not recorded"),
        batch=lat["batch_size"], runs=lat["n_runs"], warmup=lat["warmup_runs"], phases=phases, curve=curve or "| — | — | — | — |",
        size=json.dumps(row["size_mb"]), rss=row["peak_memory_mb"]["rss"], vram=row["peak_memory_mb"]["vram"],
        sla=f"{'meets' if sla_ok else 'MISSES'} p95 <= {config.SLA_TARGET.p95_ms} ms", parity=json.dumps(row.get("parity")),
        environment=json.dumps(row["environment"], indent=1), git_sha=row["git_sha"], timestamp=row["timestamp"],
        profile=row["profile"], val_sha=row.get("val_sha256", ""), notes=row.get("notes", ""),
    )  # fmt: skip


def main() -> None:
    written = 0
    for row in results.load():
        if row["status"] != "ok":
            continue
        folder = config.RESULTS_DIR / "cards" / f"{row['hardware_id']}_{row['profile']}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{row['stage']}__{row['variant']}.md").write_text(card(row))
        written += 1
    print(f"wrote {written} benchmark cards under results/cards/")


if __name__ == "__main__":
    main()
