"""results/results.json -> results/journey_table.md. Generated. Never edit the output by hand.

One table per (hardware, profile): a latency measured on one machine is not comparable
with one measured on another, so they never share a column. Every row names its parent,
because the stages are a tree, not a chain — a TensorRT engine built from the distilled
student does not contain the pruning rows printed above it.

    python -m tools.journey_table
"""

from __future__ import annotations

from collections import defaultdict

from src import config, cost, results

STAGE_ORDER = ["s00_profile", "s01_baseline", "s02_torch_compile", "s03_torchscript", "s04_onnx_export", "s05_pruning",
               "s06a_ptq", "s06b_qat", "s07_distillation", "s08_tensorrt", "s09_openvino", "s10_tflite_edge", "s11_triton"]  # fmt: skip
DASH = "—"


def _num(value, digits: int = 2) -> str:
    return DASH if value is None else f"{value:.{digits}f}"


def cost_cell(row: dict) -> str:
    """USD per 1M frames for an instance costing $1/hour, at the best batch size that meets the SLA."""
    point = cost.usable_point((row.get("throughput") or {}).get("curve", []))
    if point is None:
        return DASH
    cpm = cost.cost_per_million(cost.CostInputs(point["fps"], point["p95_ms"], 1.0, point["batch"]))
    return f"{cpm:.3f} (b{point['batch']})"


def sort_key(row: dict) -> tuple:
    stage = STAGE_ORDER.index(row["stage"]) if row["stage"] in STAGE_ORDER else 99
    return stage, row["variant"]


def header(rows: list[dict]) -> list[str]:
    env = next((r["environment"] for r in rows if r["status"] == "ok"), rows[0]["environment"])
    libs = env.get("libraries", {})
    shown = ", ".join(f"{k} {libs[k]}" for k in ("torch", "onnxruntime", "onnxruntime-gpu", "openvino", "nncf",
                                                  "tensorrt-cu13", "ai-edge-litert", "onnx2tf") if k in libs)  # fmt: skip
    gpu = f"{env['gpu']} (driver {env['driver']}, CUDA {env.get('cuda_runtime', '?')})" if env.get("gpu") else "none"
    threads = env["threads"]
    return [
        f"**Measured on:** {env['cpu']} · {env['ram_gb']} GB RAM · GPU: {gpu} · {env['os']} · Python {env['python']}",
        f"**Threads:** ANPR_THREADS={threads['ANPR_THREADS']}, OMP_NUM_THREADS={threads['OMP_NUM_THREADS']}, "
        f"ORT intra_op_num_threads={threads['ort_intra_op_num_threads']}, inter_op={threads['ort_inter_op_num_threads']}",
        f"**Latency batch size:** {config.SLA_TARGET.batch_size} · **SLA:** p95 <= {config.SLA_TARGET.p95_ms} ms per frame · "
        f"**Profile:** `{rows[0]['profile']}` · **Validation set sha256:** `{(next((r.get('val_sha256') for r in rows if r.get('val_sha256')), '') or '')[:12]}`",
        f"**Libraries (as loaded by the rows below):** {shown}",
    ]  # fmt: skip


def table(rows: list[dict]) -> list[str]:
    lines = ["| row | parent | runtime · device · precision | mAP@0.5 | OCR exact | p50 ms | p95 ms | ≤ SLA | fps (batch) | size MB | peak RSS MB | $/1M frames @ $1/h | status |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in sorted(rows, key=sort_key):
        parent = f"`{r['parent']}`" if r["parent"] else "(root)"
        runtime = f"{r['backend']} · {r['device']} · {r['precision']}" + (" · NMS in graph" if r.get("nms_in_graph") else "")
        if r["status"] != "ok":
            reason = (r.get("reason") or "").splitlines()[0][:90]
            lines.append(f"| `{r['id']}` | {parent} | {runtime} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {DASH} | {r['status']}: {reason} |")
            continue
        acc, lat, tput = r["accuracy"], r["latency"]["total_ms"], r.get("throughput") or {}
        ok = "yes" if lat["p95"] <= config.SLA_TARGET.p95_ms else "no"
        fps = f"{tput['fps']:.1f} (b{tput['saturating_batch']})" if tput else DASH
        lines.append(f"| `{r['id']}` | {parent} | {runtime} | {_num(acc['map50'], 3)} | {_num(acc['ocr_exact_match'], 3)} | "
                     f"{_num(lat['p50'], 1)} | {_num(lat['p95'], 1)} | {ok} | {fps} | {_num(r['size_mb']['total'], 2)} | "
                     f"{_num(r['peak_memory_mb']['rss'], 0)} | {cost_cell(r)} | ok |")  # fmt: skip
    return lines


def lineage(rows: list[dict]) -> list[str]:
    """Mermaid tree of which artifact each row was built from."""
    ids = {r["id"] for r in rows}
    node = {rid: f"n{i}" for i, rid in enumerate(sorted(ids))}
    out = ["```mermaid", "flowchart LR"]
    for rid in sorted(ids):
        out.append(f'  {node[rid]}["{rid}"]')
    for r in rows:
        if r["parent"] in ids:
            out.append(f"  {node[r['parent']]} --> {node[r['id']]}")
    out.append("```")
    return out


def groups() -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in results.load():
        grouped[(row["hardware_id"], row["profile"])].append(row)
    return dict(grouped)


def render(rows: list[dict], with_lineage: bool = True) -> str:
    lines = header(rows) + [""] + table(rows)
    if with_lineage:
        lines += ["", "<details><summary>Lineage: which artifact each row was built from</summary>", ""] + lineage(rows) + ["", "</details>"]
    return "\n".join(lines)


def main() -> None:
    parts = ["# Journey table", "", "_Generated by `python -m tools.journey_table` from `results/results.json`. Do not edit._", "",
             "`$/1M frames @ $1/h` = cost per million frames for an instance priced at $1/hour, at the highest-throughput batch "
             "whose p95 still meets the SLA. Multiply by your real hourly price — see `src/cost.py`.", ""]  # fmt: skip
    for (hw, prof), rows in sorted(groups().items()):
        parts += [f"## Hardware `{hw}` · profile `{prof}`", "", render(rows), ""]
    out = config.RESULTS_DIR / "journey_table.md"
    out.write_text("\n".join(parts) + "\n")
    print(f"wrote {out.relative_to(config.ROOT)} ({sum(len(r) for r in groups().values())} rows)")


if __name__ == "__main__":
    main()
