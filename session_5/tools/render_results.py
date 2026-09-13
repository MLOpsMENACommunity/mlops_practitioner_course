"""Fill the generated result blocks in README.md and guides/*.md from results.json.

A guide never contains a hand-typed measurement. It contains a marker:

    <!-- results:stage:s05_pruning -->
    ...everything between the markers is regenerated...
    <!-- /results -->

Markers:  results:journey      the full table, one per machine
          results:stage:<s>    only the rows of one stage (plus their parents)
          results:gate         the s00 profile gate report(s)
          results:int8-debug   the quant_debug report(s)

Re-running on an RTX 3090 re-renders every guide with the 3090's numbers alongside
this machine's — no prose has to be edited, because no prose holds a number.
"""

from __future__ import annotations

import re

from src import config
from tools import journey_table

BLOCK = re.compile(r"(<!-- results:(?P<key>[^ ]+) -->)(?P<body>.*?)(<!-- /results -->)", re.S)


def stage_block(stage: str) -> str:
    parts = []
    for (hw, prof), rows in sorted(journey_table.groups().items()):
        mine = [r for r in rows if r["stage"] == stage]
        if not mine:
            continue
        parents = {r["parent"] for r in mine}
        context = [r for r in rows if r["id"] in parents and r["stage"] != stage]
        parts += [f"_Hardware `{hw}` · profile `{prof}`_", "", "\n".join(journey_table.header(rows)), "",
                  "\n".join(journey_table.table(context + mine)), ""]  # fmt: skip
    return "\n".join(parts) if parts else "_No measured rows yet — run the stage's make target._"


def files_block(pattern: str, empty: str) -> str:
    found = sorted(config.RESULTS_DIR.glob(pattern))
    return "\n\n".join(p.read_text().strip() for p in found) if found else empty


def verdict_block() -> str:
    """The session's closing question, answered from data: did any measured row meet the SLA, and where?"""
    sla = config.SLA_TARGET
    lines = ["| hardware · profile | fastest measured row | p95 ms | meets p95 <= SLA? | accuracy within SLA drop of the baseline? | measured on the edge target? |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for (hw, prof), rows in sorted(journey_table.groups().items()):
        ok = [r for r in rows if r["status"] == "ok"]
        base = next((r for r in ok if r["id"] == "s01_baseline:eager-fp32"), None)
        if not ok or base is None:
            continue
        env = ok[0]["environment"]

        def within(r: dict, base: dict = base) -> bool:
            return (base["accuracy"]["map50"] - r["accuracy"]["map50"] <= sla.max_map50_drop
                    and base["accuracy"]["ocr_exact_match"] - r["accuracy"]["ocr_exact_match"] <= sla.max_ocr_em_drop)  # fmt: skip

        candidates = [r for r in ok if within(r)] or ok
        best = min(candidates, key=lambda r: r["latency"]["total_ms"]["p95"])
        edge = "Raspberry Pi" in env["cpu"] or "tegra" in env["os"].lower() or "Orin" in env.get("gpu", "")
        lines.append(f"| {env['cpu']}{' + ' + env['gpu'] if env.get('gpu') else ''} · `{prof}` | `{best['id']}` | "
                     f"{best['latency']['total_ms']['p95']:.1f} | {'yes' if best['latency']['total_ms']['p95'] <= sla.p95_ms else 'no'} | "
                     f"{'yes' if within(best) else 'no — no row met both'} | {'yes' if edge else 'no — a development machine, not the roadside box'} |")  # fmt: skip
    return "\n".join(lines) if len(lines) > 2 else "_No measured rows yet._"


def render_key(key: str) -> str:
    if key == "verdict":
        return verdict_block()
    if key == "journey":
        return "\n\n".join(f"#### Hardware `{hw}` · profile `{prof}`\n\n{journey_table.render(rows)}"
                           for (hw, prof), rows in sorted(journey_table.groups().items())) or "_No results yet._"  # fmt: skip
    if key.startswith("stage:"):
        return stage_block(key.split(":", 1)[1])
    if key == "gate":
        return files_block("profile/gate_*.md", "_Run `make s00` to produce the gate report._").replace("# Profile gate", "#### Profile gate")
    if key == "int8-debug":
        return files_block("int8_debug/report_*.md", "_Run `make int8-debug`._").replace("# INT8", "#### INT8").replace("\n## ", "\n##### ")
    raise KeyError(f"unknown results block: {key}")


def main() -> None:
    targets = [config.ROOT / "README.md", *sorted((config.ROOT / "guides").glob("*.md"))]
    for path in targets:
        if not path.exists():
            continue
        text = path.read_text()
        new = BLOCK.sub(lambda m: f"{m.group(1)}\n{render_key(m.group('key'))}\n{m.group(4)}", text)
        if new != text:
            path.write_text(new)
            print(f"rendered {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
