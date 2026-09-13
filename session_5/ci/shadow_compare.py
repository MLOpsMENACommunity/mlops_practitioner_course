"""Shadow deployment check: run the INT8 candidate next to the FP32 model on the same frames.

Before cutover, both pipelines see identical traffic; only FP32 answers the caller.
This measures how often they disagree — per condition, since INT8 damage is rarely
uniform — and writes it in Prometheus text format, so the Session 4 stack
(node-exporter textfile collector -> Prometheus -> Grafana) can alert on it.

    python -m ci.shadow_compare --fp32 detector_student.onnx,ocr_student.onnx \
        --candidate detector_student_int8.xml,ocr_student_int8.xml --candidate-backend openvino
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from src import config
from src.backends import REGISTRY
from src.datasets.splits import load_split
from src.metrics import iou
from src.pipeline import Pipeline


# --- snippet:shadow-compare ---
def disagreement(fp32: Pipeline, candidate: Pipeline, n_frames: int) -> dict[str, float]:
    """Fraction of plates the FP32 pipeline reads that the candidate reads differently, per condition."""
    seen, differ = defaultdict(int), defaultdict(int)
    for sample in load_split("val")[:n_frames]:
        jpeg = sample.path.read_bytes()
        ref, new = fp32.run([jpeg])[0], candidate.run([jpeg])[0]
        for plate in ref:
            seen[sample.condition] += 1
            match = [p for p in new if iou(plate.box[None, :4], p.box[None, :4])[0, 0] > 0.5]
            differ[sample.condition] += int(not match or match[0].text != plate.text)
    return {c: round(differ[c] / seen[c], 4) for c in seen}


# --- end-snippet ---


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Defaults are a matched pair: the s06a INT8 files were quantized FROM the baseline ONNX files.
    # Shadowing an INT8 student against an FP32 baseline would measure distillation and quantization at once.
    parser.add_argument("--fp32", default="detector_baseline.onnx,ocr_baseline.onnx")
    parser.add_argument("--fp32-backend", default="ort")
    parser.add_argument("--candidate", default="detector_int8_stratified.onnx,ocr_int8_stratified.onnx")
    parser.add_argument("--candidate-backend", default="ort")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--max-disagreement", type=float, default=0.05)
    args = parser.parse_args()
    fp32 = Pipeline(REGISTRY[args.fp32_backend](*args.fp32.split(",")))
    cand = Pipeline(REGISTRY[args.candidate_backend](*args.candidate.split(",")))
    rates = disagreement(fp32, cand, args.frames)
    out = config.RESULTS_DIR / "shadow" / f"shadow_{config.profile().name}.prom"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# HELP anpr_shadow_plate_disagreement_ratio Plates read differently by the candidate than by FP32",
             "# TYPE anpr_shadow_plate_disagreement_ratio gauge"]  # fmt: skip
    lines += [f'anpr_shadow_plate_disagreement_ratio{{condition="{c}",candidate="{args.candidate}"}} {r}' for c, r in rates.items()]
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))
    worst = max(rates.values(), default=0.0)
    raise SystemExit(0 if worst <= args.max_disagreement else f"shadow: worst condition disagreement {worst} > {args.max_disagreement}")


if __name__ == "__main__":
    main()
