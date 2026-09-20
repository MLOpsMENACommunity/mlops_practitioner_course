"""When INT8 breaks — the debugging loop, run on our pipeline.

    make int8-debug

1. sensitivity quantize ONE layer group at a time (everything else stays FP32), rank by accuracy lost
2. outliers    per-channel activation ranges over calibration data: one channel far wider
               than its neighbours forces a coarse scale onto all of them
3. mixed       keep the first layer, the output layers and the most sensitive groups in FP32,
               re-quantize with the SAME pre-processed graph and calibration as s06a, measure as a row
4. damage      per-condition accuracy of every s06a INT8 row (mixed included) vs the FP32 ONNX row
Writes results/int8_debug/report_<hardware>_<profile>.md (+ .json). Every number in it is measured.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src import config, environment, results
from src.backends import OrtBackend
from src.benchmark import RunSpec, run
from src.datasets import calibration
from src.datasets.splits import gt_crop, load_split
from src.metrics import FrameResult, summarize
from src.models import io
from src.pipeline import Pipeline
from src.postprocess import ctc_decode
from src.stages import common
from src.stages.s06a_ptq import decode_nodes

OUT_DIR = config.RESULTS_DIR / "int8_debug"
QUANTIZABLE = ("Conv", "MatMul", "Gemm")


def art(name: str):
    return config.artifacts_dir() / name


def pre(onnx_name: str) -> str:
    """The pre-processed graph s06a quantizes — so every row here shares its parent's recipe."""
    out = onnx_name.replace(".onnx", "_pre.onnx")
    if not art(out).exists():
        quant_pre_process(str(art(onnx_name)), str(art(out)), skip_symbolic_shape=True)
    return out


def layer_groups(onnx_name: str, depth: int, with_decode: bool = False) -> dict[str, list[str]]:
    """Quantizable node names grouped by the PyTorch module they came from.

    The dynamo exporter stores each node's module path in `pkg.torch.onnx.name_scopes`,
    which is what turns `node_Conv_497` back into `detector.c3.4` (ResNet layer1).
    `with_decode` adds the detector's decode arithmetic as one more group: it holds no conv,
    so a scan over conv groups alone can never find the damage it does.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for node in onnx.load(str(art(onnx_name))).graph.node:
        if node.op_type not in QUANTIZABLE:
            continue
        raw = next((p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.name_scopes"), None)
        scopes = ast.literal_eval(raw) if raw else [node.name]
        path = [s for s in scopes if s and not s.startswith(("_", "conv2d", "linear"))]
        groups[path[min(depth, len(path) - 1)] if path else node.name].append(node.name)
    if with_decode:
        groups["decode"] = decode_nodes(art(onnx_name))
    return dict(groups)


def quantize(src: str, dst: str, input_name: str, batches: list, include=None, exclude=None) -> None:
    """Identical settings to s06a's stratified row: QDQ, per-channel, S8S8, MinMax."""
    quantize_static(str(art(src)), str(art(dst)), calibration.OrtReader(input_name, batches),
                    quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
                    nodes_to_quantize=include, nodes_to_exclude=exclude)  # fmt: skip


def ocr_accuracy(onnx_name: str) -> dict[str, float]:
    """Recognizer alone on ground-truth val crops: overall and per condition."""
    cache = np.load(config.data_dir() / "val_ocr.npz")
    crops = np.stack([gt_crop(c, b, np.full(4, 0.08)) for c, b in zip(cache["crops"], cache["inner"])])[:, None]
    sess = ort.InferenceSession(str(art(onnx_name)), providers=["CPUExecutionProvider"])
    logits = np.concatenate([sess.run(None, {"crops": crops[i : i + 64]})[0] for i in range(0, len(crops), 64)])
    hit = np.array(ctc_decode(logits)) == cache["texts"]
    return {"all": round(float(hit.mean()), 4)} | {c: round(float(hit[cache["conditions"] == c].mean()), 4) for c in config.CONDITIONS}


SENSITIVITY_FRAMES = 200  # ~300 plates: one plate is ~0.003 exact match, well under the 0.02 budget


def detector_accuracy(onnx_name: str, n_frames: int = SENSITIVITY_FRAMES) -> dict[str, float]:
    """mAP@0.5 AND end-to-end plate exact match, with the FP32 recognizer reading this detector's crops."""
    pipe = Pipeline(OrtBackend(onnx_name, "ocr_baseline.onnx"))
    frames = []
    for s in load_split("val")[:n_frames]:
        plates = pipe.run([s.path.read_bytes()])[0]
        dets = np.array([p.box for p in plates], np.float32).reshape(-1, 5)
        frames.append(FrameResult(s.condition, np.array(s.boxes, np.float32).reshape(-1, 4), s.texts, dets,
                                  [p.text for p in plates]))  # fmt: skip
    acc = summarize(frames)
    return {"map50": acc["map50"], "em": acc["ocr_exact_match"]}


# --- snippet:layer-sensitivity ---
def sensitivity() -> dict:
    """Quantize one group at a time; the accuracy it costs is its sensitivity.

    Detector groups are ranked by end-to-end EXACT MATCH, not mAP: INT8 can move a box a few
    pixels — no change at IoU 0.5 — and the crop cut from it no longer reads. Ranked by mAP,
    the group doing that damage looks harmless.
    """
    ocr_cal, det_cal = calibration.ocr_batches("stratified"), calibration.detector_batches(calibration.frames("stratified", 32))
    ocr_fp32, det_fp32 = ocr_accuracy("ocr_baseline.onnx")["all"], detector_accuracy("detector_baseline.onnx")
    report = {"ocr_fp32_exact_match": ocr_fp32, "detector_fp32": det_fp32 | {"frames": SENSITIVITY_FRAMES},
              "ocr": {}, "detector": {}, "detector_map50_lost": {}}  # fmt: skip
    for group, nodes in layer_groups(pre("ocr_baseline.onnx"), depth=1).items():
        quantize(pre("ocr_baseline.onnx"), "ocr_probe.onnx", "crops", ocr_cal, include=nodes)
        report["ocr"][group] = round(ocr_fp32 - ocr_accuracy("ocr_probe.onnx")["all"], 4)
    for group, nodes in layer_groups(pre("detector_baseline.onnx"), depth=2, with_decode=True).items():
        quantize(pre("detector_baseline.onnx"), "detector_probe.onnx", "images", det_cal, include=nodes)
        probe = detector_accuracy("detector_probe.onnx")
        report["detector"][group] = round(det_fp32["em"] - probe["em"], 4)
        report["detector_map50_lost"][group] = round(det_fp32["map50"] - probe["map50"], 4)
    return report


# --- end-snippet ---


# --- snippet:outlier-channels ---
@torch.no_grad()
def outlier_channels(top: int = 8) -> list[dict]:
    """Per-channel max |activation| after BatchNorm, and how far the widest channel sticks out.

    Hooked on BatchNorm (and Conv1d) outputs, not on Conv2d: in the exported graph each
    BatchNorm is folded into its conv, so the tensor ONNX Runtime quantizes is the
    post-BatchNorm one. Measuring before BatchNorm would describe a tensor that never exists.
    """
    stats: dict[str, torch.Tensor] = {}

    def hook(name):
        def record(_, __, out):
            per_channel = out.abs().amax(dim=[d for d in range(out.dim()) if d != 1])
            stats[name] = torch.maximum(stats[name], per_channel) if name in stats else per_channel

        return record

    for model_name, batches in ((common.DET_BASE, calibration.detector_batches(calibration.frames("stratified", 32))),
                                (common.OCR_BASE, calibration.ocr_batches("stratified"))):  # fmt: skip
        model = io.load(model_name)
        watched = [(n, m) for n, m in model.named_modules() if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.Conv1d))]
        handles = [m.register_forward_hook(hook(f"{model_name}.{n}")) for n, m in watched]
        for b in batches:
            model(torch.from_numpy(b))
        for h in handles:
            h.remove()
    rows = [{"layer": k, "max": round(float(v.max()), 3), "median": round(float(v.median()), 3),
             "spread": round(float(v.max() / v.median().clamp(min=1e-6)), 1)} for k, v in stats.items()]  # fmt: skip
    return sorted(rows, key=lambda r: -r["spread"])[:top]


# --- end-snippet ---


# --- snippet:mixed-precision ---
def mixed_precision(sens: dict, keep_worst: int = 2) -> dict:
    """First layer, output layers and the most sensitive groups stay FP32; the rest go INT8."""
    ocr_pre, det_pre = pre("ocr_baseline.onnx"), pre("detector_baseline.onnx")
    ocr_groups, det_groups = layer_groups(ocr_pre, 1), layer_groups(det_pre, 2, with_decode=True)
    worst_ocr = sorted(sens["ocr"], key=lambda g: -sens["ocr"][g])[:keep_worst]
    worst_det = sorted(sens["detector"], key=lambda g: -sens["detector"][g])[:keep_worst]
    ocr_keep = set(worst_ocr) | {next(iter(ocr_groups)), "fc"}
    det_keep = set(worst_det) | {next(iter(det_groups)), "detector.obj", "detector.box"}
    exclude_ocr = [n for g in ocr_keep if g in ocr_groups for n in ocr_groups[g]]
    exclude_det = [n for g in det_keep if g in det_groups for n in det_groups[g]]
    quantize(ocr_pre, "ocr_int8_mixed.onnx", "crops", calibration.ocr_batches("stratified"), exclude=exclude_ocr)
    quantize(det_pre, "detector_int8_mixed.onnx", "images",
             calibration.detector_batches(calibration.frames("stratified")), exclude=exclude_det)  # fmt: skip
    return {"ocr_kept_fp32": sorted(ocr_keep), "detector_kept_fp32": sorted(det_keep)}


# --- end-snippet ---


# --- snippet:calibration-ranges ---
def calibration_ranges() -> dict:
    """How different are the activation ranges the daytime-only and stratified calibration sets produced?

    MinMax sets each range from the most extreme activation it sees. A ratio (daytime / stratified)
    near 1 means daytime frames already contained those extremes, so leaving the other conditions
    out changed nothing. A ratio well below 1 means ranges measured on daytime frames clip
    activations that night or rain produce — the failure the calibration guide describes.
    """

    def scales(name: str) -> dict[str, float]:
        model = onnx.load(str(art(name)))
        inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        return {n.input[0]: float(inits[n.input[1]]) for n in model.graph.node  # per-tensor activation scales
                if n.op_type == "QuantizeLinear" and n.input[1] in inits and inits[n.input[1]].size == 1}  # fmt: skip

    out = {}
    for model in ("detector", "ocr"):
        strat, day = scales(f"{model}_int8_stratified.onnx"), scales(f"{model}_int8_daytime.onnx")
        ratio = np.array([day[k] / strat[k] for k in strat if k in day and strat[k] > 0])
        out[model] = {"tensors": int(ratio.size), "min": round(float(ratio.min()), 3), "median": round(float(np.median(ratio)), 3),
                      "max": round(float(ratio.max()), 3), "share_10pct_narrower": round(float((ratio < 0.9).mean()), 3)}  # fmt: skip
    return out


# --- end-snippet ---


def damage(hw: str) -> list[dict]:
    """Every INT8 row's accuracy minus the FP32 ONNX row's, overall and per condition."""
    prof = config.profile().name
    fp32 = results.find("s04_onnx_export:ort-cpu-fp32", hw, prof)
    out = []
    for row in results.load():
        if not (fp32 and row["stage"] in ("s06a_ptq", "s06b_qat") and row["hardware_id"] == hw
                and row["profile"] == prof and row["status"] == "ok"):  # fmt: skip
            continue
        d = {"variant": row["id"]}
        for metric, short in (("map50", "mAP"), ("ocr_exact_match", "EM")):
            d[f"{short} all"] = round(row["accuracy"][metric] - fp32["accuracy"][metric], 4)
            for c in config.CONDITIONS:
                d[f"{short} {c}"] = round(row["accuracy"]["per_condition"][c][metric] - fp32["accuracy"]["per_condition"][c][metric], 4)
        out.append(d)
    return out


def main() -> None:
    common.banner("int8 debugging loop")
    common.require_files("detector_int8_stratified.onnx", "ocr_int8_stratified.onnx", fix="make s06a")
    hw = environment.hardware_id(environment.capture())
    report = {"sensitivity": sensitivity(), "outlier_channels": outlier_channels(), "calibration_ranges": calibration_ranges()}
    report["mixed_precision"] = mixed_precision(report["sensitivity"])
    spec = RunSpec("s06a_ptq", "ort-static-int8-mixed", "s06a_ptq:ort-static-int8-stratified", "ort",
                   {"detector": "detector_int8_mixed.onnx", "ocr": "ocr_int8_mixed.onnx"}, branch="quantization",
                   precision="int8-mixed", notes=f"FP32 kept: {report['mixed_precision']}")  # fmt: skip
    common.gate(spec, OrtBackend("detector_baseline.onnx", "ocr_baseline.onnx"), OrtBackend(**spec.options), strict=False)
    run(spec)
    report["damage"] = damage(hw)  # after the mixed row exists, so it is in the table

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"report_{hw}_{config.profile().name}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1))
    lines = ["# INT8 debugging report (generated by src/quant_debug.py)", "",
             "## 1. Damage vs the FP32 ONNX row (absolute deltas; negative = worse)", ""]  # fmt: skip
    if report["damage"]:
        keys = list(report["damage"][0])
        lines += ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        lines += ["| " + " | ".join(str(d[k]) for k in keys) + " |" for d in report["damage"]]
    s = report["sensitivity"]
    lines += ["", "## 2. Sensitivity — accuracy lost when ONLY this group is INT8", "",
              f"Recognizer FP32 exact match on ground-truth crops: {s['ocr_fp32_exact_match']}", "",
              "| OCR group | exact match lost |", "|---|---|"]  # fmt: skip
    lines += [f"| {g} | {v} |" for g, v in sorted(s["ocr"].items(), key=lambda kv: -kv[1])]
    fp32 = s["detector_fp32"]
    lines += ["", f"Detector FP32 on {fp32['frames']} val frames, FP32 recognizer: mAP@0.5 {fp32['map50']}, exact match {fp32['em']}", "",
              "| detector group | exact match lost | mAP@0.5 lost |", "|---|---|---|"]  # fmt: skip
    lines += [f"| {g} | {v} | {s['detector_map50_lost'][g]} |" for g, v in sorted(s["detector"].items(), key=lambda kv: -kv[1])]
    lines += ["", "## 3. Outlier channels (max / median per-channel activation, after BatchNorm)", "",
              "| layer | max | median | spread |", "|---|---|---|---|"]  # fmt: skip
    lines += [f"| {r['layer']} | {r['max']} | {r['median']} | {r['spread']}x |" for r in report["outlier_channels"]]
    lines += ["", "## 4. Mixed precision", "",
              f"Kept in FP32: `{report['mixed_precision']}` — measured as row `s06a_ptq:ort-static-int8-mixed`."]  # fmt: skip
    lines += ["", "## 5. Calibration ranges — daytime-only scale / stratified scale, per activation tensor", "",
              "A ratio near 1: daytime frames already held the extremes MinMax keeps. Below 1: daytime ranges clip other conditions.", "",
              "| model | tensors | min | median | max | share ≥10% narrower |", "|---|---|---|---|---|---|"]  # fmt: skip
    lines += [f"| {m} | {r['tensors']} | {r['min']} | {r['median']} | {r['max']} | {r['share_10pct_narrower']:.0%} |"
              for m, r in report["calibration_ranges"].items()]  # fmt: skip
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"  wrote {stem.with_suffix('.md').relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
