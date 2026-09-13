"""Stage 0 — profile before you optimize.

You do not touch the model until you have attributed the latency. This stage:
  1. measures the baseline pipeline with the harness (it IS the s01 baseline row)
     and draws where each millisecond goes, for the median frame and the slowest 5%;
  2. runs torch.profiler inside the model phases, with shapes and Python stacks;
  3. applies the gate — if the model is under half of the p95 tail, preprocessing
     and I/O come first — and measures fixes for exactly that, right here.

    make s00           # everything above
    make pyspy         # py-spy flame graph of the same workload (Linux; sudo on macOS)
"""

from __future__ import annotations

import argparse

from src import config, environment, results
from src.benchmark import RunSpec, not_run, run
from src.stages import common, s01_baseline

PROFILE_DIR = config.RESULTS_DIR / "profile"
MODEL_PHASES = ("detect", "ocr")


def baseline_row() -> dict:
    """The measured baseline, measuring it first if this machine has no row yet."""
    env = environment.capture()
    row = results.find(common.BASELINE_ROW, environment.hardware_id(env), config.profile().name)
    return row if row and row["status"] == "ok" else run(s01_baseline.spec())


# --- snippet:profile-gate ---
def model_share(row: dict, where: str = "p95_tail") -> float:
    """Fraction of the slowest 5% of frames spent inside the two models."""
    phases = row["latency"]["p95_tail_phases_ms"] if where == "p95_tail" else {
        p: s["mean"] for p, s in row["latency"]["phases_ms"].items()}  # fmt: skip
    return sum(phases[p] for p in MODEL_PHASES) / sum(phases.values())


# --- end-snippet ---


def draw_breakdown(row: dict) -> str:
    """Stacked bars of mean ms per phase: median-ish frame vs the p95 tail. Returns the md table."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lat = row["latency"]
    bars = {"all frames (mean)": {p: s["mean"] for p, s in lat["phases_ms"].items()},
            "slowest 5% (mean)": lat["p95_tail_phases_ms"]}  # fmt: skip
    fig, ax = plt.subplots(figsize=(9, 2.8))
    colours = ["#8a8f98", "#b0b6c0", "#2f6fdb", "#e0a526", "#c3c8cf", "#1a9e77"]
    for y, (label, phases) in enumerate(bars.items()):
        left = 0.0
        for colour, (phase, ms) in zip(colours, phases.items()):
            ax.barh(y, ms, left=left, color=colour, label=phase if y == 0 else None, edgecolor="white")
            left += ms
    ax.set_yticks(range(len(bars)), list(bars))
    ax.set_xlabel("milliseconds per frame (batch 1)")
    ax.set_title(f"Baseline pipeline — {row['environment']['cpu']}{' + ' + row['environment']['gpu'] if row['environment'].get('gpu') else ''}", fontsize=9)
    ax.axvline(config.SLA_TARGET.p95_ms, color="#d64545", linestyle="--", linewidth=1, label="30 ms budget")
    ax.legend(ncol=7, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.tight_layout()
    stem = f"phases_{row['hardware_id']}_{row['profile']}"
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PROFILE_DIR / f"{stem}.png", dpi=140)
    lines = ["| phase | mean ms (all frames) | share | mean ms (slowest 5%) | share |", "|---|---|---|---|---|"]
    all_total, tail_total = sum(bars["all frames (mean)"].values()), sum(bars["slowest 5% (mean)"].values())
    for p in row["latency"]["phases_ms"]:
        a, t = bars["all frames (mean)"][p], bars["slowest 5% (mean)"][p]
        lines.append(f"| {p} | {a:.2f} | {a / all_total:.0%} | {t:.2f} | {t / tail_total:.0%} |")
    return "\n".join(lines)


def torch_op_profile(n_frames: int = 20) -> None:
    """torch.profiler around each phase: which operators, at which shapes, from which lines."""
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    from src.backends import TorchBackend
    from src.pipeline import Pipeline

    dev = common.device()
    pipe = Pipeline(TorchBackend(common.DET_BASE, common.OCR_BASE, device=dev))
    jpegs = common.sample_jpegs(n_frames)
    for j in jpegs[:5]:
        pipe.run([j])  # warm up outside the profile window
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev == "cuda" else [])
    # --- snippet:torch-profiler ---
    with profile(activities=activities, record_shapes=True, with_stack=True, profile_memory=True) as prof:
        for jpeg in jpegs:
            with record_function("decode"):
                frames = pipe.decode([jpeg])
            with record_function("preprocess"):
                batch, lbs = pipe.preprocess(frames)
            with record_function("detect"):
                raw = pipe.detect(batch)
            with record_function("nms"):
                dets = pipe.nms(raw)
            with record_function("crop"):
                crops = pipe.crop(frames, dets, lbs)
            with record_function("ocr"):
                pipe.ocr(crops)
    sort = "self_cuda_time_total" if dev == "cuda" else "self_cpu_time_total"
    by_shape = prof.key_averages(group_by_input_shape=True).table(sort_by=sort, row_limit=25)
    # --- end-snippet ---
    by_stack = prof.key_averages(group_by_stack_n=4).table(sort_by=sort, row_limit=12)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    hw = environment.hardware_id(environment.capture())
    out = PROFILE_DIR / f"torch_ops_{hw}_{config.profile().name}.txt"
    header = ("# torch.profiler over the baseline pipeline. Profiler overhead inflates every number here:\n"
              "# use this file to ATTRIBUTE time to operators, never to quote latency. The harness does that.\n\n")  # fmt: skip
    out.write_text(header + "## by operator and input shape\n" + by_shape + "\n\n## by Python stack\n" + by_stack)
    prof.export_chrome_trace(str(config.artifacts_dir() / "s00_trace.json"))  # open in chrome://tracing or Perfetto
    print(f"  wrote {out.relative_to(config.ROOT)} and artifacts/.../s00_trace.json")
    del torch


def fixes(parent: str) -> None:
    """Preprocessing and post-processing fixes, each measured as a branch off the baseline."""
    base = {"detector": common.DET_BASE, "ocr": common.OCR_BASE, "device": common.device()}
    run(RunSpec("s00_profile", "fast-resize", parent, "torch", base, fast_resize=True, branch="pre",
                device=common.device(), notes="Image.reduce(2) instead of bilinear resize"))  # fmt: skip
    run(RunSpec("s00_profile", "nms-in-graph", parent, "torch", base | {"nms_in_graph": True}, branch="pre",
                device=common.device(), notes="top-K + Fast NMS as tensor ops inside the model"))  # fmt: skip
    gpu = RunSpec("s00_profile", "gpu-decode", parent, "torch", base, branch="pre", device="cuda", pipeline="gpu",
                  notes="nvJPEG batched decode + GPU letterbox")  # fmt: skip
    run(gpu) if common.has_cuda() else not_run(gpu, "no CUDA device: nvJPEG decode needs an NVIDIA GPU")


def workload(n: int) -> None:
    """A plain loop for py-spy to sample: no harness, no profiler, just the pipeline."""
    from src.backends import TorchBackend
    from src.pipeline import Pipeline

    pipe = Pipeline(TorchBackend(common.DET_BASE, common.OCR_BASE, device=common.device()))
    jpegs = common.sample_jpegs(16)
    for i in range(n):
        pipe.run([jpegs[i % len(jpegs)]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workload", type=int, help="just run N frames (for py-spy)")
    args = parser.parse_args()
    common.require(common.DET_BASE, common.OCR_BASE)
    if args.workload:
        return workload(args.workload)
    common.banner("s00 profile first")
    row = baseline_row()
    table = draw_breakdown(row)
    torch_op_profile()
    share, share_mean = model_share(row), model_share(row, "mean")
    verdict = ("the MODELS dominate the tail — model-level optimization is the right next step"
               if share >= 0.5 else
               "the models are UNDER half of the tail — fix preprocessing, NMS and I/O before touching the model")  # fmt: skip
    gate = (f"# Profile gate — {row['environment']['cpu']}, profile {row['profile']}\n\n"
            f"p50 {row['latency']['total_ms']['p50']} ms, p95 {row['latency']['total_ms']['p95']} ms, batch 1\n\n"
            f"{table}\n\nModel share of the slowest 5%: **{share:.0%}** (of all frames: {share_mean:.0%}) — {verdict}.\n")  # fmt: skip
    (PROFILE_DIR / f"gate_{row['hardware_id']}_{row['profile']}.md").write_text(gate)
    print("\n" + gate)
    fixes(common.BASELINE_ROW)


if __name__ == "__main__":
    main()
