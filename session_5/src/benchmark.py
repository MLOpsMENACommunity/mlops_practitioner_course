"""THE harness. Nothing else in this repo times anything.

    from src.benchmark import RunSpec, run
    run(RunSpec("s04_onnx_export", "ort-cpu-fp32", parent="s01_baseline:eager-fp32",
                backend="ort", options={"detector": "detector.onnx", "ocr": "ocr.onnx"}))

Every measurement runs in a FRESH worker process (python -m src.benchmark '<spec>'):
  - peak RSS then belongs to this backend alone, not to whatever the stage script
    trained, exported or imported before it;
  - OMP_NUM_THREADS is set before torch/ORT import, which is the only time it counts.

Inside the worker, two passes that never share a loop:
  1. accuracy — every validation frame, no clock running
  2. timing   — its own loop over a fixed frame list; predictions are discarded
Reading predictions out of the timing loop gives you a 32-frame accuracy estimate
dressed up as a metric.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from src import config, environment, results


@dataclass
class RunSpec:
    stage: str
    variant: str
    parent: str | None  # lineage: which row's artifact this one was built from
    backend: str  # key into src.backends.REGISTRY
    options: dict = field(default_factory=dict)
    fast_resize: bool = False
    pipeline: str = "cpu"  # "gpu": nvJPEG decode + GPU letterbox (src/pipeline_gpu.py)
    branch: str = "main"  # "main" rows chain; other branches fork off `parent`
    precision: str = "fp32"
    device: str = "cpu"
    throughput: bool = True
    concurrent: bool = False  # throughput from N concurrent single-frame clients (servers) instead of client batches
    parity: dict | None = None  # report from src.parity.check, attached by the stage
    notes: str = ""

    @property
    def id(self) -> str:
        return f"{self.stage}:{self.variant}"


# --- snippet:peak-memory ---
class PeakMemory:
    """Samples resident memory (and GPU memory through NVML) every 5 ms on a thread.

    Peak RSS, not file size, decides whether a pipeline fits on a 4 GB edge box:
    runtimes allocate workspaces, arenas and caches that no file on disk reflects.
    """

    def __init__(self, device: str) -> None:
        import psutil

        self.proc, self.rss, self.vram, self._stop = psutil.Process(), 0, None, threading.Event()
        self._nvml = None
        if device.startswith("cuda"):
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
                self.vram = 0
            except Exception:  # noqa: BLE001 — no NVML: VRAM is reported as unmeasured, not zero
                self._nvml = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.rss = max(self.rss, self.proc.memory_info().rss)
            if self._nvml:
                nv, handle = self._nvml
                for p in nv.nvmlDeviceGetComputeRunningProcesses(handle):
                    if p.pid == self.proc.pid and p.usedGpuMemory:
                        self.vram = max(self.vram, p.usedGpuMemory)
            time.sleep(0.005)

    def __enter__(self) -> PeakMemory:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


# --- end-snippet ---


# --- snippet:timing-loop ---
def timed_call(pipe, jpegs: list[bytes]) -> dict[str, float]:
    """One pass through the pipeline, milliseconds per phase."""
    marks, last = {}, time.perf_counter()  # perf_counter: monotonic, high resolution; time.time is neither

    def lap(phase: str) -> None:
        nonlocal last
        pipe.backend.sync()  # GPU calls return when work is queued; stop the clock only when it is done
        now = time.perf_counter()
        marks[phase], last = (now - last) * 1000, now

    frames = pipe.decode(jpegs)
    lap("decode")
    batch, lbs = pipe.preprocess(frames)
    lap("preprocess")
    raw = pipe.detect(batch)
    lap("detect")
    dets = pipe.nms(raw)
    lap("nms")
    crops = pipe.crop(frames, dets, lbs)
    lap("crop")
    pipe.ocr(crops)
    lap("ocr")
    marks["total"] = sum(marks.values())
    return marks


# --- end-snippet ---


def accuracy_pass(pipe) -> dict:
    """Every validation frame, untimed. The split is hash-verified first."""
    from src.datasets.splits import load_split
    from src.metrics import FrameResult, summarize

    frames = []
    for sample in load_split("val", verify=True):
        plates = pipe.run([sample.path.read_bytes()])[0]
        dets = np.array([p.box for p in plates], np.float32).reshape(-1, 5)
        gt = np.array(sample.boxes, np.float32).reshape(-1, 4)
        frames.append(FrameResult(sample.condition, gt, sample.texts, dets, [p.text for p in plates]))
    return summarize(frames)


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values)
    return {q: round(float(np.percentile(arr, p)), 3) for q, p in (("p50", 50), ("p95", 95), ("p99", 99))} | {
        "mean": round(float(arr.mean()), 3)
    }


def latency_pass(pipe, jpegs: list[bytes], batch_size: int, runs: int) -> dict:
    """Fixed frames, fixed batch size, >= 20 warm-up calls, predictions discarded."""
    batches = [jpegs[i : i + batch_size] for i in range(0, len(jpegs) - batch_size + 1, batch_size)]
    # Warm-up: the first calls pay for lazy allocation, kernel/algorithm selection,
    # compilation and cold caches. None of that is steady-state latency — but the
    # very first call is recorded on its own, because a camera that reboots pays it.
    cold = timed_call(pipe, batches[0])["total"]
    for i in range(1, config.WARMUP_RUNS):
        timed_call(pipe, batches[i % len(batches)])
    calls = [timed_call(pipe, batches[i % len(batches)]) for i in range(runs)]
    phases = {p: _stats([c[p] for c in calls]) for p in pipe.PHASES}
    total = _stats([c["total"] for c in calls])
    tail = [c for c in calls if c["total"] >= total["p95"]] or calls
    tail_phases = {p: round(float(np.mean([c[p] for c in tail])), 3) for p in pipe.PHASES}
    return {"batch_size": batch_size, "n_runs": runs, "warmup_runs": config.WARMUP_RUNS,
            "cold_first_call_ms": round(cold, 3), "total_ms": total, "phases_ms": phases,
            "p95_tail_phases_ms": tail_phases}  # fmt: skip


def throughput_pass(pipe, jpegs: list[bytes], batch_sizes: tuple[int, ...], iters: int) -> dict:
    """Frames/sec per batch size. Every frame in a batch waits for the whole batch."""
    curve = []
    for bs in batch_sizes:
        lat = latency_pass(pipe, jpegs, bs, iters)
        fps = bs / (lat["total_ms"]["mean"] / 1000)
        curve.append({"batch": bs, "fps": round(fps, 2), "p50_ms": lat["total_ms"]["p50"], "p95_ms": lat["total_ms"]["p95"]})
    best = max(curve, key=lambda p: p["fps"])
    return {"saturating_batch": best["batch"], "fps": best["fps"], "curve": curve}


def concurrent_pass(make_pipe, jpegs: list[bytes], clients: tuple[int, ...], iters: int) -> dict:
    """Frames/sec with N concurrent single-frame clients — how cameras actually hit a server.

    A client-side batch (throughput_pass) never exercises a server's dynamic batcher:
    the batcher merges SEPARATE requests that arrive close together, so the load must
    be separate requests arriving close together.
    """
    from concurrent.futures import ThreadPoolExecutor

    curve = []
    for n in clients:
        pipes = [make_pipe() for _ in range(n)]
        for p in pipes:
            timed_call(p, jpegs[:1])  # warm each connection
        with ThreadPoolExecutor(n) as pool:
            start = time.perf_counter()
            per_client = list(pool.map(lambda p: [timed_call(p, [jpegs[i % len(jpegs)]])["total"] for i in range(iters)], pipes))
            wall = time.perf_counter() - start
        totals = [t for client in per_client for t in client]
        curve.append({"batch": n, "clients": n, "fps": round(len(totals) / wall, 2),
                      "p50_ms": round(float(np.percentile(totals, 50)), 3), "p95_ms": round(float(np.percentile(totals, 95)), 3)})  # fmt: skip
    best = max(curve, key=lambda p: p["fps"])
    return {"mode": "concurrent-clients", "saturating_batch": best["batch"], "fps": best["fps"], "curve": curve}


def _worker(spec: RunSpec) -> dict:
    """Runs inside the fresh process."""
    from src.backends import REGISTRY
    from src.datasets.splits import fingerprint, load_split
    from src.pipeline import Pipeline

    prof = config.profile()

    def make_pipe():
        b = REGISTRY[spec.backend](**spec.options)
        if spec.pipeline == "gpu":
            from src.pipeline_gpu import GpuPipeline

            return GpuPipeline(b)
        if spec.pipeline == "remote":
            from src.pipeline_remote import RemotePipeline

            return RemotePipeline(b)
        return Pipeline(b, fast_resize=spec.fast_resize)

    pipe = make_pipe()
    backend = pipe.backend
    if inject := float(os.environ.get("ANPR_INJECT_LATENCY_MS", "0")):
        # CI demo only (make gate-demo): a deliberate regression, so the gate can be seen failing.
        real_detect = backend.detect
        backend.detect = lambda x: (time.sleep(inject / 1000), real_detect(x))[1]
    jpegs = [s.path.read_bytes() for s in load_split("val")]  # in memory: disk I/O is not the pipeline
    with PeakMemory(spec.device) as mem:
        # The process's very first call pays lazy initialisation, torch.compile compilation and
        # TensorRT EP engine builds. Timed before the accuracy pass, which would otherwise absorb it.
        first = time.perf_counter()
        pipe.run(jpegs[:1])
        backend.sync()
        process_first_call_ms = round((time.perf_counter() - first) * 1000, 3)
        accuracy = accuracy_pass(pipe)
        latency = latency_pass(pipe, jpegs, config.SLA_TARGET.batch_size, prof.latency_runs)
        if not spec.throughput:
            tput = None
        elif spec.concurrent:
            tput = concurrent_pass(make_pipe, jpegs, prof.throughput_batches, prof.throughput_iters)
        else:
            tput = throughput_pass(pipe, jpegs, prof.throughput_batches, prof.throughput_iters)
    out = {"accuracy": accuracy, "latency": latency, "throughput": tput, "size_mb": backend.size_mb(),
           "process_first_call_ms": process_first_call_ms,
           "peak_memory_mb": {"rss": round(mem.rss / 2**20, 1),
                              "vram": None if mem.vram is None else round(mem.vram / 2**20, 1)},
           "val_sha256": fingerprint("val")}  # fmt: skip
    if hasattr(backend, "active_providers"):
        out["active_providers"] = backend.active_providers
    if hasattr(backend, "inference_precision"):
        out["inference_precision"] = backend.inference_precision
    return out


def _row(spec: RunSpec, env: dict, status: str, **fields) -> dict:
    return {"id": spec.id, "stage": spec.stage, "variant": spec.variant, "parent": spec.parent,
            "branch": spec.branch, "backend": spec.backend, "device": spec.device, "precision": spec.precision,
            "fast_resize": spec.fast_resize, "nms_in_graph": spec.options.get("nms_in_graph", False),
            "options": spec.options, "pipeline": spec.pipeline, "concurrent": spec.concurrent,
            "status": status, "profile": config.profile().name, "hardware_id": environment.hardware_id(env),
            "environment": env, "git_sha": environment.git_sha(), "parity": spec.parity, "notes": spec.notes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}  # fmt: skip


def run(spec: RunSpec, record: bool = True) -> dict:
    """Measure `spec` in a fresh process and append the row to results/results.json.

    record=False measures without writing — the CI gate's re-measurement must never
    overwrite the row a stage recorded.
    """
    env = os.environ | {"OMP_NUM_THREADS": str(config.THREADS)}
    print(f"  benchmarking {spec.id} ({spec.backend}, {spec.device}) ...", flush=True)
    proc = subprocess.run([sys.executable, "-m", "src.benchmark", json.dumps(asdict(spec))],
                          capture_output=True, text=True, env=env, cwd=config.ROOT)  # fmt: skip
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if proc.returncode != 0 or not lines:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
        row = _row(spec, environment.capture(), "failed", reason="\n".join(tail))
        if record:
            results.record(row)
        print("    FAILED:\n      " + "\n      ".join(tail), flush=True)
        return row
    measured = json.loads(lines[-1])
    row = _row(spec, measured.pop("environment"), "ok", **measured)
    if record:
        results.record(row)
    lat, acc = row["latency"]["total_ms"], row["accuracy"]
    print(f"    mAP@0.5 {acc['map50']}  OCR EM {acc['ocr_exact_match']}  p50 {lat['p50']} ms  p95 {lat['p95']} ms  "
          f"size {row['size_mb']['total']} MB  RSS {row['peak_memory_mb']['rss']} MB", flush=True)  # fmt: skip
    return row


def failed(spec: RunSpec, reason: str) -> dict:
    """Record a stage that ran and failed — a parity gate, a converter crash."""
    row = _row(spec, environment.capture(), "failed", reason=reason)
    results.record(row)
    return row


def not_run(spec: RunSpec, reason: str) -> dict:
    """Record that a stage could not run here, and why. Its table cells render as a dash."""
    row = _row(spec, environment.capture(), "not_run", reason=reason)
    results.record(row)
    print(f"  NOT RUN {spec.id}: {reason}", flush=True)
    return row


if __name__ == "__main__":
    spec = RunSpec(**json.loads(sys.argv[1]))
    measured = _worker(spec)
    measured["environment"] = environment.capture()  # after the imports, so versions are the loaded ones
    print(json.dumps(measured))
