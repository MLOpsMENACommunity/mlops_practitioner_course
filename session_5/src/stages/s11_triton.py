"""Stage 11 — Triton Inference Server: the pipeline as a service, in Docker.

1. copy the distilled student ONNX files into triton_serving/model_repository/*/1/
2. start nvcr.io/nvidia/tritonserver:26.05-py3 in explicit model-control mode
3. prove it: triton_serving/client.py compares server output with the local pipeline, HTTP and gRPC
4. hot reload: add version 2 of `ocr`, load it, confirm the server switched without a restart
5. measure, with concurrent clients (that is what dynamic batching batches):
     triton-grpc-client-pipeline   decode/letterbox/NMS/crop in the client, models on the server
     triton-http-client-pipeline   the same over HTTP
     triton-grpc-nobatch           the same models with dynamic batching OFF
     triton-bls-server-pipeline    the whole pipeline server side (anpr, Python backend BLS)
6. read the batching the server actually did from the metrics endpoint

    make s11        # needs Docker; on an NVIDIA box also the NVIDIA Container Toolkit
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
import urllib.request

from src import config
from src.benchmark import RunSpec, not_run, run
from src.stages import common

IMAGE = "nvcr.io/nvidia/tritonserver:26.05-py3"  # TensorRT 10.16.1.11, the same as requirements-gpu.txt
CONTAINER = "anpr-triton"
PARENT = "s07_distillation:student-distilled-onnx"
MODELS = {"detector": "detector_student.onnx", "detector_nobatch": "detector_student.onnx",
          "detector_nms": "detector_student_nms.onnx", "ocr": "ocr_student.onnx"}  # fmt: skip


def populate() -> None:
    for model, onnx_name in MODELS.items():
        version = config.TRITON_REPO / model / "1"
        version.mkdir(parents=True, exist_ok=True)
        shutil.copy(config.artifacts_dir() / onnx_name, version / "model.onnx")
        stale = config.TRITON_REPO / model / "2"
        shutil.rmtree(stale, ignore_errors=True)


def docker_problem() -> str | None:
    if shutil.which("docker") is None:
        return "docker is not installed"
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "the Docker daemon is not running"
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        return f"image {IMAGE} not pulled (docker pull {IMAGE}, ~10 GB)"
    return None


# --- snippet:triton-docker-run ---
def start() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    gpus = ["--gpus", "all"] if common.has_cuda() else []  # needs the NVIDIA Container Toolkit
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER, *gpus,
                    "-p", "8000:8000", "-p", "8001:8001", "-p", "8002:8002",  # HTTP, gRPC, Prometheus metrics
                    "-v", f"{config.TRITON_REPO}:/models", IMAGE,
                    "tritonserver", "--model-repository=/models",
                    "--model-control-mode=explicit", "--load-model=*"], check=True)  # fmt: skip
    # --- end-snippet ---
    import tritonclient.http as http

    client = http.InferenceServerClient("localhost:8000")
    for _ in range(120):
        try:
            if client.is_server_ready() and all(client.is_model_ready(m) for m in [*MODELS, "anpr"]):
                return
        except Exception:  # noqa: BLE001 — not listening yet
            pass
        time.sleep(2)
    logs = subprocess.run(["docker", "logs", "--tail", "30", CONTAINER], capture_output=True, text=True)
    raise RuntimeError(f"Triton did not become ready:\n{logs.stdout}{logs.stderr}")


# --- snippet:triton-hot-reload ---
def hot_reload() -> str:
    """Drop in version 2 and load it — no restart. version_policy `latest: 1` serves the newest."""
    import tritonclient.http as http

    client = http.InferenceServerClient("localhost:8000")
    shutil.copytree(config.TRITON_REPO / "ocr" / "1", config.TRITON_REPO / "ocr" / "2", dirs_exist_ok=True)
    client.load_model("ocr")  # explicit mode: the repository is re-read for this model only
    ready = [v for v in ("1", "2") if client.is_model_ready("ocr", v)]
    return f"after load_model('ocr') with a new 2/ directory, ready versions: {ready}"


# --- end-snippet ---


def counters(model: str) -> tuple[float, float]:
    """(successful requests, executions) for one model, read from the Prometheus endpoint."""
    text = urllib.request.urlopen("http://localhost:8002/metrics", timeout=5).read().decode()

    def total(name: str) -> float:
        return sum(float(v) for v in re.findall(rf'{name}{{[^}}]*model="{model}"[^}}]*}} ([0-9.e+]+)', text))

    return total("nv_inference_request_success"), total("nv_inference_exec_count")


# --- snippet:triton-batch-size-from-metrics ---
def batch_size_under_load(model: str, clients: int = 8, requests: int = 64) -> float:
    """Average batch the server formed for `model` during a burst of concurrent single-frame requests.

    Taken as a DELTA of lifetime counters around the burst only — the benchmark's
    one-at-a-time accuracy pass would otherwise drag the average towards 1.
    No clock runs here: this reads what the server did, not how long it took.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from src.backends import TritonBackend
    from src.pipeline import Pipeline

    frame = Pipeline(TritonBackend(detector=model)).preprocess(Pipeline(TritonBackend()).decode(common.sample_jpegs(1)))[0]
    local = threading.local()

    def send(_: int) -> None:
        if not hasattr(local, "backend"):
            local.backend = TritonBackend(detector=model)
        local.backend.detect(frame)

    before = counters(model)
    with ThreadPoolExecutor(clients) as pool:
        list(pool.map(send, range(requests)))
    after = counters(model)
    return round((after[0] - before[0]) / max(after[1] - before[1], 1), 2)


# --- end-snippet ---


def specs() -> list[RunSpec]:
    client = {"detector": "detector", "ocr": "ocr"}
    return [
        RunSpec("s11_triton", "triton-grpc-client-pipeline", PARENT, "triton", client | {"protocol": "grpc"},
                branch="serving", concurrent=True),
        RunSpec("s11_triton", "triton-http-client-pipeline", PARENT, "triton",
                client | {"protocol": "http", "url": "localhost:8000"}, branch="serving", concurrent=True),
        RunSpec("s11_triton", "triton-grpc-nobatch", "s11_triton:triton-grpc-client-pipeline", "triton",
                {"detector": "detector_nobatch", "ocr": "ocr", "protocol": "grpc"}, branch="serving", concurrent=True),
        RunSpec("s11_triton", "triton-bls-server-pipeline", PARENT, "triton", {"protocol": "grpc"}, pipeline="remote",
                branch="serving", concurrent=True, notes="letterbox + detector_nms + crop + ocr inside the server"),
    ]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--up-only", action="store_true")
    args = parser.parse_args()
    common.banner("s11 Triton Inference Server")
    if problem := docker_problem():
        for spec in specs():
            not_run(spec, problem)
        return
    common.require_files("detector_student.onnx", "detector_student_nms.onnx", "ocr_student.onnx", fix="make s07")
    populate()
    start()
    if args.up_only:
        print("  Triton is up: HTTP :8000, gRPC :8001, metrics :8002")
        return
    from triton_serving.client import prove

    proof = {p: prove(p, url, 40) for p, url in (("grpc", "localhost:8001"), ("http", "localhost:8000"))}
    reload_note = hot_reload()
    print(f"  {reload_note}")
    batching = {m: batch_size_under_load(m) for m in ("detector", "detector_nobatch")}
    print(f"  average batch size the server formed under 8 concurrent clients (from /metrics deltas): {batching}")
    for spec in specs():
        spec.parity = proof["grpc"] if spec.pipeline == "remote" else None
        spec.notes = "; ".join(n for n in (spec.notes, f"server avg batch size under 8 concurrent clients: {batching}", reload_note) if n)
        run(spec)


if __name__ == "__main__":
    main()
