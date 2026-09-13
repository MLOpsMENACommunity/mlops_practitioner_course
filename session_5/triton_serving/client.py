"""Prove the Triton deployment serves the two-model pipeline — over HTTP and gRPC.

    python -m triton_serving.client --protocol grpc --frames 40
    python -m triton_serving.client --protocol http --url localhost:8000

For each validation frame: send the decoded frame to the server-side `anpr` model,
get boxes + plate strings back, and compare them with the same student models run
locally in ONNX Runtime. Exits non-zero if the server disagrees with the local pipeline.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from src.backends import OrtBackend, TritonBackend
from src.datasets.splits import load_split
from src.metrics import FrameResult, iou, summarize
from src.pipeline import Pipeline
from src.preprocess import decode_jpeg


def prove(protocol: str, url: str, n_frames: int, min_agreement: float = 0.95) -> dict:
    remote = TritonBackend(url=url, protocol=protocol)
    if not remote.client.is_server_ready():
        raise SystemExit(f"Triton at {url} is not ready — make triton-up")
    local = Pipeline(OrtBackend("detector_student_nms.onnx", "ocr_student.onnx", nms_in_graph=True), fast_resize=True)
    agree, total, frames = 0, 0, []
    for sample in [s for s in load_split("val") if s.boxes][:n_frames]:
        jpeg = sample.path.read_bytes()
        dets, texts = remote.pipeline(decode_jpeg(jpeg))
        mine = local.run([jpeg])[0]
        for plate in mine:  # every locally read plate must be found and read identically by the server
            total += 1
            if len(dets):
                j = int(iou(plate.box[None, :4], dets[:, :4])[0].argmax())
                agree += int(iou(plate.box[None, :4], dets[j : j + 1, :4])[0, 0] > 0.9 and texts[j] == plate.text)
        frames.append(FrameResult(sample.condition, np.array(sample.boxes, np.float32), sample.texts, dets, texts))
    report = {"protocol": protocol, "frames": len(frames), "local_plates": total,
              "server_agrees": round(agree / max(total, 1), 4), "server_accuracy": summarize(frames)}  # fmt: skip
    print(f"  {protocol}: {len(frames)} frames, server agrees with local pipeline on {report['server_agrees']:.1%} "
          f"of {total} plates; server mAP@0.5 {report['server_accuracy']['map50']}, "
          f"exact match {report['server_accuracy']['ocr_exact_match']}")  # fmt: skip
    if report["server_agrees"] < min_agreement:
        raise SystemExit(f"server and local pipeline disagree beyond tolerance: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["grpc", "http"], default="grpc")
    parser.add_argument("--url", help="default localhost:8001 (gRPC) / localhost:8000 (HTTP)")
    parser.add_argument("--frames", type=int, default=40)
    args = parser.parse_args()
    url = args.url or ("localhost:8001" if args.protocol == "grpc" else "localhost:8000")
    prove(args.protocol, url, args.frames)


if __name__ == "__main__":
    sys.exit(main())
