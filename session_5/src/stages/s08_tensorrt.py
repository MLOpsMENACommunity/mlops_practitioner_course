"""Stage 8 — TensorRT: build engines from ONNX, FP32 / FP16 / INT8. NVIDIA GPU + TensorRT 10.x.

Rows (each engine's parent is the ONNX row it was built from):
  baseline-fp16            the server model, for the "what does a big GPU buy" reference
  student-fp32             strict parity gate against the ONNX student
  student-fp16             FP16 kernels; Ampere has FP16 tensor cores
  student-int8-stratified  implicit INT8 through IInt8EntropyCalibrator2 on our stratified calibration set
  student-int8-daytime     the same calibrator on daytime frames only
FP8 is out of scope: it needs an Ada (SM 8.9) or newer GPU; the RTX 3090 is Ampere (SM 8.6).

Targets TensorRT 10.16 (NGC 26.05). TensorRT 11 removed the implicit INT8 calibrator
and the FP16/INT8 builder flags this file uses; there, INT8 means a Q/DQ ONNX (s06a).
On a machine without CUDA + TensorRT 10, every row is recorded as not_run with the reason.

    make s08
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import config
from src.benchmark import RunSpec, not_run, run
from src.datasets import calibration
from src.stages import common

ONNX_PARENT = "s07_distillation:student-distilled-onnx"


def unavailable() -> str | None:
    if not common.has_cuda():
        return "no CUDA device: TensorRT engines are built for, and run on, an NVIDIA GPU"
    if not common.has_module("tensorrt"):
        return "tensorrt is not installed (make setup-gpu)"
    import tensorrt as trt

    if not trt.__version__.startswith("10."):
        return f"TensorRT {trt.__version__}: this stage targets 10.x (NGC 26.05); 11.x removed the INT8 calibrator"
    return None


def calibrator_class():
    """Defined lazily: subclassing a tensorrt type at import time would crash on a CPU laptop."""
    import tensorrt as trt
    import torch

    # --- snippet:int8-calibrator ---
    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """Feeds our calibration batches to TensorRT, which picks per-tensor ranges by KL divergence."""

        def __init__(self, batches: list[np.ndarray], cache: Path) -> None:
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.batches, self.cache, self.i = batches, cache, 0
            self.buffer = torch.empty(batches[0].shape, dtype=torch.float32, device="cuda")

        def get_batch_size(self) -> int:
            return self.batches[0].shape[0]

        def get_batch(self, names: list[str]) -> list[int] | None:
            if self.i == len(self.batches):
                return None  # tells TensorRT calibration is finished
            self.buffer.copy_(torch.from_numpy(self.batches[self.i]))
            self.i += 1
            return [int(self.buffer.data_ptr())]

        def read_calibration_cache(self) -> bytes | None:
            # A cache from a different calibration set is silently reused. Delete it when the data changes.
            return self.cache.read_bytes() if self.cache.exists() else None

        def write_calibration_cache(self, cache) -> None:
            self.cache.write_bytes(bytes(cache))

    # --- end-snippet ---
    return EntropyCalibrator


# --- snippet:trt-build ---
def build_engine(onnx_path: Path, engine_path: Path, precision: str, input_name: str,
                 shapes: tuple[tuple, tuple, tuple], calibrator=None, workspace_gb: int = 4) -> None:  # fmt: skip
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # explicit batch is the only mode in TRT 10
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)  # scratch space tactics may use
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED  # keep layer names for the inspector
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, *shapes)  # kernels tuned for `opt`, valid anywhere in [min, max]
    cfg.add_optimization_profile(profile)
    if precision in ("fp16", "int8"):
        cfg.set_flag(trt.BuilderFlag.FP16)  # INT8 builds keep FP16 for layers with no INT8 kernel
    if precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calibrator
        cfg.set_calibration_profile(profile)
    serialized = builder.build_serialized_network(network, cfg)
    if serialized is None:
        raise RuntimeError(f"TensorRT build failed for {onnx_path.name} ({precision})")
    engine_path.write_bytes(serialized)


# --- end-snippet ---


def inspect_layers(engine_path: Path) -> dict:
    """How many layers survived fusion, and which ones fused (TensorRT joins their names with '+')."""
    import json

    import tensorrt as trt

    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_path.read_bytes())
    info = json.loads(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    names = [layer if isinstance(layer, str) else layer.get("Name", "") for layer in info["Layers"]]
    return {"layers": len(names), "fused": [n for n in names if "+" in n][:12]}


def build_all() -> dict[str, dict]:
    art, prof = config.artifacts_dir(), config.profile()
    max_b = max(prof.throughput_batches)
    det_shapes = ((1, 3, config.DET_H, config.DET_W), (1, 3, config.DET_H, config.DET_W), (max_b, 3, config.DET_H, config.DET_W))
    ocr_shapes = ((1, 1, config.OCR_H, config.OCR_W), (config.MAX_PLATES, 1, config.OCR_H, config.OCR_W),
                  (max_b * config.MAX_PLATES, 1, config.OCR_H, config.OCR_W))  # fmt: skip
    Calibrator = calibrator_class()
    built = {}
    plan = [("baseline", "fp16", None), ("student", "fp32", None), ("student", "fp16", None),
            ("student", "int8", "stratified"), ("student", "int8", "daytime")]  # fmt: skip
    for model, precision, strategy in plan:
        tag = f"{model}_{precision}" + (f"_{strategy}" if strategy else "")
        det_cal = ocr_cal = None
        if strategy:
            # Calibration batches must fit inside the optimization profile's [min, max] shapes.
            db, ob = min(8, max_b), min(16, max_b * config.MAX_PLATES)
            frames = calibration.detector_batches(calibration.frames(strategy))
            det_cal = Calibrator([np.concatenate(frames[i : i + db]) for i in range(0, len(frames) - db + 1, db)], art / f"calib_det_{strategy}.cache")
            crops = calibration.ocr_batches(strategy)
            ocr_cal = Calibrator([np.concatenate(crops[i : i + ob]) for i in range(0, len(crops) - ob + 1, ob)], art / f"calib_ocr_{strategy}.cache")
        build_engine(art / f"detector_{model}.onnx", art / f"detector_{tag}.plan", precision, "images", det_shapes, det_cal)
        build_engine(art / f"ocr_{model}.onnx", art / f"ocr_{tag}.plan", precision, "crops", ocr_shapes, ocr_cal)
        built[tag] = inspect_layers(art / f"detector_{tag}.plan")
        print(f"  built {tag}: detector has {built[tag]['layers']} layers after fusion", flush=True)
    return built


def specs() -> list[RunSpec]:
    rows = []
    for tag, parent in (("baseline_fp16", "s04_onnx_export:ort-cpu-fp32"), ("student_fp32", ONNX_PARENT), ("student_fp16", ONNX_PARENT),
                        ("student_int8_stratified", ONNX_PARENT), ("student_int8_daytime", ONNX_PARENT)):  # fmt: skip
        precision = tag.split("_")[1]
        rows.append(RunSpec("s08_tensorrt", tag.replace("_", "-"), parent, "tensorrt",
                            {"detector": f"detector_{tag}.plan", "ocr": f"ocr_{tag}.plan"},
                            branch="tensorrt", device="cuda", precision=precision))  # fmt: skip
    return rows


def main() -> None:
    common.banner("s08 TensorRT")
    if reason := unavailable():
        for spec in specs():
            not_run(spec, reason)
        return
    from src.backends import OrtBackend, TensorRTBackend

    common.require_files("detector_baseline.onnx", "ocr_baseline.onnx", fix="make s04")
    common.require_files("detector_student.onnx", "ocr_student.onnx", fix="make s07")
    layers = build_all()
    ref = OrtBackend("detector_student.onnx", "ocr_student.onnx")
    for spec in specs():
        tag = spec.variant.replace("-", "_")
        spec.notes = f"detector engine: {layers[tag]}"
        if tag.startswith("student"):
            common.gate(spec, ref, TensorRTBackend(**spec.options), strict=spec.precision == "fp32")
        run(spec)


if __name__ == "__main__":
    main()
