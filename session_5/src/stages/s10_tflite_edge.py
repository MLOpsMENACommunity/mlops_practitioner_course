"""Stage 10 — TFLite (LiteRT) for the edge: ONNX -> onnx2tf -> .tflite, full-integer INT8.

Runs in .venv-edge (make s10): onnx2tf pins onnx 1.20.1 / onnxruntime 1.26.0.

Rows (parent: the distilled student ONNX row):
  tflite-fp32               XNNPACK on this CPU
  tflite-int8-full          weights AND activations INT8, int8 input/output tensors, per-tensor scales
  tflite-int8-full-nms      the same with NMS in the graph + fast resize
  tflite-fp16               recorded, with the reason it does not load
  raspberry-pi-5            not_run unless measured on the device itself

Pain points hit while building this, each handled below rather than hidden:
  - onnx2tf shells out to the `onnxsim` CLI: the venv's bin/ must be on PATH or simplification is skipped
  - per-channel INT8 on Conv1d-derived tensors fails onnx2tf's strict validator
    ("quantized_dimension must be in range [0, 1). Was 3") -> quant_type="per-tensor"
  - the optional INT8-weights/INT16-activations variant fails validation after the INT8 files are written
  - the float16 file keeps float16 input tensors, which LiteRT's CONV_2D kernel refuses

    make s10
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

from src import config
from src.benchmark import RunSpec, failed, not_run, run
from src.datasets import calibration
from src.stages import common

PARENT = "s07_distillation:student-distilled-onnx"


def art(name: str) -> Path:
    return config.artifacts_dir() / name


# --- snippet:onnx2tf-convert ---
def convert(onnx_name: str, input_name: str, calib: np.ndarray) -> tuple[Path, str | None]:
    """ONNX (NCHW) -> TFLite (NHWC): float32, float16 and full-integer INT8 in one call."""
    import onnx2tf

    os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"  # onnx2tf calls `onnxsim`
    out = art(f"tflite_{Path(onnx_name).stem}")
    shutil.rmtree(out, ignore_errors=True)
    np.save(out.with_suffix(".calib.npy"), calib.transpose(0, 2, 3, 1))  # representative data, NHWC like the model
    channels = calib.shape[1]
    static_shape = f"{input_name}:1," + ",".join(str(d) for d in calib.shape[1:])
    try:
        onnx2tf.convert(
            input_onnx_file_path=str(art(onnx_name)), output_folder_path=str(out),
            batch_size=1,
            # A fully static input. With only batch_size=1 the dynamic ONNX batch survives far enough
            # that every padded stride-2 conv gets runtime "zero-safe pad" scaffolding built from
            # SHAPE/FILL ops — and FILL has no full-integer kernel, so INT8 conversion fails.
            overwrite_input_shape=[static_shape],
            copy_onnx_input_output_names_to_tflite=True,
            output_integer_quantized_tflite=True, quant_type="per-tensor",
            custom_input_op_name_np_data_path=[[input_name, str(out.with_suffix(".calib.npy")),
                                                [[[[0.0] * channels]]], [[[[1.0] * channels]]]]],
            non_verbose=True,
        )  # fmt: skip
    except Exception as exc:  # noqa: BLE001 — e.g. the INT16-activation variant fails AFTER the INT8 files exist
        # Whatever was written is kept; main() verifies every file loads before measuring it.
        root = exc
        while root.__cause__ is not None:  # "fast path failed" wraps the converter's real reason
            root = root.__cause__
        reason = f"{type(root).__name__}: {str(root).splitlines()[0][:300]}"
        print(f"  onnx2tf raised for {onnx_name}: {reason} — keeping the files it produced")
        return out, reason
    return out, None


# --- end-snippet ---


def loads(path: Path) -> str | None:
    """None if LiteRT can allocate and run it, otherwise the error."""
    from ai_edge_litert.interpreter import Interpreter

    try:
        interp = Interpreter(model_path=str(path))
        interp.allocate_tensors()
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc).splitlines()[0][:200]


def main() -> None:
    common.banner("s10 TFLite / LiteRT edge")
    if not common.has_module("onnx2tf") or not common.has_module("ai_edge_litert"):
        spec = RunSpec("s10_tflite_edge", "tflite-fp32", PARENT, "tflite", branch="edge")
        not_run(spec, "onnx2tf / ai-edge-litert not in this interpreter — run with .venv-edge (make setup-edge; make s10)")
        return
    common.require_files("detector_student.onnx", "detector_student_nms.onnx", "ocr_student.onnx", fix="make s07")
    det_cal = np.concatenate(calibration.detector_batches(calibration.frames("stratified")))
    ocr_cal = np.concatenate(calibration.ocr_batches("stratified"))
    folders = {name: convert(name, inp, cal) for name, inp, cal in (
        ("detector_student.onnx", "images", det_cal), ("detector_student_nms.onnx", "images", det_cal),
        ("ocr_student.onnx", "crops", ocr_cal))}  # fmt: skip

    def pick(onnx_name: str, kind: str) -> str:
        path = folders[onnx_name][0] / f"{Path(onnx_name).stem}_{kind}.tflite"
        return str(path.relative_to(config.artifacts_dir()))

    def conversion_reason(opts: dict) -> dict:
        """For every file onnx2tf did not write, the converter's own error for that model."""
        missing = {}
        for key in ("detector", "ocr"):
            if not (config.artifacts_dir() / opts[key]).exists():
                onnx_name = next(n for n in folders if opts[key].startswith(f"tflite_{Path(n).stem}/"))
                missing[key] = folders[onnx_name][1] or "file not written"
        return missing

    from src.backends import OrtBackend, TfliteBackend

    rows = [
        ("tflite-fp32", {"detector": pick("detector_student.onnx", "float32"), "ocr": pick("ocr_student.onnx", "float32")}, "fp32", False),
        ("tflite-int8-full", {"detector": pick("detector_student.onnx", "full_integer_quant"),
                              "ocr": pick("ocr_student.onnx", "full_integer_quant")}, "int8", False),
        ("tflite-int8-full-nms", {"detector": pick("detector_student_nms.onnx", "full_integer_quant"),
                                  "ocr": pick("ocr_student.onnx", "full_integer_quant"), "nms_in_graph": True}, "int8", True),
        # What converts today: the full-integer recognizer with the float32 detector next to it.
        ("tflite-int8-ocr-fp32-detector", {"detector": pick("detector_student.onnx", "float32"),
                                           "ocr": pick("ocr_student.onnx", "full_integer_quant")}, "int8-ocr", False),
        ("tflite-fp16", {"detector": pick("detector_student.onnx", "float16"), "ocr": pick("ocr_student.onnx", "float16")}, "fp16", False),
    ]  # fmt: skip
    for variant, opts, precision, edge in rows:
        spec = RunSpec("s10_tflite_edge", variant, PARENT, "tflite", opts, fast_resize=edge, branch="edge", precision=precision)
        if unwritten := conversion_reason(opts):
            failed(spec, f"onnx2tf did not produce the file: {unwritten}")
            print(f"  FAILED {spec.id}: {unwritten}")
            continue
        problems = {k: loads(config.artifacts_dir() / v) for k, v in opts.items() if k in ("detector", "ocr")}
        if any(problems.values()):
            failed(spec, f"LiteRT cannot load the converted file: {problems}")
            print(f"  FAILED {spec.id}: {problems}")
            continue
        nms = opts.get("nms_in_graph", False)
        ref = OrtBackend("detector_student_nms.onnx" if nms else "detector_student.onnx", "ocr_student.onnx", nms_in_graph=nms)
        common.gate(spec, ref, TfliteBackend(**opts), strict=precision == "fp32")
        run(spec)

    if on_raspberry_pi():
        return  # on the Pi itself, the tflite-* rows above ARE the device measurements, under the Pi's hardware id
    not_run(RunSpec("s10_tflite_edge", "raspberry-pi-5", "s10_tflite_edge:tflite-int8-full", "tflite", branch="edge",
                    precision="int8"),
            "not measured on a Raspberry Pi: on the device, run `make setup-edge data && make s10` — the tflite-* rows "
            "then appear under the Pi's own hardware heading")  # fmt: skip


def on_raspberry_pi() -> bool:
    try:
        return "Raspberry Pi" in Path("/proc/device-tree/model").read_text()
    except OSError:
        return False


if __name__ == "__main__":
    main()
