"""Stage 9 — OpenVINO: IR conversion, NNCF INT8, and what the performance hints change.

Rows:
  baseline-fp32-latency     the server model on OpenVINO CPU, for reference
  student-fp32-latency      ov.convert_model(ONNX) -> IR; PERFORMANCE_HINT=LATENCY
  student-fp32-throughput   same IR; PERFORMANCE_HINT=THROUGHPUT + an AsyncInferQueue across a batch
  student-int8-nncf         nncf.quantize on the IR with the stratified calibration set
  edge-candidate            INT8 student + NMS in the graph + fast resize: every fix that composes
  intel-gpu                 not_run unless an Intel GPU is present (the GPU plugin does not drive NVIDIA)

    make s09
"""

from __future__ import annotations

import nncf
import openvino as ov

from src import config
from src.backends import OpenVinoBackend, OrtBackend
from src.benchmark import RunSpec, failed, not_run, run
from src.datasets import calibration
from src.stages import common

STUDENT_ONNX_ROW = "s07_distillation:student-distilled-onnx"


def art(name: str) -> str:
    return str(config.artifacts_dir() / name)


# --- snippet:ov-convert ---
def to_ir(onnx_name: str, xml_name: str) -> None:
    model = ov.convert_model(art(onnx_name))  # the Python API; `ovc` is the CLI. `mo` no longer exists.
    ov.save_model(model, art(xml_name), compress_to_fp16=False)  # the default (True) silently stores FP16 weights


# --- end-snippet ---


# --- snippet:nncf-quantize ---
def nncf_int8(xml_name: str, out_name: str, batches: list) -> None:
    model = ov.Core().read_model(art(xml_name))
    quantized = nncf.quantize(
        model,
        nncf.Dataset(batches),  # items are already model inputs, so no transform function is needed
        preset=nncf.QuantizationPreset.PERFORMANCE,  # symmetric weights + activations: fastest on CPU
        subset_size=len(batches),
    )
    ov.save_model(quantized, art(out_name), compress_to_fp16=False)


# --- end-snippet ---


def main() -> None:
    common.banner("s09 OpenVINO")
    common.require_files("detector_baseline.onnx", "ocr_baseline.onnx", fix="make s04")
    common.require_files("detector_student.onnx", "detector_student_nms.onnx", "ocr_student.onnx", fix="make s07")
    problems: dict[str, str] = {}  # artifact -> why it could not be built; its rows record `failed`

    def attempt(artifact_name: str, fn, *args) -> None:
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001 — one converter failure must not hide the other rows
            problems[artifact_name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
            print(f"  FAILED to build {artifact_name}: {problems[artifact_name]}")

    for onnx_name in ("detector_baseline.onnx", "ocr_baseline.onnx", "detector_student.onnx",
                      "detector_student_nms.onnx", "ocr_student.onnx"):  # fmt: skip
        attempt(onnx_name.replace(".onnx", ".xml"), to_ir, onnx_name, onnx_name.replace(".onnx", ".xml"))
    det_cal = calibration.detector_batches(calibration.frames("stratified"))
    ocr_cal = calibration.ocr_batches("stratified")
    attempt("detector_student_int8.xml", nncf_int8, "detector_student.xml", "detector_student_int8.xml", det_cal)
    attempt("detector_student_nms_int8.xml", nncf_int8, "detector_student_nms.xml", "detector_student_nms_int8.xml", det_cal)
    attempt("ocr_student_int8.xml", nncf_int8, "ocr_student.xml", "ocr_student_int8.xml", ocr_cal)

    student = {"detector": "detector_student.xml", "ocr": "ocr_student.xml"}
    int8 = {"detector": "detector_student_int8.xml", "ocr": "ocr_student_int8.xml"}
    rows = [
        (RunSpec("s09_openvino", "baseline-fp32-latency", "s04_onnx_export:ort-cpu-fp32", "openvino",
                 {"detector": "detector_baseline.xml", "ocr": "ocr_baseline.xml"}, branch="openvino"), "strict"),
        (RunSpec("s09_openvino", "student-fp32-latency", STUDENT_ONNX_ROW, "openvino", student, branch="openvino"), "strict"),
        (RunSpec("s09_openvino", "student-fp32-throughput", "s09_openvino:student-fp32-latency", "openvino",
                 student | {"hint": "THROUGHPUT", "async_jobs": 4}, branch="openvino",
                 notes="THROUGHPUT hint: the plugin creates parallel streams; a batch fans out over AsyncInferQueue"), "strict"),
        (RunSpec("s09_openvino", "student-int8-nncf", "s09_openvino:student-fp32-latency", "openvino", int8,
                 branch="openvino", precision="int8"), "report"),
        (RunSpec("s09_openvino", "edge-candidate", "s09_openvino:student-int8-nncf", "openvino",
                 {"detector": "detector_student_nms_int8.xml", "ocr": "ocr_student_int8.xml", "nms_in_graph": True},
                 fast_resize=True, branch="openvino", precision="int8",
                 notes="distilled student + NNCF INT8 + NMS in graph + Image.reduce resize"), "report"),
    ]  # fmt: skip
    for spec, mode in rows:
        missing = {v: problems[v] for k, v in spec.options.items() if k in ("detector", "ocr") and v in problems}
        if missing:
            failed(spec, f"artifact build failed: {missing}")
            continue
        nms = spec.options.get("nms_in_graph", False)
        is_base = spec.variant.startswith("baseline")
        ref = OrtBackend("detector_baseline.onnx" if is_base else ("detector_student_nms.onnx" if nms else "detector_student.onnx"),
                         "ocr_baseline.onnx" if is_base else "ocr_student.onnx", nms_in_graph=nms)  # fmt: skip
        common.gate(spec, ref, OpenVinoBackend(**spec.options), strict=mode == "strict")
        run(spec)

    gpu = RunSpec("s09_openvino", "intel-gpu", "s09_openvino:student-fp32-latency", "openvino",
                  student | {"ov_device": "GPU"}, branch="openvino", device="gpu")  # fmt: skip
    if "GPU" in ov.Core().available_devices:
        run(gpu)
    else:
        not_run(gpu, f"no OpenVINO GPU device (available: {ov.Core().available_devices}); the GPU plugin targets Intel GPUs")


if __name__ == "__main__":
    main()
