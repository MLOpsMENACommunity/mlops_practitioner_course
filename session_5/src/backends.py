"""One thin adapter per runtime. The pipeline talks to every one of them the same way.

    detect(float32 NCHW batch) -> (scores [B,N], boxes [B,N,4])   or  [B,TOPK,5] with NMS in graph
    ocr(float32 [N,1,32,128])  -> logits [N,32,37]
    sync()                     -> block until queued device work is finished

Everything runtime-specific — sessions, engines, interpreters, device copies,
CUDA streams — lives in exactly one class below. Heavy imports happen inside
__init__, so a TFLite run never pays torch's memory just by importing this file.
Adapters are looked up by name in REGISTRY; there is no if-chain over runtimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src import config

REGISTRY: dict[str, type[Backend]] = {}


def register(kind: str):
    def wrap(cls: type[Backend]) -> type[Backend]:
        cls.kind = kind
        REGISTRY[kind] = cls
        return cls

    return wrap


def artifact(name: str) -> Path:
    return config.artifacts_dir() / name


def select_quantized_engine() -> str:
    """PyTorch INT8 kernels exist per CPU family: `x86` (fbgemm) on Intel/AMD, `qnnpack` on ARM.

    A quantized model run with the wrong engine either errors or silently falls back
    to slow reference kernels. Note these are CPU-only: PyTorch-native INT8 does not
    run on CUDA — on a GPU, INT8 means TensorRT (s08).
    """
    import platform

    import torch

    engines = torch.backends.quantized.supported_engines
    preferred = "qnnpack" if platform.machine().lower() in ("arm64", "aarch64") else "x86"
    torch.backends.quantized.engine = preferred if preferred in engines else engines[-1]
    return torch.backends.quantized.engine


class Backend:
    kind = "base"
    device = "cpu"
    precision = "fp32"

    def __init__(self, detector: str, ocr: str, nms_in_graph: bool = False, **_: Any) -> None:
        self.detector_name, self.ocr_name, self.nms_in_graph = detector, ocr, nms_in_graph

    def detect(self, x: np.ndarray):
        raise NotImplementedError

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sync(self) -> None:
        """CPU runtimes return finished results; nothing to wait for."""

    def artifacts(self) -> list[Path]:
        return [artifact(self.detector_name), artifact(self.ocr_name)]

    def size_mb(self) -> dict[str, float]:
        def mb(paths: list[Path]) -> float:
            return round(sum(p.stat().st_size for p in paths if p.exists()) / 2**20, 3)

        det, ocr = self.artifacts()[:1], self.artifacts()[1:]
        det = det + [p.with_suffix(".bin") for p in det if p.suffix == ".xml"] + [Path(f"{p}.data") for p in det]
        ocr = ocr + [p.with_suffix(".bin") for p in ocr if p.suffix == ".xml"] + [Path(f"{p}.data") for p in ocr]
        return {"detector": mb(det), "ocr": mb(ocr), "total": round(mb(det) + mb(ocr), 3)}


@register("torch")
class TorchBackend(Backend):
    """Eager PyTorch, optionally torch.compile'd. Checkpoints come from src/models/io.py."""

    def __init__(self, detector: str, ocr: str, device: str = "cpu", nms_in_graph: bool = False,
                 compile: bool = False, **kw: Any) -> None:  # fmt: skip
        super().__init__(detector, ocr, nms_in_graph)
        import torch

        from src.models import io
        from src.models.detector import DetectorExport

        torch.set_num_threads(config.THREADS)
        select_quantized_engine()  # harmless for FP32; required for the dynamic-INT8 recognizer
        self.torch, self.device = torch, device
        self.det = DetectorExport(io.load(detector, device), nms_in_graph).eval()
        self.rec = io.load(ocr, device)
        if compile:
            # dynamic=True on the recognizer: the number of plates per frame varies,
            # and without it every new crop count triggers a recompile mid-benchmark.
            self.det, self.rec = torch.compile(self.det), torch.compile(self.rec, dynamic=True)

    def detect(self, x):
        with self.torch.inference_mode():
            x = self.torch.from_numpy(x) if isinstance(x, np.ndarray) else x  # GpuPipeline hands over CUDA tensors
            out = self.det(x.to(self.device))
            return out.cpu().numpy() if self.nms_in_graph else tuple(o.cpu().numpy() for o in out)

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        with self.torch.inference_mode():
            return self.rec(self.torch.from_numpy(crops).to(self.device)).cpu().numpy()

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()  # CUDA kernels are queued, not run, when the call returns

    def artifacts(self) -> list[Path]:
        return [artifact(f"{self.detector_name}.pt"), artifact(f"{self.ocr_name}.pt")]


@register("torchscript")
class TorchScriptBackend(TorchBackend):
    """A .ts file produced by torch.jit.trace — no Python model class needed to load it."""

    def __init__(self, detector: str, ocr: str, device: str = "cpu", nms_in_graph: bool = False, **kw: Any) -> None:
        Backend.__init__(self, detector, ocr, nms_in_graph)
        import torch

        torch.set_num_threads(config.THREADS)
        select_quantized_engine()  # a traced quantized model (s06b) needs the right INT8 kernels
        self.torch, self.device = torch, device
        self.det = torch.jit.load(str(artifact(detector)), map_location=device).eval()
        self.rec = torch.jit.load(str(artifact(ocr)), map_location=device).eval()

    def artifacts(self) -> list[Path]:
        return Backend.artifacts(self)


@register("ort")
class OrtBackend(Backend):
    """ONNX Runtime with any execution provider list: CPU, CUDA, TensorRT, OpenVINO."""

    def __init__(self, detector: str, ocr: str, providers: list | None = None, nms_in_graph: bool = False,
                 device: str = "cpu", **kw: Any) -> None:  # fmt: skip
        super().__init__(detector, ocr, nms_in_graph)
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.THREADS  # pinned: the default is "all cores", per machine
        opts.inter_op_num_threads = 1
        providers = [tuple(p) if isinstance(p, list) else p for p in (providers or ["CPUExecutionProvider"])]
        self.device = device
        self.det = ort.InferenceSession(str(artifact(detector)), opts, providers=providers)
        self.rec = ort.InferenceSession(str(artifact(ocr)), opts, providers=providers)
        # ORT silently falls back to CPU when a provider fails to load. Record what ran.
        self.active_providers = self.det.get_providers()

    def detect(self, x: np.ndarray):
        out = self.det.run(None, {"images": x})
        return out[0] if self.nms_in_graph else (out[0], out[1])

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        return self.rec.run(None, {"crops": crops})[0]


@register("openvino")
class OpenVinoBackend(Backend):
    """OpenVINO IR (.xml + .bin) compiled for a device with a performance hint."""

    def __init__(self, detector: str, ocr: str, hint: str = "LATENCY", ov_device: str = "CPU",
                 nms_in_graph: bool = False, async_jobs: int = 0, precision_hint: str | None = None,
                 **kw: Any) -> None:  # fmt: skip
        super().__init__(detector, ocr, nms_in_graph)
        import openvino as ov

        core = ov.Core()
        props = {"PERFORMANCE_HINT": hint, "INFERENCE_NUM_THREADS": config.THREADS}
        if precision_hint:
            # The CPU plugin picks its own inference precision per device: on ARM CPUs the
            # default is f16, so an "FP32" IR silently runs in FP16 unless you ask for f32.
            props["INFERENCE_PRECISION_HINT"] = precision_hint
        self.det = core.compile_model(str(artifact(detector)), ov_device, props)
        self.rec = core.compile_model(str(artifact(ocr)), ov_device, props)
        self.inference_precision = str(self.det.get_property("INFERENCE_PRECISION_HINT"))  # what actually ran
        self.det_req, self.rec_req = self.det.create_infer_request(), self.rec.create_infer_request()
        self.queue = ov.AsyncInferQueue(self.det, async_jobs) if async_jobs else None
        if self.queue:
            self.queue.set_callback(self._collect)

    def _collect(self, request, index: int) -> None:
        self._pending[index] = [request.get_output_tensor(i).data.copy() for i in range(len(self.det.outputs))]

    # --- snippet:ov-async-queue ---
    def detect(self, x: np.ndarray):
        if self.queue and len(x) > 1:  # one request per frame, run concurrently on the plugin's streams
            self._pending = [None] * len(x)
            for i, frame in enumerate(x):
                self.queue.start_async({0: frame[None]}, userdata=i)
            self.queue.wait_all()
            outs = [np.concatenate(o) for o in zip(*self._pending)]
        else:
            res = self.det_req.infer({0: x})
            outs = [res[o] for o in self.det.outputs]
        return outs[0] if self.nms_in_graph else (outs[0], outs[1])

    # --- end-snippet ---

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        return self.rec_req.infer({0: crops})[self.rec.outputs[0]]


@register("tflite")
class TfliteBackend(Backend):
    """LiteRT interpreter (XNNPACK on CPU). onnx2tf emits NHWC inputs and a fixed batch of 1."""

    def __init__(self, detector: str, ocr: str, nms_in_graph: bool = False, **kw: Any) -> None:
        super().__init__(detector, ocr, nms_in_graph)
        from ai_edge_litert.interpreter import Interpreter

        self.det = Interpreter(model_path=str(artifact(detector)), num_threads=config.THREADS)
        self.rec = Interpreter(model_path=str(artifact(ocr)), num_threads=config.THREADS)
        for interp in (self.det, self.rec):
            interp.allocate_tensors()

    @staticmethod
    def _invoke(interp, x: np.ndarray) -> list[np.ndarray]:
        inp = interp.get_input_details()[0]
        scale, zero = inp["quantization"]
        if inp["dtype"] == np.int8:  # full-integer model: quantize at the boundary
            x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        outs = []
        for det in sorted(interp.get_output_details(), key=lambda d: d["name"]):
            y = interp.get_tensor(det["index"])
            if det["dtype"] == np.int8:
                s, z = det["quantization"]
                y = (y.astype(np.float32) - z) * s
            outs.append(y)
        return outs

    def detect(self, x: np.ndarray):
        per_frame = [self._invoke(self.det, f[None].transpose(0, 2, 3, 1)) for f in x]
        if self.nms_in_graph:
            return np.concatenate([o[0] for o in per_frame])
        # Sorted by output name: "boxes" before "scores".
        return np.concatenate([o[1] for o in per_frame]), np.concatenate([o[0] for o in per_frame])

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        return np.concatenate([self._invoke(self.rec, c[None].transpose(0, 2, 3, 1))[0] for c in crops])


@register("tensorrt")
class TensorRTBackend(Backend):
    """A serialized TensorRT engine, driven through the TRT 10 tensor-address API.

    Device buffers are torch CUDA tensors, so no pycuda/cuda-python dependency.
    """

    def __init__(self, detector: str, ocr: str, nms_in_graph: bool = False, **kw: Any) -> None:
        super().__init__(detector, ocr, nms_in_graph)
        import tensorrt as trt
        import torch

        self.torch, self.device = torch, "cuda"
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.det = _TrtEngine(runtime, artifact(detector))
        self.rec = _TrtEngine(runtime, artifact(ocr))

    def detect(self, x: np.ndarray):
        out = self.det(self.torch.from_numpy(x))
        return out["detections"] if self.nms_in_graph else (out["scores"], out["boxes"])

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        return self.rec(self.torch.from_numpy(crops))["logits"]

    def sync(self) -> None:
        self.torch.cuda.synchronize()


class _TrtEngine:
    def __init__(self, runtime, path: Path) -> None:
        import tensorrt as trt
        import torch

        self.torch = torch
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{path}: engine failed to deserialize — built with another TensorRT version or GPU?")
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if n not in self.inputs]
        self.dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16,
                       trt.DataType.INT32: torch.int32, trt.DataType.INT64: torch.int64}  # fmt: skip

    def __call__(self, x) -> dict[str, np.ndarray]:
        torch = self.torch
        x = x.cuda().contiguous()
        self.ctx.set_input_shape(self.inputs[0], tuple(x.shape))
        self.ctx.set_tensor_address(self.inputs[0], x.data_ptr())
        outs = {}
        for name in self.outputs:
            dtype = self.dtypes[self.engine.get_tensor_dtype(name)]
            outs[name] = torch.empty(tuple(self.ctx.get_tensor_shape(name)), dtype=dtype, device="cuda")
            self.ctx.set_tensor_address(name, outs[name].data_ptr())
        stream = torch.cuda.current_stream()
        self.ctx.execute_async_v3(stream.cuda_stream)
        # execute_async_v3 returns as soon as the work is QUEUED. Stop a timer here and
        # you have measured submission, not inference.
        stream.synchronize()
        return {k: v.float().cpu().numpy() for k, v in outs.items()}


@register("triton")
class TritonBackend(Backend):
    """Remote models on Triton Inference Server, over gRPC or HTTP."""

    def __init__(self, detector: str = "detector", ocr: str = "ocr", url: str = "localhost:8001",
                 protocol: str = "grpc", nms_in_graph: bool = False, **kw: Any) -> None:  # fmt: skip
        super().__init__(detector, ocr, nms_in_graph)
        if protocol == "grpc":
            import tritonclient.grpc as client
        else:
            import tritonclient.http as client
        self.client_mod, self.client = client, client.InferenceServerClient(url)

    def _infer(self, model: str, name: str, x: np.ndarray, outputs: list[str]) -> list[np.ndarray]:
        inp = self.client_mod.InferInput(name, list(x.shape), "FP32")
        inp.set_data_from_numpy(x)
        req = [self.client_mod.InferRequestedOutput(o) for o in outputs]
        res = self.client.infer(model, [inp], outputs=req)
        return [res.as_numpy(o) for o in outputs]

    def detect(self, x: np.ndarray):
        if self.nms_in_graph:
            return self._infer(self.detector_name, "images", x, ["detections"])[0]
        return tuple(self._infer(self.detector_name, "images", x, ["scores", "boxes"]))

    def ocr(self, crops: np.ndarray) -> np.ndarray:
        return self._infer(self.ocr_name, "crops", crops, ["logits"])[0]

    def pipeline(self, frame: np.ndarray, model: str = "anpr") -> tuple[np.ndarray, list[str]]:
        """The whole pipeline on the server: one decoded frame in, boxes and plate strings out."""
        inp = self.client_mod.InferInput("frame", list(frame.shape), "UINT8")
        inp.set_data_from_numpy(np.ascontiguousarray(frame, dtype=np.uint8))
        outs = [self.client_mod.InferRequestedOutput(n) for n in ("detections", "plates")]
        res = self.client.infer(model, [inp], outputs=outs)
        texts = [t.decode() if isinstance(t, bytes) else str(t) for t in res.as_numpy("plates")]
        return res.as_numpy("detections").reshape(-1, 5), texts

    def artifacts(self) -> list[Path]:
        repo = config.TRITON_REPO
        return [repo / self.detector_name / "1" / "model.onnx", repo / self.ocr_name / "1" / "model.onnx"]
