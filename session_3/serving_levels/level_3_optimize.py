"""
LEVEL 3 — Optimize the runtime (layer 3).

Levels 1 and 2 only touched layers 1 and 2. The model itself is still eager
FP32 PyTorch. Three independent levers, and they multiply with the Level 2
batching win rather than replacing it:

  Lever A  Graph optimization  Conv+BN+ReLU fuse into one kernel, constants
                               fold, memory is planned ahead of time.
                               torch.compile / TorchScript / ONNX Runtime /
                               TensorRT / OpenVINO.       ~2-3x, no accuracy cost
  Lever B  Reduced precision   FP32 -> FP16 -> INT8.      ~2-4x, small accuracy cost
  Lever C  Device concurrency  2-4 model copies per device so compute overlaps
                               host<->device transfers.            ~20-50%

This script measures A and B on whatever hardware you are actually on, then
delivers the punchline: after all of it, preprocessing is the bottleneck.

Run:  python level_3_optimize.py            # --quick skips INT8 build
"""

import argparse
import io
import statistics
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

IMAGE = Path(__file__).parent / "assets" / "dog.jpg"
ARTIFACTS = Path(__file__).parent / "assets"
BATCH_SIZES = (1, 8, 32)


def bench(fn, batch, iters=9, warmup=3) -> float:
    """Median ms per call. Median, not mean: one stray GC pause or a thermal
    blip should not decide which runtime you ship."""
    for _ in range(warmup):
        fn(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip INT8 (slow to build)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).eval().to(device)
    print(f"device: {device}   torch: {torch.__version__}   "
          f"threads: {torch.get_num_threads()}\n")

    variants: dict[str, callable] = {}

    # -- baseline: eager, FP32, op by op through the Python interpreter -------
    @torch.inference_mode()
    def eager(x):
        return model(x.to(device))

    variants["pytorch eager fp32"] = eager

    # -- Lever A: memory layout ----------------------------------------------
    # CUDA only, on purpose. channels_last feeds NHWC to cuDNN's tensor-core
    # kernels; on CPU it is typically a 2x PESSIMIZATION. A lever that helps on
    # one device and hurts on another is the normal case, not the exception --
    # which is why you measure on your target hardware instead of copying a
    # blog post's flag list.
    if device == "cuda":
        m_cl = model.to(memory_format=torch.channels_last)

        @torch.inference_mode()
        def channels_last(x):
            return m_cl(x.to(device, memory_format=torch.channels_last))

        variants["+ channels_last"] = channels_last

    # -- Lever A: graph capture + kernel fusion -------------------------------
    try:
        compiled = torch.compile(model)

        @torch.inference_mode()
        def compiled_fn(x):
            return compiled(x.to(device))

        variants["+ torch.compile"] = compiled_fn
    except Exception as exc:
        print(f"  torch.compile unavailable: {exc}\n")

    # -- Lever A: ONNX Runtime. Works on CPU too -- this is the lever that
    #    matters when you are not paying for a GPU at all. ---------------------
    onnx_path = ARTIFACTS / "resnet50.onnx"
    have_onnx = False
    try:
        import onnxruntime as ort

        if not onnx_path.exists():
            print("exporting ONNX graph ...")
            torch.onnx.export(
                model.to("cpu"),
                torch.randn(1, 3, 224, 224),
                str(onnx_path),
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                dynamo=False,
            )
            model.to(device)

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        variants["onnxruntime fp32"] = lambda x: sess.run(None, {"input": x.numpy()})
        have_onnx = True
    except Exception as exc:
        print(f"  onnxruntime unavailable: {exc}\n")

    # -- Lever B: reduced precision -------------------------------------------
    # INT8 weights with dynamic activation scales. Production PTQ calibrates on
    # a few hundred real images (static quantization) and usually lands under 1%
    # top-1 loss for ResNet; this zero-calibration version is the lower bound.
    if have_onnx and not args.quick:
        try:
            import onnxruntime as ort
            from onnxruntime.quantization import QuantType, quantize_dynamic

            int8_path = ARTIFACTS / "resnet50.int8.onnx"
            if not int8_path.exists():
                print("quantizing to INT8 ...")
                quantize_dynamic(str(onnx_path), str(int8_path),
                                 weight_type=QuantType.QUInt8)
            sess8 = ort.InferenceSession(str(int8_path),
                                         providers=["CPUExecutionProvider"])
            variants["onnxruntime int8"] = lambda x: sess8.run(None, {"input": x.numpy()})
        except Exception as exc:
            print(f"  int8 quantization unavailable: {exc}\n")

    # -- run the grid ---------------------------------------------------------
    print("throughput, images/sec (higher is better); x is vs eager at that batch\n")
    print(f"{'variant':<22}" + "".join(f"{'bs=' + str(b):>16}" for b in BATCH_SIZES))
    print("-" * (22 + 16 * len(BATCH_SIZES)))

    results: dict[str, dict[int, float]] = {}
    base: dict[int, float] = {}
    for name, fn in variants.items():
        row, results[name] = f"{name:<22}", {}
        for bs in BATCH_SIZES:
            x = torch.randn(bs, 3, 224, 224)
            try:
                ips = bs / (bench(fn, x) / 1000)
            except Exception:
                row += f"{'n/a':>16}"
                continue
            results[name][bs] = ips
            base.setdefault(bs, ips)
            row += f"{ips:>10.1f}/s{ips / base[bs]:>5.1f}x"
        print(row)

    # -- what the grid actually says on THIS machine --------------------------
    eager_1 = results["pytorch eager fp32"].get(1, 0)
    eager_32 = results["pytorch eager fp32"].get(32, 0)
    print(f"""
Read the columns, not just the rows.

  bs=1 -> bs=32 on the SAME row is the Level 2 batching win. On a GPU that is
  5-10x, because a batch of 32 costs about what a batch of 1 costs -- the
  device was idle waiting for work. On this CPU it is {eager_32 / max(eager_1, 1e-9):.2f}x for eager
  PyTorch: a CPU is already compute-saturated by one image, so a batch of 32 is
  genuinely 32x the work, and past a point it falls off a memory-bandwidth
  cliff. Batching is a device-utilization fix. No device idle, no win.

  Down a column is the Level 3 runtime win, and it is the lever that still
  pays off on CPU. Note which runtimes hold up at bs=32 and which collapse.

On an NVIDIA GPU the ladder continues past what this machine can run:
    pytorch eager fp32   ->  baseline
    + onnxruntime        ->  ~2x      (works on CPU too)
    + tensorrt fp16      ->  ~4x
    + tensorrt int8      ->  ~7-10x""")

    # -- the punchline --------------------------------------------------------
    raw = IMAGE.read_bytes()
    pre = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    decode_ms = bench(lambda _: pre(Image.open(io.BytesIO(raw)).convert("RGB")),
                      None, iters=15, warmup=3)

    best = max((n for n in results if results[n].get(1)), key=lambda n: results[n][1])
    fwd_ms = 1000 / results[best][1]

    print(f"""
Now the lesson that actually matters
{"-" * 60}
  JPEG decode + resize + normalize      {decode_ms:6.2f} ms   1 image, Python
  forward pass, {best:<22}  {fwd_ms:6.2f} ms   fastest runtime here
  preprocessing is {decode_ms / fwd_ms:.1f}x the cost of the model

We spent all of Level 3 optimizing the forward pass, and preprocessing is now
comparable to it or worse. On a GPU with TensorRT the forward pass drops to
~3ms while that JPEG decode stays ~12ms -- the model becomes a rounding error
and every remaining millisecond is image handling.

Optimize what is actually slow, not what you assume is slow.

Fixes: NVIDIA DALI, GPU-side JPEG decode (nvJPEG), or move preprocessing into
a separate pipeline stage that scales independently of the accelerator.""")


if __name__ == "__main__":
    main()
