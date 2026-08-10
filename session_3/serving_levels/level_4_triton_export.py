"""
LEVEL 4, Triton variant — export ResNet-50 and build a real Triton model repository.

Same model as every other level in this repo. BentoML served the PyTorch module
directly; Triton wants a portable graph instead, so the model is exported to ONNX
first. That conversion step is the main difference in effort between the two, and
what you get for it is a C++ server with no GIL anywhere on the request path.

This script:
  1. loads the same torchvision ResNet-50 as level_0..level_4
  2. exports it to ONNX with a dynamic batch dimension
  3. verifies the ONNX output matches PyTorch (the step people skip)
  4. writes level_4_triton/model_repository/resnet50/1/model.onnx

Run:  python level_4_triton_export.py
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torchvision.models import ResNet50_Weights, resnet50

HERE = Path(__file__).resolve().parent
REPO = HERE / "level_4_triton" / "model_repository" / "resnet50"
VERSION_DIR = REPO / "1"          # Triton's version directory. "2" would be v2.

model = resnet50(weights=ResNet50_Weights.DEFAULT).eval()
print(f"loaded {type(model).__name__} (torchvision, IMAGENET1K weights)")

VERSION_DIR.mkdir(parents=True, exist_ok=True)
out = VERSION_DIR / "model.onnx"

dummy = torch.randn(1, 3, 224, 224)
print("exporting ONNX graph ...")
torch.onnx.export(
    model,
    dummy,
    str(out),
    input_names=["input"],
    output_names=["output"],
    # The dynamic batch axis is what lets Triton's dynamic_batching stanza feed
    # this graph 1 image or 32. Hard-code the batch size here and dynamic
    # batching cannot work at all -- the model would reject every batch but one.
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    dynamo=False,
)
print(f"wrote {out.relative_to(HERE)}  ({out.stat().st_size / 1e6:.0f} MB)")

# ---------------------------------------------------------------------------
# Verify the conversion, at more than one batch size. A silently-wrong export is
# a genuinely nasty production bug: the server is healthy, latency is great, and
# the numbers are subtly wrong. Checking bs=1 only would also miss a broken
# dynamic axis, which is the most common way this export goes wrong.
# ---------------------------------------------------------------------------
sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

print(f"\n{'batch':>7}{'max abs diff':>16}{'top-1 agree':>14}")
ok = True
for bs in (1, 8, 32):
    x = torch.randn(bs, 3, 224, 224)
    with torch.inference_mode():
        torch_out = model(x).numpy()
    onnx_out = sess.run(None, {"input": x.numpy()})[0]

    diff = float(np.max(np.abs(torch_out - onnx_out)))
    agree = bool((torch_out.argmax(1) == onnx_out.argmax(1)).all())
    ok &= diff < 1e-3 and agree
    print(f"{bs:>7}{diff:>16.2e}{str(agree):>14}")

assert ok, "ONNX export does not match PyTorch -- do not ship this"
print("\nconversion verified across batch sizes")

print(f"""
model repository ready:

  {REPO.parent.relative_to(HERE)}/
    resnet50/
      config.pbtxt
      1/model.onnx        <- this run
      2/model.onnx        <- next version, hot-loaded without a restart

Serve it (see level_4_triton/README.md for the platform caveat):

  docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \\
    -v {REPO.parent}:/models \\
    nvcr.io/nvidia/tritonserver:24.09-py3 \\
    tritonserver --model-repository=/models""")
