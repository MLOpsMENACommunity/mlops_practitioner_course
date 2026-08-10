"""
LEVEL 0 — The notebook.

    model = torchvision.models.resnet50(weights="DEFAULT")
    model.eval()
    img = preprocess(Image.open("dog.jpg")).unsqueeze(0)
    out = model(img)

What we have:       a prediction.
What we don't have: anything anyone else can use.

Run:  python level_0_notebook.py
"""

import os
import time
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

IMAGE = Path(__file__).parent / "assets" / "dog.jpg"


# --- this is the entire "system" -------------------------------------------
t0 = time.perf_counter()
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
load_s = time.perf_counter() - t0

preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = preprocess(Image.open(IMAGE).convert("RGB")).unsqueeze(0)   # [1, 3, 224, 224]

with torch.inference_mode():
    t0 = time.perf_counter()
    out = model(img)
    infer_ms = (time.perf_counter() - t0) * 1000

probs = torch.softmax(out, dim=-1)[0]
top5 = torch.topk(probs, k=5)

print("\nPrediction")
print("-" * 52)
for score, idx in zip(top5.values, top5.indices):
    print(f"  {weights.meta['categories'][idx]:<28} {float(score):.4f}")


# --- now the uncomfortable part --------------------------------------------
# Everything below is diagnostics, not modelling. It exists to make the failure
# mode concrete rather than rhetorical.
print(f"""
It works. Now look at what it actually is
{"-" * 52}
  model load                {load_s:6.2f} s   paid again on every kernel restart
  forward pass              {infer_ms:6.2f} ms  the only part that is "the model"
  batch size                     1        the GPU wants 32-64
  callable by                    1 person over 0 network interfaces
  process                   pid {os.getpid():<8} dies with this terminal

The failure is not accuracy. The model is fine.
The failure is that this prediction lives in one kernel session on one laptop,
and nothing else in the world has any way to ask for one.

Level 1 fixes exactly one of those lines -- the "0 network interfaces" one.
It fixes none of the others, and that is the whole point of the session.
""")
