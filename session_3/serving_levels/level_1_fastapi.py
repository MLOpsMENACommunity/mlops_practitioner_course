"""
LEVEL 1 — FastAPI alone.

What we gained: a real HTTP endpoint. Anyone on the network can call it.
                OpenAPI docs at /docs. Request validation. Genuine progress,
                and for many projects genuinely enough.

What we did NOT gain is the point of this file. It is instrumented so that
`python bench.py --url http://localhost:8001/predict -c 32` makes all four
problems show up as numbers on /metrics rather than as claims on a slide:

  1. batch size is always 1              -> inference_batch_size_avg == 1.00
  2. preprocessing is on the request path -> preprocess_ms > forward_ms (!)
  3. no model versioning                  -> /version is hardcoded, restart to change
  4. eager FP32 PyTorch                   -> see level_3_optimize.py

Run:  uvicorn level_1_fastapi:app --port 8001
Then: python bench.py --port 8001 -c 32 -n 200
"""

import io
import time
from contextlib import asynccontextmanager

import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image
from pydantic import BaseModel
from torchvision.models import ResNet50_Weights, resnet50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VERSION = "v1"          # problem 3: this is a constant. Changing it = redeploy.


class Metrics(BaseModel):
    requests: int = 0
    preprocess_s: float = 0.0
    forward_s: float = 0.0
    # tracked so Level 1 and Level 2 expose the *same* metric names
    batches: int = 0
    batched_items: int = 0


class Classifier:
    def __init__(self) -> None:
        weights = ResNet50_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        # Problem 4: eager, FP32, op-by-op through the Python interpreter.
        self.model = resnet50(weights=weights).eval().to(DEVICE)
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, raw: bytes) -> torch.Tensor:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.preprocess(img)

    @torch.inference_mode()
    def predict(self, tensor: torch.Tensor) -> list[dict]:
        # Problem 1: unsqueeze(0) -> batch of exactly one, every single time.
        # The GPU is built to do 32-64 of these at once. Utilization ~8-15%.
        batch = tensor.unsqueeze(0).to(DEVICE)
        probs = torch.softmax(self.model(batch), dim=-1)[0]
        top5 = torch.topk(probs, k=5)
        return [
            {"label": self.categories[i], "score": round(float(s), 4)}
            for s, i in zip(top5.values.cpu(), top5.indices.cpu())
        ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.metrics = Metrics()
    app.state.model = Classifier()
    app.state.model.predict(torch.zeros(3, 224, 224))     # warm-up
    yield


app = FastAPI(title=f"ResNet50 — Level 1 ({MODEL_VERSION})", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    raw = await file.read()
    m: Metrics = app.state.metrics

    # Problem 2: JPEG decode + resize + normalize runs HERE, on the request
    # path, in Python. Note this handler is `async` but does blocking CPU work
    # -- so it holds the event loop and blocks *other* connections from being
    # accepted while it decodes. Benchmark it: preprocess usually costs MORE
    # wall-clock than the forward pass.
    t0 = time.perf_counter()
    try:
        tensor = app.state.model.encode(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")
    t1 = time.perf_counter()

    preds = app.state.model.predict(tensor)
    t2 = time.perf_counter()

    m.requests += 1
    m.preprocess_s += t1 - t0
    m.forward_s += t2 - t1
    m.batches += 1
    m.batched_items += 1        # always 1. That is the finding.

    return {
        "version": MODEL_VERSION,
        "predictions": preds,
        "preprocess_ms": round((t1 - t0) * 1000, 2),
        "forward_ms": round((t2 - t1) * 1000, 2),
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": DEVICE, "version": MODEL_VERSION}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    m: Metrics = app.state.metrics
    n = max(m.requests, 1)
    return (
        f"level 1\n"
        f"inference_requests_total {m.requests}\n"
        f"inference_batch_size_avg {m.batched_items / max(m.batches, 1):.2f}\n"
        f"preprocess_seconds_avg {m.preprocess_s / n:.5f}\n"
        f"forward_seconds_avg {m.forward_s / n:.5f}\n"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
