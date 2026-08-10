"""
LEVEL 2 — FastAPI + a hand-written dynamic batcher.

  Request A -+
  Request B -+--> [ queue: wait 8ms OR until 32 items ] --> model(batch) --> split
  Request C -+

What we gained: the single biggest win available. Same hardware, same model.
                On a GPU: commonly 5-10x throughput, because the device was
                sitting idle at 8-15% utilization waiting for work.
                On a CPU (what most laptops in this room have): expect ~1.4x
                throughput but a much better tail -- measured here, p99 went
                1220ms -> 466ms. A CPU was never idle, so the win comes from
                controlling contention rather than from filling the device.
                Run bench.py against Level 1 and Level 2 and see for yourself.

What it cost:   the ~90 lines between the two SERVING LAYER markers. Read them.
                Every one of the five hard parts is labelled [COST-n] inline:

                  [COST-1] async queue with a timeout
                  [COST-2] correlating each response back to its request
                  [COST-3] variable input shapes
                  [COST-4] backpressure when the queue outruns the GPU
                  [COST-5] partial batch failure

                This is where self-rolled serving stacks develop their worst
                bugs -- and it is one config line in every framework in Level 4.

Run:  uvicorn level_2_batching:app --port 8002 --workers 1
      ^ workers MUST be 1: the batcher holds in-process state and the GPU.
        Scale with replicas/pods, not uvicorn workers.
Then: python bench.py --port 8002 -c 32 -n 200      # compare against level 1
"""

import asyncio
import io
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Bigger batch = more throughput, worse tail latency -- but only up to the point
# where the device stops absorbing it for free. Measured on an M-series CPU,
# ResNet50 at bs=32 takes ~2.5s per batch (a memory-bandwidth cliff) while bs=8
# is ~100ms. Capping at 8 on CPU is not a style choice; run level_3_optimize.py
# and read the bs=32 column to see what the uncapped version costs you.
MAX_BATCH_SIZE = 32 if DEVICE == "cuda" else 8
BATCH_WINDOW_MS = 8           # how long we are willing to wait to fill a batch
QUEUE_MAXSIZE = 256           # [COST-4] bound it, or you queue until you OOM
MODEL_VERSION = "v1"


@dataclass
class Metrics:
    requests: int = 0
    rejected: int = 0
    batches: int = 0
    batched_items: int = 0
    preprocess_s: float = 0.0
    queue_s: float = 0.0
    forward_s: float = 0.0


class Classifier:
    def __init__(self) -> None:
        weights = ResNet50_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.model = resnet50(weights=weights).eval().to(DEVICE)
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, raw: bytes) -> torch.Tensor:
        """[COST-3] Resize+CenterCrop force every image to [3,224,224] so the
        batch stacks. A model with genuinely variable input shapes (detection,
        ASR, seq2seq) cannot batch this naively -- you need bucketing or padding."""
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.preprocess(img)

    @torch.inference_mode()
    def predict_batch(self, tensors: list[torch.Tensor]) -> list[list[dict]]:
        """THE forward pass. One call, N images. This is the whole point.

        Safe here ONLY because every request costs the same compute: a batch of
        32 takes roughly as long as a batch of 1. That assumption is exactly
        what breaks for autoregressive LLMs, where each sequence decodes a
        different number of steps -- which is why vLLM exists.
        """
        batch = torch.stack(tensors).to(DEVICE)
        probs = torch.softmax(self.model(batch), dim=-1)
        top5 = torch.topk(probs, k=5, dim=-1)
        return [
            [{"label": self.categories[i], "score": round(float(s), 4)}
             for s, i in zip(scores, idxs)]
            for scores, idxs in zip(top5.values.cpu(), top5.indices.cpu())
        ]


# ===========================================================================
# SERVING LAYER — begins
# Everything to the next marker is what TorchServe / BentoML / Triton hand you
# for free. Written by hand here so the cost is visible rather than asserted.
# ===========================================================================
@dataclass
class Job:
    tensor: torch.Tensor
    # [COST-2] The future is how a result finds its way back to the right
    # client. It MUST come from the *running* loop -- binding it at import time
    # attaches every future to a loop uvicorn never runs, and you get
    # "got Future attached to a different loop" on request #1.
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    enqueued_at: float = field(default_factory=time.perf_counter)


class Batcher:
    def __init__(self, model: Classifier, metrics: Metrics) -> None:
        self.model = model
        self.metrics = metrics
        self.queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, tensor: torch.Tensor) -> list[dict]:
        job = Job(tensor=tensor)
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            # [COST-4] Backpressure. Shedding load with a fast 503 is a feature:
            # the alternative is unbounded queueing, climbing latency, then OOM.
            self.metrics.rejected += 1
            raise HTTPException(status_code=503, detail="overloaded, retry")
        return await job.future

    async def _collect(self) -> list[Job]:
        """[COST-1] Block for the first job, then greedily take whatever lands
        inside the window. The timeout is what bounds added latency; without it
        a quiet server would wait forever for a batch that never fills."""
        jobs = [await self.queue.get()]
        deadline = time.perf_counter() + BATCH_WINDOW_MS / 1000
        while len(jobs) < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                jobs.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return jobs

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = await self._collect()
            now = time.perf_counter()
            self.metrics.queue_s += sum(now - j.enqueued_at for j in jobs)

            t0 = time.perf_counter()
            try:
                # Run the GPU work in a thread, off the event loop -- otherwise
                # HTTP stalls for the whole forward pass and you have serialised
                # the very thing you just parallelised.
                results = await loop.run_in_executor(
                    None, self.model.predict_batch, [j.tensor for j in jobs]
                )
                for job, res in zip(jobs, results):
                    if not job.future.done():       # client may have disconnected
                        job.future.set_result(res)
            except Exception as exc:
                # [COST-5] Partial batch failure. One malformed tensor must not
                # kill the batcher task -- if this loop dies, every subsequent
                # request hangs forever on a future nobody will ever resolve.
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)

            self.metrics.forward_s += time.perf_counter() - t0
            self.metrics.batches += 1
            self.metrics.batched_items += len(jobs)
# ===========================================================================
# SERVING LAYER — ends
# ===========================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.metrics = Metrics()
    app.state.model = Classifier()
    app.state.batcher = Batcher(app.state.model, app.state.metrics)
    app.state.batcher.start()
    await app.state.batcher.submit(torch.zeros(3, 224, 224))     # warm-up
    yield
    await app.state.batcher.stop()


app = FastAPI(title=f"ResNet50 — Level 2 ({MODEL_VERSION})", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    raw = await file.read()
    loop = asyncio.get_running_loop()
    m: Metrics = app.state.metrics

    # Unlike Level 1, preprocessing goes to a threadpool so the event loop stays
    # free to accept connections. It is still Python, still on the request path,
    # and still fighting the GIL -- Level 3 shows it becomes the bottleneck.
    t0 = time.perf_counter()
    try:
        tensor = await loop.run_in_executor(None, app.state.model.encode, raw)
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")
    m.preprocess_s += time.perf_counter() - t0

    m.requests += 1
    preds = await app.state.batcher.submit(tensor)
    return {"version": MODEL_VERSION, "predictions": preds}


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": DEVICE, "version": MODEL_VERSION}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    m: Metrics = app.state.metrics
    n = max(m.requests, 1)
    return (
        f"level 2\n"
        f"inference_requests_total {m.requests}\n"
        f"inference_rejected_total {m.rejected}\n"
        f"inference_batches_total {m.batches}\n"
        f"inference_batch_size_avg {m.batched_items / max(m.batches, 1):.2f}\n"
        f"preprocess_seconds_avg {m.preprocess_s / n:.5f}\n"
        f"queue_seconds_avg {m.queue_s / max(m.batched_items, 1):.5f}\n"
        f"forward_seconds_per_batch_avg {m.forward_s / max(m.batches, 1):.5f}\n"
    )
