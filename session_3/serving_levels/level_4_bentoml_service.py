"""
LEVEL 4 — A purpose-built serving runtime (BentoML).

Compare this file against level_2_batching.py. They do the same job.

  Level 2, by hand                      Level 4, BentoML
  ------------------------------------  -----------------------------------
  ~90 lines between the SERVING LAYER   batchable=True
  markers: queue, window, futures,      max_batch_size=8
  backpressure, partial-batch failure   max_latency_ms=<your SLA>
  uvicorn --workers 1 (batcher state)   workers=1 in @bentoml.service
  hand-rolled /metrics endpoint         Prometheus at /metrics, free
  MODEL_VERSION = "v1" constant         model store: versioned, hot-swappable
  write your own Dockerfile             bentoml build && bentoml containerize
  UploadFile + io.BytesIO + try/except  annotate the param as PIL.Image

The five [COST-n] problems annotated in Level 2 become three config lines.
That is the whole argument for not writing it yourself.

Setup:  python level_4_save_model.py
Run:    bentoml serve level_4_bentoml_service:ResNet50 --port 8004
Then:   python bench.py --port 8004 --field images -c 16 -n 60
        curl -s http://localhost:8004/metrics | grep -i batch
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

import bentoml

CATEGORIES = ResNet50_Weights.DEFAULT.meta["categories"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH = 32 if DEVICE == "cuda" else 8       # same CPU cliff as Level 2

_preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@bentoml.service(
    name="resnet50_classifier",
    # workers=1 for exactly the reason Level 2 needed --workers 1: one process
    # owns the accelerator. Scale with replicas, not workers.
    workers=1,
    resources={"cpu": "2"},
    # concurrency is BentoML's admission limit -- its version of Level 2's
    # QUEUE_MAXSIZE, and it answers with the same 503. It MUST be >= the batch
    # size, or requests get rejected before enough of them accumulate to fill a
    # batch and you have paid for a batching system that never batches. Leave it
    # at the default and this service returns "process is overloaded" under the
    # very load it was built for.
    traffic={"timeout": 60, "concurrency": 64},
)
class ResNet50:
    # A versioned entry from the model store, resolved to a concrete hash at
    # startup. This is the model versioning Level 1 did not have: canarying v2
    # is a second service pinned to another tag, not a second server you operate
    # by hand. `bentoml models list` shows every version you have built.
    bento_model = bentoml.models.BentoModel("resnet50:latest")

    def __init__(self) -> None:
        self.model = resnet50()
        self.model.load_state_dict(
            torch.load(self.bento_model.path_of("model.pt"), weights_only=True)
        )
        self.model.eval().to(DEVICE)

    # ---- THE ENTIRE LEVEL 2 SERVING LAYER, AS FOUR KEYWORD ARGUMENTS -------
    @bentoml.api(
        batchable=True,             # [COST-1] queue + window: handled
        batch_dim=0,                # [COST-2] response correlation: handled
        max_batch_size=MAX_BATCH,   # [COST-4] backpressure: handled
        # NOT the same knob as Level 2's BATCH_WINDOW_MS. This is a latency SLA:
        # BentoML's dispatcher measures how long batches actually take, tunes
        # the wait window itself, and sheds requests it predicts cannot finish
        # in time. Set this to 8 "to match Level 2" and the server 503s almost
        # everything, because a batch of 8 ResNet50 images takes ~100ms on CPU.
        # You declare the deadline; the framework picks the window.
        max_latency_ms=10_000,
    )
    @torch.inference_mode()
    def predict(self, images: list[Image.Image]) -> list[dict]:
        """Each client POSTs ONE image. BentoML merges concurrent requests into
        `images`, calls this once, then slices the returned list back to the
        client that asked for each element. Watch it happen: the `batch_size`
        field below reports how many requests were merged into your call.

        Annotating the parameter as PIL.Image is all it takes to accept a
        multipart upload and get a decoded image -- no UploadFile, no
        io.BytesIO, no hand-written 400 on a corrupt JPEG.

        Note what did NOT get solved: preprocessing is still Python, still on
        the request path, still costing what Level 3 measured. No framework
        makes JPEG decode free.
        """
        batch = torch.from_numpy(
            np.stack([_preprocess(im.convert("RGB")).numpy() for im in images])
        ).to(DEVICE)

        probs = torch.softmax(self.model(batch), dim=-1)
        top5 = torch.topk(probs, k=5, dim=-1)
        return [
            {
                "version": self.bento_model.tag.version,
                "batch_size": len(images),      # the Level 2 metric, for free
                "predictions": [
                    {"label": CATEGORIES[i], "score": round(float(s), 4)}
                    for s, i in zip(scores, idxs)
                ],
            }
            for scores, idxs in zip(top5.values.cpu(), top5.indices.cpu())
        ]
