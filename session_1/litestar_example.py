from litestar import Litestar, post, get
from litestar.di import Provide
from pydantic import BaseModel
from src.model import RideDurationModel


# ── Schema ──────────────────────────────────────────
class PredictRequest(BaseModel):
    distance: float
    passengers: int = 1


class PredictResponse(BaseModel):
    duration_min: float
    status: str = "ok"


# ── Dependency factory ───────────────────────────────
def get_model() -> RideDurationModel:
    return RideDurationModel()


# ── Handlers ─────────────────────────────────────────
@post("/predict")
async def predict(
    data: PredictRequest,
    model: RideDurationModel,  # injected
) -> PredictResponse:
    dur = model.predict([data.distance, data.passengers])
    return PredictResponse(duration_min=dur)


@get("/health")
async def health() -> dict:
    return {"status": "healthy"}


# ── App ───────────────────────────────────────────────
app = Litestar(
    route_handlers=[predict, health],
    # use_cache=True     → build the model once, reuse it for every request
    # sync_to_thread=False → the factory is non-blocking, so don't offload it
    dependencies={"model": Provide(get_model, use_cache=True, sync_to_thread=False)},
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
