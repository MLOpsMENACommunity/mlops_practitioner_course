from litestar import Litestar, post, get
from litestar.di import Provide
from pydantic import BaseModel, Field
from src.model import RideDurationModel


# ── Schema ──────────────────────────────────────────
class PredictRequest(BaseModel):
    distance_km: float = Field(..., gt=0)
    passengers: int = Field(1, ge=1, le=8)


class PredictResponse(BaseModel):
    duration_min: float


# ── Dependency factory ───────────────────────────────
def get_model() -> RideDurationModel:
    return RideDurationModel()


# ── Handlers ─────────────────────────────────────────
@post("/predict")
async def predict(
    data: PredictRequest,
    model: RideDurationModel,  # injected
) -> PredictResponse:
    dur = model.predict([data.distance_km, data.passengers])
    return PredictResponse(duration_min=round(dur, 2))


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
