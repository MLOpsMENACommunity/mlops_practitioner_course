from fastapi import FastAPI
from pydantic import BaseModel
from src.model import RideDurationModel

app = FastAPI(title="Ride Duration API")
model = RideDurationModel()


class PredictRequest(BaseModel):
    distance: float
    passengers: int = 1


@app.post("/predict")
async def predict(req: PredictRequest) -> dict:
    duration = model.predict([req.distance, req.passengers])
    return {"duration_min": duration, "status": "ok"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# from fastapi.responses import RedirectResponse

# @app.get("/")
# async def root():
#     return RedirectResponse(url="/docs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
