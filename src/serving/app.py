import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.features.feature_engineering import build_features
from src.monitoring.metrics import (
    FRAUD_PREDICTIONS,
    NON_FRAUD_PREDICTIONS,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Modelo não encontrado em: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    yield  # aplicação roda aqui


app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="API de inferência para detecção de fraude.",
    lifespan=lifespan,
)

MODEL_PATH = Path(
    "mlruns/1/models/m-b9bd8edb6d8f4e459e9de9ae28543096/artifacts/model.pkl"
)


model: Any | None = None


class TransactionRequest(BaseModel):
    valor: float = Field(..., gt=0)
    hora: int = Field(..., ge=0, le=23)
    dispositivo_novo: bool
    tentativas_24h: int = Field(..., ge=0)
    distancia_km: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Fraud Detection API online"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest) -> PredictionResponse:
    global model

    if model is None:
        REQUEST_ERRORS.inc()
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    REQUEST_COUNT.inc()
    start_time = time.perf_counter()

    try:
        raw_df = pd.DataFrame(
            [
                {
                    "valor": request.valor,
                    "hora": request.hora,
                    "dispositivo_novo": request.dispositivo_novo,
                    "tentativas_24h": request.tentativas_24h,
                    "distancia_km": request.distancia_km,
                }
            ]
        )

        features_df = build_features(raw_df)

        prediction = int(model.predict(features_df)[0])
        probability = float(model.predict_proba(features_df)[0][1])

        if prediction == 1:
            FRAUD_PREDICTIONS.inc()
        else:
            NON_FRAUD_PREDICTIONS.inc()

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
        )

    except Exception as exc:
        REQUEST_ERRORS.inc()
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao gerar predição: {str(exc)}",
        ) from exc

    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start_time)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
