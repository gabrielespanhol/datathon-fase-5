import logging
import time
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from src.agent.rag_pipeline import SimpleRAGPipeline

from src.features.feature_engineering import build_features
from src.monitoring.metrics import (
    FRAUD_PREDICTIONS,
    NON_FRAUD_PREDICTIONS,
    REGISTRY,
    REQUEST_COUNT,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "fraud_detection"
model = None


class AskRequest(BaseModel):
    question: str


def load_model():
    logger.info("Carregando modelo do MLflow (Production)...")

    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    try:
        model = load_model()
        logger.info("Modelo carregado com sucesso (Production)")
    except Exception as e:
        logger.error("Erro ao carregar modelo: %s", e)
        raise RuntimeError("Modelo não disponível no MLflow")

    yield

    global rag_pipeline

    rag_pipeline = SimpleRAGPipeline(
        docs_paths=[
            "docs/MODEL_CARD.md",
            "docs/PROJECT_ANALYSIS.md",
        ],
    )

    rag_pipeline.build_index()


app = FastAPI(
    title="Fraud Detection API",
    version="0.1.0",
    description="API de inferência para detecção de fraude.",
    lifespan=lifespan,
)


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


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline não inicializado.")

    return rag_pipeline.ask(request.question)
