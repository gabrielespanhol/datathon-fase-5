import logging
import time
from contextlib import asynccontextmanager
import os
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
from src.agent.simple_agent import SimpleFraudAgent


logger = logging.getLogger(__name__)

model = None
rag_pipeline = None
agent = None


class AskRequest(BaseModel):
    question: str


class AgentRequest(BaseModel):
    question: str


def load_model():
    model_path = "./src/models/fraud_detection"
    logger.info("Carregando modelo de: %s", model_path)
    return mlflow.sklearn.load_model(model_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global rag_pipeline
    global agent

    rag_pipeline = SimpleRAGPipeline(
        docs_paths=[
            "docs/MODEL_CARD.md",
            "DATASET.md",
        ],
    )

    try:
        model = load_model()
        logger.info("Modelo carregado com sucesso (Production)")

    except Exception as e:
        logger.error("Erro ao carregar modelo: %s", e)
        raise RuntimeError("Modelo não disponível no MLflow")

    try:
        rag_pipeline.build_index()
        logger.info("Índice RAG construído com sucesso")

    except Exception as e:
        logger.error("Erro ao construir índice RAG: %s", e)
        raise RuntimeError("Falha na indexação do pipeline RAG")

    try:
        agent = SimpleFraudAgent(model=model, rag_pipeline=rag_pipeline)
        logger.info("Agente de fraude inicializado com sucesso")

    except Exception as e:
        logger.error("Erro ao inicializar o agente: %s", e)
        raise RuntimeError("Não foi possível instanciar o SimpleFraudAgent")

    yield


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


@app.post("/agent")
def run_agent(request: AgentRequest) -> dict:
    if agent is None:
        raise HTTPException(status_code=503, detail="Agente não inicializado.")

    return agent.run(request.question)
