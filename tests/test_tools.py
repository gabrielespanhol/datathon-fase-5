from unittest.mock import Mock, patch

import pandas as pd

from src.agent.tools import (
    explain_risk_tool,
    predict_fraud_tool,
    rag_docs_tool,
)


class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return [self.prediction]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


@patch("src.agent.tools.build_features")
def test_predict_fraud_tool_fraude(mock_build_features):
    transaction = {"valor": 3000}
    features = pd.DataFrame([{"valor": 3000}])
    mock_build_features.return_value = features

    model = FakeModel(prediction=1)

    result = predict_fraud_tool(model, transaction)

    mock_build_features.assert_called_once()
    assert result == {
        "prediction": 1,
        "probability": 0.8,
        "label": "fraude",
    }


@patch("src.agent.tools.build_features")
def test_predict_fraud_tool_nao_fraude(mock_build_features):
    transaction = {"valor": 100}
    features = pd.DataFrame([{"valor": 100}])
    mock_build_features.return_value = features

    model = FakeModel(prediction=0)

    result = predict_fraud_tool(model, transaction)

    assert result == {
        "prediction": 0,
        "probability": 0.8,
        "label": "não fraude",
    }


def test_explain_risk_tool_sem_fatores():
    transaction = {
        "valor": 100,
        "hora": 12,
        "dispositivo_novo": False,
        "tentativas_24h": 1,
        "distancia_km": 10,
    }

    result = explain_risk_tool(transaction)

    assert result == "Nenhum fator de risco relevante foi identificado."


def test_explain_risk_tool_com_todos_os_fatores_hora_madrugada():
    transaction = {
        "valor": 2000,
        "hora": 5,
        "dispositivo_novo": True,
        "tentativas_24h": 3,
        "distancia_km": 1000,
    }

    result = explain_risk_tool(transaction)

    assert result == (
        "Fatores de risco identificados: valor elevado, horário incomum, "
        "uso de dispositivo novo, múltiplas tentativas recentes, distância elevada."
    )


def test_explain_risk_tool_hora_noite():
    transaction = {
        "valor": 100,
        "hora": 23,
        "dispositivo_novo": False,
        "tentativas_24h": 1,
        "distancia_km": 10,
    }

    result = explain_risk_tool(transaction)

    assert result == "Fatores de risco identificados: horário incomum."


def test_rag_docs_tool():
    rag_pipeline = Mock()
    rag_pipeline.ask.return_value = {
        "answer": "Resposta teste",
        "sources": ["doc1"],
    }

    result = rag_docs_tool(rag_pipeline, "Pergunta teste?")

    rag_pipeline.ask.assert_called_once_with("Pergunta teste?")
    assert result == {
        "answer": "Resposta teste",
        "sources": ["doc1"],
    }
