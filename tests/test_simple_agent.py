from unittest.mock import Mock, patch

from src.agent.simple_agent import (
    SimpleFraudAgent,
    extract_transaction,
    infer_risk_from_factors,
    normalize_question,
)


def test_infer_risk_alto_moderado_baixo():
    assert (
        infer_risk_from_factors(
            {
                "valor": 2000,
                "hora": 5,
                "dispositivo_novo": True,
                "tentativas_24h": 0,
                "distancia_km": 0,
            }
        )
        == "Alta probabilidade de fraude"
    )

    assert (
        infer_risk_from_factors(
            {
                "valor": 1,
                "hora": 12,
                "dispositivo_novo": False,
                "tentativas_24h": 3,
                "distancia_km": 0,
            }
        )
        == "Risco moderado"
    )

    assert (
        infer_risk_from_factors(
            {
                "valor": 1,
                "hora": 12,
                "dispositivo_novo": False,
                "tentativas_24h": 0,
                "distancia_km": 0,
            }
        )
        == "Baixo risco de fraude"
    )


def test_infer_risk_cobre_demais_condicoes():
    assert (
        infer_risk_from_factors(
            {
                "valor": 1,
                "hora": 23,
                "dispositivo_novo": False,
                "tentativas_24h": 0,
                "distancia_km": 1000,
            }
        )
        == "Risco moderado"
    )


def test_normalize_question():
    assert normalize_question("  Fraude13h Valor2000  ") == "fraude 13h valor 2000"
    assert normalize_question("valor 13 h") == "valor 13h"
    assert normalize_question("13horas") == "13 horas"


def test_extract_transaction_com_defaults_e_hora_por_campo():
    result = extract_transaction("hora 22")

    assert result == {
        "valor": 1,
        "hora": 22,
        "dispositivo_novo": False,
        "tentativas_24h": 0,
        "distancia_km": 0,
    }


def test_extract_transaction_variacoes_dispositivo_novo():
    assert extract_transaction("dispositivo_novo true")["dispositivo_novo"] is True
    assert extract_transaction("dispositivo_novo: true")["dispositivo_novo"] is True
    assert extract_transaction("dispositivo_novo=true")["dispositivo_novo"] is True


def test_classify_intent_prediction_docs_hybrid_default():
    agent = SimpleFraudAgent(model=Mock(), rag_pipeline=Mock())

    assert agent.classify_intent("isso é fraude valor 100?") == "prediction"
    assert agent.classify_intent("quais features do modelo?") == "docs"
    assert (
        agent.classify_intent("fraude valor 100 e quais features do modelo?")
        == "hybrid"
    )
    assert agent.classify_intent("olá") == "docs"


@patch("src.agent.simple_agent.explain_risk_tool")
@patch("src.agent.simple_agent.predict_fraud_tool")
def test_run_prediction(mock_predict, mock_explain):
    mock_predict.return_value = {"prediction": 1, "probability": 0.9, "label": "fraude"}
    mock_explain.return_value = "Fatores de risco identificados: valor elevado."

    agent = SimpleFraudAgent(model=Mock(), rag_pipeline=Mock())

    result = agent.run("fraude valor 2500 hora 12")

    assert result["intent"] == "prediction"
    assert result["tools_used"] == ["predict_fraud_tool", "explain_risk_tool"]
    assert result["prediction"]["label"] == "fraude"
    assert result["explanation"] == "Fatores de risco identificados: valor elevado."
    assert result["answer"] == (
        "Risco moderado. Motivo: Fatores de risco identificados: valor elevado."
    )


@patch("src.agent.simple_agent.rag_docs_tool")
def test_run_docs(mock_rag):
    mock_rag.return_value = {
        "answer": "Documentação do modelo",
        "sources": ["doc1"],
    }

    agent = SimpleFraudAgent(model=Mock(), rag_pipeline=Mock())

    result = agent.run("explique o treinamento do modelo")

    assert result == {
        "intent": "docs",
        "tools_used": ["rag_docs_tool"],
        "answer": "Documentação do modelo",
        "sources": ["doc1"],
    }


@patch("src.agent.simple_agent.rag_docs_tool")
def test_run_docs_sem_sources(mock_rag):
    mock_rag.return_value = {"answer": "Sem fontes"}

    agent = SimpleFraudAgent(model=Mock(), rag_pipeline=Mock())

    result = agent.run("documentação")

    assert result["sources"] == []


@patch("src.agent.simple_agent.rag_docs_tool")
@patch("src.agent.simple_agent.explain_risk_tool")
@patch("src.agent.simple_agent.predict_fraud_tool")
def test_run_hybrid(mock_predict, mock_explain, mock_rag):
    mock_predict.return_value = {
        "prediction": 1,
        "probability": 0.95,
        "label": "fraude",
    }
    mock_explain.return_value = "Fatores de risco identificados: valor elevado."
    mock_rag.return_value = {
        "answer": "O modelo usa features transacionais.",
        "sources": ["doc1"],
    }

    agent = SimpleFraudAgent(model=Mock(), rag_pipeline=Mock())

    result = agent.run("fraude valor 2500 e quais features do modelo?")

    assert result["intent"] == "hybrid"
    assert result["tools_used"] == [
        "predict_fraud_tool",
        "explain_risk_tool",
        "rag_docs_tool",
    ]
    assert result["prediction"]["label"] == "fraude"
    assert result["docs_answer"] == "O modelo usa features transacionais."
    assert result["sources"] == ["doc1"]
    assert result["answer"] == (
        "Risco moderado. Motivo: Fatores de risco identificados: valor elevado.\n\n"
        "Complemento: O modelo usa features transacionais."
    )
