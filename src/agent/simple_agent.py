import re

from src.agent.tools import (
    explain_risk_tool,
    predict_fraud_tool,
    rag_docs_tool,
)


def normalize_question(question: str) -> str:
    q = question.lower().strip()

    # separa letras e números: fraude13h -> fraude 13h
    q = re.sub(r"([a-zA-ZçÇãõáéíóúâêôàü]+)(\d)", r"\1 \2", q)
    q = re.sub(r"(\d)([a-zA-ZçÇãõáéíóúâêôàü]+)", r"\1 \2", q)

    # junta de volta padrão de hora: 13 h -> 13h
    q = re.sub(r"\b(\d{1,2})\s*h\b", r"\1h", q)

    return q


def extract_transaction(question: str) -> dict:
    q = normalize_question(question)

    def get_number(pattern: str, default: float = 0) -> float:
        match = re.search(rf"(?:{pattern})\s*[:=]?\s*(\d+(?:\.\d+)?)", q, re.I)
        return float(match.group(1)) if match else default

    hora_match = re.search(r"\b(\d{1,2})h\b", q)
    hora = int(hora_match.group(1)) if hora_match else int(get_number("hora", 12))

    return {
        "valor": get_number("valor", 1),
        "hora": hora,
        "dispositivo_novo": any(
            term in q
            for term in [
                "dispositivo novo",
                "dispositivo_novo true",
                "dispositivo_novo: true",
                "dispositivo_novo=true",
            ]
        ),
        "tentativas_24h": int(get_number("tentativas_24h|tentativas", 0)),
        "distancia_km": get_number("distancia_km|distância|distancia", 0),
    }


class SimpleFraudAgent:
    def __init__(self, model, rag_pipeline):
        self.model = model
        self.rag_pipeline = rag_pipeline

    def classify_intent(self, question: str) -> str:
        q = normalize_question(question)

        has_compact_hour = re.search(r"\b\d{1,2}h\b", q) is not None

        has_transaction_data = has_compact_hour or any(
            field in q
            for field in [
                "valor",
                "hora",
                "tentativas",
                "distancia",
                "distância",
                "dispositivo",
            ]
        )

        asks_prediction = any(
            word in q
            for word in [
                "fraude",
                "risco",
                "prever",
                "predição",
                "predicao",
                "classificar",
                "transação",
                "transacao",
            ]
        )

        asks_docs = any(
            word in q
            for word in [
                "modelo",
                "treino",
                "treinamento",
                "métrica",
                "metrica",
                "mlflow",
                "model card",
                "documentação",
                "documentacao",
                "features",
                "variáveis",
                "variaveis",
                "pipeline",
            ]
        )

        if asks_prediction and has_transaction_data and asks_docs:
            return "hybrid"

        if asks_prediction and has_transaction_data:
            return "prediction"

        if asks_docs:
            return "docs"

        return "docs"

    def _run_prediction(self, question: str) -> dict:
        transaction = extract_transaction(question)

        prediction = predict_fraud_tool(self.model, transaction)
        explanation = explain_risk_tool(transaction)

        return {
            "intent": "prediction",
            "tools_used": [
                "predict_fraud_tool",
                "explain_risk_tool",
            ],
            "transaction": transaction,
            "prediction": prediction,
            "explanation": explanation,
            "answer": (
                f"A transação foi classificada como {prediction['label']} "
                f"com probabilidade de {prediction['probability']:.2%}. "
                f"{explanation}"
            ),
        }

    def _run_docs(self, question: str) -> dict:
        rag_result = rag_docs_tool(self.rag_pipeline, question)

        return {
            "intent": "docs",
            "tools_used": ["rag_docs_tool"],
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
        }

    def _run_hybrid(self, question: str) -> dict:
        transaction = extract_transaction(question)

        prediction = predict_fraud_tool(self.model, transaction)
        explanation = explain_risk_tool(transaction)
        rag_result = rag_docs_tool(self.rag_pipeline, question)

        return {
            "intent": "hybrid",
            "tools_used": [
                "predict_fraud_tool",
                "explain_risk_tool",
                "rag_docs_tool",
            ],
            "transaction": transaction,
            "prediction": prediction,
            "explanation": explanation,
            "docs_answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
            "answer": (
                f"A transação foi classificada como {prediction['label']} "
                f"com probabilidade de {prediction['probability']:.2%}. "
                f"{explanation}\n\n"
                f"Com base na documentação do projeto: {rag_result['answer']}"
            ),
        }

    def run(self, question: str) -> dict:
        intent = self.classify_intent(question)

        if intent == "prediction":
            return self._run_prediction(question)

        if intent == "hybrid":
            return self._run_hybrid(question)

        return self._run_docs(question)
