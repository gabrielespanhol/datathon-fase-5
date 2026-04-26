import re

from src.agent.tools import (
    explain_risk_tool,
    predict_fraud_tool,
    rag_docs_tool,
)


def extract_transaction(question: str) -> dict:
    def get_number(name: str, default: float = 0) -> float:
        match = re.search(rf"{name}\s*[:=]?\s*(\d+(\.\d+)?)", question, re.I)
        return float(match.group(1)) if (match and match.group(1)) else default

    return {
        "valor": get_number("valor", 1),
        "hora": int(get_number("hora", 12)),
        "dispositivo_novo": "dispositivo novo" in question.lower()
        or "dispositivo_novo true" in question.lower(),
        "tentativas_24h": int(get_number("tentativas_24h|tentativas", 0)),
        "distancia_km": get_number("distancia_km|distância|distancia", 0),
    }


class SimpleFraudAgent:
    def __init__(self, model, rag_pipeline):
        self.model = model
        self.rag_pipeline = rag_pipeline

    def run(self, question: str) -> dict:
        lower_question = question.lower()

        if any(
            word in lower_question
            for word in ["modelo", "treino", "métrica", "mlflow", "model card"]
        ):
            rag_result = rag_docs_tool(self.rag_pipeline, question)
            return {
                "tool_used": "rag_docs_tool",
                "answer": rag_result["answer"],
                "sources": rag_result.get("sources", []),
            }

        if any(
            word in lower_question
            for word in ["fraude", "transação", "risco", "prever", "predição"]
        ):
            transaction = extract_transaction(question)

            prediction = predict_fraud_tool(self.model, transaction)
            explanation = explain_risk_tool(transaction)

            return {
                "tool_used": "predict_fraud_tool + explain_risk_tool",
                "transaction": transaction,
                "prediction": prediction,
                "explanation": explanation,
                "answer": (
                    f"A transação foi classificada como {prediction['label']} "
                    f"com probabilidade {prediction['probability']:.2%}. "
                    f"{explanation}"
                ),
            }

        rag_result = rag_docs_tool(self.rag_pipeline, question)
        return {
            "tool_used": "rag_docs_tool",
            "answer": rag_result["answer"],
            "sources": rag_result.get("sources", []),
        }
