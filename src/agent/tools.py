import pandas as pd

from src.features.feature_engineering import build_features


def predict_fraud_tool(model, transaction: dict) -> dict:
    print("USANDO MODELO:", type(model))
    print("TRANSAÇÃO RECEBIDA:", transaction)
    raw_df = pd.DataFrame([transaction])
    features_df = build_features(raw_df)

    print("FEATURES ENVIADAS AO MODELO:")
    print(features_df)

    prediction = int(model.predict(features_df)[0])
    probability = float(model.predict_proba(features_df)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
        "label": "fraude" if prediction == 1 else "não fraude",
    }


def explain_risk_tool(transaction: dict) -> str:
    reasons = []

    if transaction["valor"] > 2000:
        reasons.append("valor alto")
    if transaction["hora"] < 6:
        reasons.append("transação em horário incomum")
    if transaction["dispositivo_novo"]:
        reasons.append("uso de dispositivo novo")
    if transaction["tentativas_24h"] >= 5:
        reasons.append("muitas tentativas nas últimas 24h")
    if transaction["distancia_km"] > 1000:
        reasons.append("distância elevada em relação ao padrão esperado")

    if not reasons:
        return "Nenhum fator de risco forte foi identificado."

    return "Fatores de risco identificados: " + ", ".join(reasons) + "."


def rag_docs_tool(rag_pipeline, question: str) -> dict:
    return rag_pipeline.ask(question)
