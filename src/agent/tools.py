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
    factors = []

    if transaction["valor"] >= 2000:
        factors.append("valor elevado")

    if transaction["hora"] <= 5 or transaction["hora"] >= 23:
        factors.append("horário incomum")

    if transaction["dispositivo_novo"]:
        factors.append("uso de dispositivo novo")

    if transaction["tentativas_24h"] >= 3:
        factors.append("múltiplas tentativas recentes")

    if transaction["distancia_km"] >= 1000:
        factors.append("distância elevada")

    if not factors:
        return "Nenhum fator de risco relevante foi identificado."

    return "Fatores de risco identificados: " + ", ".join(factors) + "."


def rag_docs_tool(rag_pipeline, question: str) -> dict:
    return rag_pipeline.ask(question)
