import os
import random
from datetime import datetime

import pandas as pd


def gerar_transacao():
    valor = round(random.uniform(10, 5000), 2)
    hora = random.randint(0, 23)
    dispositivo_novo = random.choice([True, False])
    tentativas = random.randint(0, 5)
    distancia = round(random.uniform(0, 2000), 2)

    score = 0

    # Regras de risco
    if valor > 1000:
        score += 1
    if hora > 22 or hora < 6:
        score += 1
    if dispositivo_novo:
        score += 1
    if tentativas > 3 and distancia > 1000:
        score += 1

    fraude = 1 if score >= 3 else 0

    return {
        "valor": valor,
        "hora": hora,
        "dispositivo_novo": dispositivo_novo,
        "tentativas_24h": tentativas,
        "distancia_km": distancia,
        "fraude": fraude,
    }


def gerar_dataset(n):
    return pd.DataFrame([gerar_transacao() for _ in range(n)])


def salvar_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    N_AMOSTRAS = 200000

    df = gerar_dataset(N_AMOSTRAS)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"data/raw/fraud_dataset_{timestamp}.csv"

    salvar_csv(df, output_path)

    print(f"Dataset gerado com {len(df)} linhas")
    print(f"Salvo em: {output_path}")
    print("\nDistribuição de fraude:")
    print(df["fraude"].value_counts(normalize=True))
