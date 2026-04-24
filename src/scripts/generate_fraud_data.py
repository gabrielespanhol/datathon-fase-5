import hashlib
import json
import os
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

OUTPUT_PATH = "data/raw/fraud_dataset.csv"
N_AMOSTRAS = 200000
RANDOM_SEED = 42


def gerar_transacao():
    valor = round(random.uniform(10, 5000), 2)
    hora = random.randint(0, 23)
    dispositivo_novo = random.choice([True, False])
    tentativas = random.randint(0, 5)
    distancia = round(random.uniform(0, 2000), 2)

    score = 0

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


def gerar_dataset(n: int) -> pd.DataFrame:
    return pd.DataFrame([gerar_transacao() for _ in range(n)])


def salvar_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def file_md5(path: str) -> str:
    hash_md5 = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def main() -> None:
    df = gerar_dataset(N_AMOSTRAS)
    salvar_csv(df, OUTPUT_PATH)

    data_hash = file_md5(OUTPUT_PATH)

    metadata = {
        "data_version": data_hash,
        "generated_at": datetime.utcnow().isoformat(),
        "n_rows": len(df),
        "fraud_rate": float(df["fraude"].mean()),
        "raw_path": OUTPUT_PATH,
    }

    metadata_path = Path("data/raw/dataset_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Dataset gerado com {len(df)} linhas")
    print(f"Data version: {data_hash}")
    print(f"Salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
