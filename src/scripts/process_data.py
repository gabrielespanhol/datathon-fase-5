from pathlib import Path

import pandas as pd

from src.features.feature_engineering import build_features

RAW_DATA_PATH = Path("data/raw/fraud_dataset.csv")
PROCESSED_DATA_PATH = Path("data/processed/features.parquet")


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Arquivo raw não encontrado: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    df_processed = build_features(df)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(PROCESSED_DATA_PATH, index=False)

    print(f"Dados processados salvos em: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
