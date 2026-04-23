import pandas as pd

from src.features.feature_engineering import build_features

RAW_DATA_PATH = "data/raw/fraud_dataset_20260421_232806.csv"
PROCESSED_DATA_PATH = "data/processed/features.parquet"


def main() -> None:
    df = pd.read_csv(RAW_DATA_PATH)
    df_processed = build_features(df)
    df_processed.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"Dados processados salvos em: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()