import pandas as pd

from src.features.feature_engineering import build_features
from src.models.baseline import run_baseline


def train():
    df = pd.read_csv("data/raw/fraud_dataset_20260421_232806.csv")

    df = build_features(df)

    df.to_parquet("data/processed/features.parquet", index=False)

    return run_baseline()


if __name__ == "__main__":
    results = train()
    print(results)
