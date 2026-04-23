import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformações de feature engineering ao dataframe."""
    df = df.copy()

    if "tentativas_24h" in df.columns:
        df = df.drop(columns=["tentativas_24h"])

    if "hora" in df.columns:
        df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
        df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
        df = df.drop(columns=["hora"])

    if "valor" in df.columns:
        df["valor"] = np.log1p(df["valor"])

    if "distancia_km" in df.columns:
        df["distancia_km"] = np.log1p(df["distancia_km"])

    if "dispositivo_novo" in df.columns:
        df["dispositivo_novo"] = df["dispositivo_novo"].astype(int)

    return df


def process_and_save(input_path: str, output_path: str) -> None:
    """Lê dados brutos, aplica feature engineering e salva dados processados."""
    df = pd.read_csv(input_path)
    df_processed = build_features(df)
    df_processed.to_parquet(output_path, index=False)
