import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("data/processed/features.parquet")
TARGET_COL = "fraude"
EXPERIMENT_NAME = "fraude-baseline"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega os dados processados."""
    return pd.read_parquet(data_path)


def load_dataset_metadata(path: Path = Path("data/raw/dataset_metadata.json")) -> dict:
    with open(path) as f:
        return json.load(f)


def split_data(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa X e y e faz split estratificado."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series,
) -> dict[str, float]:
    """Calcula métricas de classificação."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def build_model() -> Pipeline:
    """Cria pipeline do modelo baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def set_standard_tags():
    """Define metadata padronizada do modelo."""
    mlflow.set_tag("model_name", "fraud_detection")
    mlflow.set_tag("model_version", "v1")
    mlflow.set_tag("model_type", "classification")
    mlflow.set_tag("owner", "grupo-xx")
    mlflow.set_tag("risk_level", "high")
    mlflow.set_tag("git_sha", os.getenv("GIT_SHA", "dev"))


def run_baseline(data_path: Path = DATA_PATH) -> dict[str, float]:
    """
    Executa baseline com Regressão Logística,
    com tracking + registry + governança.
    """

    logger.info("Iniciando treinamento baseline...")

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_model()

    dataset_metadata = load_dataset_metadata()

    with mlflow.start_run(run_name="logistic_regression_v1"):

        # 🔥 Metadata padronizada
        set_standard_tags()

        # Treino
        model.fit(X_train, y_train)

        # Predição
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        # Métricas
        metrics = compute_metrics(y_test, y_pred, y_score)

        # Params organizados
        mlflow.log_params(
            {
                "model_name": "logistic_regression",
                "test_size": 0.2,
                "random_state": 42,
                "n_samples_train": len(X_train),
                "n_samples_test": len(X_test),
                "n_features": X_train.shape[1],
                "training_data_version": dataset_metadata["data_version"],
                "training_data_rows": dataset_metadata["n_rows"],
                "training_data_fraud_rate": dataset_metadata["fraud_rate"],
            }
        )

        mlflow.set_tag("training_data_version", dataset_metadata["data_version"])
        mlflow.set_tag("training_data_path", dataset_metadata["raw_path"])

        # Log métricas
        mlflow.log_metrics(metrics)

        # 🔥 REGISTRO DO MODELO (ESSENCIAL)
        mlflow.sklearn.log_model(
            model, name="model", registered_model_name="fraud_detection"
        )

        logger.info("Treinamento concluído.")
        logger.info("Métricas: %s", metrics)

        return metrics, model, X_test, y_test


if __name__ == "__main__":
    results = run_baseline()
    print(results)
