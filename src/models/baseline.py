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


def load_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega os dados processados."""
    return pd.read_parquet(data_path)


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


def run_baseline(data_path: Path = DATA_PATH) -> dict[str, float]:
    """Executa o baseline com Regressão Logística e registra no MLflow."""
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)

    model = Pipeline(
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

    with mlflow.start_run(run_name="logistic_regression"):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_score)

        mlflow.log_param("model_name", "logistic_regression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("n_samples_train", len(X_train))
        mlflow.log_param("n_samples_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

        return metrics


if __name__ == "__main__":
    results = run_baseline()
    print(results)
