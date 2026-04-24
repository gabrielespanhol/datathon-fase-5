import logging

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score

from src.models.baseline import run_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_scores(model, X):
    """
    Extrai scores do modelo de forma robusta.

    Prioridade:
    1. predict_proba
    2. decision_function
    3. fallback para predict
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    elif hasattr(model, "decision_function"):
        return model.decision_function(X)

    else:
        # fallback (menos ideal)
        return model.predict(X)


def evaluate(model, X, y):
    """
    Avalia modelo usando ROC AUC com score apropriado.
    """
    try:
        y_score = get_model_scores(model, X)
        return roc_auc_score(y, y_score)

    except Exception as e:
        # fallback final (segurança)
        print("Erro ao calcular ROC AUC:", e)
        y_pred = model.predict(X)
        return roc_auc_score(y, y_pred)


def promote_if_better(
    challenger_model,
    X_test,
    y_test,
    model_name: str = "fraud_detection",
    min_improvement: float = 0.005,  # 0.5%
):
    client = MlflowClient()

    try:
        champion = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

        champion_score = evaluate(champion, X_test, y_test)

    except Exception:
        logger.warning("Nenhum champion encontrado — promovendo challenger")
        champion_score = 0

    challenger_score = evaluate(challenger_model, X_test, y_test)

    logger.info("Champion score: %.4f", champion_score)
    logger.info("Challenger score: %.4f", challenger_score)

    if challenger_score > champion_score + min_improvement:
        versions = client.get_latest_versions(model_name)
        latest_version = versions[0].version

        client.transition_model_version_stage(
            name=model_name, version=latest_version, stage="Production"
        )

        logger.info("🚀 Challenger promovido para Production")
        return "promoted"

    logger.info("Champion mantido")
    return "kept"


def train():
    logger.info("Iniciando pipeline de treinamento...")

    metrics, model, X_test, y_test = run_baseline()

    promote_if_better(model, X_test, y_test)

    logger.info("Treinamento finalizado")
    return metrics


if __name__ == "__main__":
    train()
