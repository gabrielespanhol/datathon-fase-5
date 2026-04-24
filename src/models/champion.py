# src/models/champion.py
import mlflow
from sklearn.metrics import roc_auc_score

from src.models.baseline import build_model


def evaluate(model, X, y):
    y_pred = model.predict(X)
    return roc_auc_score(y, y_pred)


def run_champion_challenger(X_train, y_train, X_test, y_test):
    """Compara modelo atual vs novo modelo"""

    # Challenger (novo)
    challenger = build_model()
    challenger.fit(X_train, y_train)

    challenger_score = evaluate(challenger, X_test, y_test)

    # Champion (último modelo registrado)
    try:
        model_uri = "models:/fraud_detection/Production"
        champion = mlflow.sklearn.load_model(model_uri)
        champion_score = evaluate(champion, X_test, y_test)
    except Exception:
        champion = None
        champion_score = 0

    print("Champion:", champion_score)
    print("Challenger:", challenger_score)

    if challenger_score > champion_score:
        print("Promovendo challenger 🚀")
        mlflow.sklearn.log_model(
            challenger, "model", registered_model_name="fraud_detection"
        )
        return "challenger_promoted"

    return "champion_kept"
