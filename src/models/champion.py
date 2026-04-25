# src/models/champion.py
import mlflow
from sklearn.metrics import recall_score, roc_auc_score

from src.models.baseline import build_model


def evaluate(model, X, y):
    y_pred = model.predict(X)

    # Calculando as métricas
    auc = roc_auc_score(y, y_pred)
    recall = recall_score(y, y_pred)

    return auc, recall


def run_champion_challenger(X_train, y_train, X_test, y_test):
    """Compara modelo atual vs novo modelo usando AUC e Recall"""

    # Challenger (novo)
    challenger = build_model()
    challenger.fit(X_train, y_train)

    challenger_auc, challenger_recall = evaluate(challenger, X_test, y_test)

    # Champion (último modelo registrado)
    try:
        model_uri = "models:/fraud_detection/Production"
        champion = mlflow.sklearn.load_model(model_uri)
        champion_auc, champion_recall = evaluate(champion, X_test, y_test)
    except Exception:
        champion = None
        champion_auc, champion_recall = 0, 0

    print(f"Champion   | AUC: {champion_auc:.4f} | Recall: {champion_recall:.4f}")
    print(f"Challenger | AUC: {challenger_auc:.4f} | Recall: {challenger_recall:.4f}")

    # Lógica de promoção: Aqui você decide qual métrica tem prioridade.
    # Exemplo: Promove se o AUC for melhor, ou se o AUC for igual mas o Recall for superior.
    if challenger_auc > champion_auc:
        print("Promovendo challenger por melhor performance global (AUC) 🚀")
        promote = True
    elif (challenger_auc == champion_auc) and (challenger_recall > champion_recall):
        print("Promovendo challenger por melhor critério de desempate (Recall) 🚀")
        promote = True
    else:
        promote = False

    if promote:
        mlflow.sklearn.log_model(
            challenger,
            "model",
            registered_model_name="fraud_detection",
            metrics={"auc": challenger_auc, "recall": challenger_recall},
        )
        return "challenger_promoted"

    return "champion_kept"
