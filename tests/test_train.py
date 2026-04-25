from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Importando as funções do seu arquivo original
from src.models.train import evaluate, get_model_scores, promote_if_better, train

# Definimos o caminho do módulo para facilitar os patches
MODULE_PATH = "src.models.train"


@pytest.fixture
def dummy_data():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    return X, y


# --- Testes para get_model_scores (Cobre as 3 ramificações) ---


def test_get_model_scores_predict_proba():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])
    X = np.array([[1], [2]])
    scores = get_model_scores(model, X)
    assert scores[0] == 0.8


def test_get_model_scores_decision_function():
    # Usamos spec para garantir que o hasattr ignore predict_proba
    model = MagicMock(spec=["decision_function"])
    model.decision_function.return_value = np.array([0.5, 0.9])
    scores = get_model_scores(model, [[1], [2]])
    assert scores[0] == 0.5


def test_get_model_scores_fallback_predict():
    model = MagicMock(spec=["predict"])
    model.predict.return_value = np.array([0, 1])
    scores = get_model_scores(model, [[1], [2]])
    assert scores[1] == 1


# --- Testes para evaluate (Cobre try e except) ---


def test_evaluate_success(dummy_data):
    X, y = dummy_data
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])
    score = evaluate(model, X, y)
    assert isinstance(score, float)


def test_evaluate_exception_fallback(dummy_data):
    X, y = dummy_data
    model = MagicMock()
    # Forçamos falha no get_model_scores para entrar no except
    model.predict_proba.side_effect = Exception("Erro de score")
    model.predict.return_value = np.array([0, 1])

    score = evaluate(model, X, y)
    assert isinstance(score, float)
    model.predict.assert_called()


# --- Testes para promote_if_better (Cobre lógica de promoção e MLflow) ---


@patch(f"{MODULE_PATH}.mlflow.sklearn.load_model")
@patch(f"{MODULE_PATH}.MlflowClient")
@patch(f"{MODULE_PATH}.evaluate")
def test_promote_if_better_scenarios(
    mock_evaluate, mock_client_class, mock_load_model, dummy_data
):
    X, y = dummy_data
    mock_client = mock_client_class.return_value

    # CENÁRIO 1: Sem Champion (Linhas 60-62)
    mock_load_model.side_effect = Exception("No production model")
    mock_evaluate.return_value = 0.90
    # Mock para pegar a versão do modelo
    mock_version = MagicMock(version="1")
    mock_client.get_latest_versions.return_value = [mock_version]

    res = promote_if_better(MagicMock(), X, y)
    assert res == "promoted"

    # CENÁRIO 2: Champion existe mas melhora é insuficiente (Linhas 80-81)
    mock_load_model.side_effect = None
    mock_load_model.return_value = MagicMock()
    mock_evaluate.side_effect = [
        0.90,
        0.901,
    ]  # Champion 0.90, Challenger 0.901 (diff < 0.005)

    res = promote_if_better(MagicMock(), X, y)
    assert res == "kept"

    # CENÁRIO 3: Challenger é significativamente melhor (Linhas 70-78)
    mock_evaluate.side_effect = [0.80, 0.90]  # Melhora de 10%
    res = promote_if_better(MagicMock(), X, y)
    assert res == "promoted"
    assert mock_client.transition_model_version_stage.called


# --- Testes para train e Main (Linhas 88-99) ---


@patch(f"{MODULE_PATH}.run_baseline")
@patch(f"{MODULE_PATH}.promote_if_better")
def test_train_function(mock_promote, mock_run):
    mock_run.return_value = ({"roc_auc": 0.95}, MagicMock(), "X", "y")
    metrics = train()
    assert metrics["roc_auc"] == 0.95
    mock_promote.assert_called_once()
