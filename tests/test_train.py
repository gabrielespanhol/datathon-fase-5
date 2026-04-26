from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models.train import evaluate, get_model_scores, promote_if_better, train

MODULE_PATH = "src.models.train"


@pytest.fixture
def dummy_data():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    return X, y


def test_get_model_scores_predict_proba():
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.2, 0.8], [0.3, 0.7]])

    scores = get_model_scores(model, np.array([[1], [2]]))

    assert scores[0] == 0.8


def test_get_model_scores_decision_function():
    model = MagicMock(spec=["decision_function"])
    model.decision_function.return_value = np.array([0.5, 0.9])

    scores = get_model_scores(model, [[1], [2]])

    assert scores[0] == 0.5


def test_get_model_scores_fallback_predict():
    model = MagicMock(spec=["predict"])
    model.predict.return_value = np.array([0, 1])

    scores = get_model_scores(model, [[1], [2]])

    assert scores[1] == 1


def test_evaluate_success(dummy_data):
    X, y = dummy_data
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9]])

    score = evaluate(model, X, y)

    assert isinstance(score, float)


def test_evaluate_exception_fallback(dummy_data):
    X, y = dummy_data
    model = MagicMock()
    model.predict_proba.side_effect = Exception("Erro de score")
    model.predict.return_value = np.array([0, 1])

    score = evaluate(model, X, y)

    assert isinstance(score, float)
    model.predict.assert_called()


@patch(f"{MODULE_PATH}.mlflow.sklearn.load_model")
@patch(f"{MODULE_PATH}.save_champion")
@patch(f"{MODULE_PATH}.evaluate")
def test_promote_if_better_scenarios(
    mock_evaluate,
    mock_save_champion,
    mock_load_model,
    dummy_data,
):
    X, y = dummy_data

    challenger = MagicMock(name="challenger_model")
    champion = MagicMock(name="champion_model")

    mock_load_model.side_effect = Exception("No local model")

    result = promote_if_better(challenger, X, y)

    assert result == "promoted"
    mock_save_champion.assert_called_once_with(challenger)
    mock_save_champion.reset_mock()

    mock_load_model.side_effect = None
    mock_load_model.return_value = champion
    mock_evaluate.side_effect = [0.90, 0.901]

    result = promote_if_better(challenger, X, y)

    assert result == "kept"
    mock_save_champion.assert_not_called()

    mock_evaluate.side_effect = [0.80, 0.90]

    result = promote_if_better(challenger, X, y)

    assert result == "promoted"
    mock_save_champion.assert_called_once_with(challenger)


@patch(f"{MODULE_PATH}.save_champion")
@patch(f"{MODULE_PATH}.run_baseline")
@patch(f"{MODULE_PATH}.promote_if_better")
def test_train_function(mock_promote, mock_run, mock_save_champion):
    mock_run.return_value = (
        {"roc_auc": 0.95},
        MagicMock(name="trained_model"),
        "X",
        "y",
    )
    mock_promote.return_value = "promoted"

    metrics = train()

    assert metrics["roc_auc"] == 0.95
    mock_promote.assert_called_once()
    mock_save_champion.assert_not_called()
