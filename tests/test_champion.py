import pytest
from unittest.mock import MagicMock, patch
from src.models.champion import evaluate, run_champion_challenger


def test_evaluate():
    # Mock do modelo e dos dados
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # Execução (ROC AUC de predição perfeita deve ser 1.0)
    score = evaluate(mock_model, X, y)

    assert score == 1.0
    mock_model.predict.assert_called_once_with(X)


@patch("src.models.champion.mlflow.sklearn")
@patch("src.models.champion.build_model")
@patch("src.models.champion.evaluate")
def test_run_champion_challenger_promotes_new_model(
    mock_evaluate, mock_build_model, mock_mlflow_sklearn
):
    """Cenário: Challenger é melhor que o Champion atual"""
    # Configuração dos mocks
    mock_challenger = MagicMock()
    mock_build_model.return_value = mock_challenger

    mock_champion = MagicMock()
    mock_mlflow_sklearn.load_model.return_value = mock_champion

    # Mock do evaluate: primeiro chamado (challenger) retorna 0.9, segundo (champion) retorna 0.8
    mock_evaluate.side_effect = [0.9, 0.8]

    result = run_champion_challenger(None, None, None, None)

    assert result == "challenger_promoted"
    mock_mlflow_sklearn.log_model.assert_called_once()


@patch("src.models.champion.mlflow.sklearn")
@patch("src.models.champion.build_model")
@patch("src.models.champion.evaluate")
def test_run_champion_challenger_keeps_champion(
    mock_evaluate, mock_build_model, mock_mlflow_sklearn
):
    """Cenário: Champion atual ainda é melhor que o Challenger"""
    mock_challenger = MagicMock()
    mock_build_model.return_value = mock_challenger

    mock_champion = MagicMock()
    mock_mlflow_sklearn.load_model.return_value = mock_champion

    # Challenger 0.7 vs Champion 0.8
    mock_evaluate.side_effect = [0.7, 0.8]

    result = run_champion_challenger(None, None, None, None)

    assert result == "champion_kept"
    mock_mlflow_sklearn.log_model.assert_not_called()


@patch("src.models.champion.mlflow.sklearn")
@patch("src.models.champion.build_model")
@patch("src.models.champion.evaluate")
def test_run_champion_challenger_no_existing_champion(
    mock_evaluate, mock_build_model, mock_mlflow_sklearn
):
    """Cenário: Erro ao carregar champion (ex: primeira execução do pipeline)"""
    mock_challenger = MagicMock()
    mock_build_model.return_value = mock_challenger

    # Simula erro ao carregar o modelo do MLflow (cobre o bloco except)
    mock_mlflow_sklearn.load_model.side_effect = Exception("Model not found")

    # Score do challenger
    mock_evaluate.return_value = 0.9

    result = run_champion_challenger(None, None, None, None)

    assert result == "challenger_promoted"
    # Garante que o score do champion foi tratado como 0
    mock_mlflow_sklearn.log_model.assert_called_once()
