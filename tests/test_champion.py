from unittest.mock import MagicMock, patch

from src.models.champion import evaluate, run_champion_challenger


def test_evaluate():
    # Mock do modelo e dos dados
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    # Execução (Predição perfeita deve retornar 1.0 para AUC e 1.0 para Recall)
    auc, recall = evaluate(mock_model, X, y)

    assert auc == 1.0
    assert recall == 1.0
    mock_model.predict.assert_called_once_with(X)


@patch("src.models.champion.mlflow.sklearn")
@patch("src.models.champion.build_model")
@patch("src.models.champion.evaluate")
def test_run_champion_challenger_promotes_new_model(
    mock_evaluate, mock_build_model, mock_mlflow_sklearn
):
    """Cenário: Challenger é melhor que o Champion atual no AUC"""
    mock_challenger = MagicMock()
    mock_build_model.return_value = mock_challenger

    # Challenger (0.9 AUC, 0.5 Recall) vs Champion (0.8 AUC, 0.5 Recall)
    mock_evaluate.side_effect = [(0.9, 0.5), (0.8, 0.5)]

    result = run_champion_challenger(None, None, None, None)

    assert result == "challenger_promoted"
    mock_mlflow_sklearn.log_model.assert_called_once()


@patch("src.models.champion.mlflow.sklearn")
@patch("src.models.champion.build_model")
@patch("src.models.champion.evaluate")
def test_run_champion_challenger_promotes_on_recall_tiebreak(
    mock_evaluate, mock_build_model, mock_mlflow_sklearn
):
    """Cenário: AUC igual, mas Challenger tem melhor Recall"""
    mock_challenger = MagicMock()
    mock_build_model.return_value = mock_challenger

    # Challenger (0.8 AUC, 0.9 Recall) vs Champion (0.8 AUC, 0.7 Recall)
    mock_evaluate.side_effect = [(0.8, 0.9), (0.8, 0.7)]

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
    # Challenger (0.7 AUC, 0.9 Recall) vs Champion (0.8 AUC, 0.9 Recall)
    mock_evaluate.side_effect = [(0.7, 0.9), (0.8, 0.9)]

    result = run_champion_challenger(None, None, None, None)

    assert result == "champion_kept"
    mock_mlflow_sklearn.log_model.assert_not_called()


@patch("src.models.champion.mlflow.sklearn")
@patch("src.models.champion.build_model")
@patch("src.models.champion.evaluate")
def test_run_champion_challenger_no_existing_champion(
    mock_evaluate, mock_build_model, mock_mlflow_sklearn
):
    """Cenário: Erro ao carregar champion (bloco except)"""
    mock_mlflow_sklearn.load_model.side_effect = Exception("Model not found")

    # Score do challenger (qualquer valor > 0 resultará em promoção)
    mock_evaluate.return_value = (0.9, 0.8)

    result = run_champion_challenger(None, None, None, None)

    assert result == "challenger_promoted"
    mock_mlflow_sklearn.log_model.assert_called_once()
