import pandas as pd
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from src.serving.app import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reseta as métricas do Prometheus antes de cada teste."""
    from src.monitoring.metrics import REGISTRY

    for collector in list(REGISTRY._names_to_collectors.values()):
        if hasattr(collector, "_metrics"):
            collector._metrics.clear()
        elif hasattr(collector, "_value"):
            collector._value.set(0)
    yield


@pytest.fixture
def mock_model():
    """Cria um mock para o modelo do sklearn."""
    model = MagicMock()
    model.predict.return_value = [0]
    model.predict_proba.return_value = [[0.8, 0.2]]
    return model


# --- Testes de Ciclo de Vida (Lifespan/MLflow) ---


def test_lifespan_startup_success():
    # Mockamos o load_model que usa MLflow
    with patch("src.serving.app.mlflow.sklearn.load_model") as mock_mlflow:
        mock_mlflow.return_value = MagicMock()
        with TestClient(app) as ac:
            # Ao entrar no contexto, o lifespan é executado
            assert ac.get("/health").status_code == 200
        mock_mlflow.assert_called_once()


def test_lifespan_startup_fail():
    # Simula erro de conexão ou modelo inexistente no MLflow
    with patch(
        "src.serving.app.mlflow.sklearn.load_model",
        side_effect=Exception("MLflow Error"),
    ):
        with pytest.raises(RuntimeError) as exc:
            with TestClient(app):
                pass
    assert "Modelo não disponível no MLflow" in str(exc.value)


# --- Testes de Rotas Simples ---


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API online"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "process_cpu_seconds_total" in response.text  # Padrão do Prometheus


# --- Testes de Predição ---


def test_predict_model_not_loaded():
    # Garante que o modelo está None para testar o IF inicial
    with patch("src.serving.app.model", None):
        payload = {
            "valor": 100.0,
            "hora": 10,
            "dispositivo_novo": False,
            "tentativas_24h": 1,
            "distancia_km": 5.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "Modelo não carregado."


@pytest.mark.parametrize("prediction_val", [0, 1])
def test_predict_success(prediction_val, mock_model):
    mock_model.predict.return_value = [prediction_val]

    with (
        patch("src.serving.app.model", mock_model),
        patch("src.serving.app.build_features", return_value=pd.DataFrame([0])),
    ):
        payload = {
            "valor": 500.0,
            "hora": 14,
            "dispositivo_novo": True,
            "tentativas_24h": 0,
            "distancia_km": 10.5,
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == prediction_val
        assert "probability" in data


def test_predict_exception_during_processing(mock_model):
    # Cobre o bloco 'except Exception as exc' dentro do /predict
    with (
        patch("src.serving.app.model", mock_model),
        patch(
            "src.serving.app.build_features", side_effect=ValueError("Erro customizado")
        ),
    ):
        payload = {
            "valor": 100.0,
            "hora": 10,
            "dispositivo_novo": False,
            "tentativas_24h": 1,
            "distancia_km": 5.0,
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 400
        assert "Erro ao gerar predição" in response.json()["detail"]


def test_predict_validation_error():
    # Cobre erros do Pydantic (valor negativo onde gt=0)
    payload = {"valor": -10, "hora": 25}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
