from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Ajuste o import para o local correto do seu arquivo
from src.serving.app import MODEL_PATH, app


# Fixture corrigida para resetar métricas usando a API pública do Registry
@pytest.fixture(autouse=True)
def reset_metrics():
    from src.monitoring.metrics import REGISTRY

    # A forma segura de iterar nos coletores registrados
    for collector in list(REGISTRY._names_to_collectors.values()):
        if hasattr(collector, "_metrics"):
            # Limpa os valores (armazenados em dict interno)
            collector._metrics.clear()
        elif hasattr(collector, "_value"):
            # Para tipos simples se houver
            collector._value.set(0)
    yield


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = [0]
    model.predict_proba.return_value = [[0.8, 0.2]]
    return model


client = TestClient(app)

# --- Testes de Ciclo de Vida (Lifespan) ---


def test_lifespan_startup_success():
    with (
        patch.object(Path, "exists", return_value=True),
        patch("src.serving.app.joblib.load") as mock_load,
    ):
        with TestClient(app):
            mock_load.assert_called_once_with(MODEL_PATH)


def test_lifespan_startup_fail():
    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(RuntimeError) as exc:
            with TestClient(app):
                pass

    assert "Modelo não encontrado" in str(exc.value)


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
    assert "fraud_api_requests_total" in response.text


# --- Testes de Predição ---


def test_predict_model_not_loaded():
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
        assert response.json()["prediction"] == prediction_val

        from src.monitoring.metrics import REGISTRY

        if prediction_val == 1:
            assert REGISTRY.get_sample_value("fraud_model_predictions_fraud_total") == 1
        else:
            assert (
                REGISTRY.get_sample_value("fraud_model_predictions_nonfraud_total") == 1
            )


def test_predict_exception_handling(mock_model):
    with (
        patch("src.serving.app.model", mock_model),
        patch("src.serving.app.build_features", side_effect=Exception("Erro de Teste")),
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

        from src.monitoring.metrics import REGISTRY

        assert REGISTRY.get_sample_value("fraud_api_request_errors_total") == 1


def test_predict_validation_error():
    payload = {"valor": -1, "hora": 25}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
