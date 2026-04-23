from prometheus_client import REGISTRY as DEFAULT_REGISTRY

import src.monitoring.metrics as metrics


def test_metrics_initialization():
    """
    Valida se todas as métricas foram instanciadas e registradas no REGISTRY customizado.
    """
    # 1. Verificar se o Registry é o correto
    assert metrics.REGISTRY != DEFAULT_REGISTRY

    # 2. Mapeamento de nomes esperados para os objetos
    # Dica: Remova o '_total' dos nomes de Counter para bater com o '_name' interno
    expected_metrics = {
        "fraud_api_requests": metrics.REQUEST_COUNT,
        "fraud_api_request_errors": metrics.REQUEST_ERRORS,
        "fraud_api_request_latency_seconds": metrics.REQUEST_LATENCY,
        "fraud_model_predictions_fraud": metrics.FRAUD_PREDICTIONS,
        "fraud_model_predictions_nonfraud": metrics.NON_FRAUD_PREDICTIONS,
    }

    for expected_name, metric_obj in expected_metrics.items():
        # Valida se o objeto existe
        assert metric_obj is not None

        # Valida se o nome base está correto no objeto
        assert expected_name == metric_obj._name

        # Valida se a métrica está de fato registrada no REGISTRY customizado
        # O método collect() retorna os objetos de métrica registrados
        collectors = [c._name for c in metrics.REGISTRY._names_to_collectors.values()]
        assert expected_name in collectors


def test_metrics_functionality():
    """
    Valida se os coletores permitem incrementos de forma relativa.
    """
    # Pegamos o valor atual para não sofrer com o acúmulo de outros testes
    start_val = metrics.REGISTRY.get_sample_value("fraud_api_requests_total") or 0

    # Executamos o incremento
    metrics.REQUEST_COUNT.inc()

    # Validamos se o valor agora é o inicial + 1
    current_val = metrics.REGISTRY.get_sample_value("fraud_api_requests_total")
    assert current_val == start_val + 1

    # Repita a lógica para os outros ou apenas valide que não são None
    metrics.REQUEST_LATENCY.observe(0.5)
    assert (
        metrics.REGISTRY.get_sample_value("fraud_api_request_latency_seconds_count")
        >= 1
    )
