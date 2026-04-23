from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total de requisições recebidas pela API de fraude",
    registry=REGISTRY,
)

REQUEST_ERRORS = Counter(
    "fraud_api_request_errors_total",
    "Total de erros nas requisições da API de fraude",
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "Latência das requisições da API de fraude em segundos",
    registry=REGISTRY,
)

FRAUD_PREDICTIONS = Counter(
    "fraud_model_predictions_fraud_total",
    "Número de predições classificadas como fraude",
    registry=REGISTRY,
)

NON_FRAUD_PREDICTIONS = Counter(
    "fraud_model_predictions_nonfraud_total",
    "Número de predições classificadas como não fraude",
    registry=REGISTRY,
)
