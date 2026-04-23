import logging

from src.models.baseline import run_baseline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train() -> dict[str, float]:
    """Executa o treinamento do modelo baseline e retorna as métricas."""
    logger.info("Iniciando treinamento do modelo baseline...")

    metrics = run_baseline()

    logger.info("Treinamento concluído.")
    logger.info("Métricas: %s", metrics)

    return metrics


if __name__ == "__main__":
    results = train()
    print(results)