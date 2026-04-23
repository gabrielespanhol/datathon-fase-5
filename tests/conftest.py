import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Retorna um dataset sintético para testes."""
    return pd.DataFrame(
        {
            "valor": [100.0, 250.0, 80.0, 500.0, 120.0, 300.0, 90.0, 450.0, 200.0, 350.0],
            "hora": [0, 6, 12, 23, 3, 8, 14, 21, 10, 18],
            "dispositivo_novo": [True, False, True, False, True, False, True, False, True, False],
            "tentativas_24h": [1, 2, 3, 4, 1, 2, 3, 4, 2, 1],
            "distancia_km": [1.0, 10.0, 50.0, 100.0, 5.0, 20.0, 60.0, 80.0, 15.0, 30.0],
            "fraude": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )