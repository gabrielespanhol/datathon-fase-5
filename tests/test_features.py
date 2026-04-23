import pandas as pd

from src.features.feature_engineering import build_features


def test_build_features_returns_dataframe(sample_data: pd.DataFrame) -> None:
    """A função deve retornar um DataFrame."""
    result = build_features(sample_data)
    assert isinstance(result, pd.DataFrame)


def test_row_count_is_preserved(sample_data: pd.DataFrame) -> None:
    """O número de linhas deve ser preservado após a transformação."""
    result = build_features(sample_data)
    assert len(result) == len(sample_data)


def test_tentativas_24h_is_removed(sample_data: pd.DataFrame) -> None:
    """A coluna tentativas_24h deve ser removida."""
    result = build_features(sample_data)
    assert "tentativas_24h" not in result.columns


def test_hora_is_removed(sample_data: pd.DataFrame) -> None:
    """A coluna original hora deve ser removida após transformação cíclica."""
    result = build_features(sample_data)
    assert "hora" not in result.columns


def test_hora_cyclical_features_are_created(sample_data: pd.DataFrame) -> None:
    """As colunas hora_sin e hora_cos devem ser criadas."""
    result = build_features(sample_data)
    assert "hora_sin" in result.columns
    assert "hora_cos" in result.columns


def test_dispositivo_novo_becomes_integer(sample_data: pd.DataFrame) -> None:
    """A coluna dispositivo_novo deve ser convertida para inteiro."""
    result = build_features(sample_data)
    assert pd.api.types.is_integer_dtype(result["dispositivo_novo"])


def test_no_nulls_are_created(sample_data: pd.DataFrame) -> None:
    """A transformação não deve criar valores nulos."""
    result = build_features(sample_data)
    assert result.isnull().sum().sum() == 0