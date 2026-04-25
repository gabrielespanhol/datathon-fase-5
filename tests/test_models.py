import pandas as pd

from src.models.baseline import compute_metrics, run_baseline, split_data


def test_split_data_returns_four_objects(sample_data: pd.DataFrame) -> None:
    """split_data deve retornar X_train, X_test, y_train e y_test."""
    X_train, X_test, y_train, y_test = split_data(sample_data)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


def test_split_data_removes_target_from_features(sample_data: pd.DataFrame) -> None:
    """A coluna target não deve estar em X_train e X_test."""
    X_train, X_test, _, _ = split_data(sample_data)

    assert "fraude" not in X_train.columns
    assert "fraude" not in X_test.columns


def test_split_data_preserves_total_number_of_rows(sample_data: pd.DataFrame) -> None:
    """A soma das linhas de treino e teste deve ser igual ao total do dataset."""
    X_train, X_test, y_train, y_test = split_data(sample_data)

    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) + len(y_test) == len(sample_data)


def test_compute_metrics_returns_expected_keys() -> None:
    """compute_metrics deve retornar todas as métricas esperadas."""
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.4])

    metrics = compute_metrics(y_true, y_pred, y_score)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected_keys


def test_compute_metrics_values_are_between_zero_and_one() -> None:
    """Todas as métricas devem estar no intervalo [0, 1]."""
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.4])

    metrics = compute_metrics(y_true, y_pred, y_score)

    for value in metrics.values():
        assert 0.0 <= value <= 1.0


def test_run_baseline_returns_metrics_dict(
    monkeypatch, sample_data: pd.DataFrame
) -> None:
    """run_baseline deve retornar um dicionário com as métricas esperadas."""

    def fake_load_data(data_path=None):
        return sample_data

    monkeypatch.setattr("src.models.baseline.load_data", fake_load_data)
    monkeypatch.setattr("src.models.baseline.mlflow.set_experiment", lambda name: None)
    monkeypatch.setattr(
        "src.models.baseline.mlflow.log_param", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.models.baseline.mlflow.log_metrics", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.models.baseline.mlflow.sklearn.log_model", lambda *args, **kwargs: None
    )

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(
        "src.models.baseline.mlflow.start_run", lambda **kwargs: DummyRun()
    )

    metrics, _, _, _ = run_baseline()

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == expected_keys


def test_run_baseline_metrics_are_between_zero_and_one(
    monkeypatch, sample_data: pd.DataFrame
) -> None:
    """As métricas retornadas por run_baseline devem estar no intervalo [0, 1]."""

    def fake_load_data(data_path=None):
        return sample_data

    monkeypatch.setattr("src.models.baseline.load_data", fake_load_data)
    monkeypatch.setattr("src.models.baseline.mlflow.set_experiment", lambda name: None)
    monkeypatch.setattr(
        "src.models.baseline.mlflow.log_param", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.models.baseline.mlflow.log_metrics", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "src.models.baseline.mlflow.sklearn.log_model", lambda *args, **kwargs: None
    )

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    monkeypatch.setattr(
        "src.models.baseline.mlflow.start_run", lambda **kwargs: DummyRun()
    )

    metrics, _, _, _ = run_baseline()

    for value in metrics.values():
        assert 0.0 <= value <= 1.0
