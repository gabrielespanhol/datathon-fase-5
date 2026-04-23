from src.models.train import train


def test_train_returns_metrics_from_run_baseline(monkeypatch) -> None:
    """train deve retornar exatamente as métricas vindas de run_baseline."""
    expected_metrics = {
        "accuracy": 0.82,
        "precision": 0.51,
        "recall": 0.80,
        "f1": 0.63,
        "roc_auc": 0.92,
    }

    def fake_run_baseline():
        return expected_metrics

    monkeypatch.setattr("src.models.train.run_baseline", fake_run_baseline)

    result = train()

    assert result == expected_metrics


def test_train_calls_run_baseline_once(monkeypatch) -> None:
    """train deve chamar run_baseline uma vez."""
    calls = {"count": 0}

    def fake_run_baseline():
        calls["count"] += 1
        return {
            "accuracy": 0.82,
            "precision": 0.51,
            "recall": 0.80,
            "f1": 0.63,
            "roc_auc": 0.92,
        }

    monkeypatch.setattr("src.models.train.run_baseline", fake_run_baseline)

    train()

    assert calls["count"] == 1