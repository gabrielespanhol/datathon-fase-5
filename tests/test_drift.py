import json
import pandas as pd
import pytest
from pathlib import Path
from src.monitoring.drift import run_drift_detection


@pytest.fixture
def data_paths(tmp_path):
    """Cria arquivos parquet temporários para teste."""
    ref_path = tmp_path / "ref.parquet"
    curr_path = tmp_path / "curr.parquet"

    # Dados base
    df_ref = pd.DataFrame({"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]})
    df_ref.to_parquet(ref_path)

    return ref_path, curr_path, df_ref


def test_run_drift_detection_no_drift(data_paths):
    """Caso onde os dados são idênticos (0% drift)."""
    ref_path, curr_path, df_ref = data_paths
    df_ref.to_parquet(curr_path)  # Criando o atual igual ao de referência

    output = run_drift_detection(str(ref_path), str(curr_path))

    assert output["drift_share"] == 0
    assert output["n_drifted_features"] == 0
    assert Path("reports/drift_report.json").exists()


def test_run_drift_detection_with_drift(data_paths):
    """Caso onde os dados são muito diferentes (drift alto)."""
    ref_path, curr_path, _ = data_paths
    # Valores totalmente fora da distribuição original
    df_curr = pd.DataFrame(
        {"col1": [100, 200, 300, 400, 500], "col2": [10, 21, 29, 41, 50]}
    )
    df_curr.to_parquet(curr_path)

    output = run_drift_detection(str(ref_path), str(curr_path))

    # Pelo menos col1 deve apresentar drift
    assert output["drift_share"] > 0
    assert output["n_drifted_features"] >= 1


def test_run_drift_detection_no_numeric_columns(tmp_path):
    """Cobre a linha do 'if total_columns > 0' e o caso de zero colunas."""
    ref_path = tmp_path / "ref_str.parquet"
    curr_path = tmp_path / "curr_str.parquet"

    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    df.to_parquet(ref_path)
    df.to_parquet(curr_path)

    output = run_drift_detection(str(ref_path), str(curr_path))

    assert output["drift_share"] == 0
    assert output["total_features_checked"] == 0


def test_json_content_correctness(data_paths):
    """Verifica se o arquivo JSON salvo contém as chaves certas."""
    ref_path, curr_path, df_ref = data_paths
    df_ref.to_parquet(curr_path)

    run_drift_detection(str(ref_path), str(curr_path))

    with open("reports/drift_report.json", "r") as f:
        data = json.load(f)

    assert "timestamp" in data
    assert "drift_share" in data
    assert data["n_reference"] == 5
