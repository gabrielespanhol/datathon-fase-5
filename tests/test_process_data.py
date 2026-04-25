import runpy
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.scripts import process_data


@pytest.fixture
def mock_df():
    return pd.DataFrame({"valor": [100], "hora": [10]})


## 1. Teste de Sucesso (Cobre definições de Path, Fluxo Main e Print)
@patch("src.scripts.process_data.RAW_DATA_PATH")
@patch("src.scripts.process_data.PROCESSED_DATA_PATH")
@patch("src.scripts.process_data.pd.read_csv")
@patch("src.scripts.process_data.build_features")
@patch("pandas.DataFrame.to_parquet")
@patch("builtins.print")
def test_main_success(
    mock_print,
    mock_to_parquet,
    mock_build,
    mock_read,
    mock_processed_path,
    mock_raw_path,
    mock_df,
):
    """Cobre o caminho feliz, criação de pastas, salvamento e a linha do print."""
    # Setup
    mock_raw_path.exists.return_value = True
    mock_read.return_value = mock_df
    mock_build.return_value = mock_df

    # Mock do comportamento do Path para o print e mkdir
    mock_processed_path.parent.mkdir = MagicMock()
    mock_processed_path.__str__.return_value = "data/processed/features.parquet"

    # Execução
    process_data.main()

    # Verificações
    mock_read.assert_called_once()
    mock_build.assert_called_once()
    mock_processed_path.parent.mkdir.assert_called_once_with(
        parents=True, exist_ok=True
    )
    mock_to_parquet.assert_called_once()
    mock_print.assert_called_with(f"Dados processados salvos em: {mock_processed_path}")


## 2. Teste de Erro (Cobre o Raise)
@patch("src.scripts.process_data.RAW_DATA_PATH")
def test_main_file_not_found(mock_raw_path):
    """Cobre o raise FileNotFoundError."""
    mock_raw_path.exists.return_value = False

    with pytest.raises(FileNotFoundError, match="Arquivo raw não encontrado"):
        process_data.main()


## 3. Teste do Bloco de Execução Principal (Cobre a Linha 25)
def test_script_execution_as_main():
    """
    Usa runpy para simular 'python process_data.py'.
    É a forma mais limpa de cobrir a linha 25 sem redundância de reload.
    """
    script_path = "src/scripts/process_data.py"

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "pandas.read_csv", return_value=pd.DataFrame({"valor": [100], "hora": [10]})
        ),
        patch(
            "src.features.feature_engineering.build_features",
            return_value=pd.DataFrame({"valor": [100], "hora": [10]}),
        ),
        patch("pandas.DataFrame.to_parquet") as mock_to_parquet,
    ):
        runpy.run_path(script_path, run_name="__main__")

    assert mock_to_parquet.called


## 4. Teste de Sanidade/Importação
def test_module_structure():
    """Garante que as constantes e funções básicas estão acessíveis."""
    assert hasattr(process_data, "RAW_DATA_PATH")
    assert hasattr(process_data, "main")
    assert process_data.build_features is not None
