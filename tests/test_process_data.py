import os
import pytest
from unittest.mock import patch
import pandas as pd
from src.scripts.process_data import main
import importlib
import src.scripts.process_data as process_data


## 1. Teste de Sucesso do Fluxo Principal
@patch("src.scripts.process_data.pd.read_csv")
@patch("src.scripts.process_data.build_features")
@patch("src.scripts.process_data.pd.DataFrame.to_parquet")
def test_main_flow_success(mock_to_parquet, mock_build_features, mock_read_csv):
    """
    Testa se o main lê o CSV, chama a feature engineering e salva em parquet.
    """
    # Configura um DataFrame mockado para a leitura
    mock_df_raw = pd.DataFrame({"valor": [100], "hora": [10]})
    mock_read_csv.return_value = mock_df_raw

    # Configura o retorno da função build_features
    mock_df_processed = pd.DataFrame(
        {"valor": [100], "hora": [10], "feature_nova": [1]}
    )
    mock_build_features.return_value = mock_df_processed

    # Executa a função main
    main()

    # Verificações (Assertions)
    mock_read_csv.assert_called_once()
    mock_build_features.assert_called_once_with(mock_df_raw)
    mock_to_parquet.assert_called_once()

    # Valida se o caminho de saída está correto (conforme definido no seu script)
    args, kwargs = mock_to_parquet.call_args
    assert "data/processed/features.parquet" in args[0]


## 2. Teste de Tratamento de Erro (Arquivo não encontrado)
@patch("src.scripts.process_data.pd.read_csv")
def test_main_file_not_found(mock_read_csv):
    """Verifica o comportamento caso o arquivo raw não exista."""
    mock_read_csv.side_effect = FileNotFoundError("Arquivo não encontrado")

    with pytest.raises(FileNotFoundError):
        main()


## 3. Teste de Integração (Opcional - com arquivo real temporário)
def test_main_integration_with_temp_files(tmp_path):
    """
    Teste de integração real usando arquivos temporários.
    """
    # Criar caminhos temporários
    tmp_dir = tmp_path / "data"
    raw_dir = tmp_dir / "raw"
    proc_dir = tmp_dir / "processed"
    raw_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    test_csv = raw_dir / "fraud_dataset_test.csv"
    test_parquet = proc_dir / "features.parquet"

    # Criar dados de teste
    df = pd.DataFrame({"valor": [100], "hora": [5]})
    df.to_csv(test_csv, index=False)

    # Patch nos paths do script original para usar os temporários
    with patch("src.scripts.process_data.RAW_DATA_PATH", str(test_csv)):
        with patch("src.scripts.process_data.PROCESSED_DATA_PATH", str(test_parquet)):
            # Patch na build_features para não precisar testar a lógica dela aqui
            with patch(
                "src.scripts.process_data.build_features", side_effect=lambda x: x
            ):
                main()

    assert os.path.exists(test_parquet)
