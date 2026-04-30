from unittest.mock import patch

import numpy as np
import pytest

# Ajuste conforme o nome do seu arquivo real
from src.agent.rag_pipeline import SimpleRAGPipeline


@pytest.fixture
def mock_dependencies():
    """
    Mock do SentenceTransformer e LocalLLM.
    O patch deve apontar para onde as classes são USADAS (no seu arquivo rag_pipeline).
    """
    # Se o seu arquivo estiver em src/agent/rag_pipeline.py, o caminho do patch é:
    path_st = "src.agent.rag_pipeline.SentenceTransformer"
    path_llm = "src.agent.rag_pipeline.LocalLLM"

    with patch(path_st) as mock_st, patch(path_llm) as mock_llm:

        # Configura o mock do SentenceTransformer
        instance_st = mock_st.return_value
        # Simula retorno de vetores: se for lista (build_index), retorna matriz; se for str (retrieve), retorna array
        instance_st.encode.side_effect = lambda x, **kwargs: (
            np.ones((len(x), 384)) if isinstance(x, list) else np.ones(384)
        )

        # Configura o mock do LocalLLM
        instance_llm = mock_llm.return_value
        instance_llm.generate.return_value = "Resposta mockada para teste."

        yield instance_st, instance_llm


## Testes de Indexação


def test_build_index_success(tmp_path, mock_dependencies):
    # Cria um arquivo real temporário para o pathlib ler
    f = tmp_path / "documento_teste.txt"
    f.write_text(
        "Este é um conteúdo de teste para validar o fatiamento.", encoding="utf-8"
    )

    pipeline = SimpleRAGPipeline(docs_paths=[str(f)], chunk_size=20, chunk_overlap=5)
    pipeline.build_index()

    assert len(pipeline.chunks) > 0
    assert pipeline.embeddings is not None
    assert isinstance(pipeline.embeddings, np.ndarray)


def test_build_index_empty_or_missing_files(mock_dependencies):
    # Testa erro ao passar lista vazia ou arquivos que não existem
    pipeline = SimpleRAGPipeline(docs_paths=["arquivo_fantasma.txt"])
    with pytest.raises(ValueError, match="Nenhum documento encontrado para indexar."):
        pipeline.build_index()


## Testes de Recuperação (Retrieval)


def test_retrieve_without_index(mock_dependencies):
    # Testa se dispara erro ao tentar buscar sem indexar primeiro
    pipeline = SimpleRAGPipeline(docs_paths=[])
    with pytest.raises(RuntimeError, match="Índice ainda não foi construído"):
        pipeline.retrieve("Qualquer pergunta")


## Testes de Geração e Fluxo Completo


def test_ask_flow(tmp_path, mock_dependencies):
    _, mock_llm = mock_dependencies

    f = tmp_path / "info.txt"
    f.write_text("O sistema de fraude usa IA.", encoding="utf-8")

    pipeline = SimpleRAGPipeline(docs_paths=[str(f)])
    pipeline.build_index()

    response = pipeline.ask("Como funciona o sistema?")

    assert response["question"] == "Como funciona o sistema?"
    assert response["answer"] == "Resposta mockada para teste."
    assert len(response["sources"]) > 0
    assert "score" in response["sources"][0]


def test_chunk_text_logic():
    """Teste isolado da lógica de overlap e tamanho de chunk."""
    pipeline = SimpleRAGPipeline(docs_paths=[], chunk_size=10, chunk_overlap=2)
    text = "abcdefghijklmnop"  # 16 caracteres

    chunks = pipeline._chunk_text(text)

    # Chunk 1: "abcdefghij" (10)
    # Start: 10 - 2 = 8. Chunk 2: "ijklmnop" (8)
    assert len(chunks) == 2
    assert chunks[0] == "abcdefghij"
