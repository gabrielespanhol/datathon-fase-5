from pathlib import Path
from typing import NamedTuple

import numpy as np
from src.agent.local_llm import LocalLLM
from sentence_transformers import SentenceTransformer


class RetrievedChunk(NamedTuple):
    source: str
    text: str
    score: float


class SimpleRAGPipeline:
    def __init__(
        self,
        docs_paths: list[str],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        top_k: int = 3,
    ) -> None:
        self.docs_paths = docs_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = LocalLLM()

        self.chunks: list[tuple[str, str]] = []
        self.embeddings: np.ndarray | None = None

    def build_index(self) -> None:
        chunks = []

        for doc_path in self.docs_paths:
            path = Path(doc_path)

            if not path.exists():
                continue

            text = path.read_text(encoding="utf-8")
            for chunk in self._chunk_text(text):
                chunks.append((str(path), chunk))

        if not chunks:
            raise ValueError("Nenhum documento encontrado para indexar.")

        self.chunks = chunks
        texts = [chunk_text for _, chunk_text in chunks]

        self.embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )

    def ask(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        context = "\n\n".join(
            f"[Fonte: {chunk.source}]\n{chunk.text}" for chunk in retrieved
        )

        prompt = f"""
Você é um assistente técnico do projeto de detecção de fraude.

Responda usando apenas o contexto abaixo.
Se a resposta não estiver no contexto, diga que não há informação suficiente.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""".strip()

        answer = self.llm.generate(prompt)

        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": chunk.source,
                    "score": chunk.score,
                    "text": chunk.text[:300],
                }
                for chunk in retrieved
            ],
        }

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if self.embeddings is None:
            raise RuntimeError("Índice ainda não foi construído. Rode build_index().")

        query_embedding = self.embedding_model.encode(
            [question],
            normalize_embeddings=True,
        )[0]

        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][: self.top_k]

        return [
            RetrievedChunk(
                source=self.chunks[i][0],
                text=self.chunks[i][1],
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def _chunk_text(self, text: str) -> list[str]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks
