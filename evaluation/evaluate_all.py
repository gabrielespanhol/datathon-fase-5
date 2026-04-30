# evaluation/evaluate_all.py
import json
import re
from pathlib import Path
from typing import Any

import mlflow.sklearn
import numpy as np
from ragas.embeddings import BaseRagasEmbeddings
from sentence_transformers import SentenceTransformer

from src.agent.local_llm import LocalLLM
from src.agent.rag_pipeline import SimpleRAGPipeline
from src.agent.simple_agent import SimpleFraudAgent

GOLDEN_SET_PATH = Path("data/golden_set/golden_set.json")
OUTPUT_PATH = Path("evaluation/evaluation_results.json")
MODEL_PATH = "src/models/fraud_detection"


class LocalEmbeddingsForRagas(BaseRagasEmbeddings):
    """Embeddings locais para evitar dependência de OpenAI no RAGAS."""

    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts).tolist()

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


def load_golden_set(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_agent_response(response: Any) -> tuple[str, list[str]]:
    if isinstance(response, dict):
        answer = (
            response.get("answer")
            or response.get("explanation")
            or response.get("result")
            or response.get("message")
            or str(response)
        )

        sources = response.get("sources") or response.get("contexts") or []

        contexts: list[str] = []
        for source in sources:
            if isinstance(source, dict):
                contexts.append(
                    source.get("text")
                    or source.get("content")
                    or source.get("page_content")
                    or str(source)
                )
            else:
                contexts.append(str(source))

        if not contexts:
            contexts = [str(answer)]

        return str(answer), contexts

    return str(response), [str(response)]


def create_agent_for_evaluation() -> SimpleFraudAgent:
    model = mlflow.sklearn.load_model(MODEL_PATH)

    docs_paths = [
        "docs/FRAUD_KNOWLEDGE_BASE.md",
        "docs/MODEL_CARD.md",
        "docs/DATASET.md",
    ]

    rag_pipeline = SimpleRAGPipeline(docs_paths=docs_paths)
    rag_pipeline.build_index()

    return SimpleFraudAgent(
        model=model,
        rag_pipeline=rag_pipeline,
    )


def run_agent_on_golden_set(
    golden_set: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    agent = create_agent_for_evaluation()
    results = []

    for item in golden_set:
        query = item["query"]
        expected = item["expected_answer"]

        response = agent.run(query)
        answer, contexts = normalize_agent_response(response)

        results.append(
            {
                "query": query,
                "expected_answer": expected,
                "answer": answer,
                "contexts": contexts,
                "type": item.get("type"),
                "difficulty": item.get("difficulty"),
                "risk_level": item.get("risk_level"),
                "features": item.get("features"),
            }
        )

    return results




def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_np = np.array(a)
    b_np = np.array(b)

    denom = np.linalg.norm(a_np) * np.linalg.norm(b_np)
    if denom == 0:
        return 0.0

    return float(np.dot(a_np, b_np) / denom)


def run_ragas(results: list[dict[str, Any]]) -> dict[str, float]:
    """
    Avaliação local robusta, sem RAGAS/OpenAI.

    Métricas:
    - context_availability: % de exemplos com contexto retornado
    - context_semantic_similarity: similaridade semântica entre esperado+query e contexto
    - answer_semantic_similarity: similaridade semântica entre resposta esperada e resposta do agente
    - retrieval_score: média das métricas de contexto
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    context_availability_scores = []
    context_similarity_scores = []
    answer_similarity_scores = []

    for item in results:
        query = item["query"]
        expected = item["expected_answer"]
        answer = item["answer"]
        contexts = item.get("contexts") or []

        context_text = " ".join(contexts).strip()

        has_context = bool(context_text)
        context_availability_scores.append(1.0 if has_context else 0.0)

        reference_text = f"{query} {expected}"

        if has_context:
            ref_emb = model.encode(reference_text).tolist()
            ctx_emb = model.encode(context_text).tolist()
            context_similarity_scores.append(cosine_similarity(ref_emb, ctx_emb))
        else:
            context_similarity_scores.append(0.0)

        expected_emb = model.encode(expected).tolist()
        answer_emb = model.encode(answer).tolist()
        answer_similarity_scores.append(cosine_similarity(expected_emb, answer_emb))

    context_availability = float(np.mean(context_availability_scores))
    context_semantic_similarity = float(np.mean(context_similarity_scores))
    answer_semantic_similarity = float(np.mean(answer_similarity_scores))

    retrieval_score = float(
        np.mean(
            [
                context_availability,
                context_semantic_similarity,
            ]
        )
    )

    return {
        "context_availability": round(context_availability, 4),
        "context_semantic_similarity": round(context_semantic_similarity, 4),
        "answer_semantic_similarity": round(answer_semantic_similarity, 4),
        "retrieval_score": round(retrieval_score, 4),
    }


def build_judge_prompt(query: str, expected: str, answer: str) -> str:
    return f"""
Você é um avaliador técnico de um sistema de detecção de fraude.

Avalie a resposta do sistema comparando com a resposta esperada.

Pergunta:
{query}

Resposta esperada:
{expected}

Resposta do sistema:
{answer}

Critérios:
1. correctness: a resposta está correta?
2. clarity: a resposta é clara?
3. explanation: a justificativa é boa?
4. business_alignment: a resposta está alinhada ao contexto de fraude financeira?

Dê notas de 0 a 10.

Responda somente em JSON válido, sem markdown:

{{
  "correctness": 0,
  "clarity": 0,
  "explanation": 0,
  "business_alignment": 0,
  "reason": "breve justificativa"
}}
""".strip()


def parse_judge_output(text: str) -> dict[str, Any]:
    try:
        start = text.find("{")
        end = text.rfind("}") + 1

        if start >= 0 and end > start:
            parsed = json.loads(text[start:end])
            return {
                "correctness": float(parsed.get("correctness", 0)),
                "clarity": float(parsed.get("clarity", 0)),
                "explanation": float(parsed.get("explanation", 0)),
                "business_alignment": float(parsed.get("business_alignment", 0)),
                "reason": parsed.get("reason", ""),
            }
    except Exception:
        pass

    numbers = [float(n) for n in re.findall(r"\b(?:10|[0-9])(?:\.\d+)?\b", text)]

    return {
        "correctness": numbers[0] if len(numbers) > 0 else 0.0,
        "clarity": numbers[1] if len(numbers) > 1 else 0.0,
        "explanation": numbers[2] if len(numbers) > 2 else 0.0,
        "business_alignment": numbers[3] if len(numbers) > 3 else 0.0,
        "reason": "Não foi possível extrair JSON válido do judge.",
        "raw_output": text,
    }


def run_llm_judge(
    llm: LocalLLM,
    query: str,
    expected: str,
    answer: str,
) -> dict[str, Any]:
    prompt = build_judge_prompt(query, expected, answer)
    raw_output = llm.generate(prompt)
    return parse_judge_output(raw_output)


def run_judge(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llm = LocalLLM()
    judged_results = []

    for idx, item in enumerate(results, start=1):
        print(f"Judge {idx}/{len(results)}...")

        judge_scores = run_llm_judge(
            llm=llm,
            query=item["query"],
            expected=item["expected_answer"],
            answer=item["answer"],
        )

        judged_results.append(
            {
                **item,
                "judge": judge_scores,
            }
        )

    return judged_results


def summarize_judge(results: list[dict[str, Any]]) -> dict[str, float]:
    keys = ["correctness", "clarity", "explanation", "business_alignment"]
    summary = {}

    for key in keys:
        values = [
            float(item["judge"].get(key, 0))
            for item in results
            if isinstance(item.get("judge"), dict)
        ]

        summary[key] = round(sum(values) / len(values), 2) if values else 0.0

    summary["overall_judge_score"] = round(
        sum(summary.values()) / len(summary),
        2,
    )

    return summary


def main() -> None:
    golden_set = load_golden_set(GOLDEN_SET_PATH)

    print(f"Rodando avaliação com {len(golden_set)} exemplos...")

    base_results = run_agent_on_golden_set(golden_set)

    print("Calculando RAGAS...")
    ragas_scores = run_ragas(base_results)

    print("Rodando LLM-as-judge...")
    judged_results = run_judge(base_results)

    judge_summary = summarize_judge(judged_results)

    final_report = {
        "total_examples": len(golden_set),
        "local_rag_eval": ragas_scores,
        "judge_summary": judge_summary,
        "results": judged_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)

    print(f"Avaliação final salva em: {OUTPUT_PATH}")
    print(
        json.dumps(
            {
                "ragas": ragas_scores,
                "judge_summary": judge_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
