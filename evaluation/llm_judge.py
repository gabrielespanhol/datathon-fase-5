# evaluation/llm_judge.py

import json
from typing import Dict, List
from src.agent.simple_agent import SimpleFraudAgent


# 👉 você pode trocar depois pelo seu LLM local
def llm_judge(prompt: str) -> str:
    from src.agent.local_llm import LocalLLM

    llm = LocalLLM()
    return llm.generate(prompt)


def build_prompt(query: str, expected: str, answer: str) -> str:
    return f"""
Você é um avaliador de respostas de um sistema de detecção de fraude.

Pergunta:
{query}

Resposta esperada:
{expected}

Resposta do sistema:
{answer}

Avalie com base nos critérios:

1. Correção (0-10)
2. Clareza (0-10)
3. Explicação (0-10)

Responda no formato JSON:

{{
  "correctness": X,
  "clarity": X,
  "explanation": X
}}
"""


def evaluate_agent(golden_path: str) -> List[Dict]:
    agent = SimpleFraudAgent()

    with open(golden_path) as f:
        golden_set = json.load(f)

    results = []

    for item in golden_set:
        query = item["query"]
        expected = item["expected_answer"]

        # 🔹 resposta do seu sistema
        response = agent.run(query)
        answer = response.get("answer", str(response))

        # 🔹 prompt de avaliação
        prompt = build_prompt(query, expected, answer)

        # 🔹 julgamento
        judge_output = llm_judge(prompt)

        results.append(
            {
                "query": query,
                "expected": expected,
                "answer": answer,
                "judge": judge_output,
            }
        )

    return results


if __name__ == "__main__":
    results = evaluate_agent("data\\golden_set\\golden_set.json")

    # salvar resultado
    with open("evaluation/judge_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Avaliação concluída 🚀")
