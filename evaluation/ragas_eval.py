# evaluation/ragas_eval.py

import json

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def run_eval(rag_fn):
    with open("data/golden_set/golden_set.json") as f:
        data = json.load(f)

    model = SentenceTransformer("all-MiniLM-L6-v2")

    context_scores = []
    answer_scores = []
    availability_scores = []

    for item in data:
        query = item["query"]
        expected = item["expected_answer"]

        answer, contexts = rag_fn(query)

        context_text = " ".join(contexts) if contexts else ""

        # 🔹 Context availability
        availability_scores.append(1.0 if context_text else 0.0)

        # 🔹 Context similarity
        ref_text = f"{query} {expected}"

        if context_text:
            ref_emb = model.encode(ref_text)
            ctx_emb = model.encode(context_text)
            context_scores.append(cosine_similarity(ref_emb, ctx_emb))
        else:
            context_scores.append(0.0)

        # 🔹 Answer similarity
        expected_emb = model.encode(expected)
        answer_emb = model.encode(answer)
        answer_scores.append(cosine_similarity(expected_emb, answer_emb))

    results = {
        "context_availability": round(float(np.mean(availability_scores)), 4),
        "context_similarity": round(float(np.mean(context_scores)), 4),
        "answer_similarity": round(float(np.mean(answer_scores)), 4),
    }

    print("\n📊 RAG Evaluation Results:")
    print(results)

    return results
