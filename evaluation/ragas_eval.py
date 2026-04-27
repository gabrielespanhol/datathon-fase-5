# evaluation/ragas_eval.py
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import json


def run_eval(rag_fn):
    with open("evaluation/golden_set.json") as f:
        data = json.load(f)

    results = []
    for item in data:
        answer, contexts = rag_fn(item["query"])

        results.append(
            {
                "question": item["query"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["expected_answer"],
            }
        )

    dataset = Dataset.from_list(results)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    print(scores)
