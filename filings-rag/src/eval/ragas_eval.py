"""Ragas evaluation pipeline for the filings-rag chain.

Runs the chain over a labelled QA test set and computes the standard Ragas
metrics (faithfulness, answer_relevancy, context_precision, context_recall)
plus a custom `refusal_accuracy` on the adversarial subset.
"""

import json
from pathlib import Path
from typing import Callable, Iterable, List

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)


REFUSAL_SENTINEL = "No relevant content found in the corpus."


def load_qa_set(path: Path) -> List[dict]:
    """Read a JSONL QA file into a list of row dicts."""
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def run_chain_on_qa(
    qa_set: Iterable[dict],
    chain_fn: Callable[[str], dict],
) -> List[dict]:
    """Execute `chain_fn` on every QA row and shape results for Ragas."""
    rows: List[dict] = []
    for row in qa_set:
        result = chain_fn(row["question"])
        rows.append(
            {
                "user_input": row["question"],
                "response": result["answer"],
                "retrieved_contexts": [c["text"] for c in result.get("chunks", [])],
                "reference": row.get("ground_truth_answer", ""),
                "category": row.get("category", "unknown"),
            }
        )
    return rows


def compute_refusal_accuracy(rows: List[dict]) -> float:
    """Fraction of `adversarial` rows where the response is a clean refusal."""
    adversarial = [r for r in rows if r["category"] == "adversarial"]
    if not adversarial:
        return 0.0
    return sum(REFUSAL_SENTINEL in r["response"] for r in adversarial) / len(adversarial)


def run_ragas_eval(rows: List[dict]) -> dict:
    """Run the full Ragas metric suite and return a flat metric dict."""
    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
    )
    metrics = result.to_pandas().mean(numeric_only=True).to_dict()
    metrics["refusal_accuracy"] = compute_refusal_accuracy(rows)
    return metrics
