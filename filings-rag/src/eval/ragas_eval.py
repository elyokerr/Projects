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


PLACEHOLDER_PREFIX = "PLACEHOLDER"


def _build_ragas_components():
    """Wire Ragas to use OUR Groq LLM + a local HF embedder (default = OpenAI, which we don't have)."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_huggingface import HuggingFaceEmbeddings

    from src.generation.llm_client import build_llm

    llm = LangchainLLMWrapper(build_llm())
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    return llm, embeddings


def _references_are_real(rows: List[dict]) -> bool:
    """True only if at least one non-adversarial row has a non-PLACEHOLDER reference."""
    non_adv = [r for r in rows if r.get("category") != "adversarial"]
    return any(
        r.get("reference") and not str(r["reference"]).startswith(PLACEHOLDER_PREFIX)
        for r in non_adv
    )


def run_ragas_eval(rows: List[dict]) -> dict:
    """Run the Ragas metric suite that's available given the labels we have.

    Faithfulness and AnswerRelevancy do not require a `reference` (ground-truth
    answer) — they're always computed. ContextPrecision and ContextRecall *do*
    require references, so we skip them when references are PLACEHOLDER values.
    """
    import os
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()

    # AnswerRelevancy needs the LLM to return n>1 completions — Groq rejects this with
    # "'n' : number must be at most 1". Skip when using Groq; include when using Gemini.
    metric_set = [Faithfulness()]
    if provider != "groq":
        metric_set.append(AnswerRelevancy())
    if _references_are_real(rows):
        metric_set.extend([ContextPrecision(), ContextRecall()])
        print(f"  Ragas metrics: {[m.__class__.__name__ for m in metric_set]} (reference-aware)")
    else:
        print(f"  Ragas metrics: {[m.__class__.__name__ for m in metric_set]} (reference-free; references are PLACEHOLDER)")

    llm, embeddings = _build_ragas_components()

    dataset = EvaluationDataset.from_list(rows)
    result = evaluate(dataset, metrics=metric_set, llm=llm, embeddings=embeddings)
    metrics = result.to_pandas().mean(numeric_only=True).to_dict()
    metrics["refusal_accuracy"] = compute_refusal_accuracy(rows)
    return metrics
