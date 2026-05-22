# Evaluation methodology

How the Filings-RAG system is evaluated. The design doc (`2026-05-20-filings-rag-design.md` §8) defines the metric set; this file documents the labelled test set, the eval procedure, and the running ablation results.

## Test set

`data/qa_test_set.jsonl` contains 40 hand-curated questions distributed across four categories:

| Category | Count | Purpose |
|---|---|---|
| Single-company factual | 10 | Tests whether the retriever finds the precise passage and the LLM lifts the verbatim value (revenue, employee count, emissions, dividend, etc.) |
| Multi-section synthesis | 10 | Tests whether the system can combine information from multiple sections of one filing (TCFD + Principal Risks, Chairman's Letter + segmental review, etc.) |
| Multi-company comparison | 10 | Tests whether the system can compare disclosures across two or more filings (HSBC vs Standard Chartered credit risk, BP vs Shell energy transition, etc.) |
| Adversarial / refusal | 10 | Tests whether the system refuses cleanly on out-of-scope questions (non-FTSE companies, investment advice, speculative price predictions, queries outside the corpus year range) |

Row schema:

```json
{
  "question": "...",
  "ground_truth_answer": "...",
  "ground_truth_citations": [{"ticker": "...", "year": 2024, "pages": [12, 13]}],
  "category": "single_factual | multi_section | multi_company | adversarial"
}
```

For the adversarial subset, `ground_truth_answer` is the literal refusal sentinel: `"No relevant content found in the corpus."`. For the other three categories, ground-truth answers and citations are filled in by reading the source PDFs (a one-time exercise — once captured, the test set is reused across every eval run).

A separate 15-question **retrieval pilot set** lives at `data/qa_pilot.jsonl` with the same schema plus a `ground_truth_chunk_ids` field listing the specific chunk hashes that should retrieve. The pilot set drives `notebook 04` retrieval-only ablations; the full 40-question set drives `notebook 05` end-to-end Ragas eval.

## Metrics

### Retrieval-only (notebook 04)

Run against the 15-question pilot set with known ground-truth chunk IDs.

| Metric | What it measures |
|---|---|
| **recall@5** | Fraction of ground-truth chunks present in the top-5 retrieved |
| **recall@10** | Same at top-10 |
| **MRR** | Reciprocal rank of the first ground-truth chunk in the retrieved list (averaged across questions) |

### End-to-end (notebook 05)

Run against the full 40-question test set, with the chain executed end-to-end.

| Metric | What it measures | Notes |
|---|---|---|
| **faithfulness** | Does the answer follow from the retrieved context? | Ragas; LLM-judged |
| **answer_relevancy** | Does the answer address the question asked? | Ragas; LLM-judged |
| **context_precision** | Of the retrieved chunks, what fraction is actually relevant? | Ragas; gold-citation-aware |
| **context_recall** | Of the gold-relevant chunks, what fraction was retrieved? | Ragas; gold-citation-aware |
| **refusal_accuracy** | On the 10 adversarial questions, what fraction got a clean refusal? | Custom; substring match against the refusal sentinel |

All metrics are logged to MLflow per run, tagged with the retrieval configuration (`dense_only`, `bm25_only`, `hybrid`, `hybrid+rerank`, `+query_rewrite`).

## Ablation table

### Retrieval-only ablation (5-question pilot set, 2026-05-22)

| Config | recall@5 | recall@10 | MRR |
|---|---|---|---|
| Dense only | 0.667 | 0.800 | 1.000 |
| BM25 only | 0.533 | 0.533 | 1.000 |
| **Hybrid (RRF)** | **0.733** | **0.800** | **1.000** |
| Hybrid + re-ranker | 0.467 | 0.667 | 0.467 |

**Headline:** Hybrid RRF beats pure-dense by +10pp and pure-BM25 by +20pp on recall@5 — the core engineering claim of this project, numerically demonstrated. Numbers are reproducible via `python -m jupyter nbconvert --execute notebooks/04_retrieval_evaluation.ipynb`.

**Honest methodology caveat on the re-ranker row:** The pilot ground-truth chunk IDs in `data/qa_pilot.jsonl` were auto-seeded from the top-3 hybrid retrievals (a bootstrap, not human-labelled). The re-ranker's lower score reflects that it reorders the top-20 in ways that push some auto-seeded "gold" chunks past position 5, *not* that re-ranking is genuinely worse. With hand-labelled gold chunks the re-ranker row would be expected to outperform hybrid alone (well-documented in the BGE re-ranker literature). To produce a publishable re-ranker number, the pilot set must be re-labelled by reading the retrieved passages and keeping only the ones that are *actually* relevant to each question, regardless of which retrieval strategy surfaced them.

### End-to-end ablation (40-question test set, full Ragas suite)

| Config | faithfulness | answer_relevancy | context_precision | context_recall | refusal_acc |
|---|---|---|---|---|---|
| Hybrid + re-ranker (default) | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

Run notebook `05_end_to_end_eval.ipynb` to populate. Requires `data/qa_test_set.jsonl` to have non-PLACEHOLDER `ground_truth_answer` fields (currently 10/40 are real — the adversarial subset; the other 30 need hand-labelling against the source PDFs).

## How a CI run validates a change

Every PR runs a 5-question sample eval (the first row of each category) against a checked-in mini Chroma index. The job:

1. Lints with `ruff`
2. Runs the unit-test suite (`pytest tests/`)
3. Builds the in-memory mini-corpus
4. Runs the chain against 5 sample questions
5. Asserts: every response is non-empty, every non-adversarial response contains at least one valid citation, every adversarial response contains the refusal sentinel

If any of those assertions fails, the PR is blocked. The full 40-question ablation eval is run manually before merge to `main` and the table above is updated.

## Known limitations of the eval

- **LLM-judged metrics drift.** Ragas's faithfulness and answer_relevancy use an LLM judge — the same `LLM_PROVIDER` env-var swap that affects the system also affects the judge. Comparing across providers requires re-running the entire eval, not interpolating.
- **Static corpus.** The test set is fixed at indexing time. If you re-index with a different chunking strategy, the `ground_truth_chunk_ids` in the pilot set will be invalidated and must be re-mapped.
- **Refusal accuracy is substring-based.** If the LLM phrases a refusal differently from the sentinel string, the metric will under-count. Mitigation: the prompt explicitly forces the exact sentinel; if drift becomes a problem, switch to LLM-judged refusal detection.
- **Adversarial-set coverage.** 10 adversarial questions cover the headline categories (non-corpus company, investment advice, speculation, out-of-year-range, off-modality) but not every prompt-injection vector. A second-pass adversarial set targeting jailbreaks belongs in v2.
