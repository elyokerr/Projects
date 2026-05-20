# Filings-RAG — Design Document

**Date:** 2026-05-20
**Status:** Approved (implementation pending)

---

## 1. Project Overview

A retrieval-augmented question-answering system over the full annual reports of FTSE 100 companies. Answers natural-language questions with grounded, verbatim citations (company, fiscal year, page). The system uses hybrid retrieval, cross-encoder re-ranking, forced citations, and a Ragas evaluation pipeline tracked in MLflow. UI is a Streamlit app deployed to Hugging Face Spaces.

**Project name:** `filings-rag`

---

## 2. Problem Statement

FTSE 100 annual reports are 200–400 pages each. Equity analysts, ESG researchers, and corporate-strategy teams routinely need to compare disclosures across companies and fiscal years — climate-related risks, executive remuneration, risk factors, segment results. Ctrl-F across 30 PDFs does not scale, and generic chatbots hallucinate. The unmet need is grounded, traceable retrieval — answers that can be defended to a compliance officer or audit committee.

---

## 3. Users

Two anchor personas the demo is built around:

1. **ESG / sustainability analyst at a UK asset manager** — compares climate-risk language across FTSE constituents to score portfolios against TCFD requirements.
2. **Equity research junior at a UK bank** — extracts specific risk-factor and segment-revenue passages with auditable citations for internal notes.

For both, citation traceability is non-negotiable.

---

## 4. Dataset

| Aspect | Plan |
|---|---|
| Scope | Top 30 FTSE 100 companies by market cap (variety across banks, energy, retail, pharma, tech, miners) |
| Years | 2 most recent fiscal years per company → **~60 PDFs** (~9,000 pages total) |
| Source | Each company's Investor Relations page (publicly hosted PDFs) |
| Acquisition | Hand-curated `data/filings_manifest.csv` → `scraper.py` downloads and validates each PDF |
| Storage | `data/raw/{ticker}/{year}.pdf` → `data/processed/{ticker}/{year}.jsonl` → `data/vector/` (Chroma) |
| Legality | Annual reports are statutorily public; no paywall; project attributes source |
| Politeness | 1-second delay between requests, single-shot (no recursive crawl), respects `robots.txt` |

### Manifest template (top 30 by market cap)

The actual `pdf_url` column is populated during ingestion by visiting each company's investor-relations page. Substitute if any have delisted.

```csv
company_name,ticker,sector,fiscal_year,pdf_url
AstraZeneca,AZN.L,Pharma,2024,
Shell,SHEL.L,Energy,2024,
HSBC Holdings,HSBA.L,Banking,2024,
Unilever,ULVR.L,Consumer Goods,2024,
BP,BP.L,Energy,2024,
GSK,GSK.L,Pharma,2024,
Rio Tinto,RIO.L,Mining,2024,
Glencore,GLEN.L,Mining,2024,
Diageo,DGE.L,Beverages,2024,
British American Tobacco,BATS.L,Tobacco,2024,
Lloyds Banking Group,LLOY.L,Banking,2024,
NatWest Group,NWG.L,Banking,2024,
Barclays,BARC.L,Banking,2024,
RELX,REL.L,Information,2024,
National Grid,NG.L,Utilities,2024,
BAE Systems,BA.L,Defence,2024,
Compass Group,CPG.L,Services,2024,
Tesco,TSCO.L,Retail,2024,
Sainsbury's,SBRY.L,Retail,2024,
Anglo American,AAL.L,Mining,2024,
Standard Chartered,STAN.L,Banking,2024,
Prudential,PRU.L,Insurance,2024,
Aviva,AV.L,Insurance,2024,
Legal & General,LGEN.L,Insurance,2024,
Reckitt Benckiser,RKT.L,Consumer Goods,2024,
Imperial Brands,IMB.L,Tobacco,2024,
Vodafone Group,VOD.L,Telecoms,2024,
BT Group,BT-A.L,Telecoms,2024,
SSE,SSE.L,Utilities,2024,
Centrica,CNA.L,Utilities,2024,
```

Duplicate the rows for `fiscal_year=2023` to reach 60 PDFs total.

---

## 5. Technology Stack

| Layer | Tool | Technical justification |
|---|---|---|
| Primary LLM | Groq — `llama-3.3-70b-versatile` | Free tier; ~750 tok/s; sufficient quality for grounded synthesis |
| Fallback LLM | Google AI Studio — `gemini-2.0-flash` | Generous free quota; swap via `LLM_PROVIDER` env var |
| Embeddings | `BAAI/bge-small-en-v1.5` | 384-dim; top of MTEB at this size; runs on free Colab T4 |
| Re-ranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder for high-precision top-k re-ranking |
| Vector store | ChromaDB (persistent) | Local single-folder backend; no external services |
| Keyword index | `bm25s` | Pure-Python BM25; in-process |
| Orchestration | LangChain (LCEL) | Composable chain syntax; mature ecosystem for RAG components |
| PDF parsing | PyMuPDF (`fitz`) + `unstructured` | PyMuPDF for fast text + page metadata; `unstructured` for table/heading detection |
| Eval | Ragas | Established RAG eval framework with faithfulness, answer-relevancy, context metrics |
| Tracking | MLflow | Run-level metric logging and comparison UI |
| UI | Streamlit | Single-file chat UI; HF Spaces native support |
| Hosting | Hugging Face Spaces (Docker SDK) | Free, persistent public URL; same Docker image runs locally |
| Container | Docker + docker-compose | Reproducibility |
| CI | GitHub Actions | Lint, tests, sample eval on PR |
| Tests | pytest | Standard |
| Heavy compute | Google Colab (T4 free tier) | Embedding the full corpus |

Library versions pinned in `requirements.txt`; APIs verified against current docs before implementation, since LangChain and Ragas evolve quickly.

---

## 6. System Architecture

```
┌─────────────────────────┐
│  FTSE 100 IR pages      │       OFFLINE PIPELINE (Colab, one-shot)
│  (manifest CSV)         │
└──────────┬──────────────┘
           │ scraper.py (polite, 1s delay)
           ▼
┌─────────────────────────┐
│  data/raw/{ticker}/     │
│   {year}.pdf            │
└──────────┬──────────────┘
           │ PyMuPDF + unstructured  → page+heading-aware chunks
           ▼
┌─────────────────────────────────────────────┐
│  Chunks w/ metadata                         │
│  {text, ticker, year, page, section, hash}  │
└──────────┬──────────────────────────────────┘
           │ BGE-small embeddings (Colab GPU)
           ▼
┌─────────────────────────┐     ┌──────────────────────────┐
│  ChromaDB (vectors)     │     │  BM25 index (bm25s)      │
└──────────┬──────────────┘     └──────────┬───────────────┘
           │                                │
           └────────────┬───────────────────┘
                        │
        ╔═══════════════╧════════════════════════════════════╗
        ║       ONLINE QUERY PATH (HF Spaces / local)         ║
        ╠═════════════════════════════════════════════════════╣
        ║   User query → Streamlit chat                       ║
        ║       │                                             ║
        ║   LLM query rewriter                                ║
        ║       │                                             ║
        ║   Hybrid retrieval (Top-50 dense + Top-50 BM25,     ║
        ║                     Reciprocal Rank Fusion)         ║
        ║       │                                             ║
        ║   BGE re-ranker → top-5 chunks                      ║
        ║       │                                             ║
        ║   Groq Llama 3.3 70B with grounded-citation prompt  ║
        ║       │                                             ║
        ║   Citation validator (reject + regenerate if bad)   ║
        ║       │                                             ║
        ║   Streamlit: answer + click-through citations       ║
        ╚═════════════════════════════════════════════════════╝

        ┌───────────────────────────────────────────┐
        │ EVAL TRACK (offline, on every code change) │
        │  qa_test_set.jsonl  →  full pipeline  →    │
        │  Ragas metrics  →  logged to MLflow        │
        └───────────────────────────────────────────┘
```

---

## 7. Repository Structure

Follows the standard layout in `Projects/CONTRIBUTING.md`:

```
filings-rag/
├── README.md                        ← 9-section per CONTRIBUTING.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example                     ← LLM keys, paths
├── .github/workflows/ci.yml
├── notebooks/
│   ├── 01_eda_corpus.ipynb
│   ├── 02_chunking_experiments.ipynb
│   ├── 03_embedding_index_build.ipynb   ← Colab
│   ├── 04_retrieval_evaluation.ipynb
│   └── 05_end_to_end_eval.ipynb
├── src/
│   ├── ingestion/
│   │   ├── scraper.py
│   │   ├── pdf_parser.py
│   │   └── chunker.py
│   ├── retrieval/
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── bm25_index.py
│   │   ├── hybrid.py                ← RRF fusion
│   │   └── reranker.py
│   ├── generation/
│   │   ├── llm_client.py            ← provider-agnostic
│   │   ├── query_rewriter.py
│   │   └── prompts.py
│   ├── eval/
│   │   ├── qa_dataset.py
│   │   ├── ragas_eval.py
│   │   └── mlflow_logger.py
│   └── utils/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/                         ← gitignored
│   ├── processed/                   ← gitignored
│   ├── vector/                      ← gitignored
│   ├── filings_manifest.csv         ← committed
│   └── qa_test_set.jsonl            ← committed
├── tests/
│   ├── test_chunker.py
│   ├── test_hybrid.py
│   ├── test_llm_client.py
│   ├── test_prompts.py
│   └── test_scraper.py
├── reports/figures/
└── docs/
    ├── architecture.md
    ├── eval_methodology.md
    └── 2026-05-20-filings-rag-design.md   ← this file
```

---

## 8. Evaluation Methodology

### QA test set (40 questions, hand-labelled)

| Category | Count | Example |
|---|---|---|
| Single-company factual | 10 | "What was BP's adjusted operating profit in 2024?" |
| Multi-section synthesis | 10 | "Summarise Tesco's main climate-related risks across their TCFD section and risk factors." |
| Multi-company comparison | 10 | "How do HSBC and Barclays differ in how they describe credit-risk exposure to commercial real estate?" |
| Adversarial / refusal | 10 | "What was Tesla's revenue in 2024?" (not in corpus — system should refuse) |

Row schema in `qa_test_set.jsonl`:

```json
{"question": "...", "ground_truth_answer": "...", "ground_truth_citations": [{"ticker": "...", "year": 2024, "pages": [12, 13]}], "category": "single_factual | multi_section | multi_company | adversarial"}
```

### Question templates (25 of 40; the remaining 15 are authored against the corpus)

**Single-company factual (7):**
1. What did `{COMPANY}` report as adjusted operating profit for fiscal year `{YEAR}`?
2. How many employees did `{COMPANY}` have at the close of `{YEAR}`?
3. What were `{COMPANY}`'s total revenues by reporting segment in `{YEAR}`?
4. What was `{COMPANY}`'s reported Scope 1 + Scope 2 greenhouse-gas emissions for `{YEAR}`?
5. What dividend did `{COMPANY}` declare per share for `{YEAR}`?
6. Who is the CEO of `{COMPANY}` as named in the `{YEAR}` annual report?
7. What was `{COMPANY}`'s reported R&D spend (or equivalent) in `{YEAR}`?

**Multi-section synthesis (6):**
8. Summarise `{COMPANY}`'s climate-related risks across the TCFD section and the principal-risks section in `{YEAR}`.
9. How does `{COMPANY}`'s `{YEAR}` annual report describe its approach to AI and automation, across the strategy and risk sections?
10. What are the principal cybersecurity risks `{COMPANY}` disclosed in `{YEAR}`, and what mitigations are described?
11. How does `{COMPANY}` describe its capital allocation policy across the strategic and financial review sections of `{YEAR}`?
12. Summarise `{COMPANY}`'s `{YEAR}` discussion of geopolitical risk, drawing from both the risk factors and the chairman's statement.
13. What does `{COMPANY}` disclose about executive remuneration structure and any pay-ratio commentary in `{YEAR}`?

**Multi-company comparison (7):**
14. How do `HSBC` and `Barclays` differ in how they describe credit-risk exposure to commercial real estate in `{YEAR}`?
15. Compare the way `BP` and `Shell` describe their energy-transition strategies in their `{YEAR}` annual reports.
16. How do `Tesco` and `Sainsbury's` describe inflation impact on their cost base in `{YEAR}`?
17. Compare the climate transition-risk language used by `Rio Tinto` and `Anglo American` in their `{YEAR}` filings.
18. How do `AstraZeneca` and `GSK` describe their R&D pipeline risks in `{YEAR}`?
19. Compare how `Vodafone` and `BT Group` discuss capex commitments in `{YEAR}`.
20. How does executive remuneration disclosure differ between `Lloyds`, `NatWest`, and `Barclays` in `{YEAR}`?

**Adversarial / refusal (5):**
21. What was Tesla's revenue in `{YEAR}`? *(not a FTSE 100 company; expected: refusal)*
22. What is the predicted share price of `{COMPANY}` for next year? *(speculative; expected: refusal)*
23. Should I invest in `{COMPANY}` based on its `{YEAR}` filing? *(financial advice; expected: refusal with explanation)*
24. What did `{COMPANY}` discuss in its 2019 annual report? *(outside corpus year range; expected: refusal)*
25. What did `{COMPANY}`'s CEO say in their TV interview last month? *(out-of-corpus modality; expected: refusal)*

### Metrics tracked (logged to MLflow per run)

- **Retrieval-only** (notebook 04): `recall@5`, `recall@10`, `MRR` against ground-truth chunks
- **End-to-end** (notebook 05, Ragas): `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`
- **Refusal accuracy** on the 10 adversarial questions
- **Citation validity** — % of cited (ticker, year, page) tuples that exist in retrieval results

### Ablation table (populated in `eval_methodology.md`)

| Config | recall@5 | faithfulness | answer_relevancy | refusal_acc |
|---|---|---|---|---|
| Dense only | — | — | — | — |
| BM25 only | — | — | — | — |
| Hybrid (RRF) | — | — | — | — |
| Hybrid + re-ranker | — | — | — | — |
| + query rewriting | — | — | — | — |

CI runs a 5-question sample eval on every PR and posts the diff against `main` in the PR comment.

---

## 9. Error Handling

| Failure mode | Handling |
|---|---|
| Groq rate-limit / 429 | Exponential backoff (1, 2, 4s); fall through to Gemini after 3 retries |
| Both LLM providers down | User-facing Streamlit error; log to MLflow |
| Empty retrieval (no chunks above similarity threshold) | Prompt forces refusal; no LLM generation attempted |
| Invalid citation generated | Reject + regenerate once with stricter prompt; if still invalid, return answer with "citation could not be verified" warning |
| PDF parse failure during indexing | Log + skip + surface in run report; annotate manifest |
| Context overflow after retrieval | Truncate to top-3 chunks; log warning |
| HF Spaces cold start | UI shows "warming up — first query may take 20s" message |

---

## 10. Testing Strategy

| Test | Scope |
|---|---|
| `test_chunker.py` | Page metadata preserved; no chunks span PDFs; max-token boundary respected |
| `test_hybrid.py` | RRF math deterministic on synthetic inputs; stable tie-breaking |
| `test_llm_client.py` | Provider switch logic (mocked Groq + Gemini); retry/backoff |
| `test_prompts.py` | Citation-extraction regex on well-formed and malformed model outputs |
| `test_scraper.py` | URL validation; content-type check; no scraping beyond manifest |
| CI smoke test | 5-chunk in-memory mini-corpus + mocked LLM; full pipeline runs <30s; asserts non-empty answer + valid citation |
| Nightly integration | 2-PDF Chroma index in-memory + 3 fixed queries; asserts grounded answers |

---

## 11. Deployment

**Primary:** Hugging Face Spaces (Docker SDK) — free, persistent public URL.
**Backups:** Streamlit Community Cloud (no Docker), Render free tier (sleeps after inactivity), Fly.io free tier.
**Local:** `docker compose up` runs the full stack (Streamlit + Chroma + persisted index volume).

---

## 12. Scaling Path

| Dimension | Change |
|---|---|
| Corpus 60 → 10k PDFs | ChromaDB → Qdrant or pgvector with HNSW; embed job moves to Prefect/Airflow batched pipeline; indices in object storage |
| Multilingual filings | Swap BGE-small for `bge-m3`; add language detection in chunking |
| Multi-tenant | Per-user auth + isolated collections; Streamlit → FastAPI + JS frontend |
| Live updates | Scheduled ingestion job (daily/weekly); incremental embedding; versioned indices |
| Free LLM → paid SLA | Provider-agnostic client already supports it; add Redis caching for repeat queries |
| Read-only QA → analyst workflows | Saved queries, exported PDF reports, comparison tables across companies |

---

## 13. Definition of Done

- [ ] 60 PDFs ingested and indexed (Chroma + BM25)
- [ ] Streamlit app runs locally via `docker compose up` and on a public HF Spaces URL
- [ ] All 5 notebooks executed top-to-bottom without errors, results committed
- [ ] Ragas eval baseline logged to MLflow; ablation table populated in `eval_methodology.md`
- [ ] CI green (lint + pytest + sample-eval)
- [ ] README (9-section per `CONTRIBUTING.md`) includes hero metrics, screenshots, demo URL, and the eval ablation table
- [ ] Portfolio root `README.md` updated with Project #2 entry
