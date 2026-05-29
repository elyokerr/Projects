# AI Career Gap Analyst

> A tool-using AI agent that turns live UK job postings into a personalised, evidence-backed skills-gap plan. Paste your CV, name a target role and location, and get a ranked list of the skills you are missing, with each gap backed by how many real postings asked for it.

Built with a single LangGraph agent, skill matching against the ESCO taxonomy, Langfuse tracing, and a mobile-responsive FastAPI web app you can open on your phone.

---

## Hero results

| Item | Value |
|---|---|
| Architecture | Single LangGraph tool-calling agent with a deterministic groundedness check and a hard iteration cap |
| Skill matching | LLM extraction normalised to the **ESCO** skills taxonomy by embedding similarity (skill entity-linking) |
| Evaluation | Component eval (precision / recall / F1) over a hand-labelled gold set, plus agent groundedness checks |
| Observability | Langfuse tracing of every agent step, tool call, latency, and token cost |
| Tests | **36 passing**, 1 gated end-to-end eval, lint clean |
| Reproducibility | Runs with **zero secrets** on a committed Adzuna snapshot and a committed ESCO index |

> **A note on numbers.** This repository ships with a small committed stub ESCO index (67 skills) and a hand-seeded job snapshot so it clones and runs with no API keys. The headline precision / recall / F1 figure is produced by running the gated evaluation (`RUN_SLOW=1`) with a free LLM key and the full ESCO index, as described in [Evaluation](#evaluation). No metric is reported here that has not been measured.

---

## The business problem

Anyone targeting a role, say "Data Scientist, London", faces dozens of job postings whose skill requirements overlap but are written inconsistently ("PyTorch", "torch", "deep learning" all mean the same thing). Reading them by hand to answer two questions, *what is actually in demand for this role?* and *what am I missing?*, is slow and subjective, and goes stale as the market moves.

This agent automates that research. It pulls real current postings, distils the genuinely in-demand skills, normalises them to a standard vocabulary so demand can be counted reliably, compares the result against your CV, and returns a ranked plan where every recommended skill cites how many postings asked for it.

---

## What this demonstrates

| Capability | Where to look |
|---|---|
| Agentic AI (LangGraph state machine, tool-calling loop, reflection, iteration cap) | `src/agent/graph.py`, `src/agent/nodes.py` |
| Tool design for an LLM agent | `src/agent/tools.py` |
| Skill entity-linking to a taxonomy (embedding similarity) | `src/skills/esco_matcher.py`, `src/data/esco_loader.py` |
| LLM skill extraction with robust parsing | `src/skills/extractor.py`, `src/generation/prompts.py` |
| Provider-agnostic LLM client with fallback | `src/generation/llm_client.py` |
| LLMOps / observability | `src/eval/langfuse_setup.py` |
| Rigorous component evaluation (P/R/F1 vs gold) | `src/eval/component_eval.py`, `data/fixtures/gold_skills.json` |
| External-API client with secret-free fallback | `src/data/adzuna_client.py` |
| Mobile-responsive web app | `app/main.py`, `app/templates/` |
| Containerisation and CI | `Dockerfile`, `.github/workflows/career-gap-agent-ci.yml` |

---

## Quick start

Runs with no API keys (uses the committed snapshot and ESCO index). Needs Python 3.11.

```bash
git clone https://github.com/elyokerr/Projects.git
cd Projects/career-gap-agent

python -m venv .venv
.venv\Scripts\activate          # Windows. On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite (no secrets needed)
pytest tests -q

# Start the web app
uvicorn app.main:app
# open http://localhost:8000
```

The form, health check, and CV validation all work with no keys. To run a real analysis, add a free LLM key to a `.env` file (copy `.env.example`):

- `GROQ_API_KEY` (free at <https://console.groq.com/keys>) or `GOOGLE_API_KEY` (free at <https://aistudio.google.com/apikey>)
- Optional: `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` (free at <https://developer.adzuna.com/>) for live postings instead of the snapshot
- Optional: `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` for tracing

### Docker

```bash
docker compose up --build
# open http://localhost:7860
```

---

## Project structure

```
career-gap-agent/
├── app/
│   ├── main.py                 FastAPI: GET / , POST /analyze , GET /health
│   ├── templates/              index.html + _report.html (Jinja2 + HTMX + Tailwind)
│   └── static/                 app.css
├── src/
│   ├── agent/
│   │   ├── state.py            AgentState + Posting / Gap / GapReport
│   │   ├── tools.py            AgentDeps + the search_jobs / analyse_postings tools
│   │   ├── nodes.py            planner, tool, reflection nodes + groundedness check
│   │   └── graph.py            build_graph: LangGraph wiring + iteration cap
│   ├── data/
│   │   ├── adzuna_client.py    job search with committed-snapshot fallback
│   │   ├── esco_loader.py      load committed ESCO parquet + embedding index
│   │   └── cv_parser.py        PDF / text to clean CV text
│   ├── skills/
│   │   ├── extractor.py        LLM candidate-skill extraction
│   │   ├── esco_matcher.py     embedding match to canonical ESCO skills
│   │   └── gap.py              demand aggregation + gap scoring
│   ├── generation/
│   │   ├── llm_client.py       Groq primary, Gemini fallback
│   │   └── prompts.py          extraction prompt
│   └── eval/
│       ├── component_eval.py   precision / recall / F1 over the gold set
│       └── langfuse_setup.py   optional tracing callbacks
├── notebooks/                  01 ESCO EDA (executed) · 02 matching eval · 03 agent walkthrough · 04 component eval
├── scripts/                    build_esco_index.py · build_stub_esco_index.py · build_snapshot.py
├── data/
│   ├── esco/                   committed stub index (skills.parquet + skill_embeddings.npy)
│   └── fixtures/               adzuna_snapshot.json · gold_skills.json
├── tests/                      14 test modules (36 tests + 1 gated e2e)
├── Dockerfile · docker-compose.yml · requirements.txt · .env.example
└── docs/                       design doc
```

---

## Methodology

**1. Retrieve.** `search_jobs(role, location)` calls the Adzuna API, or falls back to the committed snapshot when no key is present, returning a set of postings.

**2. Extract.** For each posting and for the CV, the LLM extracts candidate skill phrases (`extract_skills`). The parser tolerates code-fenced or malformed JSON and returns an empty list rather than crashing.

**3. Normalise.** Each candidate phrase is embedded with a BGE-small model (fastembed, ONNX, CPU) and matched by cosine similarity to the nearest canonical skill in the ESCO taxonomy, above a tuned threshold. This collapses inconsistent wording ("torch" and "deep learning") onto standard labels, which is what makes demand counting reliable.

**4. Aggregate and compare.** Normalised skills are counted across postings to build a demand table, then differenced against the CV skills. Each remaining gap is scored by how many postings asked for it.

**5. Reflect.** Before the report is returned, a deterministic check confirms every reported gap maps to at least one real posting and is a valid ESCO skill. If the check fails, the agent gets one corrective pass. A hard iteration cap guarantees the loop always terminates.

**6. Serve.** A FastAPI app renders the ranked gap report as an HTML fragment via HTMX, so the page updates without a reload and works on a phone.

The agent is a single LangGraph state machine: `planner -> tools -> reflection -> end`. The planner (an LLM bound to the tools) decides which tool to call; a custom tool node executes it and writes structured results into the shared state.

---

## Evaluation

The analytical core is evaluated, not just demonstrated.

- **Skill extraction and normalisation.** `data/fixtures/gold_skills.json` holds hand-labelled postings whose true ESCO skills are known. `src/eval/component_eval.py` reports macro precision, recall, and F1 of the extraction-plus-normalisation pipeline against this gold set. Notebook `02` sweeps the cosine threshold and picks the operating point from the precision/recall curve.
- **Agent groundedness.** The reflection check is unit-tested, and a regression test confirms the iteration cap terminates even when the planner never produces a grounded report.
- **Observability.** With Langfuse keys set, every run is traced (steps, tool calls, latency, token cost).

To produce the headline P/R/F1 on real data, set a free LLM key and run:

```bash
RUN_SLOW=1 pytest tests/test_component_eval_e2e.py -v
```

The committed repo uses a small stub ESCO index so it runs with no keys. Swap in the full ESCO taxonomy by downloading `skills_en.csv` and running `scripts/build_esco_index.py` (see `data/README.md`).

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Agent framework | LangGraph | Explicit control over a tool-calling loop with a reflection node and iteration cap |
| Language model | Groq Llama 3.3 70B, Gemini 2.0 Flash fallback | Free hosted models behind one provider-agnostic client |
| Embeddings | bge-small-en-v1.5 via fastembed (ONNX) | CPU-only skill matching, no GPU needed |
| Skill taxonomy | ESCO | Canonical skill names enable reliable demand counting and a P/R/F1 eval surface |
| PDF parsing | PyMuPDF | CV text extraction |
| Backend | FastAPI, Uvicorn | Small REST surface, serves the UI |
| Frontend | Jinja2, HTMX, Tailwind (CDN) | Mobile-responsive, no build step |
| Observability | Langfuse | Agent tracing |
| Tests, lint | pytest, ruff | Unit, graph, API, and gated eval tests |
| Packaging, CI | Docker, GitHub Actions | Reproducible image and free CI |
| Hosting | Hugging Face Spaces (Docker SDK) | Free public URL, mobile accessible |

---

## Limitations and next steps

- **Stub ESCO index by default.** The committed index has 67 skills so the repo runs with no download. The full ESCO taxonomy (about 13,900 skills) is wired in via `scripts/build_esco_index.py`.
- **Real analysis needs a free LLM key.** The app boots and validates input with no keys, but extraction and the agent need a free Groq or Gemini key. This is by design, to keep the clone-and-run path secret-free.
- **Adzuna field names are best-effort from the docs** until confirmed against a live response, with a snapshot fallback so nothing breaks without a key.
- **Gold set is a starter set.** It expands toward about 25 hand-labelled postings as the full ESCO index is adopted.
- **Next:** more posting sources behind the same `search_jobs` tool (Reed, arbeitnow); a scheduled refresh so demand becomes a time series of rising and falling skills; trajectory-level agent evaluation; a fine-tuned matcher for higher ESCO precision.

---

## Data attribution

Job postings come from the [Adzuna API](https://developer.adzuna.com/) under its terms of use. Skills are matched against the [ESCO classification](https://esco.ec.europa.eu/) published by the European Commission. See [`data/README.md`](data/README.md) for how to regenerate the committed artifacts.
