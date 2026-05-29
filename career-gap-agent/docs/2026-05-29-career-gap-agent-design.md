# AI Career Gap Analyst: Design

An agent that turns live UK job postings into a personalised, evidence-backed skills-gap plan.

---

## 1. Project Overview

The Career Gap Analyst is a tool-using AI agent. Given a CV and a target role and location, it retrieves current UK job postings for that role, distils the genuinely in-demand skills from them, compares those skills against the CV, and returns a prioritised "skills to learn" report in which every recommended gap is backed by how many of the retrieved postings demanded that skill.

The system is built around a single LangGraph tool-calling agent. The language model decides which tools to invoke and in what order, loops until it has gathered enough information, and synthesises a final report. A reflection step checks the report for groundedness before it is returned.

The repository runs end-to-end with no secrets against a committed snapshot of postings, and runs in live mode against the Adzuna API when an API key is supplied.

---

## 2. Problem Statement

A job-seeker targeting a role such as "Data Scientist, London" faces dozens of postings whose skill requirements overlap but are described inconsistently ("PyTorch", "torch", "deep learning"). Reading them manually to answer two questions, *what is actually in demand for this role?* and *what am I missing?*, is slow, subjective, and hard to keep current as the market moves.

The Career Gap Analyst automates that research. It grounds its answer in real, current postings rather than generic advice, normalises inconsistent skill language to a standard taxonomy so demand can be counted reliably, and produces an output that is both ranked (most in-demand gaps first) and evidence-backed (each gap cites posting counts).

---

## 3. Users

- **Primary:** a job-seeker who supplies a CV (pasted text or an uploaded PDF) and a target role plus location, and receives a ranked skills-gap report with a supporting demand table.
- **Secondary:** an engineer browsing the repository who wants a worked example of a controllable, evaluated tool-using agent with a measurable analytical core.

Out of scope: automated job applications, CV rewriting, and any retrieval from sites whose terms prohibit programmatic access.

---

## 4. Dataset

**Adzuna API (job postings).** Live UK postings for a `(role, location)` query, returning title, company, description, salary where present, and location. Access uses a free `app_id` and `app_key`. A committed JSON **snapshot** of one frozen pull (roughly 50–100 postings across a few demo roles) lets the test suite, CI, and the public demo run without any key.

**ESCO taxonomy (skills).** The European Skills, Competences, Qualifications and Occupations classification, a free, downloadable CSV of approximately 13,900 skills, applicable to the UK/EU labour market. It is committed in processed form as a parquet file plus a precomputed skill-embedding index, so the repository needs no live download to run. (O\*NET is the United States analogue and is not used here.)

**CV.** Supplied at request time as pasted text or an uploaded PDF; PDF text is extracted with PyMuPDF. CV content is not stored.

Source attribution for Adzuna and ESCO is recorded in `data/README.md`.

---

## 5. Tech Stack

| Layer | Tool | Justification |
|---|---|---|
| Agent framework | LangGraph | Explicit state-machine control over a tool-calling loop, with a dedicated reflection node and an iteration cap. |
| Language model | Groq Llama 3.3 70B (primary), Google Gemini 2.0 Flash (fallback) | Free-tier hosted models behind one provider-agnostic client; fallback covers per-provider quota limits. |
| Embeddings | `bge-small-en-v1.5` via fastembed (ONNX) | CPU-only skill-to-ESCO matching; no GPU required. |
| Skill taxonomy | ESCO | Canonical skill names enable reliable demand counting and a precision/recall evaluation surface. |
| PDF parsing | PyMuPDF | CV text extraction. |
| Backend | FastAPI + Uvicorn | Serves the agent behind a small REST surface and renders the UI. |
| Frontend | Jinja2 templates + HTMX + Tailwind (CDN) | Mobile-responsive server-rendered pages; the analyse call is asynchronous with no full-page reload and no frontend build step. |
| Observability | Langfuse | Traces every agent step, tool call, latency, token cost, and error. |
| Tabular data | pandas, pyarrow | ESCO processing and demand tables. |
| Tests / lint | pytest, ruff | Unit, graph, and gated end-to-end eval tests; style enforcement. |
| Packaging | Docker, docker-compose | Local parity and the deployment image. |
| CI / hosting | GitHub Actions, Hugging Face Spaces (Docker SDK) | Free CI for public repos; a single free live deployment with a public URL. |

---

## 6. Architecture

### 6.1 Agent graph

A single LangGraph agent over a typed `AgentState`:

```
planner (LLM, tool-calling) → tool node → [loop back to planner until done] → reflection → END
```

- **Planner node.** The language model is bound to the tool set and chooses the next tool call (or signals completion) based on the conversation and accumulated state.
- **Tool node.** Executes the requested tool and writes its result back into the state.
- **Reflection node.** Runs deterministic groundedness checks on the draft report: every reported gap must map to at least one retrieved posting, and every named skill must be a real ESCO entity. If a check fails, it returns one corrective message to the planner; otherwise it finalises. A hard iteration cap bounds the loop.

`AgentState` is the single shared contract between nodes and carries: the message history, the target role and location, the retrieved postings, extracted and ESCO-normalised skills, the CV skill set, the demand table, and the draft report.

### 6.2 Tools

| Tool | Responsibility |
|---|---|
| `search_jobs(role, location, n)` | Retrieve postings via the Adzuna client, or from the committed snapshot when no key is present. |
| `parse_cv(text_or_pdf)` | Produce clean CV text from pasted text or an uploaded PDF. |
| `extract_skills(text)` | Use the language model to extract candidate skill phrases from a posting or the CV. |
| `normalise_to_esco(phrases)` | Embedding-match each candidate phrase to its nearest ESCO skill above a cosine threshold; return the standardised skill set. |
| `aggregate_demand(postings)` | Frequency-rank ESCO skills across all retrieved postings into a demand table. |
| `compute_gap(cv_skills, demand)` | Set-difference the CV skills against demand and score each gap by demand frequency, producing the ranked gap list. |

### 6.3 Serving

FastAPI exposes `GET /` (the form), `POST /analyze` (runs the agent and returns an HTML report fragment), and `GET /health`. The frontend posts the CV and target role via HTMX and swaps in the returned report fragment. The agent layer is independent of the web layer and is exercised directly by the test suite.

---

## 7. Repository Structure

```
career-gap-agent/
├── README.md  requirements.txt  .env.example  Dockerfile  docker-compose.yml
├── src/
│   ├── agent/      graph.py · state.py · nodes.py · tools.py
│   ├── data/       adzuna_client.py · esco_loader.py · cv_parser.py
│   ├── skills/     extractor.py · esco_matcher.py · gap.py
│   ├── generation/ llm_client.py · prompts.py
│   └── eval/       component_eval.py · langfuse_setup.py
├── app/
│   ├── main.py            FastAPI routes
│   ├── templates/         index.html + report fragment (Jinja2 + HTMX)
│   └── static/            css / js
├── notebooks/
│   ├── 01_esco_eda.ipynb
│   ├── 02_skill_matching_eval.ipynb
│   ├── 03_agent_walkthrough.ipynb
│   └── 04_component_eval.ipynb
├── data/
│   ├── esco/              committed processed parquet + embedding index
│   ├── fixtures/          Adzuna snapshot + gold-labelled postings
│   └── raw/               gitignored
├── tests/
├── reports/figures/
└── docs/
    └── 2026-05-29-career-gap-agent-design.md
```

---

## 8. Evaluation Methodology

**Skill extraction and ESCO normalisation accuracy.** A hand-labelled gold set of roughly 25 postings, each annotated with its true ESCO skills, is the reference. The `extract_skills → normalise_to_esco` pipeline is scored on precision, recall, and F1 against this gold set. Notebook `02` sweeps the cosine-match threshold and selects the operating point from the resulting precision/recall curve.

**Gap-report quality.** Two layers. A deterministic groundedness check reports the proportion of reported gaps that map to at least one real posting and the proportion of named skills that are valid ESCO entities. An LLM-as-judge applies a short rubric (groundedness, relevance, actionability) over a handful of CV-by-role cases.

**Agent behaviour.** Tool-success rate, refusal accuracy on out-of-scope inputs (empty CV, unknown role), and mean steps, tokens, and latency per run.

**Observability.** Langfuse traces every run; a representative trace is captured in the README.

---

## 9. Error Handling

| Failure | Handling |
|---|---|
| No Adzuna key | Fall back to the committed snapshot; the fallback is logged and surfaced in the UI. |
| Adzuna rate-limit or HTTP error | Retry with backoff, then fall back to the snapshot. |
| Zero postings for the role and location | Report that no postings were found and suggest broadening the search; no fabrication. |
| Language-model provider error or quota | Fall back from Groq to Gemini via the provider-agnostic client. |
| Malformed tool arguments | Schema validation followed by one corrective re-prompt. |
| Skill not present in ESCO | Caught by the reflection node and dropped or flagged. |
| Unparseable or empty CV | Return a clear error with guidance; no silent empty run. |
| Report still ungrounded after one corrective pass | Return the report with an explicit caveat; the hard iteration cap prevents an infinite loop. |

---

## 10. Testing Strategy

- **Unit tests:** the Adzuna client (mocked transport plus snapshot), the ESCO loader, the CV parser, the skill extractor (mocked language model), the ESCO matcher (deterministic over fixtures), the gap function (pure), and the provider-agnostic LLM client (provider switch and retry).
- **Graph tests:** a stub language model returning scripted tool calls drives the LangGraph, asserting the state transitions, that the reflection node triggers on an ungrounded draft, and that the iteration cap fires.
- **API tests:** FastAPI `TestClient` covers `/health` and `/analyze` (happy path, empty CV, unknown role).
- **Component eval:** a `RUN_SLOW=1`-gated test runs the gold-set evaluation end-to-end.

All tests run without secrets, using the committed snapshot and fixtures.

---

## 11. Deployment

The application is a single FastAPI service run under Uvicorn, deployed to Hugging Face Spaces using the Docker SDK. Adzuna and language-model keys are provided as Space secrets for live mode; without them the deployment serves the committed snapshot. The UI is mobile-responsive and reachable from any device at the Space's public URL. A `Dockerfile` and `docker-compose.yml` give local parity, and GitHub Actions runs ruff and pytest on every push, filtered to this project's paths.

---

## 12. Scaling Path

- Replace the embedding model with `bge-large` and add a fine-tuned skill matcher to raise ESCO matching precision.
- Cache postings and add a scheduled refresh so demand tables accumulate over time, enabling rising- and falling-demand trends per skill.
- Add further posting sources behind the existing `search_jobs` tool interface; the tool abstraction makes additional sources additive.
- Add trajectory-level agent evaluation and an online feedback loop logged to Langfuse.

---

## 13. Definition of Done

- The agent runs end-to-end on the committed snapshot with no secrets; `uvicorn app.main:app` and `pytest` are both clean.
- Live mode works against the real Adzuna API with a key.
- The gold-set component evaluation produces real precision, recall, and F1 figures, reported as the README headline metric.
- Langfuse tracing is live and a representative trace is captured in the README.
- The LangGraph reflection step and iteration cap are verified by the stub-language-model graph test.
- The README (nine sections), this design doc, and `data/README.md` (Adzuna and ESCO attribution) are complete, and CI is green.
- The application is deployed to Hugging Face Spaces with a live, mobile-accessible URL recorded in the README.
