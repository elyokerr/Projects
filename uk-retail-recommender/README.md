# UK Retail Recommender

> A two-stage personalised product recommender built on real UK e-commerce data. Give it a customer, get back the products they are most likely to buy next, retrieved and ranked the way production recommenders do it.

Multi-source retrieval (popularity, item-to-item co-purchase, ALS matrix factorisation with FAISS, and a neural two-tower model) feeds a LightGBM LambdaMART ranker, evaluated on a temporal split and served through a mobile-friendly FastAPI demo.

---

## Hero results

Model ladder on the committed sample, recall / NDCG / MAP at k = 10 (114 evaluated customers, temporal split):

| Rung | Recall@10 | NDCG@10 | MAP@10 |
|---|---|---|---|
| Popularity (baseline) | 0.081 | 0.064 | 0.025 |
| Item-to-item co-purchase | 0.419 | 0.423 | 0.272 |
| ALS matrix factorisation | 0.219 | 0.244 | 0.126 |
| **Two-stage (retrieval + ranker)** | **0.580** | **0.698** | **0.571** |

The retrieval recall ceiling is 0.602, so the two-stage ranker recovers about 96% of the items the retrieval stage could possibly surface, and it beats every single retriever. The headline comes from the real evaluation run, not a placeholder.

> **A note on the data.** The repository ships a committed sample with latent customer-cluster structure so it clones, tests, and runs the full ladder with no download. The same code runs on the full Online Retail II dataset (about 1.07M rows) via `scripts/download_data.py` and `scripts/train_all.py`.

---

## The business problem

An online retailer with thousands of products wants to show each returning customer the items they are most likely to buy next, to lift basket size and repeat purchases. Scoring every product for every customer in real time is too slow, so production systems use two stages: a fast retrieval stage narrows thousands of products to a few hundred candidates, then a slower, more accurate ranking stage orders those candidates. This project builds that system on a real UK retailer's transaction history, which is implicit feedback (purchases, not star ratings), as most production recommenders are.

---

## What this demonstrates

| Capability | Where to look |
|---|---|
| Two-stage retrieval-and-ranking architecture | `src/pipeline.py` |
| Implicit-feedback matrix factorisation (ALS) | `src/retrieval/als.py` |
| Approximate nearest-neighbour retrieval (FAISS) | `src/retrieval/faiss_index.py` |
| Neural two-tower retrieval (in-batch negatives) | `src/retrieval/two_tower.py` |
| Item-to-item co-purchase | `src/retrieval/item2item.py` |
| Learning-to-rank (LightGBM LambdaMART) | `src/ranking/ranker.py`, `src/ranking/features.py` |
| Rank-based offline evaluation + retrieval ceiling | `src/eval/metrics.py`, `src/eval/ladder.py` |
| Leakage-safe temporal evaluation | `src/data/split.py` |
| FastAPI serving with a mobile demo | `app/main.py`, `app/templates/` |
| Containerisation and CI | `Dockerfile`, `.github/workflows/uk-retail-recommender-ci.yml` |

---

## Quick start

Runs with no downloads or secrets (uses the committed sample). Needs Python 3.11.

```bash
git clone https://github.com/elyokerr/Projects.git
cd Projects/uk-retail-recommender

python -m venv .venv
.venv\Scripts\activate          # Windows. On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite (no secrets needed)
pytest tests -q

# See the full model-ladder evaluation (real numbers)
RUN_SLOW=1 pytest tests/test_e2e_eval.py -v -s

# Start the web app
uvicorn app.main:app
# open http://localhost:8000
```

The first web request trains the pipeline on the sample (a few seconds), then recommendations are instant. To run on the full data, run `python scripts/download_data.py` then `python scripts/train_all.py`.

### Docker

```bash
docker compose up --build
# open http://localhost:7860
```

---

## Project structure

```
uk-retail-recommender/
├── app/
│   ├── main.py                 FastAPI: GET / , GET /recommend , GET /similar , GET /health
│   ├── templates/              index.html + _results.html + _similar.html (Jinja2 + HTMX + Tailwind)
│   └── static/                 app.css
├── src/
│   ├── data/
│   │   ├── clean.py            drop cancellations, non-positive quantities, null customers
│   │   ├── split.py            temporal train/test split (leakage-safe)
│   │   └── interactions.py     customer-by-item implicit matrix
│   ├── retrieval/
│   │   ├── popularity.py       global top sellers
│   │   ├── item2item.py        co-purchase within baskets
│   │   ├── als.py              implicit-feedback matrix factorisation
│   │   ├── faiss_index.py      approximate nearest-neighbour search
│   │   ├── two_tower.py        neural two-tower retriever (Colab-trained)
│   │   └── candidates.py       union and dedup of all sources
│   ├── ranking/
│   │   ├── features.py         per-candidate features + FEATURE_COLUMNS
│   │   └── ranker.py           LightGBM LambdaMART
│   ├── eval/
│   │   ├── metrics.py          recall@k, ndcg@k, map@k, coverage
│   │   ├── evaluate.py         per-recommender evaluation + retrieval ceiling
│   │   └── ladder.py           the full model-ladder evaluation
│   └── pipeline.py             RecommenderPipeline: train + recommend
├── notebooks/                  01 EDA · 02 retrieval · 03 two-tower (Colab) · 04 ranking + eval
├── scripts/                    download_data.py · build_sample.py · train_all.py
├── data/                       data/README.md (download steps); raw/ gitignored
├── tests/                      unit + API tests; fixtures/ holds the committed sample; RUN_SLOW ladder eval
├── Dockerfile · docker-compose.yml · requirements.txt
└── docs/                       design doc
```

---

## Methodology

**1. Clean and split.** Transactions are cleaned (cancellations, non-positive quantities, and rows without a customer are removed) and split temporally: the model trains on the earlier period and is evaluated on each returning customer's held-out later purchases. A leakage test asserts no test interaction predates the cutoff.

**2. Retrieve (stage 1).** Four candidate generators each return scored items: global popularity, item-to-item co-purchase from the customer's history, ALS matrix factorisation with FAISS nearest-neighbour search, and a neural two-tower retriever. Their candidates are unioned and de-duplicated into a few hundred per customer. The two-tower is trained on Colab and is optional: the system runs without it.

**3. Rank (stage 2).** A LightGBM LambdaMART model scores the candidates using features built from the customer and the items (the per-source retrieval scores, item price and popularity, and the customer's recency, frequency, and monetary values). Positives are the customer's held-out purchases; the model is grouped by customer.

**4. Evaluate.** The full ladder is scored on the temporal split with recall@k, NDCG@k, and MAP@k, plus catalogue coverage. The retrieval recall ceiling (the fraction of held-out items that reach the candidate set) is reported alongside end-to-end recall, which separates the retrieval contribution from the ranking contribution.

**5. Serve.** A FastAPI app loads the fitted pipeline and exposes `/recommend`, `/similar`, and `/health`. The HTMX demo lets you pick a customer and see the ranked recommendations with each item's score and retrieval source.

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Data handling | pandas, scipy, pyarrow | Cleaning and the sparse interaction matrix |
| Matrix factorisation | `implicit` (ALS) | Standard implicit-feedback factorisation |
| Nearest-neighbour search | FAISS | Cosine retrieval over learned vectors |
| Neural retrieval | PyTorch | Two-tower model with in-batch negatives |
| Ranking | LightGBM (LambdaMART) | Learning-to-rank over candidate features |
| Backend | FastAPI, Uvicorn | API plus the demo |
| Frontend | Jinja2, HTMX, Tailwind (CDN) | Mobile-responsive, no build step |
| Tests, lint | pytest, ruff | Unit, API, and a gated end-to-end eval |
| Packaging, CI | Docker, GitHub Actions | Reproducible image and free CI |
| Heavy training | Google Colab (T4) | Two-tower training off the local machine |
| Hosting | Hugging Face Spaces (Docker SDK) | Free public URL, mobile accessible |

---

## Limitations and next steps

- **Sample by default.** The committed sample is synthetic with latent structure so the ladder runs with no download. The full Online Retail II run is reproducible via `scripts/download_data.py` and `scripts/train_all.py`.
- **Two-tower is the one Colab step.** It trains on a GPU in `notebooks/03_two_tower_colab.ipynb` and exports embeddings the pipeline loads when present; the system runs without it.
- **Invoice-level data, not clickstream.** This is a next-purchase recommender, not a real-time session model.
- **Next:** sequence-based retrieval (GRU4Rec or SASRec) as another source; FAISS IVF or HNSW for larger catalogues; online evaluation through an A/B test on click-through and conversion; scheduled retraining.

---

## Data attribution

Transaction data is the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset from the UCI Machine Learning Repository. See [`data/README.md`](data/README.md) for download and regeneration steps.
