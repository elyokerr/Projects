# UK Retail Recommender: Design

A two-stage personalised product recommender built on real UK e-commerce transaction data.

---

## 1. Project Overview

The UK Retail Recommender predicts the products a returning customer is most likely to buy next. It uses the two-stage architecture that production e-commerce recommenders use: a fast retrieval stage narrows the catalogue down to a few hundred candidate products, then a learned ranking stage orders those candidates by predicted relevance.

Retrieval draws candidates from several sources (global popularity, item-to-item co-purchase, ALS matrix factorisation with approximate-nearest-neighbour search, and a neural two-tower retriever). A LightGBM LambdaMART model then ranks the unioned candidates using features built from the customer and the items. The system is served through a FastAPI service with a small mobile-friendly demo, and the full model ladder is evaluated offline on a temporal split.

The repository runs on a committed sample of the data with no downloads or secrets, and on the full dataset after a one-line download.

---

## 2. Problem Statement

An online retailer with thousands of products wants to show each returning customer the items they are most likely to buy next, to increase basket size and repeat purchases. Scoring every product for every customer in real time is too slow, so production systems split the work into two stages: retrieval narrows thousands of products to a few hundred candidates quickly, and ranking orders those candidates accurately.

This project builds that two-stage system on a real UK retailer's transaction history. The data is implicit feedback (purchases, not star ratings), which is the common case in industry and changes how candidates are generated and how the system is evaluated.

---

## 3. Users

- **Primary:** a backend or service that calls the API with a customer id and a count, and receives a ranked list of recommended products with scores.
- **Secondary:** a person browsing the demo or the repository who selects a customer, sees that customer's purchase history, and sees the ranked recommendations along with the score and the retrieval source behind each one.

Out of scope: real-time clickstream personalisation (the data is invoice-level), image or text content recommendation, and any LLM component.

---

## 4. Dataset

**Online Retail II** (UCI Machine Learning Repository): roughly 1.07 million transaction rows for a real UK-based online retailer between December 2009 and December 2011. Fields include invoice number, stock code, description, quantity, invoice date, unit price, customer id, and country. Access is a direct download, recorded in `data/README.md`.

Cleaning removes cancelled invoices (invoice numbers beginning with `C`), non-positive quantities, and rows without a customer id, then builds a customer-by-item implicit interaction matrix where a purchase is an interaction weighted by count.

A committed sample of a few hundred customers and items lives under `data/fixtures/` so the test suite, CI, and the demo run quickly with no download.

Evaluation uses a temporal split: the model trains on the earlier period and is evaluated on each returning customer's held-out later purchases.

---

## 5. Tech Stack

| Layer | Tool | Justification |
|---|---|---|
| Data handling | pandas, pyarrow | Transaction cleaning and the interaction matrix |
| Matrix factorisation | `implicit` (ALS) | Standard, fast library for implicit-feedback factorisation |
| Approximate NN search | FAISS | Retrieves nearest items to a user or item vector at scale |
| Neural retrieval | PyTorch | Two-tower retriever with in-batch negatives |
| Ranking | LightGBM (LambdaMART) | Learning-to-rank on candidate features, grouped by customer |
| Numerics | NumPy, SciPy | Sparse matrices and metric computation |
| Backend | FastAPI, Uvicorn | API plus the demo, auto-generated Swagger docs |
| Frontend | Jinja2, HTMX, Tailwind (CDN) | Mobile-responsive demo with no build step |
| Tests, lint | pytest, ruff | Unit, pipeline, API, and gated end-to-end tests |
| Packaging, CI | Docker, GitHub Actions | Reproducible image and free CI |
| Heavy training | Google Colab (T4) | Two-tower training off the local machine |
| Hosting | Hugging Face Spaces (Docker SDK) | Free public URL, mobile accessible |

---

## 6. Architecture

### 6.1 Pipeline

```
transactions -> clean -> interaction matrix -> temporal split
                                                   |
                         +-------------------------+-------------------------+
                         | Stage 1: retrieval (candidate generators)         |
                         |  popularity | item-to-item | ALS+FAISS | two-tower|
                         +-------------------------+-------------------------+
                                                   | union + dedup
                         +-------------------------+-------------------------+
                         | Stage 2: ranking (LightGBM LambdaMART on features)|
                         +-------------------------+-------------------------+
                                                   |
                                              top-N recommendations
```

### 6.2 Stage 1: retrieval

Each generator returns candidate item ids with a score:

- **Popularity:** the global top sellers. Serves as a baseline and as the cold-start fallback for customers with no usable history.
- **Item-to-item co-purchase:** items frequently bought alongside the customer's previously purchased items, from a co-occurrence table.
- **ALS matrix factorisation:** the `implicit` library learns user and item vectors; a FAISS index returns the nearest items to the customer's vector.
- **Two-tower neural retriever:** a user tower and an item tower trained with in-batch negatives; item embeddings are indexed in FAISS. Trained on Colab; the system runs without it when its embeddings are absent.

Candidates from all available generators are unioned and de-duplicated into a few hundred per customer.

### 6.3 Stage 2: ranking

A feature builder produces one row per (customer, candidate item) with features including the ALS score, the co-purchase score, the two-tower score, item popularity and price, the customer's recency, frequency, and monetary values, the customer's country, and whether the item was previously bought. A LightGBM LambdaMART model (objective `lambdarank`, grouped by customer) scores the candidates. Positives are the customer's actual held-out purchases; negatives are sampled non-purchased candidates. The model returns the final ranked top-N.

### 6.4 Serving

FastAPI loads the serialised artifacts at startup (ALS factors, the FAISS index, the co-purchase table, the two-tower embeddings if present, and the LightGBM model). The endpoints are `GET /recommend`, `GET /similar`, and `GET /health`. The HTMX demo renders a customer's history and recommendations. The service performs only retrieval, feature building, and ranking; all training happens offline.

---

## 7. Repository Structure

```
uk-retail-recommender/
├── README.md  requirements.txt  .env.example  Dockerfile  docker-compose.yml
├── src/
│   ├── data/        download.py · clean.py · split.py · interactions.py
│   ├── retrieval/   popularity.py · item2item.py · als.py · two_tower.py · faiss_index.py · candidates.py
│   ├── ranking/     features.py · ranker.py
│   ├── eval/        metrics.py · evaluate.py
│   └── pipeline.py  end-to-end recommend(customer_id, k)
├── app/
│   ├── main.py            FastAPI: GET /recommend , GET /similar , GET /health
│   ├── templates/         index.html + fragments (Jinja2 + HTMX)
│   └── static/
├── notebooks/
│   ├── 01_eda_clean.ipynb
│   ├── 02_retrieval.ipynb
│   ├── 03_two_tower_colab.ipynb
│   └── 04_ranking_eval.ipynb
├── scripts/         download_data.py · build_sample.py · train_all.py
├── models/          serialised artifacts (gitignored; sample artifacts rebuilt from fixtures)
├── data/
│   ├── fixtures/    committed customer/item sample
│   └── raw/ processed/   gitignored
├── tests/
└── docs/
    └── 2026-05-29-uk-retail-recommender-design.md
```

---

## 8. Evaluation Methodology

Evaluation is offline on the temporal split. For each returning customer in the test period, the held-out later purchases are the ground truth. Metrics are Recall@k, NDCG@k, and MAP@k at k of 10 (also reported at 5 and 20), plus catalogue coverage to detect models that recommend the same popular items to everyone.

Every rung of the ladder is scored end-to-end on the same test customers:

1. Popularity
2. Item-to-item co-purchase
3. ALS retrieval (top-k directly)
4. Two-tower retrieval (top-k directly)
5. Two-stage: all candidate sources unioned, then the LightGBM ranker

Two two-stage diagnostics are reported: the retrieval recall ceiling, which is the fraction of held-out items that reach the candidate set and the most the ranker could recover, and the end-to-end recall after ranking. Together they separate the retrieval contribution from the ranking contribution. The results state which retriever wins and whether the ranker improves on the best single retriever. Headline numbers come from the real evaluation run.

---

## 9. Error Handling

| Situation | Handling |
|---|---|
| Unknown customer id | Cold-start: popularity-based recommendations, flagged, not an error |
| All candidates already purchased | The already-bought filter is configurable; an empty result is handled cleanly |
| Two-tower embeddings absent | That source is skipped; the system runs on ALS, item-to-item, and popularity |
| Requested k larger than the candidate pool | Return what is available |
| Missing or corrupt artifact at startup | Clear error; the app boots on the committed sample artifacts |
| Data download fails | Clear message; tests and CI use the committed sample |

---

## 10. Testing Strategy

- **Unit tests:** cleaning (cancellations and non-positive quantities removed), the temporal split, the interaction-matrix build, item-to-item co-occurrence, popularity, the ALS wrapper (a small fit on the fixture), the FAISS index (build and query), candidate union and dedup, the feature builder, and the ranker (a small fit and the resulting ordering).
- **Metrics tests:** Recall@k, NDCG@k, and MAP@k asserted against hand-worked tiny examples.
- **Leakage test:** asserts that no test interaction predates the train cutoff. This is the central correctness guard.
- **Pipeline and API tests:** `recommend(customer_id, k)` on the fixture returns k items and handles an unknown customer and a missing two-tower; the FastAPI `/health`, `/recommend` (known and unknown customer), and `/similar` endpoints via the test client.
- A `RUN_SLOW=1` end-to-end test trains the small-sample ladder and runs the evaluation, asserting the two-stage ranker recall is at least the best single retriever.

All tests run CPU-only and with no secrets on the committed sample. The two-tower path is gated when embeddings are absent.

---

## 11. Deployment

A single FastAPI service serves the API and a mobile-responsive HTMX demo. The demo lets a user select a customer id, view that customer's purchase history, and view the top-N recommendations with each item's score and retrieval source, plus a similar-products view. The service is containerised and deployed to Hugging Face Spaces using the Docker SDK on port 7860, with a public URL reachable from any device. The committed sample artifacts let the deployed demo run without the full dataset; training on the full data and loading those artifacts gives a richer demo. A `Dockerfile` and `docker-compose.yml` provide local parity, and GitHub Actions runs ruff and pytest on every push, filtered to this project's paths.

---

## 12. Scaling Path

- Replace the FAISS flat index with IVF or HNSW for larger catalogues.
- Add an online feature store and real-time ranking for production latency.
- Add a sequence-based retriever (GRU4Rec or SASRec) as another candidate source.
- Connect online evaluation through an A/B test on click-through and conversion, and add scheduled retraining.

---

## 13. Definition of Done

- The full ladder is evaluated on the temporal split, producing real Recall@k, NDCG@k, and MAP@k for the README headline, with the winner reported and no placeholders.
- `recommend(customer_id, k)` runs end-to-end on the committed sample with no secrets; pytest and ruff are clean.
- The FastAPI app boots on the sample artifacts, `/recommend` returns ranked items with scores, and the demo works on desktop and phone.
- The leakage test and the metrics tests pass.
- The two-tower is trained on Colab with embeddings that load into the service, and the system also runs without it.
- The README (nine sections), this design doc, and `data/README.md` (Online Retail II attribution) are complete, and CI is green.
- The service is deployed to Hugging Face Spaces with a live, mobile-accessible URL in the README.
