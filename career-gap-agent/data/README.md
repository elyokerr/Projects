# Data

This project commits two small data artifacts so the repo clones and runs with no API keys, and gitignores all working data (`raw/`, `interim/`, `processed/`, `external/`).

## Committed artifacts

| Path | What it is | How to regenerate |
|---|---|---|
| `esco/skills.parquet` | Canonical skill labels (`conceptUri`, `preferredLabel`) | `python scripts/build_stub_esco_index.py` (stub) or `python scripts/build_esco_index.py` (full ESCO) |
| `esco/skill_embeddings.npy` | Unit-normalised BGE-small embeddings, one row per skill | built alongside the parquet by the same script |
| `fixtures/adzuna_snapshot.json` | A frozen set of UK job postings for secret-free runs | `python scripts/build_snapshot.py` (needs an Adzuna key) |
| `fixtures/gold_skills.json` | Hand-labelled postings with their true ESCO skills, for evaluation | hand-edited |

## ESCO skills taxonomy

ESCO (European Skills, Competences, Qualifications and Occupations) is published by the European Commission and is free to download.

- Source: <https://esco.ec.europa.eu/en/use-esco/download>
- This repo ships a small **stub** index (67 common data/ML/engineering skills) built by `scripts/build_stub_esco_index.py`, so nothing needs downloading to run the tests or the app.
- To use the full taxonomy: download the classification CSV (`skills_en.csv`, v1.2), place it in `data/raw/esco/skills_en.csv`, then run:
  ```bash
  python scripts/build_esco_index.py
  ```
  This rewrites `esco/skills.parquet` and `esco/skill_embeddings.npy` with the full skill set (about 13,900 skills). The first run downloads the BGE-small ONNX model into the fastembed cache.

## Adzuna job postings

Adzuna offers a free developer API for UK job listings.

- Get a free key: <https://developer.adzuna.com/>
- Set `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in your `.env` (copy `.env.example`).
- Without a key, `src/data/adzuna_client.py` falls back to `fixtures/adzuna_snapshot.json`, so the app and tests still run.
- To refresh the snapshot from live data: set the keys, then `python scripts/build_snapshot.py`.

Postings are used under the Adzuna API terms of use.
