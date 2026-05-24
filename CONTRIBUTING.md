# Contributing to this Portfolio

This repository is a curated collection of data-science and machine-learning projects. Every project follows the same internal structure so the repo stays clean, navigable, and easy to present.

---

## Adding a new project

1. **Copy the template** - duplicate `_template/` into a new folder at the repo root.
2. **Name the folder in `kebab-case`**, describing what the project *does*. Examples:
   - `saas-churn-prediction`
   - `urban-jungle-price-estimator`
   - `retail-demand-forecasting`
3. **Fill in the README** following the 9-section template (see below).
4. **Place files in the correct standard folders** (see the structure below).
5. **Commit with a clear message** describing the project's purpose.

---

## Standard project structure

Every project in this repo follows this layout:

```
<project-name>/
├── README.md                ← Project overview (see template)
├── requirements.txt         ← Python dependencies
├── .gitignore               ← Project-specific ignores (large data, etc.)
│
├── notebooks/               ← Numbered Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── src/                     ← Reusable Python modules
│   ├── __init__.py
│   ├── data/                ← Loading & ingestion
│   ├── features/            ← Feature engineering
│   ├── models/              ← Training, evaluation, prediction
│   └── utils/               ← Shared helpers
│
├── data/                    ← Datasets (contents gitignored, structure kept)
│   ├── raw/                 ← Original, immutable data
│   ├── interim/             ← Intermediate transformations
│   ├── processed/           ← Final feature sets ready for modelling
│   └── external/            ← Third-party data
│
├── models/                  ← Serialized model artifacts (.joblib, .pkl)
│
├── reports/
│   └── figures/             ← Generated plots, screenshots, exported PDFs
│
├── app/                     ← Optional: Streamlit / FastAPI / dashboard
│
├── tests/                   ← Pytest tests
│
└── docs/                    ← Optional: extended documentation
```

Not every project needs every folder. Omit ones you don't use, but keep the names consistent with the ones you do.

---

## README template (9 sections)

Every project README should follow this structure so visitors know what to expect:

1. **Title + one-line tagline** - what the project does in plain English
2. **Hero results** - a small table of 3–5 headline metrics
3. **The business problem** - *why* this matters
4. **What this demonstrates** - skills shown, with pointers to files
5. **Quick start** - how to run it (1–2 commands)
6. **Project structure** - annotated tree of the project folder
7. **Methodology** - how the solution works, step-by-step
8. **Tech stack** - table of tools used and why
9. **Limitations & next steps** - honest assessment of what could be better

A starter version of this lives in `_template/README.md`.

---

## Conventions

| Concern | Convention |
|---|---|
| Folder names | `kebab-case` |
| Python files | `snake_case.py` |
| Notebook names | `NN_short_description.ipynb` (zero-padded number, kebab in description optional) |
| Branch | All work on `main` for now (small portfolio) |
| Commits | Imperative mood, describe the *why*: `Add Urban Jungle price estimator` not `update files` |
| Large files | Anything > 100 MB is gitignored. Document where the data comes from in `data/README.md` |

---

## Repo-wide gitignore

The root `.gitignore` already excludes:

- `mlruns/`, `mlartifacts/` - MLflow experiment artifacts (regenerated locally)
- `data/raw/`, `data/interim/`, `data/processed/`, `data/external/` - keep the folder structure, ignore the contents
- `__pycache__/`, `.ipynb_checkpoints/`, `.pytest_cache/`
- IDE clutter (`.vscode/`, `.idea/`, `.claude/`)
- Common binary blobs (`*.bin`, `*.h5`)

Add project-specific ignores in the project's own `.gitignore`.
