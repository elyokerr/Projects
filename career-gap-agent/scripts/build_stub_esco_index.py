"""Build a small stub ESCO index (real embeddings) so the repo runs without the full ESCO CSV.
Replace by running scripts/build_esco_index.py on the real skills_en.csv before deploy."""

from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

SKILLS = [
    "python (computer programming)",
    "sql",
    "r (programming language)",
    "java (computer programming)",
    "scala (programming language)",
    "machine learning",
    "deep learning",
    "natural language processing",
    "computer vision",
    "statistics",
    "data visualisation",
    "pandas",
    "numpy",
    "scikit-learn",
    "pytorch",
    "tensorflow",
    "keras",
    "xgboost",
    "data analysis",
    "data engineering",
    "etl (extract transform load)",
    "data warehousing",
    "apache spark",
    "apache kafka",
    "apache airflow",
    "dbt (data build tool)",
    "snowflake",
    "google bigquery",
    "amazon web services",
    "microsoft azure",
    "google cloud platform",
    "docker",
    "kubernetes",
    "git (version control)",
    "ci/cd",
    "fastapi",
    "flask",
    "rest api",
    "tableau",
    "power bi",
    "looker",
    "microsoft excel",
    "hypothesis testing",
    "a/b testing",
    "experimental design",
    "time series analysis",
    "forecasting",
    "regression analysis",
    "classification",
    "clustering",
    "feature engineering",
    "mlops",
    "model deployment",
    "langchain",
    "large language models",
    "prompt engineering",
    "vector databases",
    "hadoop",
    "mysql",
    "postgresql",
    "mongodb",
    "data modelling",
    "business intelligence",
    "communication",
    "stakeholder management",
    "problem solving",
    "agile methodology",
]


def main() -> None:
    out = Path("data/esco")
    out.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    embs = np.array(list(model.embed([s.lower() for s in SKILLS])), dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    df = pd.DataFrame(
        {
            "conceptUri": [f"stub:{i}" for i in range(len(SKILLS))],
            "preferredLabel": SKILLS,
        }
    )
    df.to_parquet(out / "skills.parquet", index=False)
    np.save(out / "skill_embeddings.npy", embs)
    print(f"Wrote stub index: {len(SKILLS)} skills, embeddings {embs.shape}", flush=True)


if __name__ == "__main__":
    main()
