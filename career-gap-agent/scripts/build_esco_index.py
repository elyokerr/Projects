"""One-off: build the committed ESCO skill index.

Download skills_en.csv from the ESCO portal (v1.2) into data/raw/esco/ first.
Outputs (committed, not gitignored): data/esco/skills.parquet + skill_embeddings.npy
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fastembed import TextEmbedding

RAW = Path("data/raw/esco/skills_en.csv")
OUT = Path("data/esco")
MODEL = "BAAI/bge-small-en-v1.5"


def main() -> None:
    df = pd.read_csv(RAW)
    df = df[df["skillType"] == "skill/competence"].copy()
    df = df[["conceptUri", "preferredLabel"]].dropna().drop_duplicates("preferredLabel")
    df = df.reset_index(drop=True)
    labels = df["preferredLabel"].str.lower().tolist()

    model = TextEmbedding(model_name=MODEL)
    embs = np.array(list(model.embed(labels)), dtype=np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)  # unit-normalise for cosine via dot

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "skills.parquet", index=False)
    np.save(OUT / "skill_embeddings.npy", embs)
    print(f"Wrote {len(df)} skills + embeddings {embs.shape}", flush=True)


if __name__ == "__main__":
    main()
