from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_ESCO_DIR = Path(__file__).resolve().parents[2] / "data" / "esco"


@dataclass
class EscoIndex:
    labels: list[str]
    uris: list[str]
    embeddings: np.ndarray  # (N, 384), unit-normalised

    @classmethod
    def load(cls, esco_dir: Path = _ESCO_DIR) -> EscoIndex:
        df = pd.read_parquet(esco_dir / "skills.parquet")
        embs = np.load(esco_dir / "skill_embeddings.npy")
        return cls(
            labels=df["preferredLabel"].tolist(),
            uris=df["conceptUri"].tolist(),
            embeddings=embs,
        )
