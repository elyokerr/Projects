"""Train the full recommender pipeline and (optionally) pickle artifacts.

Loads the real parquet from ``data/raw/online_retail_II.parquet`` when present,
otherwise falls back to the committed sample fixture. Cleans, trains the
end-to-end :class:`RecommenderPipeline`, and writes the fitted pipeline to
``models/pipeline.pkl``. Convenience script; not covered by the test suite.
"""

import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path so src imports work when run as a script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.clean import clean_transactions  # noqa: E402
from src.pipeline import RecommenderPipeline  # noqa: E402

RAW = _PROJECT_ROOT / "data" / "raw" / "online_retail_II.parquet"
SAMPLE = _PROJECT_ROOT / "tests" / "fixtures" / "sample_transactions.parquet"
MODELS_DIR = _PROJECT_ROOT / "models"
OUT = MODELS_DIR / "pipeline.pkl"


def main() -> None:
    if RAW.exists():
        print(f"Loading raw data: {RAW}")
        df = clean_transactions(pd.read_parquet(RAW))
    else:
        print(f"Raw data not found; using committed sample: {SAMPLE}")
        # The sample fixture is already cleaned.
        df = pd.read_parquet(SAMPLE)

    print(f"Training on {len(df)} rows, {df['customer_id'].nunique()} customers.")
    pipe = RecommenderPipeline.train(df)
    print(f"Ranker trained: {pipe.ranker is not None}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(pipe, fh)
    print(f"Wrote artifact: {OUT}")


if __name__ == "__main__":
    main()
