"""One-off: capture an Adzuna snapshot for secret-free runs.
Run with ADZUNA_APP_ID/KEY in the environment; writes data/fixtures/adzuna_snapshot.json.
"""

import json
import os
from pathlib import Path

import httpx

OUT = Path("data/fixtures/adzuna_snapshot.json")
BASE = "https://api.adzuna.com/v1/api/jobs/gb/search/1"


def main() -> None:
    params = {
        "app_id": os.environ["ADZUNA_APP_ID"],
        "app_key": os.environ["ADZUNA_APP_KEY"],
        "what": "data scientist",
        "where": "london",
        "results_per_page": 50,
    }
    raw = httpx.get(BASE, params=params, timeout=30).json()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"Wrote {len(raw.get('results', []))} postings to {OUT}", flush=True)


if __name__ == "__main__":
    main()
