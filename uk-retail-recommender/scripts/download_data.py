"""Download Online Retail II from UCI and cache as parquet.

Source: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
(online_retail_II.xlsx, 2 sheets).
"""
import io
import zipfile
from pathlib import Path

import httpx
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
OUT = Path("data/raw/online_retail_II.parquet")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading (~45MB)...", flush=True)
    blob = httpx.get(URL, timeout=120, follow_redirects=True).content
    zf = zipfile.ZipFile(io.BytesIO(blob))
    xlsx_name = next(n for n in zf.namelist() if n.endswith(".xlsx"))
    with zf.open(xlsx_name) as fh:
        sheets = pd.read_excel(fh, sheet_name=None, engine="openpyxl")
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUT}", flush=True)


if __name__ == "__main__":
    main()
