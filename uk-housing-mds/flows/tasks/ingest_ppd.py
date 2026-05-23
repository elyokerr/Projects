"""PPD (HM Land Registry Price Paid Data) ingest task.

Headerless 16-column CSV. Downloaded idempotently, verified, normalised on the
postcode column, and written to Parquet in the landing zone.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from prefect import task

from src.housing_mds.download import download_file, verify_csv_magic
from src.housing_mds.parquet_io import csv_to_parquet
from src.housing_mds.postcode import normalise_postcode

_BASE = (
    "http://prod1.publicdata.landregistry.gov.uk."
    "s3-website-eu-west-1.amazonaws.com"
)
_URLS = {
    "full": f"{_BASE}/pp-complete.csv",
    "increment": f"{_BASE}/pp-monthly-update-new-version.csv",
}

PPD_COLUMNS = [
    "transaction_unique_id",
    "price_paid",
    "date_of_transfer",
    "postcode",
    "property_type",
    "new_build_flag",
    "tenure",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category_type",
    "record_status",
]

_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ppd_mini.csv"


@task(retries=3, retry_delay_seconds=60)
def ingest_ppd(target_dir: Path, mode: str = "increment") -> Path:
    """Ingest PPD to a Parquet file under target_dir.

    Modes:
        - "full": download pp-complete.csv
        - "increment": download monthly update
        - "fixture": copy tests/fixtures/ppd_mini.csv (no network)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_archive = target_dir.parent.parent / "raw_archive"
    raw_archive.mkdir(parents=True, exist_ok=True)

    if mode == "fixture":
        stamp = "fixture"
        raw_csv = raw_archive / "ppd_fixture.csv"
        if not raw_csv.exists():
            shutil.copy(_FIXTURE, raw_csv)
        print(f"[ingest_ppd] fixture mode: using {raw_csv}", flush=True)
    elif mode in _URLS:
        stamp = date.today().strftime("%Y-%m")
        raw_csv = raw_archive / f"ppd_{mode}_{stamp}.csv"
        print(f"[ingest_ppd] downloading {_URLS[mode]} -> {raw_csv}", flush=True)
        download_file(_URLS[mode], raw_csv)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if not verify_csv_magic(raw_csv):
        raise RuntimeError(f"PPD CSV failed magic-byte check: {raw_csv}")

    out_path = target_dir / f"{stamp}.parquet"
    dtypes = {c: "string" for c in PPD_COLUMNS}
    dtypes["price_paid"] = "int64"
    # date_of_transfer is parsed via parse_dates, so omit from dtype map
    del dtypes["date_of_transfer"]

    print(f"[ingest_ppd] converting to parquet -> {out_path}", flush=True)
    csv_to_parquet(
        raw_csv,
        out_path,
        column_names=PPD_COLUMNS,
        dtypes=dtypes,
        header=None,
        parse_dates=["date_of_transfer"],
    )

    # Postcode normalisation: read parquet back, normalise, rewrite.
    print("[ingest_ppd] normalising postcodes", flush=True)
    df = pd.read_parquet(out_path)
    df["postcode"] = df["postcode"].map(normalise_postcode).astype("string")
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

    print(f"[ingest_ppd] done: {out_path} ({len(df)} rows)", flush=True)
    return out_path
