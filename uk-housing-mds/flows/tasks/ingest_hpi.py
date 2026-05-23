"""HPI (UK House Price Index) ingest task.

The HPI full file is published monthly. The URL follows the pattern
`https://publishing.landregistry.gov.uk/full-file/UK-HPI-full-file-YYYY-MM.csv`.

TBD: Update `_HPI_URL_TEMPLATE` and/or the resolution logic to the latest
YYYY-MM release before first prod run. Until then, `mode="fixture"` is the
only verified mode.
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from prefect import task

from src.housing_mds.download import download_file, verify_csv_magic
from src.housing_mds.parquet_io import csv_to_parquet

# Update YYYY-MM placeholder to the latest published release before first prod run.
_HPI_URL_TEMPLATE = (
    "https://publishing.landregistry.gov.uk/full-file/UK-HPI-full-file-{stamp}.csv"
)

_HPI_DTYPES = {
    "area_code": "string",
    "region_name": "string",
    "average_price": "float64",
    "index": "float64",
}

_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "hpi_mini.csv"


@task(retries=3, retry_delay_seconds=60)
def ingest_hpi(target_dir: Path, mode: str = "monthly") -> Path:
    """Ingest HPI to a Parquet file under target_dir.

    Modes:
        - "monthly": download latest UK HPI full file (URL template — TBD)
        - "fixture": use tests/fixtures/hpi_mini.csv (no network)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_archive = target_dir.parent.parent / "raw_archive"
    raw_archive.mkdir(parents=True, exist_ok=True)

    if mode == "fixture":
        stamp = "fixture"
        raw_csv = raw_archive / "hpi_fixture.csv"
        if not raw_csv.exists():
            shutil.copy(_FIXTURE, raw_csv)
        print(f"[ingest_hpi] fixture mode: using {raw_csv}", flush=True)
    elif mode == "monthly":
        stamp = date.today().strftime("%Y-%m")
        url = _HPI_URL_TEMPLATE.format(stamp=stamp)
        raw_csv = raw_archive / f"hpi_{stamp}.csv"
        print(f"[ingest_hpi] downloading {url} -> {raw_csv}", flush=True)
        download_file(url, raw_csv)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if not verify_csv_magic(raw_csv):
        raise RuntimeError(f"HPI CSV failed magic-byte check: {raw_csv}")

    out_path = target_dir / f"{stamp}.parquet"
    print(f"[ingest_hpi] converting to parquet -> {out_path}", flush=True)
    csv_to_parquet(
        raw_csv,
        out_path,
        dtypes=_HPI_DTYPES,
        parse_dates=["date"],
    )

    print(f"[ingest_hpi] done: {out_path}", flush=True)
    return out_path
