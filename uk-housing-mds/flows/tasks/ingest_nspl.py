"""NSPL (ONS National Statistics Postcode Lookup) ingest task.

NSPL is published as a ZIP on the ONS Open Geography Portal. The CSV of
interest is `Data/NSPL_*.csv`. Extraction uses Python's `zipfile` (NOT
PowerShell Compress-Archive) — see portfolio gotcha #6.

URL discovery (verified 2026-05-24): the ONS Open Geography Portal hosts NSPL
as an ArcGIS dataset. Browse https://geoportal.statistics.gov.uk/search?tags=NSPL
or data.gov.uk for the current release; click into the dataset and copy the
"Download CSV" URL — pattern is roughly
    https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/<ITEM-ID>/csv?layers=0
(`<ITEM-ID>` changes per release; Feb 2026 = 419355d8a54741f19025ba97e35da55a).

LIMITATION: the verified URL above returns a flat CSV ("hosted table" variant),
NOT the legacy ZIP-with-`Data/NSPL_*.csv` that the current `quarterly`-mode
implementation expects. Two ways forward before first prod run:
  (a) Find the legacy bulk-ZIP URL (Open Geography Portal item page → "Open in
      ArcGIS Hub" → bulk download). It does still exist for some releases.
  (b) Refactor this task to handle the CSV directly: drop the `zipfile.ZipFile`
      branch, `verify_zip_magic` → `verify_csv_magic`, write parquet from the
      single CSV. Column set is similar but verify schema against `_NSPL_DTYPES`.

Until either path is wired in, `mode="fixture"` is the only verified mode.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

from prefect import task

from src.housing_mds.download import download_file, verify_zip_magic
from src.housing_mds.parquet_io import csv_to_parquet

# Feb 2026 release, CSV variant — refactor the task to CSV before using (see docstring).
_NSPL_URL = "TBD"  # see module docstring; verified URL candidate stored below.
_NSPL_URL_CANDIDATE = (
    "https://open-geography-portalx-ons.hub.arcgis.com/api/download/v1/items/"
    "419355d8a54741f19025ba97e35da55a/csv?layers=0"
)

_NSPL_DTYPES = {
    "pcd": "string",
    "lsoa11": "string",
    "lad22cd": "string",
    "rgn": "string",
    "lat": "float64",
    "long": "float64",
    "imd": "Int64",
}

_FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "nspl_mini.zip"


def _quarter_stamp(d: date) -> str:
    return f"{d.year}-Q{((d.month - 1) // 3) + 1}"


@task(retries=3, retry_delay_seconds=60)
def ingest_nspl(target_dir: Path, mode: str = "quarterly") -> Path:
    """Ingest NSPL to a Parquet file under target_dir.

    Modes:
        - "quarterly": download from `_NSPL_URL` (TBD)
        - "fixture": use tests/fixtures/nspl_mini.zip (no network)
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_archive = target_dir.parent.parent / "raw_archive"
    raw_archive.mkdir(parents=True, exist_ok=True)

    if mode == "fixture":
        stamp = "fixture"
        zip_path = raw_archive / "nspl_fixture.zip"
        if not zip_path.exists():
            shutil.copy(_FIXTURE, zip_path)
        print(f"[ingest_nspl] fixture mode: using {zip_path}", flush=True)
    elif mode == "quarterly":
        if _NSPL_URL == "TBD":
            raise RuntimeError(
                "NSPL URL is TBD — set _NSPL_URL in flows/tasks/ingest_nspl.py "
                "to the latest ONS Open Geography Portal release before "
                "running in quarterly mode."
            )
        stamp = _quarter_stamp(date.today())
        zip_path = raw_archive / f"nspl_{stamp}.zip"
        print(f"[ingest_nspl] downloading {_NSPL_URL} -> {zip_path}", flush=True)
        download_file(_NSPL_URL, zip_path)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if not verify_zip_magic(zip_path):
        raise RuntimeError(f"NSPL ZIP failed magic-byte check: {zip_path}")

    # Extract via Python zipfile (gotcha #6: NOT PowerShell Compress-Archive)
    extract_dir = raw_archive / f"nspl_extract_{stamp}"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ingest_nspl] extracting -> {extract_dir}", flush=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
        names = zf.namelist()

    # Locate the CSV inside Data/
    csv_candidates = [n for n in names if n.startswith("Data/") and n.endswith(".csv")]
    if not csv_candidates:
        raise RuntimeError(f"No Data/*.csv inside NSPL zip: {names!r}")
    csv_inside = extract_dir / csv_candidates[0]
    print(f"[ingest_nspl] found CSV: {csv_inside}", flush=True)

    out_path = target_dir / f"{stamp}.parquet"
    print(f"[ingest_nspl] converting to parquet -> {out_path}", flush=True)
    csv_to_parquet(csv_inside, out_path, dtypes=_NSPL_DTYPES)

    print(f"[ingest_nspl] done: {out_path}", flush=True)
    return out_path
