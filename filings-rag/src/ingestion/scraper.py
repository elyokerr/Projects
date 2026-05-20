"""Polite PDF scraper for FTSE 100 annual reports.

Downloads PDFs listed in a manifest CSV, validates content-type, and
writes them to data/raw/{ticker}/{year}.pdf with a 1-second delay
between requests.
"""

from pathlib import Path
import csv
import json
import os
import time

import requests
from dotenv import load_dotenv

USER_AGENT = "filings-rag/0.1 (educational portfolio project)"


def _head(url: str):
    return requests.head(
        url,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
        timeout=30,
    )


def _get_bytes(url: str) -> bytes:
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.content


def validate_pdf_url(url: str) -> bool:
    """Return True if URL returns 200 OK with Content-Type: application/pdf."""
    resp = _head(url)
    return resp.status_code == 200 and "application/pdf" in resp.headers.get("Content-Type", "")


def download_pdf(url: str, out_path: Path, delay_s: float = 1.0) -> Path:
    """Download URL to out_path. Politeness: sleep `delay_s` after each fetch."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(_get_bytes(url))
    time.sleep(delay_s)
    return out_path


def run_from_manifest(manifest_csv: Path, raw_dir: Path) -> dict:
    """Iterate the manifest CSV and download every row with a pdf_url.

    Returns a dict with `ok` (list of local paths) and `failed` (list of
    rows annotated with the failure reason).
    """
    results: dict = {"ok": [], "failed": []}
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row.get("pdf_url", "").strip()
            if not url:
                results["failed"].append({**row, "reason": "no url"})
                continue
            out = raw_dir / row["ticker"] / f"{row['fiscal_year']}.pdf"
            if out.exists():
                results["ok"].append(str(out))
                continue
            try:
                if not validate_pdf_url(url):
                    results["failed"].append({**row, "reason": "not pdf"})
                    continue
                download_pdf(url, out)
                results["ok"].append(str(out))
            except Exception as e:  # noqa: BLE001 — surface all failures
                results["failed"].append({**row, "reason": str(e)})
    return results


if __name__ == "__main__":
    load_dotenv()
    res = run_from_manifest(
        Path("data/filings_manifest.csv"),
        Path(os.environ["RAW_DIR"]),
    )
    print(json.dumps(
        {
            "ok_count": len(res["ok"]),
            "failed_count": len(res["failed"]),
            "failed": res["failed"],
        },
        indent=2,
    ))
