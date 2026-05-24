"""Download helpers with idempotency + magic-byte verification."""

from __future__ import annotations

from pathlib import Path

import requests

_BROWSERY_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def download_file(
    url: str,
    target: Path,
    *,
    force: bool = False,
    chunk_size: int = 1 << 20,
) -> Path:
    """Stream URL to target. Skips when target exists and force=False."""
    target = Path(target)
    if target.exists() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, headers={"User-Agent": _BROWSERY_UA}, timeout=120)
    resp.raise_for_status()
    with target.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return target


def verify_csv_magic(path: Path) -> bool:
    """Reject HTML error pages masquerading as CSV. CSV is text; just check first bytes."""
    head = Path(path).read_bytes()[:200].lstrip().lower()
    return not head.startswith(b"<")


def verify_zip_magic(path: Path) -> bool:
    """ZIP files start with PK\\x03\\x04."""
    return Path(path).read_bytes()[:4] == b"PK\x03\x04"
