"""Adzuna job-search client with a committed-snapshot fallback.

Field names follow the documented Adzuna GB search response and are best-effort
pending live confirmation (gotcha #50). The HTTP call is injectable for testing.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx

from src.agent.state import Posting

_SNAPSHOT = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "adzuna_snapshot.json"
_BASE = "https://api.adzuna.com/v1/api/jobs/gb/search/1"

Transport = Callable[[str, dict], dict]


def _default_transport(url: str, params: dict) -> dict:
    resp = httpx.get(url, params=params, timeout=20.0)
    resp.raise_for_status()
    return resp.json()


def _parse_results(raw: dict) -> list[Posting]:
    out: list[Posting] = []
    for r in raw.get("results", []):
        out.append(
            Posting(
                title=r.get("title", "").strip(),
                company=(r.get("company") or {}).get("display_name", ""),
                description=r.get("description", ""),
                location=(r.get("location") or {}).get("display_name", ""),
                salary_min=r.get("salary_min"),
                salary_max=r.get("salary_max"),
            )
        )
    return out


def _load_snapshot(n: int) -> list[Posting]:
    raw = json.loads(_SNAPSHOT.read_text(encoding="utf-8"))
    return _parse_results(raw)[:n]


def search_jobs(
    role: str,
    location: str,
    n: int = 30,
    app_id: str | None = None,
    app_key: str | None = None,
    transport: Transport = _default_transport,
) -> list[Posting]:
    if not app_id or not app_key:
        return _load_snapshot(n)
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": role,
        "where": location,
        "results_per_page": n,
        "content-type": "application/json",
    }
    try:
        raw = transport(_BASE, params)
    except Exception as exc:  # noqa: BLE001 — any network failure → snapshot (gotcha #50)
        print(f"Adzuna call failed ({exc}); falling back to snapshot", flush=True)
        return _load_snapshot(n)
    return _parse_results(raw)[:n]
