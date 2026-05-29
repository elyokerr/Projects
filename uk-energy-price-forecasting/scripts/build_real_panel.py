"""Build a real GB energy panel from live Elexon data and save it to parquet.

Pulls half-hourly generation-by-fuel, demand out-turn, and the GB system
(imbalance) price for a date range, assembles them into a wide frame matching
the fixture-panel schema, and writes ``data/processed/real_panel.parquet``.

The system price (Elexon ``systemSellPrice``) is the forecast TARGET — this is
the post-Brexit, currently-published GB price series. ENTSO-E day-ahead auction
prices are unavailable for GB from 2021 onward.

Usage (from the project root, venv active):

    python scripts/build_real_panel.py --start 2023-01-01 --end 2024-12-31

No API key is required (Elexon is open). Be patient: the system-price endpoint
is one request per day, so a multi-year pull is a few thousand cached requests.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.elexon_client import (  # noqa: E402
    fetch_demand,
    fetch_generation_by_fuel,
    fetch_system_price,
)

OUT_PATH = Path("data/processed/real_panel.parquet")


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def build(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Pull all three sources and assemble the wide panel frame."""
    print(f"Fetching generation by fuel {start} -> {end} ...", flush=True)
    gen = fetch_generation_by_fuel(start, end)
    print(f"  {len(gen)} rows", flush=True)

    print("Fetching demand out-turn ...", flush=True)
    dem = fetch_demand(start, end)
    print(f"  {len(dem)} rows", flush=True)

    print("Fetching system price (one request per day) ...", flush=True)
    price = fetch_system_price(start, end)
    print(f"  {len(price)} rows", flush=True)

    # Pivot generation to wide gen_<fuel> columns.
    gen_wide = (
        gen.pivot_table(index="timestamp", columns="fuel", values="mw", aggfunc="mean")
        .add_prefix("gen_")
        .rename_axis(None, axis=1)
    )
    gen_wide.columns = [c.lower() for c in gen_wide.columns]

    dem = dem.set_index("timestamp")
    price = price.set_index("timestamp")

    # Inner-join on the common half-hourly index.
    panel = price[["price"]].join([dem[["indo", "itsdo"]], gen_wide], how="inner")
    panel = panel.sort_index()
    panel.index.name = "timestamp"
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=_parse_date, required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", type=_parse_date, required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    panel = build(args.start, args.end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.out)
    print(f"\nSaved {len(panel)} rows x {panel.shape[1]} cols to {args.out}", flush=True)
    print(f"Columns: {list(panel.columns)}", flush=True)
    print(f"Price range: {panel['price'].min():.1f} to {panel['price'].max():.1f} GBP/MWh", flush=True)


if __name__ == "__main__":
    main()
