"""One-off scaffold for the GE 0.18 file-backed context.

Creates great_expectations/ with three fluent Pandas filesystem datasources, one
per landing zone subdir (ppd, nspl, hpi). Idempotent: safe to re-run.
"""

from __future__ import annotations

from pathlib import Path

import great_expectations as gx

ROOT = Path(__file__).resolve().parents[1]
GE_DIR = ROOT / "great_expectations"
GE_DIR.mkdir(exist_ok=True)

ctx = gx.get_context(context_root_dir=str(GE_DIR))
print("context:", type(ctx).__name__)

for source in ["ppd", "nspl", "hpi"]:
    ds_name = f"landing_{source}"
    base = ROOT / "data" / "landing" / source
    existing_names = list(ctx.datasources.keys())
    if ds_name not in existing_names:
        ctx.sources.add_pandas_filesystem(name=ds_name, base_directory=str(base))
        print(f"datasource added: {ds_name}")
    ds = ctx.get_datasource(ds_name)
    asset_names = [a.name for a in ds.assets]
    if source not in asset_names:
        ds.add_parquet_asset(name=source, batching_regex=r"(?P<fname>.+)\.parquet")
        print(f"asset added: {source}")

print("done")
