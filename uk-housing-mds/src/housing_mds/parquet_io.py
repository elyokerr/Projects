"""CSV → Parquet conversion with explicit dtypes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def csv_to_parquet(
    src: Path,
    dst: Path,
    *,
    dtypes: dict[str, str] | None = None,
    column_names: list[str] | None = None,
    header: int | None = 0,
    parse_dates: list[str] | None = None,
) -> Path:
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        src,
        names=column_names,
        header=header,
        dtype=dtypes,
        parse_dates=parse_dates,
        low_memory=False,
    )
    df.to_parquet(dst, index=False, engine="pyarrow", compression="snappy")
    return dst
