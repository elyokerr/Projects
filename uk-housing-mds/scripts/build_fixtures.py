"""Build synthetic test fixtures for PPD, NSPL, and HPI.

Writes:
    tests/fixtures/ppd_mini.csv   — 50 rows, headerless, 16 cols
    tests/fixtures/nspl_mini.zip  — contains Data/NSPL_mini.csv (20 rows)
    tests/fixtures/hpi_mini.csv   — ~36 rows (12 months × 3 regions), with header

Run:
    .venv\\Scripts\\python.exe scripts/build_fixtures.py
"""

from __future__ import annotations

import csv
import io
import random
import uuid
import zipfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX_DIR = ROOT / "tests" / "fixtures"
FIX_DIR.mkdir(parents=True, exist_ok=True)

random.seed(20260523)

# Realistic UK postcodes covering several formats
POSTCODES = [
    "SW1A 1AA",
    "E1 6AN",
    "M1 1AE",
    "B33 8TH",
    "CR2 6XH",
    "DN55 1PT",
    "W1A 0AX",
    "EC1A 1BB",
    "LS1 4DT",
    "BT79 0NG",
]

DISTRICTS = [
    ("WESTMINSTER", "GREATER LONDON"),
    ("CITY OF LONDON", "GREATER LONDON"),
    ("MANCHESTER", "GREATER MANCHESTER"),
    ("BIRMINGHAM", "WEST MIDLANDS"),
    ("LEEDS", "WEST YORKSHIRE"),
]

PROPERTY_TYPES = ["D", "S", "T", "F"]


def build_ppd(path: Path, n: int = 50) -> None:
    start = date(2020, 1, 1)
    end = date(2025, 12, 31)
    span_days = (end - start).days

    rows = []
    for i in range(n):
        d = start + timedelta(days=random.randint(0, span_days))
        district, county = random.choice(DISTRICTS)
        rows.append(
            [
                "{" + str(uuid.uuid4()).upper() + "}",
                random.randint(50_000, 2_000_000),
                d.isoformat(),
                random.choice(POSTCODES),
                random.choice(PROPERTY_TYPES),
                random.choice(["Y", "N"]),
                random.choice(["F", "L"]),
                str(random.randint(1, 250)),  # paon
                "" if random.random() < 0.7 else f"FLAT {random.randint(1, 30)}",
                f"{random.choice(['HIGH', 'CHURCH', 'MILL', 'STATION'])} STREET",
                "" if random.random() < 0.5 else f"LOCALITY {i % 7}",
                random.choice(["LONDON", "MANCHESTER", "BIRMINGHAM", "LEEDS"]),
                district,
                county,
                random.choice(["A", "B"]),
                "A",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({n} rows)", flush=True)


def build_nspl(zip_path: Path, n: int = 20) -> None:
    regions = ["E12000007", "E12000002", "E12000003", "E12000005", "N99999999"]
    lads = ["E09000033", "E09000001", "E08000003", "E08000025", "E08000035"]

    buf = io.StringIO()
    w = csv.writer(buf, quoting=csv.QUOTE_MINIMAL)
    w.writerow(["pcd", "lsoa11", "lad22cd", "rgn", "lat", "long", "imd"])
    for i in range(n):
        pcd = POSTCODES[i % len(POSTCODES)]
        w.writerow(
            [
                pcd,
                f"E0100{1000 + i:04d}",
                random.choice(lads),
                random.choice(regions),
                round(random.uniform(50.0, 58.0), 6),
                round(random.uniform(-5.0, 1.5), 6),
                random.randint(1, 10),
            ]
        )

    csv_bytes = buf.getvalue().encode("utf-8")
    # Use forward slashes inside the archive (gotcha #6)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Data/NSPL_mini.csv", csv_bytes)
    print(f"wrote {zip_path} (Data/NSPL_mini.csv, {n} rows)", flush=True)


def build_hpi(path: Path) -> None:
    regions = [
        ("E12000007", "London"),
        ("E12000002", "North West"),
        ("E12000003", "Yorkshire and The Humber"),
    ]
    rows = []
    for year, month in [(2024, m) for m in range(1, 13)]:
        d = date(year, month, 1)
        for area_code, region_name in regions:
            rows.append(
                [
                    d.isoformat(),
                    area_code,
                    region_name,
                    round(random.uniform(150_000, 600_000), 2),
                    round(random.uniform(100.0, 160.0), 2),
                ]
            )

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["date", "area_code", "region_name", "average_price", "index"])
        for r in rows:
            w.writerow(r)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def main() -> None:
    build_ppd(FIX_DIR / "ppd_mini.csv", n=50)
    build_nspl(FIX_DIR / "nspl_mini.zip", n=20)
    build_hpi(FIX_DIR / "hpi_mini.csv")


if __name__ == "__main__":
    main()
