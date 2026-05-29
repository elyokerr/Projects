# Data

All data sources are free. The repository runs end-to-end without any of them via a committed synthetic fixture panel; live data is optional.

## Sources

### Elexon BMRS — the data source for everything

No API key required. Base URL `https://data.elexon.co.uk/bmrs/api/v1`. Client: `src/data/elexon_client.py`. All field names and endpoints below were **confirmed against live responses on 2026-05-29**.

- **Generation by fuel** (`/datasets/FUELHH`) → `fetch_generation_by_fuel`. Half-hourly MW per fuel type (CCGT, WIND, NUCLEAR, BIOMASS, interconnectors, etc.).
- **Demand out-turn** (`/demand/outturn`) → `fetch_demand`. Returns both INDO (`initialDemandOutturn`) and ITSDO (`initialTransmissionSystemDemandOutturn`) per settlement period in one response.
- **System (imbalance) price** (`/balancing/settlement/system-prices/{date}`) → `fetch_system_price`. This is the forecast **target**. GB operates a single imbalance price (System Sell Price = System Buy Price), so `systemSellPrice` is used. One request per day; the client iterates a date range and caches. Prices are **GBP/MWh** and can be **negative** (genuine GB market behaviour — the client never clips them).

Build a real panel with:

```bash
python scripts/build_real_panel.py --start 2023-01-01 --end 2024-12-31
```

This writes `data/processed/real_panel.parquet` (same wide schema as the fixture). Load it with `src.build.fixtures.load_panel_from_parquet(path)`.

### Why not ENTSO-E day-ahead price?

The original design targeted the GB **day-ahead auction** price from the ENTSO-E Transparency Platform. Confirmed against the live API (2026-05-29): **ENTSO-E has no GB price data from 2021 onward** — following Brexit, GB left the EU single electricity market and stopped publishing to ENTSO-E (data exists only up to end-2020). The project therefore targets the Elexon **system (imbalance) price**, which is current, post-Brexit, free, and authentically GB-specific. The ENTSO-E client (`src/data/entsoe_client.py`) is retained and works for pre-2021 GB data and other European bidding zones, but is not used in the live pipeline.

## The fixture panel

`tests/fixtures/fixture_panel.parquet` is a ~30-day, half-hourly **synthetic** panel (price, demand, generation by fuel) generated with a fixed seed (`src/build/fixtures.py`). It backs the test suite, the `RUN_SLOW=1` end-to-end smoke test, and the Streamlit demo so everything runs with zero secrets. It is **not** real market data and its backtest numbers are illustrative only.

## Directory layout

```
data/
├── raw/         ← cached API responses (gitignored)
├── interim/     ← intermediate transforms (gitignored)
└── processed/   ← aligned panel ready for modelling (gitignored)
```

Folder structure is tracked; contents are gitignored. The real panel is built by the ingestion + build pipeline; the Colab training notebook (`notebooks/03_colab_train_global.ipynb`) consumes the processed panel.
