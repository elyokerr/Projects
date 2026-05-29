# Data

All data sources are free. The repository runs end to end without any of them through a committed synthetic fixture panel; live data is optional.

## Sources

### Elexon BMRS

No API key required. Base URL `https://data.elexon.co.uk/bmrs/api/v1`. Client: `src/data/elexon_client.py`. The field names and endpoints below were confirmed against live responses on 2026-05-29.

- **Generation by fuel** (`/datasets/FUELHH`), via `fetch_generation_by_fuel`. Half-hourly MW per fuel type (CCGT, WIND, NUCLEAR, BIOMASS, interconnectors, and so on). The endpoint caps the date range, so the client fetches in 7-day chunks.
- **Demand out-turn** (`/demand/outturn`), via `fetch_demand`. Returns both INDO (`initialDemandOutturn`) and ITSDO (`initialTransmissionSystemDemandOutturn`) per settlement period in one response.
- **System (imbalance) price** (`/balancing/settlement/system-prices/{date}`), via `fetch_system_price`. This is the forecast target. GB operates a single imbalance price (System Sell Price equals System Buy Price), so `systemSellPrice` is used. The endpoint serves one day per request, so the client iterates a date range and caches. Prices are GBP/MWh and can be negative, which is normal GB market behaviour, so the client does not clip them.

Build a real panel with:

```bash
python scripts/build_real_panel.py --start 2023-01-01 --end 2024-12-31
```

This writes `data/processed/real_panel.parquet` (the same wide schema as the fixture). Load it with `src.build.fixtures.load_panel_from_parquet(path)`.

### Why not the ENTSO-E day-ahead price

The original design targeted the GB day-ahead auction price from the ENTSO-E Transparency Platform. Checking the live API (2026-05-29) showed that ENTSO-E has no GB price data from 2021 onward: after Brexit, GB left the EU single electricity market and stopped publishing to ENTSO-E, so data exists only up to end-2020. The project therefore targets the Elexon system (imbalance) price, which is current, free, and GB-specific. The ENTSO-E client (`src/data/entsoe_client.py`) is kept because it works for pre-2021 GB data and other European bidding zones, but it is not used in the live pipeline.

## The fixture panel

`tests/fixtures/fixture_panel.parquet` is a roughly 30-day, half-hourly synthetic panel (price, demand, generation by fuel) generated with a fixed seed (`src/build/fixtures.py`). It backs the test suite, the `RUN_SLOW=1` end-to-end smoke test, and the Streamlit demo, so everything runs with no secrets. It is not real market data, and its backtest numbers are illustrative only. The real headline numbers come from the Elexon pull above.

## Directory layout

```
data/
├── raw/         cached API responses (gitignored)
├── interim/     intermediate transforms (gitignored)
└── processed/   aligned panel ready for modelling (gitignored)
```

The folder structure is tracked and the contents are gitignored. The real panel is built by `scripts/build_real_panel.py`; the Colab training notebook consumes the processed panel.
