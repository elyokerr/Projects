# Data

All data sources are free. The repository runs end-to-end without any of them via a committed synthetic fixture panel; live data is optional.

## Sources

### Elexon BMRS Insights API
- **What:** generation by fuel type (`FUELHH`), national demand out-turn and initial estimate (`INDO` / `ITSDO`), embedded wind/solar forecast.
- **Access:** `https://data.elexon.co.uk/bmrs/api/v1` — no API key required.
- **Client:** `src/data/elexon_client.py`.
- **⚠️ Pending verification:** the exact response field names (`startTime`, `fuelType`, `generation`, `demand`) and query parameter names are coded against the published Elexon Insights documentation but have not been confirmed against a live response. Confirm against a real call before using live data; the public function signatures are stable.

### ENTSO-E Transparency Platform
- **What:** GB day-ahead auction price (documentType `A44`, domain `10YGB----------A`).
- **Access:** requires a **free** API token. Register at <https://transparency.entsoe.eu/>, then request API access by emailing **transparency@entsoe.eu** (free; typically granted within a day). Set the token as the `ENTSOE_TOKEN` environment variable.
- **Client:** `src/data/entsoe_client.py`.
- **⚠️ Pending verification:** the XML namespace and element paths (`TimeSeries > Period > Point > price.amount`) and the `securityToken` parameter name are coded against the ENTSO-E schema docs but not confirmed against a live A44 response.
- **Currency:** ENTSO-E publishes GB day-ahead prices in **EUR/MWh**. No FX conversion is applied — prices are stored in the published currency. A clean free GBP series (or a documented FX source) is a future addition; inventing an FX series would be dishonest.

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
