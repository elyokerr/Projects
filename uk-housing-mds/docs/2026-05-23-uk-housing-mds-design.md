# UK Housing Market — Modern Data Stack Pipeline

**Design document · 2026-05-23**

## 1. Project overview

A scheduled batch ELT pipeline that ingests three UK public housing datasets, loads them into a SQL warehouse, models them through a dbt project, validates them with multi-layer data-quality checks, and publishes the resulting marts as a public analytics site.

The system is dual-target: it runs against DuckDB locally (development, CI, distributable artefact) and against BigQuery in production (the scheduled monthly run). The dbt project, the Prefect flow, and the data-quality suites are written once and execute against either warehouse through profile switching.

## 2. Problem statement

UK residential property transactions are public data, but the relevant sources are fragmented across three government bodies and three formats. Anyone wanting to ask the obvious analytical questions — *how have transaction volumes in a given region moved against the national HPI since a given year?*, *which local authorities are trading above or below their long-run HPI benchmark?* — first has to download multiple bulk files, reconcile postcode-to-geography lookups, join against a monthly index series, and re-run the whole process every time fresh data is published.

The pipeline solves this by producing a refreshed, benchmarked, geo-enriched warehouse of UK residential transactions on a fixed monthly schedule, queryable directly (DuckDB file or BigQuery dataset) or browsed through an Evidence.dev analytics site.

## 3. Users

| User | How they consume the project |
|---|---|
| Property analysts / housing researchers | Query the marts directly — either by downloading the published DuckDB file release or running queries against the public BigQuery dataset. |
| Local-authority stakeholders | Read the Evidence.dev site for regional trend pages (transaction volume, median price, premium-to-HPI). |
| Engineers reviewing the repository | Read the dbt DAG, the Prefect flow, and the CI / data-quality test results. |

## 4. Datasets

| Source | Contents | Format | Refresh | Volume |
|---|---|---|---|---|
| HM Land Registry — Price Paid Data (PPD) | Every residential transaction in England and Wales since 1995: price, date, postcode, property type, new-build flag, tenure, PAON/SAON, locality, district, county | CSV bulk + monthly increment + monthly change file | Monthly | ~27 M rows |
| ONS National Statistics Postcode Lookup (NSPL) | UK postcode to OA / LSOA / MSOA / local authority / region / country with lat/lon and IMD decile | CSV bulk | Quarterly | ~2.6 M postcodes |
| UK House Price Index (HPI) | Average price and index by region, local authority, and property type, monthly | CSV | Monthly | ~50 k rows |

Ingestion strategy:

- **PPD.** One-off bulk backfill on first run, then monthly increment via the published "data update" URL. Idempotent loads keyed on `transaction_unique_id`.
- **NSPL.** Quarterly truncate-and-load; effective-dated in the warehouse so historical postcodes still resolve to the geography that applied at transaction time.
- **HPI.** Monthly truncate-and-load.

All three sources are unauthenticated public bulk files; the pipeline holds no source credentials.

## 5. Tech stack

| Layer | Tool | Justification |
|---|---|---|
| Ingestion | Python + `requests` + `pyarrow` | Bulk CSVs land as partitioned Parquet for warehouse parity between DuckDB and BigQuery. |
| Orchestration | Prefect (local execution mode) | Task retries, structured logging, parametrised flows, no hosted-server dependency. |
| Warehouse (dev / CI) | DuckDB | Single-file embedded warehouse; identical SQL surface to BigQuery for the modelled layers; fast local iteration. |
| Warehouse (production) | BigQuery free tier | Real cloud warehouse, free at the project's data volume (well inside the 1 TB/month query allowance). |
| Transformation | dbt (`dbt-duckdb`, `dbt-bigquery`) | Industry-standard modelling layer; profile switching gives the dual-warehouse target from one codebase. |
| Data quality (landing) | Great Expectations | Source-side schema and value validation before any warehouse load. |
| Data quality (warehouse) | dbt tests (generic + singular) | Cross-table and in-warehouse invariants. |
| BI / serving | Evidence.dev → GitHub Pages | SQL-first, code-in-repo, deployable to a free public URL. |
| Lint | `ruff`, `sqlfluff` | Python and SQL style consistency. |
| CI / scheduling | GitHub Actions | Free for public repositories; runs both the CI pipeline and the monthly cron. |

## 6. Architecture

```
                    ┌──────────────────────────┐
                    │   Prefect flow (cron)     │
                    │  monthly · 1st 06:00 UTC  │
                    └────────────┬──────────────┘
                                 │ triggers
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
       ┌────────┐           ┌────────┐           ┌────────┐
       │ ingest │           │ ingest │           │ ingest │
       │  PPD   │           │  NSPL  │           │  HPI   │
       │ (incr) │           │ (full) │           │ (full) │
       └────┬───┘           └────┬───┘           └────┬───┘
            │ parquet           │ parquet            │ parquet
            └────────┬──────────┴──────────┬─────────┘
                     ▼                     ▼
              ┌───────────────────────────────────┐
              │   landing zone (parquet)          │
              └──────────────┬────────────────────┘
                             │ load
              ┌──────────────▼────────────────────┐
              │  warehouse — DuckDB  |  BigQuery  │
              │     raw schema (1:1 with source)  │
              └──────────────┬────────────────────┘
                             │ dbt run
              ┌──────────────▼────────────────────┐
              │  staging (typed, renamed, 1:1)    │
              │     stg_ppd, stg_nspl, stg_hpi    │
              └──────────────┬────────────────────┘
                             ▼
              ┌───────────────────────────────────┐
              │  intermediate                     │
              │   int_nspl_current                │
              │   int_ppd_enriched (geo + HPI)    │
              └──────────────┬────────────────────┘
                             ▼
              ┌───────────────────────────────────┐
              │  marts/core                       │
              │   fct_transactions (incremental)  │
              │   dim_postcode  dim_la  dim_date  │
              │  marts/analytics                  │
              │   mart_la_monthly_summary         │
              │   mart_premium_to_benchmark       │
              └──────────────┬────────────────────┘
                             │ dbt tests + GE checkpoints
                             ▼
              ┌───────────────────────────────────┐
              │  Evidence.dev build               │
              │     → GitHub Pages publish        │
              └───────────────────────────────────┘
```

Notes on the architecture:

- The fact table `fct_transactions` is materialised as a dbt `incremental` model with `unique_key='transaction_unique_id'`, so monthly increments do not reprocess the full 27 M-row history.
- The landing zone holds Parquet, not the raw CSVs, so column types and partition layout are identical regardless of which warehouse subsequently consumes them.
- NSPL is loaded with effective dates so a PPD row from 2010 resolves against the postcode geography that was in force at the time of the transaction, not today's geography.

## 7. Repository structure

```
uk-housing-mds/
├── README.md                          ← 9-section per CONTRIBUTING.md
├── requirements.txt
├── pyproject.toml                     ← ruff + black config
├── .gitignore
├── .env.example
│
├── flows/                             ← Prefect flows and tasks
│   ├── monthly_refresh.py
│   ├── tasks/
│   │   ├── ingest_ppd.py
│   │   ├── ingest_nspl.py
│   │   ├── ingest_hpi.py
│   │   ├── load_warehouse.py
│   │   ├── run_dbt.py
│   │   ├── run_data_quality.py
│   │   └── build_evidence.py
│   └── schedules.py
│
├── dbt_project/
│   ├── dbt_project.yml
│   ├── profiles.yml.example           ← duckdb + bigquery profiles
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_ppd.sql
│   │   │   ├── stg_nspl.sql
│   │   │   ├── stg_hpi.sql
│   │   │   └── _staging__sources.yml
│   │   ├── intermediate/
│   │   │   ├── int_nspl_current.sql
│   │   │   └── int_ppd_enriched.sql
│   │   └── marts/
│   │       ├── core/
│   │       │   ├── fct_transactions.sql
│   │       │   ├── dim_postcode.sql
│   │       │   ├── dim_local_authority.sql
│   │       │   └── dim_date.sql
│   │       └── analytics/
│   │           ├── mart_la_monthly_summary.sql
│   │           └── mart_premium_to_benchmark.sql
│   ├── tests/                         ← singular tests
│   ├── macros/
│   └── seeds/                         ← e.g. property-type code map
│
├── great_expectations/
│   ├── expectations/
│   └── checkpoints/
│
├── evidence/
│   ├── pages/
│   │   ├── index.md
│   │   ├── regional-trends.md
│   │   ├── premium-vs-benchmark.md
│   │   └── data-quality.md
│   ├── sources/
│   └── package.json
│
├── notebooks/                         ← Exploration only, not in the pipeline
│   ├── 01_ppd_eda.ipynb
│   └── 02_join_validation.ipynb
│
├── src/
│   └── housing_mds/
│       ├── __init__.py
│       ├── download.py
│       └── parquet_io.py
│
├── data/                              ← contents gitignored
│   ├── landing/                       ← parquet, partitioned by month
│   ├── duckdb/                        ← dev warehouse file
│   └── raw_archive/                   ← downloaded CSVs
│
├── tests/
│   ├── conftest.py
│   ├── fixtures/                      ← small synthetic PPD/NSPL/HPI
│   ├── test_download.py
│   ├── test_parquet_io.py
│   └── test_flows_smoke.py
│
├── .github/workflows/
│   ├── ci.yml                         ← lint + pytest + dbt build (DuckDB) + GE
│   └── publish-evidence.yml           ← build and deploy Evidence
│
└── docs/
    └── 2026-05-23-uk-housing-mds-design.md
```

## 8. Evaluation methodology

The pipeline produces no model, so there is no accuracy or recall metric. The closest analogue to model evaluation is **pipeline health**, surfaced both internally (CI logs, Prefect run history) and externally (Evidence `data-quality.md` page). The health metrics tracked across runs are:

| Metric | Definition | Surfaced where |
|---|---|---|
| Ingest row count per source | Rows landed per source per run | Prefect task logs, Evidence DQ page |
| Run-to-run row delta | Difference vs trailing-3-month median | dbt singular test + DQ page |
| Test pass rate | (passing dbt tests + GE expectations) / total | DQ page |
| Freshness lag | Days between latest `transaction_date` in `fct_transactions` and current date | DQ page |
| Build success rate | Successful Prefect runs / scheduled Prefect runs over trailing 12 months | DQ page |

These are computed against the production warehouse and rendered on the Evidence DQ page, so the data-quality story is part of the public deliverable rather than buried in CI logs.

## 9. Error handling

| Failure | Where caught | Response |
|---|---|---|
| Source URL 404 / download fails | Prefect ingest task | Retry three times with exponential backoff. On final failure the flow exits non-zero, no partial state in the warehouse. |
| Source returns a schema-drifted CSV | Great Expectations landing suite | Flow halts. Prefect logs the full GE diff. Nothing is loaded. |
| Warehouse load fails mid-stream | `load_warehouse` task | Transactional load (BigQuery load job; DuckDB `INSERT INTO ... SELECT` inside a transaction). Rollback on failure; no partial increment visible to dbt. |
| dbt incremental sees duplicate `transaction_unique_id` | dbt `unique` test on `fct_transactions` | Flow halts. The dbt test failure is logged; the most likely cause is replay of a PPD change file. |
| Evidence build fails | `build_evidence` task | The previously published GitHub Pages site remains live. The flow run is logged failed. |
| BigQuery free-tier quota hit | `run_dbt` (BigQuery profile) | Flow falls back to the DuckDB profile for that run and tags the run with the target that actually executed. Next scheduled run retries BigQuery. |

Observability surfaces are the Prefect flow run history (task logs, retries, durations) and the Evidence DQ page (the cross-run trend view). Each Prefect run is tagged with the run target (`duckdb` / `bigquery`) and the per-source row counts so trends are inspectable.

## 10. Testing strategy

| Layer | Tool | Coverage | Where it runs |
|---|---|---|---|
| Python unit | `pytest` | `src/` helpers (download URL builder, parquet conversion, postcode normaliser) and `flows/tasks/*` pure logic, mocked at I/O boundaries | Locally and in CI on every push |
| Flow smoke | `pytest` + Prefect test mode | The full `monthly_refresh` flow against a 1 k-row PPD fixture, NSPL subset, and HPI subset, executing against an ephemeral DuckDB | CI on every push |
| dbt | `dbt build` (run + test) | Every model compiles; every generic and singular test passes | CI on every push, against ephemeral DuckDB |
| Data quality | Great Expectations checkpoints | Landing-zone suites for all three sources | CI on the smoke fixture; production on every Prefect run |
| Lint / format | `ruff`, `sqlfluff` | Python and SQL style | CI on every push |

A canonical small fixture (~1 k PPD rows, ~500 NSPL rows, ~50 HPI rows) lives under `tests/fixtures/` in git. It is synthetic, deterministic, and used by both the smoke flow and the dbt-on-DuckDB CI build. Real bulk data is never downloaded inside CI.

### Data-quality suites (Great Expectations)

| Suite | Key expectations |
|---|---|
| `ppd_landing` | Column set + types match expectation; `price_paid > 0`; `date_of_transfer` within the last 31 years; `postcode` matches a UK postcode regex; row count within ±25 % of the trailing-3-month median (early warning on a broken or partial download). |
| `nspl_landing` | `postcode` unique; `lsoa11` non-null on >99 % of rows; `region_name` ∈ a known closed list of UK regions. |
| `hpi_landing` | `region_name` ∈ a known closed list; `average_price > 0`. |

### dbt tests

Generic tests on every staging and mart model: `unique` and `not_null` on every key; `accepted_values` on `property_type`, `tenure`, `new_build_flag`; `relationships` from `fct_transactions.postcode` to `dim_postcode.postcode`. Singular tests for cross-table invariants:

- `assert_fct_transactions_no_future_dates.sql`
- `assert_premium_to_benchmark_within_bounds.sql` — median LA premium-to-HPI between −80 % and +500 %; outside that band flags a join blow-up.
- `assert_monthly_row_count_within_bounds.sql` — current vs trailing-3-month median; fails if outside three standard deviations.

## 11. Deployment

| Component | Where it runs | How it is deployed |
|---|---|---|
| Prefect flow (dev) | Developer's laptop, on demand | `python -m flows.monthly_refresh --target duckdb` |
| Prefect flow (production) | GitHub Actions cron, monthly 1st 06:00 UTC | Workflow checks out the repository, restores cached pip and dbt dependencies, runs the flow with `--target bigquery`, attaches the refreshed `housing.duckdb` to a GitHub Release |
| dbt warehouse (dev) | Local DuckDB file `data/duckdb/housing.duckdb` | Created by the flow's `load_warehouse` task |
| dbt warehouse (production) | BigQuery datasets `housing_mds_*` (free tier) | Service-account credentials stored in GitHub Actions Secrets as `GCP_SA_KEY`; dbt-bigquery resolves them from `profiles.yml` environment variables |
| Great Expectations | Same Python process as the flow | No separate service |
| Evidence.dev site | GitHub Pages | Built by `.github/workflows/publish-evidence.yml` after every successful monthly refresh; static HTML pushed to the `gh-pages` branch |

Secrets handled by the pipeline: `GCP_SA_KEY` only. All three data sources are unauthenticated public bulk files.

GitHub Actions cron is used in place of a hosted Prefect server because the project's monthly schedule and modest task graph do not benefit from a long-running scheduler, and a hosted Prefect tier would break the project's free-stack constraint. The orchestration value of Prefect — task retries, structured logging, parametrised flows, the named DAG — is preserved by running the flow in local-execution mode inside the Action.

## 12. Scaling path

| Scale axis | Trigger | Next move |
|---|---|---|
| Row volume | ~500 M rows (e.g. joining HMRC stamp duty or Companies House owner data) | Drop DuckDB local; move dev to a per-branch BigQuery `dev_<branch>` dataset; partition `fct_transactions` by `transaction_year` |
| Refresh frequency | Daily updates | Replace GitHub Actions cron with self-hosted Prefect on Fly.io free tier; add a sources-change-detection task to skip no-op runs |
| Source count | >10 sources | Move from hand-written ingest tasks to `dlt` or Singer taps to standardise the ingest layer |
| BI users | Concurrent / interactive | Replace Evidence's static-generated site with Metabase in Docker on Fly.io, or expose marts via a read-only FastAPI |
| Compute | dbt run exceeds 30 minutes | BigQuery partition pruning; cap the `fct_transactions` incremental window to the last 24 months |

None of these are in scope; they are documented here so the scaling story is explicit.

## 13. Definition of Done

**Pipeline correctness**
- One full PPD backfill has run end-to-end against BigQuery without manual intervention.
- At least two monthly increment runs have completed on the GitHub Actions schedule.
- All dbt tests pass on the production warehouse.
- All Great Expectations suites pass on the production landing zone.

**Public deliverable**
- The Evidence site is live on GitHub Pages with at least three populated pages: `regional-trends`, `premium-vs-benchmark`, `data-quality`.
- Each Evidence page renders without errors; the SQL behind each chart is visible per Evidence convention.
- README follows the 9-section template from `CONTRIBUTING.md`, with the hero-results table populated from the production warehouse (corpus size, run frequency, test pass rate, freshness lag).

**Engineering hygiene**
- GitHub Actions CI is green on `main` (lint + pytest + flow smoke + `dbt build` on DuckDB).
- `.env.example` and `profiles.yml.example` are present; no secrets are committed.
- The repository layout matches `CONTRIBUTING.md`.

**Distributable**
- The latest `housing.duckdb` is published as a GitHub Release asset so any reader can run `duckdb housing.duckdb` and query the marts directly.

**Out of scope (stated explicitly in the README's "Limitations & next steps")**
- Streaming or change-data-capture ingestion.
- Reverse-ETL (Sheet or Slack alerts).
- Predictive modelling on top of the marts.
- Authenticated or per-user views on the Evidence site.
- Sub-monthly freshness.
