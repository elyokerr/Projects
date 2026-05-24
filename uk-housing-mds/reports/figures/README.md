# Screenshots

Screenshots of the Evidence.dev BI site and CLI proofs (dbt build, pytest, flow run).
Embedded in the project [README](../../README.md) so the repo is browsable without running anything.

| File | What it shows |
|---|---|
| `01_home_hero.png` | Evidence homepage — hero, 4 BigValue stats, fixture-data callout |
| `02_home_charts.png` | Homepage area chart + top-LAs + region price table |
| `03_home_nav.png` | Three navigation tiles + "How this is built" footer |
| `04_regional_trends.png` | Regional Trends page — region dropdown + line/bar charts |
| `05_premium_vs_benchmark.png` | Premium vs HPI Benchmark page — LA dropdown + line + table |
| `06_data_quality.png` | Data Quality page — BigValue cards for pipeline health |
| `07_dbt_build.png` | Terminal output of `dbt build` — PASS=38 WARN=0 ERROR=0 |
| `08_pytest.png` | Terminal output of `pytest tests -v` — 32 passed, 1 skipped |
| `09_smoke_flow.png` | Terminal output of `RUN_SLOW=1 pytest tests/test_flows_smoke.py` — Prefect flow run + dbt build + Evidence build, all green |
| `10_github_actions.png` *(optional)* | Green CI run on the GitHub Actions tab |

All images are PNG, captured at ≥1440 px width. Light-mode (Evidence's appearance toggle is in the top-right; switch to light before capturing).
