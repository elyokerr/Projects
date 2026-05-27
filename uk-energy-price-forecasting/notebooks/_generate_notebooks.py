"""Generate all 5 Phase-6 notebooks for UK Energy Price Forecasting.

Run from the project root:
    .venv/Scripts/python notebooks/_generate_notebooks.py

Requires nbformat (pip install nbformat).
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_NB_DIR = Path(__file__).parent
_REPORTS_FIGURES = Path("reports/figures")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_KERNELSPEC = {
    "display_name": "uk-energy-forecast (.venv)",
    "language": "python",
    "name": "uk-energy-forecast-venv",
}
_LANGUAGE_INFO = {"name": "python"}

# The verbatim project-root setup cell (CELL 1 of every notebook).
_ROOT_SETUP_CODE = """\
import os, sys
from pathlib import Path
_cwd = Path.cwd()
_root = next((p for p in [_cwd] + list(_cwd.parents)
              if (p / 'requirements.txt').exists() and (p / 'src').is_dir()), None)
assert _root, f'Could not find project root from {_cwd}'
os.chdir(_root)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
print(f'Project root: {_root}')"""


def _make_nb(*cells) -> nbformat.NotebookNode:
    """Create a notebook with kernelspec + language_info set."""
    nb = new_notebook(cells=list(cells))
    nb.metadata["kernelspec"] = _KERNELSPEC
    nb.metadata["language_info"] = _LANGUAGE_INFO
    return nb


def _save(nb: nbformat.NotebookNode, name: str) -> Path:
    path = _NB_DIR / name
    nbformat.write(nb, str(path))
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# 01_eda.ipynb
# ---------------------------------------------------------------------------

def _build_01() -> nbformat.NotebookNode:
    cells = [
        new_code_cell(_ROOT_SETUP_CODE),
        new_markdown_cell(
            "# 01 — Exploratory Data Analysis\n\n"
            "This notebook explores the **synthetic fixture panel** "
            "(30 days × 48 settlement periods).  "
            "Run against real data by replacing `load_fixture_panel()` with "
            "`load_fixture_panel()` pointed at `data/processed/panel.parquet` "
            "(see `data/README.md`).\n\n"
            "We visualise:\n"
            "1. Price over time\n"
            "2. Daily price profile (mean by SP-of-day)\n"
            "3. Price vs demand scatter\n"
            "4. Price distribution + spike note\n"
            "5. Generation mix stacked-area"
        ),
        new_code_cell(
            """\
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe for nbconvert
import matplotlib.pyplot as plt

from src.build.fixtures import load_fixture_panel

# Use live panel if it exists, otherwise fall back to fixture
_LIVE_PATH = Path('data/processed/panel.parquet')
if _LIVE_PATH.exists():
    print('Loading LIVE panel from', _LIVE_PATH)
    import pandas as pd
    bundle_raw = pd.read_parquet(_LIVE_PATH)
    print('Live panel shape:', bundle_raw.shape)
    USE_LIVE = True
else:
    print('Using FIXTURE panel — see data/README.md for live data')
    USE_LIVE = False

bundle = load_fixture_panel()
print('Fixture bundle loaded.')
print('  target         :', bundle.target.n_timesteps, 'steps')
print('  past_covariates:', bundle.past_covariates.components.tolist())
print('  future_covar.  :', bundle.future_covariates.components.tolist())"""
        ),
        new_code_cell(
            """\
# ── Extract raw DataFrames for plotting ─────────────────────────────────────
price_s = bundle.target.to_series()          # price
past_df  = bundle.past_covariates.to_dataframe()   # demand + gen_*

# Add sp_of_day from future covariates for daily profile
fut_df = bundle.future_covariates.to_dataframe()
price_df = price_s.rename('price').to_frame()
price_df['sp_of_day'] = fut_df['sp_of_day'].values"""
        ),
        new_markdown_cell("## 1. Price over time"),
        new_code_cell(
            """\
fig, ax = plt.subplots(figsize=(13, 3))
ax.plot(price_s.index, price_s.values, lw=0.8, color='royalblue')
ax.set_xlabel('Timestamp (UTC)')
ax.set_ylabel('Price (£/MWh)')
ax.set_title('GB Half-hourly Settlement Price — fixture panel')
ax.grid(True, alpha=0.3)
fig.tight_layout()
_out = Path('reports/figures/01_price_timeseries.png')
_out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(_out, dpi=120)
plt.show()
print('Saved:', _out)"""
        ),
        new_markdown_cell("## 2. Daily price profile (mean by SP-of-day)"),
        new_code_cell(
            """\
profile = price_df.groupby('sp_of_day')['price'].agg(['mean', 'std'])
profile.index = profile.index.astype(int)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(profile.index, profile['mean'], color='royalblue', lw=1.5, label='Mean price')
ax.fill_between(
    profile.index,
    profile['mean'] - profile['std'],
    profile['mean'] + profile['std'],
    alpha=0.25, color='royalblue', label='±1 SD'
)
ax.set_xlabel('Settlement period of day (1 = 00:00–00:30 UTC)')
ax.set_ylabel('Price (£/MWh)')
ax.set_title('Daily price profile — fixture panel')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('reports/figures/01_daily_profile.png', dpi=120)
plt.show()
print('Saved: reports/figures/01_daily_profile.png')"""
        ),
        new_markdown_cell("## 3. Price vs demand scatter"),
        new_code_cell(
            """\
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(past_df['demand_indo'], price_s.values, s=4, alpha=0.3, color='steelblue')
ax.set_xlabel('Demand — INDemand Outturn (MW)')
ax.set_ylabel('Price (£/MWh)')
ax.set_title('Price vs Demand')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()"""
        ),
        new_markdown_cell(
            "## 4. Price distribution + spike note\n\n"
            "> **Price spikes**: The UK electricity spot market can exhibit large "
            "positive spikes (scarcity pricing, system stress events).  "
            "The fixture panel includes 12 synthetic spikes +30–100 £/MWh.  "
            "In production data, spikes can exceed several thousand £/MWh."
        ),
        new_code_cell(
            """\
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

axes[0].hist(price_s.values, bins=50, color='royalblue', edgecolor='white', lw=0.3)
axes[0].set_xlabel('Price (£/MWh)')
axes[0].set_ylabel('Count')
axes[0].set_title('Price distribution (full range)')
axes[0].grid(True, alpha=0.3)

# Zoom in to 5–130 £/MWh to show the bulk + minor spikes
bulk = price_s[price_s < 130]
axes[1].hist(bulk.values, bins=50, color='steelblue', edgecolor='white', lw=0.3)
axes[1].set_xlabel('Price (£/MWh)')
axes[1].set_title('Price distribution (bulk, <130 £/MWh)')
axes[1].grid(True, alpha=0.3)

fig.suptitle('Settlement price distribution — fixture panel', y=1.02)
fig.tight_layout()
plt.show()
print(f'p99 price: {price_s.quantile(0.99):.1f}  max: {price_s.max():.1f}')"""
        ),
        new_markdown_cell("## 5. Generation mix stacked-area"),
        new_code_cell(
            """\
gen_cols = [c for c in past_df.columns if c.startswith('gen_')]
gen_df   = past_df[gen_cols].copy()
# Normalise column labels for the legend
gen_df.columns = [c.replace('gen_', '') for c in gen_cols]

fig, ax = plt.subplots(figsize=(13, 4))
ax.stackplot(
    gen_df.index,
    [gen_df[c] for c in gen_df.columns],
    labels=gen_df.columns,
    alpha=0.8,
)
ax.legend(loc='upper left', ncol=3, fontsize=8)
ax.set_ylabel('Generation (MW)')
ax.set_xlabel('Timestamp (UTC)')
ax.set_title('Generation mix — fixture panel')
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig('reports/figures/01_gen_mix.png', dpi=120)
plt.show()
print('Saved: reports/figures/01_gen_mix.png')"""
        ),
    ]
    return _make_nb(*cells)


# ---------------------------------------------------------------------------
# 02_covariates.ipynb
# ---------------------------------------------------------------------------

def _build_02() -> nbformat.NotebookNode:
    cells = [
        new_code_cell(_ROOT_SETUP_CODE),
        new_markdown_cell(
            "# 02 — Covariate Engineering\n\n"
            "This notebook demonstrates how the project constructs covariates "
            "and enforces the **leakage discipline** that underpins honest "
            "backtesting.\n\n"
            "Key topics:\n"
            "1. `build_calendar()` — Fourier + calendar features from a DatetimeIndex\n"
            "2. Fourier term visualisation (daily sinusoids)\n"
            "3. Past vs future covariate split — why this matters for backtesting\n"
            "4. Listing all components from the PanelBundle"
        ),
        new_code_cell(
            """\
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.build.fixtures import load_fixture_panel
from src.data.calendar import build_calendar

bundle = load_fixture_panel()
# Recover the raw DatetimeIndex from the target series
ts_index = bundle.target.time_index
print('Index start:', ts_index[0], '  end:', ts_index[-1])
print('Length:', len(ts_index), 'half-hourly steps')"""
        ),
        new_markdown_cell("## 1. Build calendar features"),
        new_code_cell(
            """\
cal = build_calendar(ts_index)
print('Calendar DataFrame shape:', cal.shape)
print('\\nFirst 5 rows:')
cal.head()"""
        ),
        new_code_cell(
            """\
print('Calendar columns:')
for col in cal.columns:
    print(f'  {col}')"""
        ),
        new_markdown_cell(
            "## 2. Fourier terms — daily sinusoid\n\n"
            "The calendar builder produces 2 harmonics for each of daily, "
            "weekly, and annual seasonality.  "
            "`sin_daily_1` completes one full cycle every 48 SPs (24 hours); "
            "`sin_daily_2` completes two cycles."
        ),
        new_code_cell(
            """\
# Plot the first 4 days of sin_daily_1 and cos_daily_1
n_plot = 48 * 4  # 4 days
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

for i, (ax, col, color) in enumerate(zip(
    axes,
    ['sin_daily_1', 'cos_daily_1'],
    ['royalblue', 'darkorange']
)):
    ax.plot(cal.index[:n_plot], cal[col].values[:n_plot],
            color=color, lw=1.5)
    ax.set_ylabel(col, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5)

axes[1].set_xlabel('Timestamp (UTC)')
fig.suptitle('Fourier terms — first 4 days', y=1.02)
fig.tight_layout()
plt.show()"""
        ),
        new_code_cell(
            """\
# Show sp_of_day cycling 1–48 over a 3-day window
fig, ax = plt.subplots(figsize=(12, 2.5))
n = 48 * 3
ax.plot(cal.index[:n], cal['sp_of_day'].values[:n],
        color='seagreen', lw=1.2, marker='.', markersize=2)
ax.set_ylabel('sp_of_day (1–48)')
ax.set_xlabel('Timestamp (UTC)')
ax.set_title('Settlement period of day — 3-day window')
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()"""
        ),
        new_markdown_cell(
            "## 3. Past vs future covariate split\n\n"
            "### The leakage discipline\n\n"
            "In a half-hourly electricity market, the model is asked at time "
            "**T** to forecast prices for periods **T+1 … T+H**.  "
            "Features can only be used at prediction time if they would actually "
            "be available at time **T**:\n\n"
            "| Covariate type | Available at T? | Used as |\n"
            "|---|---|---|\n"
            "| Calendar (dow, sp_of_day, Fourier) | Yes — deterministic | **future** |\n"
            "| Demand outturn (indo / itsdo) | No — only known after metering | **past** |\n"
            "| Generation outturn (gen_*) | No — settlement data, not real-time | **past** |\n\n"
            "Darts enforces this distinction at model level: past covariates are "
            "consumed only up to the origin, future covariates are passed for the "
            "entire forecast horizon."
        ),
        new_code_cell(
            """\
print('=== bundle.past_covariates ===')
print('Components (', len(bundle.past_covariates.components), '):')
for c in bundle.past_covariates.components:
    print('  ', c)

print()
print('=== bundle.future_covariates ===')
print('Components (', len(bundle.future_covariates.components), '):')
for c in bundle.future_covariates.components:
    print('  ', c)"""
        ),
        new_code_cell(
            """\
# Confirm no calendar columns appear in past_covariates
past_cols  = set(bundle.past_covariates.components.tolist())
fut_cols   = set(bundle.future_covariates.components.tolist())
cal_tokens = {'dow', 'sp_of_day', 'sin_', 'cos_', 'is_weekend', 'is_holiday'}

leakage_candidates = [
    c for c in past_cols
    if any(tok in c for tok in cal_tokens)
]
if leakage_candidates:
    print('WARNING: calendar-like columns in past_covariates:', leakage_candidates)
else:
    print('OK — no calendar columns leaked into past_covariates.')

demand_gen_in_future = [
    c for c in fut_cols
    if c.startswith(('demand_', 'gen_'))
]
if demand_gen_in_future:
    print('WARNING: outturn columns in future_covariates:', demand_gen_in_future)
else:
    print('OK — no outturn columns leaked into future_covariates.')"""
        ),
        new_markdown_cell(
            "## 4. Covariate summary statistics\n\n"
            "A quick sanity check on the magnitude and variation of each covariate."
        ),
        new_code_cell(
            """\
past_df  = bundle.past_covariates.to_dataframe()
fut_df   = bundle.future_covariates.to_dataframe()

print('Past covariates summary:')
print(past_df.describe().T[['mean', 'std', 'min', 'max']].round(1).to_string())
print()
print('Future covariates summary (first 8 cols):')
print(fut_df.iloc[:, :8].describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())"""
        ),
    ]
    return _make_nb(*cells)


# ---------------------------------------------------------------------------
# 03_colab_train_global.ipynb  (DO NOT execute locally)
# ---------------------------------------------------------------------------

def _build_03() -> nbformat.NotebookNode:
    cells = [
        new_code_cell(_ROOT_SETUP_CODE),
        new_markdown_cell(
            "# 03 — Global Model Training (Colab T4)\n\n"
            "> **Run this notebook on Google Colab with a T4 GPU.**\n"
            "> It trains `GlobalLGBM`, `GlobalTiDE`, and `GlobalTFT` at full scale "
            "and serialises the fitted models to `models/`.  "
            "It also runs a zero-shot Chronos baseline.\n\n"
            "## Colab setup steps\n"
            "1. `File > Upload notebook` (or open from Drive).\n"
            "2. `Runtime > Change runtime type > T4 GPU`.\n"
            "3. Mount Google Drive and copy / clone this project.\n"
            "4. Run the pip-install cell below, then run all."
        ),
        new_code_cell(
            """\
# ── Colab: Mount Drive and set project root ──────────────────────────────────
# Uncomment when running on Colab.
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# PROJECT_DIR = '/content/drive/MyDrive/uk-energy-price-forecasting'
# import os, sys
# os.chdir(PROJECT_DIR)
# if PROJECT_DIR not in sys.path:
#     sys.path.insert(0, PROJECT_DIR)
# print('Working directory:', os.getcwd())
print('(Colab mount cell — skipped in local/CI run)')"""
        ),
        new_markdown_cell(
            "## Install dependencies (Colab only)\n\n"
            "These packages are not in the local venv — they require a GPU "
            "runtime and are heavy (~2–3 GB download)."
        ),
        new_code_cell(
            """\
# !pip install -q darts lightgbm
# !pip install -q "chronos-forecasting @ git+https://github.com/amazon-science/chronos-forecasting.git"
print('(Pip-install cell — uncomment when running on Colab T4)')"""
        ),
        new_markdown_cell("## Load the panel"),
        new_code_cell(
            """\
from pathlib import Path
import gc

# ── Data source: prefer live panel, fall back to fixture ────────────────────
_LIVE_PATH = Path('data/processed/panel.parquet')
if _LIVE_PATH.exists():
    print('Using LIVE panel:', _LIVE_PATH)
    import pandas as pd
    from src.build.panel import build_panel
    from src.data.calendar import build_calendar
    # Re-build bundle from saved parquet (same logic as load_fixture_panel).
    df = pd.read_parquet(_LIVE_PATH)
    from src.build.fixtures import load_fixture_panel
    bundle = load_fixture_panel()   # replace with live build if available
    print('Panel loaded (live path found but using fixture API for demo).')
else:
    print('Using FIXTURE panel — for full training upload real data to Colab')
    from src.build.fixtures import load_fixture_panel
    bundle = load_fixture_panel()

print('target steps:', bundle.target.n_timesteps)"""
        ),
        new_markdown_cell(
            "## Train GlobalLGBM\n\n"
            "LightGBM is fast enough to train on CPU in < 2 minutes for the "
            "full dataset.  On Colab it will typically finish in < 30 seconds.\n\n"
            "Sparse lags (`[-1, -2, -3, -48, -96]`) keep the 144 sub-model "
            "grid (48 horizon steps × 3 quantiles) tractable."
        ),
        new_code_cell(
            """\
from src.models.global_ml import GlobalLGBM

lgbm = GlobalLGBM(
    quantiles=(0.1, 0.5, 0.9),
    lags=[-1, -2, -3, -48, -96],
    lags_past_covariates=[-1, -2, -3, -48],
    # lags_future_covariates=(48, 0),   # enable if you want calendar look-back
    n_estimators=200,
    num_leaves=31,
    verbose=-1,
)

print('Fitting GlobalLGBM ...')
lgbm.fit(bundle)
print('GlobalLGBM fit complete.')

# Serialise
from pathlib import Path
Path('models').mkdir(exist_ok=True)
import joblib
joblib.dump(lgbm, 'models/global_lgbm.pkl')
print('Saved: models/global_lgbm.pkl')

del lgbm
gc.collect()
print('LGBM model released from memory.')"""
        ),
        new_markdown_cell(
            "## Train GlobalTiDE\n\n"
            "> **Requires GPU (T4+).**  "
            "Switch `accelerator='gpu'` for Colab; keep `'cpu'` for local smoke tests "
            "(will be slow).  Use 50 epochs on full data."
        ),
        new_code_cell(
            """\
# ── NOTE: heavy — requires GPU on Colab ──────────────────────────────────────
# Set accelerator='gpu' when running on Colab T4.
try:
    import torch
    _accel = 'gpu' if torch.cuda.is_available() else 'cpu'
except ImportError:
    _accel = 'cpu'
print(f'Using accelerator: {_accel}')

from src.models.global_dl import GlobalTiDE

tide = GlobalTiDE(
    quantiles=[0.1, 0.5, 0.9],
    input_chunk_length=96,     # 2 days look-back
    output_chunk_length=48,    # 1 day forecast
    n_epochs=50,
    batch_size=64,
    random_state=42,
    pl_trainer_kwargs={'accelerator': _accel, 'enable_progress_bar': True},
)
print('Fitting GlobalTiDE ...')
tide.fit(bundle)
print('GlobalTiDE fit complete.')

# Darts .save() serialises the full model including PyTorch weights
tide.model.save('models/global_tide.pt')
print('Saved: models/global_tide.pt')

del tide
gc.collect()
try:
    import torch
    torch.cuda.empty_cache()
    print('CUDA cache cleared.')
except Exception:
    pass"""
        ),
        new_markdown_cell(
            "## Train GlobalTFT\n\n"
            "> TFT requires `future_covariates` at both fit and predict time.  "
            "The project's bundle already provides calendar features as "
            "`bundle.future_covariates`."
        ),
        new_code_cell(
            """\
# ── NOTE: heavy — requires GPU on Colab ──────────────────────────────────────
from src.models.global_dl import GlobalTFT

tft = GlobalTFT(
    quantiles=[0.1, 0.5, 0.9],
    input_chunk_length=96,
    output_chunk_length=48,
    n_epochs=50,
    batch_size=64,
    random_state=42,
    pl_trainer_kwargs={'accelerator': _accel, 'enable_progress_bar': True},
)
print('Fitting GlobalTFT ...')
tft.fit(bundle)
print('GlobalTFT fit complete.')

tft.model.save('models/global_tft.pt')
print('Saved: models/global_tft.pt')

del tft
gc.collect()
try:
    torch.cuda.empty_cache()
    print('CUDA cache cleared.')
except Exception:
    pass"""
        ),
        new_markdown_cell(
            "## Zero-shot Chronos forecast\n\n"
            "Chronos is a pre-trained foundation model; no fine-tuning required.  "
            "It takes a 1-D NumPy context array and returns quantile forecasts "
            "directly."
        ),
        new_code_cell(
            """\
# ── Chronos zero-shot baseline ────────────────────────────────────────────────
import numpy as np
from src.models.foundation import chronos_forecast

# Use the last 336 SPs (7 days) as context
context = bundle.target.values().flatten()[-336:]

print('Running Chronos zero-shot forecast (horizon=48)...')
try:
    chronos_preds = chronos_forecast(
        series=context,
        horizon=48,
        quantiles=(0.1, 0.5, 0.9),
        model_name='amazon/chronos-t5-small',
        device='cuda',
    )
    print('Chronos forecast complete.')
    import pandas as pd
    chronos_df = pd.DataFrame(
        {f'q{int(q*100):02d}': arr for q, arr in chronos_preds.items()}
    )
    print(chronos_df.head(6))
    chronos_df.to_csv('models/chronos_sample_forecast.csv', index=False)
    print('Saved: models/chronos_sample_forecast.csv')
except ImportError as e:
    print('Chronos not installed (expected outside Colab):', e)"""
        ),
        new_markdown_cell(
            "## Summary\n\n"
            "After this notebook completes you should have:\n\n"
            "| File | Model |\n"
            "|---|---|\n"
            "| `models/global_lgbm.pkl` | GlobalLGBM (LightGBM quantile) |\n"
            "| `models/global_tide.pt` | GlobalTiDE (DL encoder-decoder) |\n"
            "| `models/global_tft.pt` | GlobalTFT (Temporal Fusion Transformer) |\n"
            "| `models/chronos_sample_forecast.csv` | Chronos zero-shot sample |\n\n"
            "These artefacts are consumed in `04_backtest_ablation.ipynb`."
        ),
    ]
    return _make_nb(*cells)


# ---------------------------------------------------------------------------
# 04_backtest_ablation.ipynb
# ---------------------------------------------------------------------------

def _build_04() -> nbformat.NotebookNode:
    cells = [
        new_code_cell(_ROOT_SETUP_CODE),
        new_markdown_cell(
            "# 04 — Backtest & Ablation Table\n\n"
            "We run a **rolling-origin backtest** over the last few days of the "
            "fixture panel, comparing:\n\n"
            "- `SeasonalNaive` (the baseline, no fitting)\n"
            "- `GlobalLGBM` (fit once, ~45 s on CPU)\n\n"
            "> **Note:** Rows for `GlobalTiDE`, `GlobalTFT`, and `Chronos` are "
            "produced in notebook 03 on Colab and can be merged here by loading "
            "the saved `models/*.pkl` / `*.pt` artefacts and re-running "
            "`run_backtest` on the same origins.\n\n"
            "The ablation table reports MAE, RMSE, CRPS, pinball, "
            "80 % coverage, and skill score vs the seasonal-naive baseline."
        ),
        new_code_cell(
            """\
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.build.fixtures import load_fixture_panel
from src.models.baselines import SeasonalNaive
from src.models.global_ml import GlobalLGBM
from src.backtest.rolling_origin import generate_origins, run_backtest, build_ablation_table

bundle = load_fixture_panel()
target_idx = bundle.target.time_index
print('Panel loaded — target has', bundle.target.n_timesteps, 'steps')
print('Target range:', target_idx[0], '→', target_idx[-1])"""
        ),
        new_markdown_cell(
            "## Generate rolling origins\n\n"
            "We use the last 4 days of the 30-day fixture for backtesting "
            "(leaving at least 25 days for training)."
        ),
        new_code_cell(
            """\
# Origins: last 4 midnights that have ≥48 steps of actual data after them
# target has 30 days → last midnight we can safely use = day 28 (day 29 is the last
# day, and we need 48 steps of future actuals after the origin).
panel_end = target_idx[-1]
# Use day 22–25 as backtest window (enough history before, enough future after)
# Note: Darts strips timezone from TimeSeries.time_index, so origins must be
# tz-naive to match.  The data is in UTC; midnight boundaries are the same.
origin_start = pd.Timestamp('2024-01-22 00:00')
origin_end   = pd.Timestamp('2024-01-25 00:00')

origins = generate_origins(
    time_index=target_idx,
    start=origin_start,
    end=origin_end,
    step='1D',
)
print(f'Origins ({len(origins)}):')
for o in origins:
    print(' ', o)"""
        ),
        new_markdown_cell("## Run backtest — SeasonalNaive"),
        new_code_cell(
            """\
HORIZON    = 48   # 24-hour forecast
QUANTILES  = (0.1, 0.5, 0.9)

naive = SeasonalNaive(season=48)
result_naive = run_backtest(
    model=naive,
    bundle=bundle,
    origins=origins,
    horizon=HORIZON,
    quantiles=QUANTILES,
    refit=False,
    model_name='seasonal_naive',
)
print('SeasonalNaive backtest done. Valid origins:', len(result_naive.origins))"""
        ),
        new_markdown_cell(
            "## Run backtest — GlobalLGBM\n\n"
            "> Fitting LightGBM with sparse lags takes ~30–60 s on CPU (144 sub-models).  "
            "The cell below is not skipped — it is a feature of this project."
        ),
        new_code_cell(
            """\
import warnings
warnings.filterwarnings('ignore')

lgbm = GlobalLGBM(
    quantiles=(0.1, 0.5, 0.9),
    lags=[-1, -2, -3, -48, -96],
    lags_past_covariates=[-1, -2, -3, -48],
    n_estimators=50,
    num_leaves=15,
    verbose=-1,
)

result_lgbm = run_backtest(
    model=lgbm,
    bundle=bundle,
    origins=origins,
    horizon=HORIZON,
    quantiles=QUANTILES,
    refit=False,
    model_name='global_lgbm',
)
print('GlobalLGBM backtest done. Valid origins:', len(result_lgbm.origins))"""
        ),
        new_markdown_cell("## Ablation table"),
        new_code_cell(
            """\
results = {
    'seasonal_naive': result_naive,
    'global_lgbm':    result_lgbm,
}

# Optional: if Colab artefacts are present, load and add them here
# import joblib
# from src.models.global_dl import GlobalTiDE
# ...

ablation = build_ablation_table(results, baseline='seasonal_naive')
print('\\n=== Ablation Table ===')
print(ablation.round(3).to_string())

# Save
Path('reports').mkdir(exist_ok=True)
ablation.to_csv('reports/ablation_table.csv')
print('\\nSaved: reports/ablation_table.csv')
ablation.round(3)"""
        ),
        new_markdown_cell("## Per-horizon MAE plot"),
        new_code_cell(
            """\
def per_horizon_mae(result) -> np.ndarray:
    \"\"\"MAE at each forecast step (averaged over origins).\"\"\"
    q_median = min(result.quantiles, key=lambda q: abs(q - 0.5))
    fc = result.forecasts[q_median]          # (n_origins, horizon)
    act = result.actuals                     # (n_origins, horizon)
    return np.mean(np.abs(act - fc), axis=0) # (horizon,)

mae_naive = per_horizon_mae(result_naive)
mae_lgbm  = per_horizon_mae(result_lgbm)

fig, ax = plt.subplots(figsize=(11, 3.5))
sp = np.arange(1, HORIZON + 1)
ax.plot(sp, mae_naive, label='SeasonalNaive', lw=1.5, color='grey', ls='--')
ax.plot(sp, mae_lgbm,  label='GlobalLGBM',   lw=1.5, color='royalblue')
ax.set_xlabel('Forecast step (settlement period offset)')
ax.set_ylabel('MAE (£/MWh)')
ax.set_title('Per-horizon MAE — fixture backtest')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('reports/figures/04_per_horizon_mae.png', dpi=120)
plt.show()
print('Saved: reports/figures/04_per_horizon_mae.png')"""
        ),
    ]
    return _make_nb(*cells)


# ---------------------------------------------------------------------------
# 05_probabilistic_eval.ipynb
# ---------------------------------------------------------------------------

def _build_05() -> nbformat.NotebookNode:
    cells = [
        new_code_cell(_ROOT_SETUP_CODE),
        new_markdown_cell(
            "# 05 — Probabilistic Evaluation\n\n"
            "Point forecasts are cheap to evaluate (MAE, RMSE).  "
            "This notebook evaluates **probabilistic calibration**:\n\n"
            "- **Reliability diagram**: nominal vs empirical coverage at multiple "
            "  interval levels — a well-calibrated model's dots should fall on the "
            "  diagonal.\n"
            "- **Fan chart**: median + quantile bands for one origin's 48-step "
            "  forecast vs actual prices.\n\n"
            "### Calibration note\n"
            "> `SeasonalNaive` produces *zero-width* intervals: all quantiles equal "
            "the same value, so empirical coverage is ~0 for any nominal level.  "
            "This is itself a teaching point — a model can have excellent MAE yet "
            "completely fail calibration.  `GlobalLGBM` with independent quantile "
            "regressors should achieve reasonable (if not perfect) calibration on "
            "the fixture panel."
        ),
        new_code_cell(
            """\
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.build.fixtures import load_fixture_panel
from src.models.baselines import SeasonalNaive
from src.models.global_ml import GlobalLGBM
from src.backtest.rolling_origin import generate_origins, run_backtest
from src.metrics.coverage import interval_coverage

import warnings
warnings.filterwarnings('ignore')

bundle = load_fixture_panel()
target_idx = bundle.target.time_index
print('Fixture bundle loaded.')"""
        ),
        new_markdown_cell("## Run backtest (re-run if coming from fresh kernel)"),
        new_code_cell(
            """\
HORIZON   = 48
QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

origins = generate_origins(
    time_index=target_idx,
    # Darts strips tz from TimeSeries.time_index; pass tz-naive timestamps.
    start=pd.Timestamp('2024-01-20 00:00'),
    end=pd.Timestamp('2024-01-25 00:00'),
    step='1D',
)
print(f'{len(origins)} origins for probabilistic evaluation.')

result_naive = run_backtest(
    model=SeasonalNaive(season=48),
    bundle=bundle,
    origins=origins,
    horizon=HORIZON,
    quantiles=QUANTILES,
    refit=False,
    model_name='seasonal_naive',
)

lgbm = GlobalLGBM(
    quantiles=list(QUANTILES),
    lags=[-1, -2, -3, -48, -96],
    lags_past_covariates=[-1, -2, -3, -48],
    n_estimators=50,
    num_leaves=15,
    verbose=-1,
)
result_lgbm = run_backtest(
    model=lgbm,
    bundle=bundle,
    origins=origins,
    horizon=HORIZON,
    quantiles=QUANTILES,
    refit=False,
    model_name='global_lgbm',
)
print('Backtests complete.')"""
        ),
        new_markdown_cell(
            "## Reliability diagram\n\n"
            "For each symmetric nominal interval `[q_low, q_high]` (e.g. 80% = [0.1, 0.9]) "
            "we compute the empirical fraction of actuals falling inside the interval.  "
            "A perfectly calibrated model lies on the diagonal."
        ),
        new_code_cell(
            """\
def reliability_points(result, quantile_pairs):
    \"\"\"Return (nominal_coverage, empirical_coverage) for each quantile pair.\"\"\"
    actuals = result.actuals.flatten()
    fc      = result.forecasts
    nominals, empiricals = [], []
    for (q_lo, q_hi) in quantile_pairs:
        lower = fc[q_lo].flatten()
        upper = fc[q_hi].flatten()
        emp   = interval_coverage(actuals, lower, upper)
        nominals.append(q_hi - q_lo)
        empiricals.append(emp)
    return np.array(nominals), np.array(empiricals)

# Symmetric pairs centred on 0.5
pairs = [
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9),
]

nom_naive, emp_naive = reliability_points(result_naive, pairs)
nom_lgbm,  emp_lgbm  = reliability_points(result_lgbm,  pairs)

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
ax.plot(nom_naive, emp_naive, 'o-', color='grey',      lw=1.5,
        markersize=7, label='SeasonalNaive (zero-width intervals)')
ax.plot(nom_lgbm,  emp_lgbm,  's-', color='royalblue', lw=1.5,
        markersize=7, label='GlobalLGBM')

ax.set_xlabel('Nominal coverage')
ax.set_ylabel('Empirical coverage')
ax.set_title('Reliability diagram — probabilistic calibration')
ax.legend(fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.tight_layout()

Path('reports/figures').mkdir(parents=True, exist_ok=True)
fig.savefig('reports/figures/05_reliability_diagram.png', dpi=120)
plt.show()
print('Saved: reports/figures/05_reliability_diagram.png')

# Print a quick table
cov_df = pd.DataFrame({
    'nominal'   : nom_lgbm,
    'naive_emp' : emp_naive,
    'lgbm_emp'  : emp_lgbm,
})
print(cov_df.round(3).to_string(index=False))"""
        ),
        new_markdown_cell(
            "## Fan chart — one origin\n\n"
            "A fan chart shows the median forecast + shaded interval bands for a "
            "single forecast origin.  This lets us visually inspect how the "
            "uncertainty widens across the 48-step horizon."
        ),
        new_code_cell(
            """\
# Pick the first valid origin from the LGBM backtest
origin_idx = 0
origin = result_lgbm.origins[origin_idx]

# Actual prices for the horizon window
actuals_window = result_lgbm.actuals[origin_idx]       # shape (48,)
sp_axis = np.arange(1, HORIZON + 1)

# Quantile arrays
q10 = result_lgbm.forecasts[0.1][origin_idx]
q20 = result_lgbm.forecasts[0.2][origin_idx]
q30 = result_lgbm.forecasts[0.3][origin_idx]
q40 = result_lgbm.forecasts[0.4][origin_idx]
q50 = result_lgbm.forecasts[0.5][origin_idx]
q60 = result_lgbm.forecasts[0.6][origin_idx]
q70 = result_lgbm.forecasts[0.7][origin_idx]
q80 = result_lgbm.forecasts[0.8][origin_idx]
q90 = result_lgbm.forecasts[0.9][origin_idx]

fig, ax = plt.subplots(figsize=(12, 4))

# Shaded bands (darkest = tightest)
alpha_levels = [0.15, 0.20, 0.25, 0.30]
band_pairs   = [(q10, q90), (q20, q80), (q30, q70), (q40, q60)]
labels       = ['10–90%', '20–80%', '30–70%', '40–60%']
for (lo, hi), a, lbl in zip(band_pairs, alpha_levels, labels):
    ax.fill_between(sp_axis, lo, hi, alpha=a, color='royalblue', label=lbl)

ax.plot(sp_axis, q50, color='royalblue', lw=2, label='Median (q50)')
ax.plot(sp_axis, actuals_window, color='black', lw=1.5, ls='--', label='Actual')

ax.set_xlabel('Forecast step (SP offset from origin)')
ax.set_ylabel('Price (£/MWh)')
ax.set_title(f'Fan chart — GlobalLGBM, origin {origin}')
ax.legend(ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('reports/figures/05_fan_chart.png', dpi=120)
plt.show()
print('Saved: reports/figures/05_fan_chart.png')"""
        ),
        new_markdown_cell(
            "## SeasonalNaive calibration teaching point\n\n"
            "Let's confirm that seasonal-naive has zero-width intervals and "
            "therefore ~0 empirical coverage at all levels."
        ),
        new_code_cell(
            """\
# All quantiles for SeasonalNaive should be identical (zero interval width)
o = 0   # first origin
q10_n = result_naive.forecasts[0.1][o]
q90_n = result_naive.forecasts[0.9][o]
width = q90_n - q10_n
print(f'SeasonalNaive 80% interval width (first origin): '
      f'min={width.min():.4f}  max={width.max():.4f}')
print('Confirms: zero-width intervals => empirical coverage ≈ 0')

print(f'\\nGlobalLGBM 80% empirical coverage (all origins): '
      f'{emp_lgbm[-1]:.3f} (nominal 0.80)')"""
        ),
    ]
    return _make_nb(*cells)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all 5 notebooks."""
    print("Generating notebooks ...")

    _NB_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)

    _save(_build_01(), "01_eda.ipynb")
    _save(_build_02(), "02_covariates.ipynb")
    _save(_build_03(), "03_colab_train_global.ipynb")
    _save(_build_04(), "04_backtest_ablation.ipynb")
    _save(_build_05(), "05_probabilistic_eval.ipynb")

    print("\nDone.  Notebooks written to:", _NB_DIR.resolve())


if __name__ == "__main__":
    main()
