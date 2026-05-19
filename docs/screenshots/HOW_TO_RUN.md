# HOW TO RUN - Urban Jungle Price Estimator

A fully self-contained, step-by-step guide. Anyone with a ZIP of this project
can be up and running in under 5 minutes — no Google Drive, no manual path
edits, no extra configuration.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Versions of every technology used](#2-versions-of-every-technology-used)
3. [Step-by-step: local setup (recommended)](#3-step-by-step-local-setup-recommended)
4. [Step-by-step: Google Colab](#4-step-by-step-google-colab)
5. [Running the interactive dashboard](#5-running-the-interactive-dashboard)
6. [Running the notebook](#6-running-the-notebook)
7. [Using the trained model in your own code](#7-using-the-trained-model-in-your-own-code)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What you need

- A computer running **Windows, macOS, or Linux**.
- **Python 3.10, 3.11, 3.12 or 3.13** installed (https://www.python.org/downloads/).
- An internet connection (only for the one-time dependency install).
- About **500 MB of free disk space** (the raw CSV is ~50 MB; trained model artefacts ~20 MB; libraries ~400 MB).

That's it. No Google account, no cloud setup, no Docker.

---

## 2. Versions of every technology used

All version ranges are pinned in `requirements.txt`. The notebook also prints
the runtime versions in its first cell.

| Layer | Technology | Version range | Purpose |
|---|---|---|---|
| Language | **Python** | 3.10 – 3.13 | Runtime |
| Numerics | **NumPy** | `>= 2.0, < 2.3` | Arrays, linear algebra |
| Dataframes | **pandas** | `>= 2.2, < 2.4` | Tabular data wrangling |
| Scientific | **SciPy** | `>= 1.13, < 1.16` | Statistical functions (skew, kurtosis, Q-Q) |
| Machine learning | **scikit-learn** | `>= 1.6, < 1.8` | All models, pipelines, validation, search |
| Explainability | **SHAP** | `>= 0.46, < 0.50` | Per-prediction explanations |
| Model serialisation | **joblib** | `>= 1.4, < 1.6` | Save / load trained models |
| Static plots | **matplotlib** | `>= 3.10, < 3.12` | Notebook figures |
| Statistical plots | **seaborn** | `>= 0.13, < 0.15` | Boxplots, heatmaps |
| Notebooks | **Jupyter** | `>= 1.1` | Notebook UI |
| Notebook kernel | **ipykernel** | `>= 6.29` | Jupyter Python kernel |
| Web framework | **Streamlit** | `>= 1.36, < 2.0` | Interactive dashboard `app.py` |
| Interactive charts | **Plotly** | `>= 5.22, < 6.0` | Gauge, band, comparison charts in app |

Last verified to run cleanly on Python 3.12 with the latest minor versions in
each range.

---

## 3. Step-by-step: local setup (recommended)

### 3.1 Extract the ZIP

Unzip the project to any folder — for example:

- **Windows:** `C:\Users\<you>\Documents\Urban Jungle Task\`
- **macOS / Linux:** `~/Documents/Urban Jungle Task/`

Confirm the folder contains at least these files at its top level:

```
UJ_price_estimator.ipynb
UJ_datatask_prices.csv
app.py
requirements.txt
README.md
HOW_TO_RUN.md
```

### 3.2 Open a terminal in that folder

- **Windows:** Open **PowerShell**, then `cd "C:\Users\<you>\Documents\Urban Jungle Task"`
- **macOS:** Open **Terminal**, then `cd "~/Documents/Urban Jungle Task"`
- **Linux:** Same as macOS.

### 3.3 (Optional but recommended) Create a virtual environment

This keeps the project's dependencies isolated from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3.4 Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything listed in [Section 2](#2-versions-of-every-technology-used). Takes ~2 minutes the first time.

### 3.5 You're ready

Go to either [Section 5 (dashboard)](#5-running-the-interactive-dashboard) or
[Section 6 (notebook)](#6-running-the-notebook).

---

## 4. Step-by-step: Google Colab

You only need this if you can't or don't want to install Python locally.

1. Open https://colab.research.google.com/
2. Click **File → Upload notebook** and select `UJ_price_estimator.ipynb` from the project ZIP.
3. In the Colab left sidebar, click the **folder icon** (Files panel).
4. Click the **upload icon** (page with up-arrow) and upload `UJ_datatask_prices.csv`.
5. Click **Runtime → Run all**.

The notebook's first cell auto-detects Colab, finds the uploaded CSV in `/content/`, and runs end-to-end. **No Google Drive mount is required.** Total runtime ~5 minutes.

> **If the upload disappears between sessions:** Colab clears `/content/` when
> the runtime times out. Just re-upload the CSV before re-running.

---

## 5. Running the interactive dashboard

The dashboard simulates how a customer-facing or internal pricing tool would
use the model.

### One-click launchers

**Windows:** Double-click `run_app.bat` in File Explorer. It installs
dependencies (if needed) and opens the app in your browser.

**macOS / Linux:**
```bash
chmod +x run_app.sh   # one-time
./run_app.sh
```

### Manual launch (any platform)

```bash
streamlit run app.py
```

Your default browser opens automatically at `http://localhost:8501`.

**First run:** The app trains the model from the CSV (~30 seconds, shown as a
progress bar) and caches it to `artifacts/uj_price_estimator_bundle.joblib`.

**Subsequent runs:** The cached model loads instantly.

### What's in the dashboard

- **Get a quote** tab: enter a customer profile → see the predicted lowest
  panel quote, 80% confidence band, gauge, and market-position insight.
- **Market insights** tab: interactive comparisons across bedrooms and
  postcodes for the current profile.
- **About this model** tab: model card, honest validation results, hyperparameters.
- **Sidebar:** model performance metrics and quick-preset customer profiles.

To stop the app, return to the terminal and press **Ctrl+C**.

---

## 6. Running the notebook

```bash
jupyter notebook UJ_price_estimator.ipynb
```

Your browser opens at `http://localhost:8888`. Click **Cell → Run All** (or
press Shift+Enter through each cell).

The first cell prints something like:

```
Environment    : local
Base dir       : C:\Users\<you>\Documents\Urban Jungle Task
Data file      : .../UJ_datatask_prices.csv  OK
─── versions ───────────────────────────────────────────
Platform   : Windows-10-...
Python     : 3.12.x
numpy      : 2.0.x
pandas     : 2.2.x
...
```

If you see `Data file: ... NOT FOUND`, see [Troubleshooting](#8-troubleshooting).

Total notebook runtime: ~5 minutes on a modern laptop. Plots are saved to
`./plots/` and the trained model to `./artifacts/`.

---

## 7. Using the trained model in your own code

After running the notebook (or the dashboard at least once), a model bundle
exists at `artifacts/uj_price_estimator_bundle.joblib`. Load and use it:

```python
import joblib
import pandas as pd
import re

bundle = joblib.load('artifacts/uj_price_estimator_bundle.joblib')

def predict_lowest_price(scenario, bundle=bundle):
    pc = scenario['INSUREDPOSTCODE']
    m = re.match(r'^([A-Z]{1,2}\d{1,2}[A-Z]?)', str(pc))
    outward = m.group(1) if m else pc
    area = re.match(r'^([A-Z]+)', str(pc)).group(1)

    row = pd.DataFrame([{**scenario,
                         'POSTCODE_OUTWARD': outward,
                         'POSTCODE_AREA': area}])[bundle['feature_cols']]
    for c in bundle['categorical_cols']:
        row[c] = pd.Categorical(row[c], categories=bundle['cat_levels'][c])

    return {
        'point': float(bundle['point_model'].predict(row)[0]),
        'q10':   float(bundle['quantile_models'][0.1].predict(row)[0]),
        'q50':   float(bundle['quantile_models'][0.5].predict(row)[0]),
        'q90':   float(bundle['quantile_models'][0.9].predict(row)[0]),
    }

result = predict_lowest_price({
    'AGE': 35, 'BEDROOMS_N': 3,
    'ACCID_CONTENTS': 1, 'ALARM_BIN': 1,
    'OCCUPATION': 'D78', 'INSUREDPOSTCODE': 'N65TX',
})
print(result)
# {'point': 130.21, 'q10': 127.83, 'q50': 129.34, 'q90': 134.55}
```

---

## 8. Troubleshooting

### "Data file: ... NOT FOUND" in the notebook

Make sure `UJ_datatask_prices.csv` is in the **same folder as the notebook**.
The notebook only looks at the current working directory by default. If your
Jupyter server was launched from a different folder, either restart it from
the correct folder or move/copy the CSV there.

### Streamlit says `streamlit: command not found`

Use the module form instead:
```bash
python -m streamlit run app.py
```

### `pip install -r requirements.txt` fails

- Confirm `pip --version` reports Python 3.10–3.13.
- Upgrade pip itself: `python -m pip install --upgrade pip`
- On Windows, if SSL errors appear, try: `python -m pip install --upgrade pip --trusted-host pypi.org --trusted-host files.pythonhosted.org`

### `ModuleNotFoundError: No module named 'streamlit'`

Either your virtual environment isn't activated, or you installed against a
different Python interpreter. Re-activate the venv and re-run
`pip install -r requirements.txt`.

### Port 8501 already in use (Streamlit)

Another Streamlit app is running. Either close it or pick a different port:
```bash
streamlit run app.py --server.port 8502
```

### Notebook is in Colab but the CSV is on my Drive

Set this in a cell **before** running the first setup cell:
```python
import os
os.environ['UJ_USE_DRIVE'] = '1'
```
This opts you back into the Drive-mount fallback. The CSV must be in
`MyDrive/Urban Jungle Task/UJ_datatask_prices.csv`.

### First model training is slow

The notebook's `RandomizedSearchCV` cell and the app's first-run training
take ~30–60 seconds each on a 2020-era laptop. This is normal. Subsequent
runs hit the cached model bundle and start instantly.

---

If you hit something not covered here, the notebook's first cell prints
exactly where it looked for the data file and what failed — start there.
