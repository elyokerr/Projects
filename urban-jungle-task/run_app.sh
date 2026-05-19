#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────
# Urban Jungle — Quote Estimator Launcher (macOS / Linux)
# ────────────────────────────────────────────────────────────────────────
set -euo pipefail

echo
echo "=== Urban Jungle Quote Estimator ==="
echo

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 not found on PATH. Install Python 3.10+ and retry."
    exit 1
fi

echo "Installing dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r requirements.txt

echo
echo "Launching Streamlit app — opening in your browser..."
echo "Press Ctrl+C to stop."
echo
python3 -m streamlit run app.py
