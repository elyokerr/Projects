#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────
# Urban Jungle — Quote Estimator Launcher (macOS / Linux)
# ────────────────────────────────────────────────────────────────────────
# Installs dependencies (first run only) and launches the Streamlit app.
# Resolves project root from this script's location, so it can be run
# from anywhere.

set -euo pipefail

echo
echo "=== Urban Jungle Quote Estimator ==="
echo

# Resolve project root (parent of this script's directory)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"
cd "$PROJECT_ROOT"

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
python3 -m streamlit run app/app.py
