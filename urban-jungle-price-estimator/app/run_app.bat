@echo off
REM ────────────────────────────────────────────────────────────────────────
REM Urban Jungle — Quote Estimator Launcher (Windows)
REM ────────────────────────────────────────────────────────────────────────
REM Installs dependencies (first run only) and launches the Streamlit app.
REM Resolves project root from this script's location, so it can be run
REM from anywhere.

echo.
echo === Urban Jungle Quote Estimator ===
echo.

REM Resolve project root (parent of this script's directory)
pushd "%~dp0\.."

REM Check for Python
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.10+ and retry.
    popd
    pause
    exit /b 1
)

REM Install/upgrade dependencies (idempotent)
echo Installing dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    popd
    pause
    exit /b 1
)

echo.
echo Launching Streamlit app — opening in your browser...
echo Press Ctrl+C to stop.
echo.
python -m streamlit run app/app.py
popd
