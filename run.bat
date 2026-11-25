@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found. Install Python 3.11 or newer from python.org, tick "Add Python to PATH", then run this again.
  pause
  exit /b 1
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if errorlevel 1 (
  echo Your Python is too old. Please install Python 3.11 or newer from python.org.
  pause
  exit /b 1
)
if not exist .venv (
  echo Creating a private environment inside this folder...
  python -m venv .venv
)
call .venv\Scripts\activate
if not exist .venv\installed.flag (
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  if errorlevel 1 (
    echo Installation failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
  echo ok> .venv\installed.flag
)
set HF_HOME=%~dp0models
streamlit run app.py --browser.gatherUsageStats false
pause