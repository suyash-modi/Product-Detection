@echo off
cd /d "%~dp0"
echo Creating venv with Python 3.12 (for logo detection model)...
py -3.12 -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt
pip install inference
echo.
echo Done. Run: run.bat  or  .venv\Scripts\python src\apps\simple_ui.py
pause
