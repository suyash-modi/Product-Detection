@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual env not found. Run: py -3.12 -m venv .venv
    echo Then: .venv\Scripts\pip install -r requirements.txt inference
    pause
    exit /b 1
)
python src\apps\simple_ui.py
pause
