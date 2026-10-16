@echo off
echo Starting InvisibleInk...

REM Determine python command
set PYTHON_CMD=python
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.12
)

REM Check if virtual environment exists
if not exist "venv" (
    if not exist "venv_app" (
        echo Creating virtual environment...
        %PYTHON_CMD% -m venv venv_app
    )
)

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist "venv_app\Scripts\activate.bat" (
    call venv_app\Scripts\activate.bat
)

echo Verifying dependencies...
pip install -r requirements.txt

echo Starting Flask Server...
python app.py
pause
