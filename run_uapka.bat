@echo off
REM UAP Kinematics Analyzer - Windows launcher. First run creates a local virtual environment.
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    py -3 -m venv .venv || python -m venv .venv || goto :nopython
    .venv\Scripts\python.exe -m pip install --upgrade pip
    .venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
)
.venv\Scripts\python.exe -m uapka gui %*
goto :eof
:nopython
echo Python 3.10+ is required: https://www.python.org/downloads/windows/
pause
goto :eof
:fail
echo Dependency installation failed.
pause
