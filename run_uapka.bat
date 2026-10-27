@echo off
REM UAP Kinematics Analyzer - Windows launcher.
REM   run_uapka.bat                      start the GUI
REM   run_uapka.bat analyze clip.mp4 ... any command-line use (info, synth, analyze)
REM   run_uapka.bat --install-only       create/refresh the virtual environment and exit
REM The first run creates .venv next to this file and installs the dependencies (needs internet once).
setlocal
set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "PY=%VENV%\Scripts\python.exe"
REM UTF-8 for console output and tracebacks regardless of the console code page.
set "PYTHONUTF8=1"
cd /d "%HERE%" || (echo Cannot change to "%HERE%". & goto :pause_fail)

if not exist "%VENV%\installed.ok" call :install || goto :pause_fail
if /i "%~1"=="--install-only" exit /b 0

if "%~1"=="" (
    "%PY%" -m uapka gui
) else (
    "%PY%" -m uapka %*
)
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo uapka exited with code %RC%. For diagnostics run:  "%PY%" tools\check_env.py
    pause
)
exit /b %RC%

:install
echo Creating the virtual environment in "%VENV%" ...
set "BOOT="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "BOOT=py -3"
if not defined BOOT (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "BOOT=python"
)
if not defined BOOT (
    echo Python 3.10 or newer was not found ^(neither the "py" launcher nor "python" on PATH^).
    echo Install it from https://www.python.org/downloads/windows/ and tick "Add python.exe to PATH".
    exit /b 1
)
if exist "%VENV%" rmdir /s /q "%VENV%"
%BOOT% -m venv "%VENV%" || (echo Could not create the virtual environment. & exit /b 1)
"%PY%" -m pip install --upgrade pip >nul 2>&1 || echo (pip could not be upgraded; continuing with the bundled version)
"%PY%" -m pip install -r "%HERE%requirements.txt" || (
    echo Dependency installation failed. Is the internet reachable? Any proxy configured for pip?
    exit /b 1
)
echo ok> "%VENV%\installed.ok"
echo Installation complete.
exit /b 0

:pause_fail
pause
exit /b 1
