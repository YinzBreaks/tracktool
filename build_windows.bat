@echo off
REM Builds a self-contained Windows folder with PyInstaller:
REM   dist\UAPKA\UAPKA.exe       GUI (windowed, no console)
REM   dist\UAPKA\uapka-cli.exe   command line (info / synth / analyze)
REM Zip dist\UAPKA\ to distribute it; no Python is needed on the target machine.
setlocal
set "HERE=%~dp0"
set "PY=%HERE%.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
cd /d "%HERE%" || (echo Cannot change to "%HERE%". & goto :fail)

call "%HERE%run_uapka.bat" --install-only || goto :fail
"%PY%" -m pip install --upgrade "pyinstaller>=6.0" || goto :fail
"%PY%" -m PyInstaller --noconfirm --clean "%HERE%uapka.spec" || goto :fail

echo.
echo Smoke test of the console build ...
"%HERE%dist\UAPKA\uapka-cli.exe" --help >nul || goto :fail
echo.
echo Built "%HERE%dist\UAPKA\" : UAPKA.exe ^(GUI^) and uapka-cli.exe ^(command line^).
exit /b 0

:fail
echo.
echo Build failed. Check the messages above; "%PY%" tools\check_env.py shows the environment.
pause
exit /b 1
