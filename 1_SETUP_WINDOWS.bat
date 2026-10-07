@echo off
title Nasdaq Bot - Setup
cd /d "%~dp0"
echo.
echo ==============================================
echo   Nasdaq Bot - one-time setup (Windows)
echo ==============================================
echo.
where py >nul 2>nul
if %errorlevel%==0 (
    set PY=py -3
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python is not installed.
        echo.
        echo 1. Go to https://www.python.org/downloads/  and download Python 3.12
        echo 2. Run the installer and TICK THE BOX "Add python.exe to PATH"
        echo 3. Then double-click this file again.
        echo.
        pause
        exit /b 1
    )
    set PY=python
)
echo Creating a private Python environment in the "venv" folder...
%PY% -m venv venv
if errorlevel 1 (
    echo Could not create the environment. Is Python 3.10 - 3.13 installed?
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo Installing required packages (this takes a minute or two)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Package install failed. Check your internet connection and try again.
    pause
    exit /b 1
)
echo.
echo Installing the speed-up package (optional, ok if this fails)...
pip install numba
echo.
echo ==============================================
echo   Setup finished!  Now double-click  2_START_WINDOWS.bat
echo ==============================================
pause
