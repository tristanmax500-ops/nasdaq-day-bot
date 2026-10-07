@echo off
title Nasdaq Bot - search then trade
cd /d "%~dp0"
set VENV=venv
if exist use_arm64.txt if exist venv_arm64\Scripts\activate.bat set VENV=venv_arm64
if "%VENV%"=="venv_arm64" set NUMBA_CPU_NAME=generic
if not exist %VENV%\Scripts\activate.bat (
    echo Please run 1_SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)
call %VENV%\Scripts\activate.bat
echo.
echo Step 1 of 2: searching for strategies (up to 2 hours - stops early when it stops improving) ...
echo.
python run.py discover --minutes 120
echo.
echo Step 2 of 2: starting the bot. Leave this window open.
echo.
python run.py start
pause
