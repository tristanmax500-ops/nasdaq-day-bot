@echo off
title Nasdaq Bot - search for strategies (trading runs on GitHub)
cd /d "%~dp0"
set VENV=venv
if exist use_arm64.txt if exist venv_arm64\Scripts\activate.bat set VENV=venv_arm64
if not exist %VENV%\Scripts\activate.bat (
    echo Please run 1_SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)
call %VENV%\Scripts\activate.bat
echo.
echo Searching for strategies (up to 2 hours - stops early when it stops improving).
echo Trading itself runs on GitHub, so you can close this window any time (Ctrl+C, then N).
echo When the search ends, the strategies it validated are sent to GitHub automatically.
echo.
python run.py discover --minutes 120
echo.
pause
