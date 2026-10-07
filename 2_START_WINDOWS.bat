@echo off
title Nasdaq Bot
cd /d "%~dp0"
if not exist venv\Scripts\activate.bat (
    echo Please run 1_SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python run.py %*
pause
