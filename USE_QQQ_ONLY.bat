@echo off
title Nasdaq Bot - switch mode
cd /d "%~dp0"
echo.
echo Make sure the bot window is CLOSED before switching.
echo.
pause
set VENV=venv
if exist use_arm64.txt if exist venv_arm64\Scripts\python.exe set VENV=venv_arm64
%VENV%\Scripts\python.exe run.py mode qqq
echo.
pause
