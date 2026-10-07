@echo off
title Nasdaq Bot - speed upgrade for ARM laptops
cd /d "%~dp0"
echo.
echo ==============================================================
echo   Speed upgrade: run Python natively on your ARM processor
echo   (your current Python is an Intel/AMD version that Windows
echo    has to translate while it runs). Takes about 10 minutes.
echo   Your current setup is NOT changed unless the new one is
echo   proven faster and passes all tests.
echo ==============================================================
echo.
set ARCH=%PROCESSOR_ARCHITECTURE%
if defined PROCESSOR_ARCHITEW6432 set ARCH=%PROCESSOR_ARCHITEW6432%
if /i not "%ARCH%"=="ARM64" (
    echo This computer does not have an ARM processor - nothing to do.
    pause
    exit /b 0
)
if not exist venv\Scripts\python.exe (
    echo Please run 1_SETUP_WINDOWS.bat first.
    pause
    exit /b 1
)
where py >nul 2>nul
if errorlevel 1 (
    echo The Python install manager "py" was not found. Nothing was changed.
    pause
    exit /b 1
)

echo Step 1 of 5: installing the ARM version of Python 3.14 ...
py install 3.14-arm64
set ARMPY=
for /f "delims=" %%p in ('py -V:3.14-arm64 -c "import sys;print(sys.executable)" 2^>nul') do set "ARMPY=%%p"
if not defined ARMPY if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-arm64\python.exe" set "ARMPY=%LOCALAPPDATA%\Python\pythoncore-3.14-arm64\python.exe"
if not defined ARMPY (
    echo Could not find the ARM Python after installing. Nothing was changed.
    pause
    exit /b 1
)
"%ARMPY%" -c "import platform,sys;sys.exit(0 if platform.machine().upper() in ('ARM64','AARCH64') else 1)"
if errorlevel 1 (
    echo The installed Python is not the ARM version. Nothing was changed.
    pause
    exit /b 1
)

echo.
echo Step 2 of 5: building a second environment (venv_arm64) ...
rem Numba's compiler does not know the Snapdragon chip model yet - compile for a generic ARM CPU
set NUMBA_CPU_NAME=generic
if exist data\speed_arm64.txt del data\speed_arm64.txt
if exist venv_arm64 rmdir /s /q venv_arm64
"%ARMPY%" -m venv venv_arm64
if errorlevel 1 goto failed
venv_arm64\Scripts\python.exe -m pip install --upgrade pip
venv_arm64\Scripts\python.exe -m pip install -r requirements.txt numba
if errorlevel 1 goto failed

echo.
echo Step 3 of 5: checking the ARM version works (unit tests) ...
venv_arm64\Scripts\python.exe -m tests.test_units
if errorlevel 1 goto failed

echo.
echo Step 4 of 5: speed test - current version ...
venv\Scripts\python.exe run.py bench --write data\speed_current.txt
if errorlevel 1 goto failed
echo.
echo Step 5 of 5: speed test - ARM version ...
venv_arm64\Scripts\python.exe run.py bench --write data\speed_arm64.txt
if errorlevel 1 goto failed

if not exist data\speed_arm64.txt goto failed
venv_arm64\Scripts\python.exe -c "a=float(open(r'data\speed_arm64.txt').read());c=float(open(r'data\speed_current.txt').read());print();print('ARM version: %%.1fx the speed of the current one' %% (a/c));raise SystemExit(0 if a>c*1.1 else 1)"
if errorlevel 1 (
    echo The ARM version is not clearly faster - keeping your current setup.
    if exist use_arm64.txt del use_arm64.txt
    pause
    exit /b 0
)
echo ARM64> use_arm64.txt
echo.
echo ==============================================================
echo   Done. The bot now uses the faster ARM version automatically.
echo   (To go back: delete the file use_arm64.txt)
echo ==============================================================
pause
exit /b 0

:failed
echo.
echo Something went wrong (see above). Your current setup was NOT changed.
if exist use_arm64.txt del use_arm64.txt
pause
exit /b 1
