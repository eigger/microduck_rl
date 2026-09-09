@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===================================================
echo   Microduck MuJoCo 3D Simulator
echo ===================================================
echo [Info] Loading trained policies and robot model...
echo [Info] 3D window will open in about 3 seconds.
echo.
set PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -u scripts\infer_policy.py %*
echo.
echo ===================================================
echo Simulator exited (Exit code: %ERRORLEVEL%)
echo ===================================================
pause
