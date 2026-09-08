@echo off
cd /d "%~dp0"
echo ===================================================
echo   Microduck RL Training (RTX 5060 GPU + TensorBoard)
echo ===================================================
echo.
set PYTHONUNBUFFERED=1
set PYTHONUTF8=1

if "%~1"=="" (
    echo [No arguments specified. Running default task: Mjlab-Velocity-Flat-MicroDuck]
    echo [Envs: 4096, Device: RTX 5060 GPU, Logger: TensorBoard]
    echo.
    .\.venv\Scripts\python.exe -m mjlab_microduck.train_cli Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
) else (
    echo [Running with custom arguments: %*]
    echo.
    .\.venv\Scripts\python.exe -m mjlab_microduck.train_cli %*
)

echo.
echo Training process exited with code %ERRORLEVEL%
pause
