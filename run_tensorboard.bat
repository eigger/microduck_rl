@echo off
cd /d "%~dp0"
echo ===================================================
echo   Starting TensorBoard (Offline RL Dashboard)
echo ===================================================
echo.
echo Opening browser: http://localhost:6006
start http://localhost:6006
echo Running TensorBoard on logs\rsl_rl...
.\.venv\Scripts\python.exe -m tensorboard.main --logdir logs\rsl_rl
pause
