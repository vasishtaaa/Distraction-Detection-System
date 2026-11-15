@echo off
title Real-Time Driver Safety & Distraction Detection System
cd /d "%~dp0"
echo =================================================================
echo   REAL-TIME DRIVER SAFETY & DISTRACTION DETECTION SYSTEM
echo   Launching with Integrated Camera (Camera Index 0)...
echo =================================================================
python main.py --source 0
if errorlevel 1 (
    echo.
    echo [ERROR] Could not start system with Camera 0.
    echo Trying Camera 1...
    python main.py --source 1
)
pause
