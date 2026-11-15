@echo off
title Distraction Detection System - Desktop Interface
cd /d "%~dp0"
echo =================================================================
echo   DISTRACTION DETECTION SYSTEM (DDS)
echo   Starting Desktop Application (Integrated Camera)...
echo =================================================================
python main.py --source 0
if errorlevel 1 (
    echo.
    echo Camera 0 not accessible. Starting with Camera 1...
    python main.py --source 1
)
pause
