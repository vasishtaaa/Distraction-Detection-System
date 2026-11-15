@echo off
title Real-Time Driver Safety System - Scenario Demo
cd /d "%~dp0"
echo =================================================================
echo   REAL-TIME DRIVER SAFETY & DISTRACTION DETECTION SYSTEM
echo   Launching Scenario Simulation Video...
echo =================================================================
python main.py --source sample_driver_demo.mp4
pause
