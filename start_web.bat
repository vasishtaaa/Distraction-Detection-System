@echo off
title Distraction Detection System - Web Dashboard
cd /d "%~dp0"
echo =================================================================
echo   DISTRACTION DETECTION SYSTEM (DDS)
echo   Starting Modern Web Dashboard at http://127.0.0.1:8000
echo =================================================================
start http://127.0.0.1:8000
python app_web.py
pause
