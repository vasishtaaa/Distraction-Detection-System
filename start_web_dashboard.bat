@echo off
title Driver Safety & Distraction Monitor - Web Dashboard
cd /d "%~dp0"
echo =================================================================
echo   STARTING DRIVER SAFETY MONITOR WEB DASHBOARD
echo   Open your browser to: http://127.0.0.1:8000
echo =================================================================
python app_web.py
pause
