@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_front_camera_lidar.ps1"
exit /b %errorlevel%
