@echo off
setlocal

cd /d "%~dp0"
REM -NoExit keeps the window open so a startup error stays readable instead of
REM flashing closed. The window stays up while the server runs; close it or
REM press Ctrl+C to stop.
powershell -NoProfile -ExecutionPolicy Bypass -NoExit -File "%~dp0Start-Backgrounder.ps1"

echo.
echo Backgrounder exited. Press any key to close this window.
pause >nul
endlocal
