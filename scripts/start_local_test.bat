@echo off
setlocal

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] powershell.exe was not found.
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_local_test.ps1" %*
set "SCRIPT_EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %SCRIPT_EXIT_CODE%
