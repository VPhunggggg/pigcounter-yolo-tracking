@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

cls
echo.
echo ================================================================================
echo   DUNG PIG COUNTER
echo ================================================================================
echo.

set "FOUND=0"
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    set "FOUND=1"
    echo Dang dung tien trinh dang chay ung dung...
    taskkill /PID %%a /F >nul 2>&1
)

if "!FOUND!"=="0" (
    echo [i] Pig Counter hien khong chay.
) else (
    echo [OK] Da dung Pig Counter thanh cong.
)

echo.
pause
