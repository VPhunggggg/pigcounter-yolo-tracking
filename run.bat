@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cd /d "%~dp0"

REM --- run.bat CHI de CHAY ung dung. Khong cai dat gi ca. ---
REM --- Neu chua cai dat, hay chay install.bat MOT LAN truoc. ---

if not exist "%~dp0.venv\Scripts\python.exe" (
    cls
    echo.
    echo ================================================================================
    echo   CHUA CAI DAT
    echo ================================================================================
    echo.
    echo [!] May tinh nay chua cai dat Pig Counter.
    echo.
    echo     Vui long bam DUP CHUOT VAO FILE "install.bat" MOT LAN
    echo     truoc khi chay "run.bat".
    echo.
    pause
    exit /b 1
)

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":5000 " ^| findstr "LISTENING"') do (
    echo.
    echo [!] Pig Counter da dang chay tai http://127.0.0.1:5000
    start "" "http://127.0.0.1:5000"
    exit /b 0
)

cls
echo.
echo ================================================================================
echo   PIG COUNTER - Dang khoi dong...
echo ================================================================================
echo.

set "PROJECT_DIR=%~dp0"
set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

call ".venv\Scripts\activate.bat"

REM --- Chay python NGAY TRONG cua so terminal nay (khong mo cua so moi) bang
REM --- "start /B", de tat ca log (server + AI + so dem qua cong...) hien
REM --- chung trong 1 cua so duy nhat, khong bi mo 2 cua so terminal.
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"
start /B "" ".venv\Scripts\python.exe" "%PROJECT_DIR%\app.py"

echo Dang cho server khoi dong (lan dau load model AI co the mat 1-2 phut)...
echo.
set /a tries=0
set /a max_tries=300

:wait_loop
timeout /t 1 /nobreak >nul
set /a tries+=1

curl -s http://127.0.0.1:5000/api/health >nul 2>nul
if not errorlevel 1 goto :server_ready

set /a mod10=!tries! %% 10
if !mod10!==0 echo    ... da cho !tries! giay, van dang khoi dong...

if !tries! lss !max_tries! goto :wait_loop

echo.
echo [!] Da cho qua lau. Kiem tra xem co loi khong, hoac tu mo trinh duyet tai:
echo     http://127.0.0.1:5000
echo.
pause
exit /b 1

:server_ready
echo.
echo [OK] Server da san sang!
echo.
echo   Trinh duyet se tu mo tai: http://127.0.0.1:5000
echo   De DUNG ung dung: bam nut "Dung ung dung" ngay trong trang web,
echo   chay file "stop.bat", HOAC dong truc tiep CUA SO NAY.
echo.
echo   (Cua so nay se hien log cua chuong trinh - AI dang xu ly gi, dem duoc
echo    bao nhieu heo... Dung dong cua so nay khi con dang dem heo.)
echo.
start "" "http://127.0.0.1:5000"

:idle_loop
timeout /t 3600 /nobreak >nul
goto :idle_loop
