@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1

cls
echo.
echo ================================================================================
echo   CAI DAT - PIG COUNTER
echo ================================================================================
echo.

echo [1/3] Kiem tra Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [LOI] Python chua duoc cai dat!
    echo Vui long tai Python tai: https://www.python.org/downloads/
    echo Khi cai dat, CHON DANH DAU "Add Python to PATH"
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version') do echo [OK] %%i
echo.

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [LOI] Python cua ban khong co san "pip". Vui long cai lai Python tu python.org.
    pause
    exit /b 1
)

echo [2/3] Tao virtual environment (.venv)...
if not exist ".venv" (
    python -m venv .venv
)
call ".venv\Scripts\activate.bat"
echo.

echo [3/3] Cai dat thu vien (co the mat 10-20 phut lan dau, tuy toc do mang)...
echo.
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [LOI] Cai dat that bai! Kiem tra ket noi Internet va thu lai.
    pause
    exit /b 1
)

if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

echo INSTALLED > ".installed"

echo.
echo ================================================================================
echo   CAI DAT THANH CONG!
echo   Ban KHONG CAN chay lai file nay nua.
echo   Tu bay gio, chi can bam file "run.bat" moi lan muon mo Pig Counter.
echo ================================================================================
echo.
pause
