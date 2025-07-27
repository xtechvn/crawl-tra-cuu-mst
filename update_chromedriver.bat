@echo off
echo ========================================
echo    CAP NHAT CHROMEDRIVER
echo ========================================
echo.

REM Kich hoat virtual environment
call .venv\Scripts\activate

REM Chay script update
python update_chromedriver.py

echo.
echo ========================================
echo    HOAN THANH
echo ========================================
pause 