# 📌 CivicPulse Windows Task Scheduler Script
# Creates automated data collection schedule for Windows

@echo off
echo CivicPulse Data Collection Scheduler
echo =====================================

set PROJECT_DIR=C:\Users\manes\OneDrive\Documents\Desktop\CivicPulse
set PYTHON_EXE=python
set SCRIPT_PATH=%PROJECT_DIR%\src\collect_all_data.py

echo Project Directory: %PROJECT_DIR%
echo Python Executable: %PYTHON_EXE%
echo Script Path: %SCRIPT_PATH%

echo.
echo Creating Windows Task Scheduler entries...
echo.

REM Morning collection (9:00 AM daily)
schtasks /create /tn "CivicPulse-Morning" /tr "%PYTHON_EXE% %SCRIPT_PATH%" /sc daily /st 09:00 /f
if %errorlevel% equ 0 (
    echo ✅ Morning task created successfully
) else (
    echo ❌ Failed to create morning task
)

REM Evening collection (7:00 PM daily)  
schtasks /create /tn "CivicPulse-Evening" /tr "%PYTHON_EXE% %SCRIPT_PATH%" /sc daily /st 19:00 /f
if %errorlevel% equ 0 (
    echo ✅ Evening task created successfully
) else (
    echo ❌ Failed to create evening task
)

echo.
echo Scheduled Tasks Created:
echo • CivicPulse-Morning: Daily at 9:00 AM
echo • CivicPulse-Evening: Daily at 7:00 PM
echo.

echo To view tasks: schtasks /query /tn "CivicPulse*"
echo To delete tasks: schtasks /delete /tn "CivicPulse-Morning" /f
echo                   schtasks /delete /tn "CivicPulse-Evening" /f
echo.

echo Setup complete! 🎉
pause