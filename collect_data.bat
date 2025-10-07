@echo off
echo.
echo ================================================== 
echo CivicPulse Data Collection Session
echo ==================================================
echo.

echo [%time%] Starting News Collection...
python src\fetch_news_unified.py
if %errorlevel% equ 0 (
    echo [%time%] News Collection: SUCCESS
) else (
    echo [%time%] News Collection: COMPLETED WITH WARNINGS
)

echo.
echo [%time%] Starting Twitter Collection...
python src\fetch_twitter_hybrid.py
if %errorlevel% equ 0 (
    echo [%time%] Twitter Collection: SUCCESS  
) else (
    echo [%time%] Twitter Collection: COMPLETED WITH WARNINGS
)

echo.
echo ==================================================
echo Collection Session Complete
echo ==================================================

echo.
echo Current Data Status:
python -c "import pandas as pd; import os; files={'News': 'data/raw/all_news_data.csv', 'Local': 'data/raw/local_news.csv', 'Twitter': 'data/raw/twitter_data.csv'}; [print(f'{name}: {len(pd.read_csv(path)) if os.path.exists(path) else 0} records') for name, path in files.items()]"

echo.
echo Data collection complete! 
pause