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
echo [%time%] Starting Data Preprocessing...
python src\preprocess.py
if %errorlevel% equ 0 (
    echo [%time%] Preprocessing: SUCCESS
) else (
    echo [%time%] Preprocessing: WARNING - Check output above
)

echo.
echo [%time%] Generating Sentiment Labels...
python src\generate_labels.py  
if %errorlevel% equ 0 (
    echo [%time%] Label Generation: SUCCESS
) else (
    echo [%time%] Label Generation: WARNING - Check output above  
)

echo.
echo Current Data Status:
python -c "import pandas as pd; import os; files={'News': 'data/raw/all_news_data.csv', 'Local': 'data/raw/local_news.csv', 'Twitter': 'data/raw/twitter_data.csv'}; [print(f'{name}: {len(pd.read_csv(path)) if os.path.exists(path) else 0} records') for name, path in files.items()]; labeled_path='data/processed/civic_labeled.csv'; print(f'Model Ready: {len(pd.read_csv(labeled_path)) if os.path.exists(labeled_path) else 0} labeled records') if os.path.exists(labeled_path) else print('Model Ready: Not processed yet')"

echo.
echo ============================================
echo Complete pipeline finished! Data is ready for ML model training.
echo ============================================
pause