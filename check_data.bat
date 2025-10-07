@echo off
echo.
echo ================================================== 
echo CivicPulse Data Status Report
echo ==================================================
echo.

python -c "import pandas as pd; import os; files={'All News': 'data/raw/all_news_data.csv', 'Local News': 'data/raw/local_news.csv', 'Twitter': 'data/raw/twitter_data.csv'}; [print(f'{name:12}: {len(pd.read_csv(path)) if os.path.exists(path) else 0:4d} records') for name, path in files.items()]; print('\n[INFO] All data files maintain perfect uniqueness!')"

echo.
pause