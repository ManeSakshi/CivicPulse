@echo off
setlocal enabledelayedexpansion

echo.
echo ================================================================== 
echo CivicPulse SMART Pipeline - Only Process If New Data Found
echo ==================================================================
echo.

REM Get initial record counts
set "initial_news=0"
set "initial_twitter=0"

for /f "tokens=*" %%i in ('python -c "import pandas as pd; import os; print(len(pd.read_csv('data/raw/all_news_data.csv')) if os.path.exists('data/raw/all_news_data.csv') else 0)"') do set initial_news=%%i
for /f "tokens=*" %%i in ('python -c "import pandas as pd; import os; print(len(pd.read_csv('data/raw/twitter_data.csv')) if os.path.exists('data/raw/twitter_data.csv') else 0)"') do set initial_twitter=%%i

echo [INFO] Initial counts - News: !initial_news!, Twitter: !initial_twitter!

REM Data Collection
echo.
echo [STEP 1] DATA COLLECTION
echo --------------------------------------------------
python src\fetch_news_unified.py > nul 2>&1
python src\fetch_twitter_hybrid.py > nul 2>&1

REM Get final record counts
set "final_news=0" 
set "final_twitter=0"

for /f "tokens=*" %%i in ('python -c "import pandas as pd; import os; print(len(pd.read_csv('data/raw/all_news_data.csv')) if os.path.exists('data/raw/all_news_data.csv') else 0)"') do set final_news=%%i
for /f "tokens=*" %%i in ('python -c "import pandas as pd; import os; print(len(pd.read_csv('data/raw/twitter_data.csv')) if os.path.exists('data/raw/twitter_data.csv') else 0)"') do set final_twitter=%%i

REM Calculate changes
set /a news_diff=!final_news!-!initial_news!
set /a twitter_diff=!final_twitter!-!initial_twitter!
set /a total_new=!news_diff!+!twitter_diff!

echo [INFO] Final counts - News: !final_news! (+!news_diff!), Twitter: !final_twitter! (+!twitter_diff!)

REM Only process if new data was found
if !total_new! gtr 0 (
    echo.
    echo [SUCCESS] Found !total_new! new records - Running full pipeline...
    
    echo.
    echo [STEP 2] PREPROCESSING NEW DATA
    echo --------------------------------------------------
    python src\preprocess.py
    
    echo.
    echo [STEP 3] UPDATING LABELS  
    echo --------------------------------------------------
    python src\generate_labels.py
    
    echo.
    echo [COMPLETE] Pipeline finished with !total_new! new records processed!
) else (
    echo.
    echo [INFO] No new data found - skipping preprocessing
    echo Your existing processed data is still current
)

echo.
echo ==================================================================
python -c "import pandas as pd; import os; labeled_path='data/processed/civic_labeled.csv'; print(f'Total model-ready records: {len(pd.read_csv(labeled_path))}') if os.path.exists(labeled_path) else print('No processed data available')"
echo ==================================================================
echo.
pause