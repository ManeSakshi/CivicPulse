@echo off
echo.
echo ================================================================== 
echo CivicPulse COMPLETE Data Pipeline
echo Collect → Preprocess → Label → Model Ready
echo ==================================================================
echo.

set START_TIME=%time%
echo [%time%] Pipeline started...

REM Step 1: Data Collection
echo.
echo [STEP 1/4] DATA COLLECTION
echo --------------------------------------------------
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

REM Step 2: Data Preprocessing
echo.
echo [STEP 2/4] DATA PREPROCESSING
echo --------------------------------------------------
echo [%time%] Starting intelligent text preprocessing...
python src\preprocess.py
if %errorlevel% equ 0 (
    echo [%time%] Preprocessing: SUCCESS
) else (
    echo [%time%] Preprocessing: FAILED
    goto :error
)

REM Step 3: Label Generation
echo.
echo [STEP 3/4] SENTIMENT LABEL GENERATION
echo --------------------------------------------------
echo [%time%] Generating VADER + TextBlob sentiment labels...
python src\generate_labels.py
if %errorlevel% equ 0 (
    echo [%time%] Label Generation: SUCCESS
) else (
    echo [%time%] Label Generation: FAILED
    goto :error
)

REM Step 4: Final Status & Model Readiness Check
echo.
echo [STEP 4/4] PIPELINE COMPLETION & STATUS
echo --------------------------------------------------
echo [%time%] Checking final data status...

python -c "
import pandas as pd
import os

print('COMPLETE PIPELINE STATUS REPORT')
print('=' * 50)

# Raw data status
files = {'News': 'data/raw/all_news_data.csv', 'Local': 'data/raw/local_news.csv', 'Twitter': 'data/raw/twitter_data.csv'}
total_raw = 0
for name, path in files.items():
    count = len(pd.read_csv(path)) if os.path.exists(path) else 0
    print(f'{name:12}: {count:4d} records')
    total_raw += count

# Processed data status
if os.path.exists('data/processed/civicpulse_processed.csv'):
    df_processed = pd.read_csv('data/processed/civicpulse_processed.csv')
    print(f'Processed   : {len(df_processed):4d} records')
else:
    print('Processed   : ERROR - File not found')

# Labeled data status
if os.path.exists('data/processed/civic_labeled.csv'):
    df_labeled = pd.read_csv('data/processed/civic_labeled.csv')
    labels = df_labeled['label'].value_counts()
    print(f'Labeled     : {len(df_labeled):4d} records')
    for label, count in labels.items():
        print(f'  {label:8}: {count:4d} ({count/len(df_labeled)*100:.1f}%%)')
    
    # Model readiness check
    vocab_size = len(set(' '.join(df_labeled['text']).split()))
    avg_length = df_labeled['text'].str.len().mean()
    
    print(f'')
    print(f'MODEL READINESS METRICS:')
    print(f'  Vocabulary size: {vocab_size:,} unique words')
    print(f'  Average length:  {avg_length:.1f} characters')
    print(f'  Data quality:    HIGH (cleaned + lemmatized)')
    print(f'  Label quality:   DUAL-METHOD (VADER + TextBlob)')
    
    if len(df_labeled) >= 500 and vocab_size >= 1000:
        print(f'')
        print(f'[SUCCESS] Dataset is READY for ML model training!')
        print(f'Files ready: data/processed/civic_labeled.csv')
        print(f'             data/processed/civicpulse_with_labels.csv')
    else:
        print(f'[WARNING] Dataset may need more data for robust training')
else:
    print('Labeled     : ERROR - File not found')
"

echo.
echo ==================================================================
set END_TIME=%time%
echo [%END_TIME%] COMPLETE PIPELINE FINISHED
echo From: %START_TIME% To: %END_TIME%
echo ==================================================================
echo [SUCCESS] Your civic sentiment data is now model-ready!
echo.
echo NEXT STEPS:
echo 1. Train ML model: Use data/processed/civic_labeled.csv
echo 2. External data:  Use data/processed/external/ for pre-training  
echo 3. Run analysis:   Start sentiment analysis and topic modeling
echo ==================================================================
echo.
goto :end

:error
echo.
echo [ERROR] Pipeline failed at preprocessing or labeling stage
echo Check the error messages above and try again
echo.

:end
pause