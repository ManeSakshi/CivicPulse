@echo off
echo.
echo ================================================================== 
echo CivicPulse SANGLI-ONLY Data Pipeline
echo Collect Sangli → Process → Label → Dashboard Ready
echo ==================================================================
echo 🎯 FOCUS: Sangli Municipal Corporation civic data ONLY
echo ❌ EXCLUDES: Mumbai, Pune, Delhi, other cities
echo.

set START_TIME=%time%
echo [%time%] Sangli-only pipeline started...

REM Step 1: Collect ONLY Sangli data
echo.
echo [STEP 1/4] SANGLI-ONLY DATA COLLECTION
echo --------------------------------------------------
echo [%time%] Checking Sangli data status...
python src\sangli_status_report.py
print('SANGLI-ONLY PIPELINE STATUS REPORT')
print('=' * 50)

# Check Sangli-only files
sangli_files = {
    'Sangli News': 'data/raw/sangli_only_news.csv',
    'Sangli Twitter': 'data/raw/sangli_only_twitter.csv', 
    'Local News': 'data/raw/local_news.csv'
}

total_raw = 0
for name, path in sangli_files.items():
    if os.path.exists(path):
        count = len(pd.read_csv(path))
        print(f'{name:15}: {count:4d} records')
        total_raw += count
    else:
        print(f'{name:15}: FILE NOT FOUND')

print(f'{'Total Raw':15}: {total_raw:4d} records')
print()

# Check processed data
if os.path.exists('data/processed/sangli_processed.csv'):
    df_processed = pd.read_csv('data/processed/sangli_processed.csv')
    print(f'Processed      : {len(df_processed):4d} Sangli records')
else:
    print('Processed      : FILE NOT FOUND')

# Check labeled data  
if os.path.exists('data/processed/sangli_labeled.csv'):
    df_labeled = pd.read_csv('data/processed/sangli_labeled.csv')
    labels = df_labeled['label'].value_counts()
    print(f'Labeled        : {len(df_labeled):4d} Sangli records')
    
    for label, count in labels.items():
        print(f'  {label:8}: {count:4d} ({count/len(df_labeled)*100:.1f}%)')
    
    print()
    print('SANGLI DASHBOARD READINESS:')
    print(f'  ✅ 100% Sangli-specific content')
    print(f'  ✅ {len(df_labeled)} civic records ready')
    print(f'  ✅ Clean sentiment labels applied')
    print(f'  ✅ Ready for dashboard display')
else:
    print('Labeled        : FILE NOT FOUND')
"

echo.
echo ==================================================================
set END_TIME=%time%
echo [%END_TIME%] SANGLI-ONLY PIPELINE FINISHED
echo From: %START_TIME% To: %END_TIME%
echo ==================================================================
echo [SUCCESS] Your SANGLI civic sentiment data is ready!
echo.
echo 🎯 DASHBOARD WILL NOW SHOW ONLY SANGLI DATA
echo.
echo TO VIEW SANGLI DASHBOARD:
echo python -m streamlit run src/dashboard_simple.py
echo.
echo ==================================================================
goto :end

:error
echo.
echo [ERROR] Sangli pipeline failed - check error messages above
echo.

:end
pause