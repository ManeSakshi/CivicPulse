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
echo [STEP 4/4] PIPELINE COMPLETION ^& STATUS
echo --------------------------------------------------
echo [%time%] Checking final data status...
python src\pipeline_status.py
if %errorlevel% equ 0 (
    echo [%time%] Status Check: SUCCESS
) else (
    echo [%time%] Status Check: COMPLETED WITH WARNINGS
)

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