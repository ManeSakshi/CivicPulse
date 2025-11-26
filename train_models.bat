@echo off
REM CivicPulse Complete ML Pipeline Execution
REM Trains sentiment analysis model, runs topic modeling, and launches dashboard

echo ======================================
echo    CIVICPULSE ML PIPELINE EXECUTION
echo ======================================
echo.

REM Check if Python environment is configured
python -c "import sys; print('Python:', sys.version)" 2>nul
if errorlevel 1 (
    echo ERROR: Python not found or not configured
    echo Please ensure Python is installed and in PATH
    pause
    exit /b 1
)

REM Navigate to project directory
cd /d "%~dp0"
if not exist "src\" (
    echo ERROR: src directory not found
    echo Please run from project root directory
    pause
    exit /b 1
)

echo Step 1: Installing required packages...
echo =====================================

REM Install core ML packages
pip install pandas numpy scikit-learn matplotlib seaborn

REM Install advanced NLP packages
pip install transformers torch spacy textblob vaderSentiment

REM Install topic modeling packages
pip install bertopic sentence-transformers umap-learn

REM Install dashboard packages  
pip install streamlit plotly

REM Download SpaCy model
python -m spacy download en_core_web_sm

echo.
echo Step 2: Training Sentiment Analysis Model...
echo ===========================================
echo This may take 10-15 minutes for full training...

python src/sentiment_infer.py
if errorlevel 1 (
    echo ERROR: Sentiment model training failed
    echo Check error messages above
    pause
    exit /b 1
)

echo ✅ Sentiment analysis model training completed!
echo.

echo Step 3: Running Topic Modeling...
echo =================================

python src/topic_model.py
if errorlevel 1 (
    echo WARNING: Topic modeling failed, continuing...
    echo This is optional - dashboard will still work
) else (
    echo ✅ Topic modeling completed!
)

echo.
echo Step 4: Checking model files...
echo ==============================

if exist "models\sentiment_model.pkl" (
    echo ✅ Sentiment model saved successfully
) else (
    echo ⚠️  Sentiment model file not found
)

if exist "models\topics\topic_results.pkl" (
    echo ✅ Topic model saved successfully
) else (
    echo ⚠️  Topic model file not found
)

echo.
echo Step 5: Launching Dashboard...
echo =============================
echo Dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the dashboard server
echo.

REM Launch Streamlit dashboard
streamlit run src/dashboard_app.py --server.port 8501 --server.headless false

echo.
echo ======================================
echo    PIPELINE EXECUTION COMPLETE
echo ======================================
pause