@echo off
echo.
echo ================================================================== 
echo SANGLI DASHBOARD - WEEKLY UPDATE
echo Collect ΓåÆ Process ΓåÆ Label ΓåÆ Topics ΓåÆ Ready
echo ==================================================================
echo 🎯 Updating 100%% Sangli Municipal Corporation civic data
echo.

set START_TIME=%time%
echo [%time%] Weekly Sangli update started...

echo.
echo 📡 STEP 1: COLLECTING FRESH SANGLI DATA
echo --------------------------------------------------
echo [%time%] Collecting Sangli news...
python src\fetch_sangli_only.py
echo.
echo [%time%] Generating Sangli civic tweets...
python src\fetch_sangli_twitter.py

echo.
echo 🔄 STEP 2: PROCESSING SANGLI DATA  
echo --------------------------------------------------
echo [%time%] Processing and cleaning text...
python src\preprocess_sangli.py

echo.
echo 🎭 STEP 3: SENTIMENT ANALYSIS
echo --------------------------------------------------  
echo [%time%] Generating sentiment labels...
python src\label_sangli.py

echo.
echo 🎯 STEP 4: TOPIC CATEGORIZATION
echo --------------------------------------------------
echo [%time%] Analyzing civic issue categories...
python src\sangli_topic_model.py

echo.
echo 🔧 STEP 5: DASHBOARD UPDATE
echo --------------------------------------------------
echo [%time%] Updating dashboard data...
python update_sangli_topics.py

echo.
echo 📊 STEP 6: VERIFICATION  
echo --------------------------------------------------
echo [%time%] Verifying update success...
python verify_sangli_data.py

echo.
echo ==================================================================
set END_TIME=%time%
echo [%END_TIME%] WEEKLY SANGLI UPDATE COMPLETE!
echo Duration: %START_TIME% to %END_TIME%
echo ==================================================================
echo.
echo ✅ SANGLI DASHBOARD UPDATED WITH FRESH DATA!
echo.
echo 🚀 TO VIEW UPDATED DASHBOARD:
echo    python -m streamlit run src/dashboard_simple.py
echo    Browser: http://localhost:8501
echo.
echo 📊 WHAT'S NEW:
echo    - Latest Sangli civic news and issues
echo    - Updated sentiment analysis  
echo    - Fresh topic categorization
echo    - 100%% Sangli-focused content
echo.
echo ==================================================================
pause