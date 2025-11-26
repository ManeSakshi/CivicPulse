@echo off
echo.
echo 🏛️ LAUNCHING SANGLI DASHBOARD...
echo ================================================================== 
echo Dashboard: CivicPulse Sangli Civic Sentiment Analysis
echo Content:   100%% Sangli Municipal Corporation data
echo Features:  Sentiment Analysis + Topic Categories
echo ==================================================================
echo.

echo 📊 Dashboard starting...
echo    URL: http://localhost:8501
echo    Content: Sangli Water, Traffic, Roads, Municipal Services
echo.
echo ⏳ Loading dashboard (may take 10-15 seconds)...
echo    Press Ctrl+C to stop dashboard
echo.

python -m streamlit run src\dashboard_simple.py

echo.
echo 📴 Dashboard stopped.
pause