#!/usr/bin/env python3
"""
CivicPulse Sangli Dashboard - Complete Update Guide
Step-by-step process to keep Sangli civic data fresh
"""

print("""
🏛️ CIVICPULSE SANGLI DASHBOARD - UPDATE GUIDE
======================================================================

📅 REGULAR MAINTENANCE SCHEDULE:

DAILY (Optional):
- No action needed - data is stable

WEEKLY (Recommended):
- Collect fresh Sangli civic data
- Update dashboard with new content  

MONTHLY (Deep Clean):
- Clean old data, optimize performance
- Update API keys if needed

======================================================================

🔄 WEEKLY UPDATE PROCESS (3 Steps):
""")

print("""
STEP 1: COLLECT FRESH SANGLI DATA
----------------------------------
Command: python src/fetch_sangli_only.py
Purpose: Get latest Sangli news & civic issues
Time:    ~3-5 minutes

What it does:
✅ Fetches new Sangli news articles
✅ Filters out Mumbai/Pune/other cities  
✅ Saves to data/raw/sangli_only_news.csv
✅ Prevents duplicates from previous runs

Command: python src/fetch_sangli_twitter.py  
Purpose: Generate new Sangli civic tweets
Time:    ~1 minute

What it does:
✅ Creates realistic Sangli civic tweets
✅ Covers all Sangli areas (Market Yard, Miraj Road, etc.)
✅ Saves to data/raw/sangli_only_twitter.csv
""")

print("""
STEP 2: PROCESS & ANALYZE NEW DATA
----------------------------------
Command: python src/preprocess_sangli.py
Purpose: Clean and prepare text data
Time:    ~2-3 minutes

What it does:
✅ Combines news + Twitter data
✅ Cleans text (removes noise, lemmatization)
✅ Saves to data/processed/sangli_processed.csv

Command: python src/label_sangli.py
Purpose: Generate sentiment labels  
Time:    ~1-2 minutes

What it does:
✅ VADER + TextBlob sentiment analysis
✅ Labels: positive, neutral, negative
✅ Saves to data/processed/sangli_labeled.csv
""")

print("""
STEP 3: UPDATE TOPIC CATEGORIES
--------------------------------
Command: python src/sangli_topic_model.py
Purpose: Categorize civic issues by topic
Time:    ~2-3 minutes

What it does:
✅ Identifies: Water, Traffic, Roads, Municipal Services
✅ Creates topic distribution charts
✅ Saves topic results for dashboard

Command: python update_sangli_topics.py
Purpose: Add topics to dashboard data
Time:    ~30 seconds

What it does:
✅ Merges topic categories with labeled data
✅ Updates dashboard-ready files
✅ Enables topic-wise analysis
""")

print("""
======================================================================

🚀 AUTOMATED UPDATE (All Steps Combined):
""")

print("""
OPTION 1: Manual Step-by-Step
-----------------------------
python src/fetch_sangli_only.py      # Collect news
python src/fetch_sangli_twitter.py   # Generate tweets  
python src/preprocess_sangli.py      # Process data
python src/label_sangli.py           # Label sentiment
python src/sangli_topic_model.py     # Categorize topics
python update_sangli_topics.py       # Update dashboard

OPTION 2: One-Click Update (Recommended)
----------------------------------------
.\run_sangli_only_pipeline.bat       # Runs all steps automatically

OPTION 3: Quick Dashboard Launch
--------------------------------
python -m streamlit run src/dashboard_simple.py  # View results
""")

print("""
======================================================================

📊 DASHBOARD VERIFICATION:
""")

print("""
After updates, your dashboard should show:

📈 OVERVIEW TAB:
- Total Sangli records count
- Sentiment percentages  
- Data freshness indicators

🎯 TOPICS TAB:
- Issue category breakdown (Water, Traffic, Roads, etc.)
- Interactive charts and filters
- Sample civic issues by category

🔍 EXPLORER TAB:
- Search Sangli civic records
- Filter by sentiment/topic/date
- Export capabilities

🔧 SYSTEM INFO TAB:  
- Data pipeline status
- Model performance metrics
- Last update timestamps
""")

print("""
======================================================================

⚠️ TROUBLESHOOTING GUIDE:
""")

print("""
ISSUE: "No new data collected"
SOLUTION: 
✅ Check internet connection
✅ Verify API keys (optional - synthetic data works)
✅ Run anyway - synthetic data provides fresh content

ISSUE: "Topic categories missing"  
SOLUTION:
✅ Run: python src/sangli_topic_model.py
✅ Then: python update_sangli_topics.py
✅ Restart dashboard

ISSUE: "Dashboard shows old data"
SOLUTION:  
✅ Check file: data/processed/sangli_labeled.csv
✅ Verify 'topic_category' column exists
✅ Refresh browser (Ctrl+F5)

ISSUE: "Dashboard won't start"
SOLUTION:
✅ Use: python -m streamlit run src/dashboard_simple.py  
✅ Check for error messages in terminal
✅ Ensure all dependencies installed
""")

print("""
======================================================================

📅 MAINTENANCE CALENDAR:
""")

print("""
WEEKLY (Every Sunday):
□ Run complete update pipeline
□ Check dashboard functionality  
□ Review new civic issues

MONTHLY (1st of month):
□ Clean old cache files
□ Update API keys if expired
□ Backup important data

QUARTERLY (Every 3 months):
□ Review model performance
□ Optimize topic categories
□ Update documentation
""")

print("""
======================================================================

🎯 SUCCESS INDICATORS:
""")

print("""
Your Sangli dashboard is working properly when:

✅ Total records increase after updates
✅ New dates appear in data timestamps  
✅ Topic categories show balanced distribution
✅ Sentiment analysis reflects current issues
✅ Search functionality finds relevant content
✅ No error messages in dashboard

Dashboard URL: http://localhost:8501
Data Focus: 100% Sangli Municipal Corporation civic issues
Update Frequency: Weekly recommended, monthly minimum

======================================================================
""")

if __name__ == "__main__":
    print("📖 Sangli Dashboard Update Guide Complete!")
    print("\n🚀 Next Action: Run weekly update")
    print("   Command: .\\run_sangli_only_pipeline.bat")
    print("   Then launch: python -m streamlit run src/dashboard_simple.py")