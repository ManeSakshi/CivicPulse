# Civic Pulse Sangli

**Final Year Project – Civic Sentiment Analysis Platform**

This project analyzes civic complaints about **Sangli city** (traffic, water, sanitation, garbage) using NLP and AI.  
It provides an **interactive dashboard** to visualize public sentiment and issue categories.

## 🚀 Features
- Collect data from Twitter (Tweepy) & NewsAPI  
- Preprocess and clean text  
- Sentiment analysis using DistilBERT  
- Explainable AI (SHAP/LIME)  
- Topic modeling with BERTopic/LDA  
- Store results in SQLite  
- Interactive dashboard (Streamlit/Flask + Plotly)  

## ⚡ Setup

### 1. Clone the Repo
git clone https://github.com/<your-username>/civic-pulse-sangli.git
cd civic-pulse-sangli

###2. Create Virtual Environment
Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate.ps1

Linux/Mac:
python3 -m venv venv
source venv/bin/activate

###3. Install Requirements
pip install -r requirements.txt

###4. Add API Keys
Copy .env.example → .env and add your keys:
TWITTER_BEARER_TOKEN=your_token_here
NEWSAPI_KEY=your_api_key_here

###5. Test Setup
python src/test_setup.py

## ▶️ Usage
python src/fetch_twitter.py      # Collect tweets
python src/fetch_news.py         # Collect news
python src/preprocess.py         # Clean data
python src/sentiment_infer.py    # Run sentiment analysis
python src/topic_model.py        # Topic modeling
streamlit run src/dashboard_app.py   # Run dashboard
