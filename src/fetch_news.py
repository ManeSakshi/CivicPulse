from newsapi import NewsApiClient
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load API key
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)

def fetch_news(query="Sangli OR Sangli Miraj Kupwad OR Sangli Maharashtra", language="en"):
    """Fetch Sangli-specific news articles"""
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        articles = newsapi.get_everything(
            q=query,
            language=language,
            from_param=from_date,
            to=to_date,
            sort_by="relevancy",
            page_size=50
        )
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return pd.DataFrame()

    news_data = []
    for article in articles.get("articles", []):
        news_data.append({
            "source": article["source"]["name"],
            "author": article.get("author"),
            "title": article.get("title"),
            "description": article.get("description"),
            "content": article.get("content"),
            "publishedAt": article.get("publishedAt"),
            "url": article.get("url")
        })

    return pd.DataFrame(news_data)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)  # ensure folder exists
    df = fetch_news()
    if not df.empty:
        df.to_csv("data/raw/news_data.csv", index=False)
        print(f"✅ News saved to data/raw/news_data.csv ({len(df)} rows)")
    else:
        print("⚠️ No news articles found for the given query.")
