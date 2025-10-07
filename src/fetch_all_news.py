import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from gnews import GNews

# -------------------------------
# 1. Load environment & setup
# -------------------------------
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
gnews = GNews(language='en', country='IN', max_results=50)

# -------------------------------
# 2. Fetch from NewsAPI
# -------------------------------
def fetch_newsapi_data(query="Sangli civic OR traffic OR water OR garbage OR sanitation"):
    print("📰 Fetching from NewsAPI...")
    from_date = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        articles = newsapi.get_everything(
            q=query,
            language="en",
            from_param=from_date,
            to=to_date,
            sort_by="relevancy",
            page_size=50
        )
        news_data = []
        for a in articles.get("articles", []):
            news_data.append({
                "source": a["source"]["name"],
                "title": a["title"],
                "description": a["description"],
                "content": a["content"],
                "publishedAt": a["publishedAt"],
                "url": a["url"],
                "provider": "NewsAPI"
            })
        print(f"✅ NewsAPI fetched {len(news_data)} articles.")
        return pd.DataFrame(news_data)
    except Exception as e:
        print("❌ Error fetching from NewsAPI:", e)
        return pd.DataFrame()

# -------------------------------
# 3. Fetch from GNews
# -------------------------------
def fetch_gnews_data(query="Sangli civic OR traffic OR water OR garbage OR sanitation"):
    print("🌐 Fetching from GNews...")
    try:
        articles = gnews.get_news(query)
        news_data = []
        for a in articles:
            news_data.append({
                "source": a.get("publisher", {}).get("title", "Unknown"),
                "title": a.get("title"),
                "description": a.get("description"),
                "content": a.get("description"),
                "publishedAt": a.get("published date"),
                "url": a.get("url"),
                "provider": "GNews"
            })
        print(f"✅ GNews fetched {len(news_data)} articles.")
        return pd.DataFrame(news_data)
    except Exception as e:
        print("❌ Error fetching from GNews:", e)
        return pd.DataFrame()

# -------------------------------
# 4. Merge & Save
# -------------------------------
def collect_all_news():
    os.makedirs("data/raw", exist_ok=True)
    df1 = fetch_newsapi_data()
    df2 = fetch_gnews_data()

    combined = pd.concat([df1, df2], ignore_index=True)
    combined.drop_duplicates(subset=["title"], inplace=True)
    print(f"🧹 Combined total: {len(combined)} unique news articles")

    save_path = "data/raw/all_news_data.csv"
    combined.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved merged news to {save_path}")

    return combined

if __name__ == "__main__":
    collect_all_news()
