from gnews import GNews
import pandas as pd
import os
from datetime import datetime

def fetch_news():
    """Fetch civic-related news for Sangli and Maharashtra."""
    google_news = GNews(language='en', country='IN', max_results=100)
    queries = ["Sangli civic", "Sangli traffic", "Sangli garbage", "Sangli water", "Sangli roads", "Maharashtra civic"]
    all_articles = []

    for q in queries:
        articles = google_news.get_news(q)
        for art in articles:
            all_articles.append({
                "title": art.get("title"),
                "description": art.get("description"),
                "url": art.get("url"),
                "publishedAt": art.get("published date"),
                "source": art.get("publisher", {}).get("title", "Unknown"),
                "query": q
            })

    df = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/gnews_data.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} GNews articles to data/raw/gnews_data.csv")

if __name__ == "__main__":
    fetch_news()
