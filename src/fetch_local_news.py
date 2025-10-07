from gnews import GNews
import pandas as pd
from datetime import datetime
import os

def fetch_news(query, lang='en'):
    google_news = GNews(language=lang, country='IN', period='14d', max_results=80)
    results = google_news.get_news(query)
    print(f"🔍 {len(results)} articles found for query='{query}' ({lang})")

    articles = []
    for item in results:
        title = item.get("title", "")
        desc = item.get("description", "")
        url = item.get("url", "")
        src = item.get("publisher", {}).get("title", "")

        # Filter strictly to Sangli mentions
        if "sangli" in (title + desc).lower():
            articles.append({
                "title": title,
                "description": desc,
                "published_date": item.get("published date"),
                "url": url,
                "source": src,
                "language": lang,
                "fetched_at": datetime.utcnow().isoformat()
            })
    return articles


def fetch_local_news():
    queries = [
        "Sangli civic OR traffic OR water OR garbage OR sanitation OR municipality",
        "सांगली नगरपालिका OR पाणी OR रस्ता OR कचरा OR वाहतूक OR शहर"
    ]

    all_articles = []
    for lang, q in zip(['en', 'mr'], queries):
        all_articles.extend(fetch_news(q, lang))

    df = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/local_news.csv", index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} Sangli news articles to data/raw/local_news.csv")


if __name__ == "__main__":
    fetch_local_news()
