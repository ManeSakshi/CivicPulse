import pandas as pd
import os
import subprocess
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
import tweepy

# --------------------------------------------------------------------
# ✅ Load environment variables
# --------------------------------------------------------------------
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# --------------------------------------------------------------------
# ✅ Setup Tweepy client (if available)
# --------------------------------------------------------------------
client = None
if BEARER_TOKEN:
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
        print("✅ Twitter API client initialized")
    except Exception as e:
        print(f"⚠️ Tweepy setup failed: {e}")
else:
    print("⚠️ No TWITTER_BEARER_TOKEN found — Tweepy API mode will be skipped.")


# --------------------------------------------------------------------
# 🐦 Fetch tweets using Tweepy (API)
# --------------------------------------------------------------------
def fetch_tweets_tweepy(query="Maharashtra civic OR traffic OR water OR garbage OR sanitation", max_pages=2):
    tweets_data = []
    if not client:
        return pd.DataFrame()

    next_token = None
    for _ in range(max_pages):
        try:
            response = client.search_recent_tweets(
                query=query + " -is:retweet lang:en",
                tweet_fields=["id", "text", "created_at"],
                max_results=50,
                next_token=next_token,
            )
            if not response.data:
                break
            for tweet in response.data:
                tweets_data.append({
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "source": "TwitterAPI"
                })
            next_token = response.meta.get("next_token")
            if not next_token:
                break
        except Exception as e:
            print("⚠️ Tweepy error:", e)
            break

    return pd.DataFrame(tweets_data)


# --------------------------------------------------------------------
# 🕵️ Fetch tweets using snscrape (No API needed)
# --------------------------------------------------------------------
def fetch_tweets_snscrape(
    query="Sangli civic OR Sangli traffic OR Sangli water OR Sangli garbage OR Sangli sanitation",
    limit_per_query=150
):
    tweets_data = []
    since_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    queries = [
        "Sangli traffic",
        "Sangli garbage",
        "Sangli water",
        "Sangli sanitation",
        "Sangli civic issues",
        "Sangli municipality",
        "Sangli roads"
    ]

    for q in queries:
        cmd = f'python -m snscrape --jsonl --max-results {limit_per_query} twitter-search "{q} since:{since_date}"'
        print(f"🔍 Running snscrape: {q}")
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

        if result.returncode != 0:
            print(f"⚠️ snscrape failed for {q}: {result.stderr}")
            continue

        for line in result.stdout.split("\n"):
            if line.strip():
                try:
                    tweet_json = json.loads(line)
                    tweets_data.append({
                        "text": tweet_json.get("content", ""),
                        "created_at": tweet_json.get("date", datetime.now()),
                        "source": "snscrape"
                    })
                except json.JSONDecodeError:
                    continue

    return pd.DataFrame(tweets_data)


# --------------------------------------------------------------------
# 🚀 Main function to combine results
# --------------------------------------------------------------------
def collect_all_tweets():
    print("🐦 Starting Twitter data collection...")

    # 1️⃣ Fetch via Tweepy (if available)
    api_df = fetch_tweets_tweepy()
    print(f"📡 Twitter API returned {len(api_df)} tweets")

    # 2️⃣ Fetch via snscrape
    scrape_df = fetch_tweets_snscrape()
    print(f"🔍 snscrape returned {len(scrape_df)} tweets")

    # 3️⃣ Combine both
    combined = pd.concat([api_df, scrape_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"], keep="first")

    print(f"✅ Total unique tweets collected: {len(combined)}")

    # 4️⃣ Save results
    os.makedirs("data/raw", exist_ok=True)
    combined.to_csv("data/raw/twitter_data.csv", index=False, encoding="utf-8-sig")
    print("💾 Saved to data/raw/twitter_data.csv")

    return combined


# --------------------------------------------------------------------
# 🏁 Entry Point
# --------------------------------------------------------------------
if __name__ == "__main__":
    collect_all_tweets()
