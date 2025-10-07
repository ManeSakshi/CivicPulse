import os
import time
import json
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Tweepy import (for API)
import tweepy

# ----------------------------------------
# 1. Setup
# ----------------------------------------
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
os.makedirs("data/raw", exist_ok=True)

# ----------------------------------------
# 2. Tweepy client setup
# ----------------------------------------
client = None
if BEARER_TOKEN:
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)
        print("✅ Tweepy client initialized")
    except Exception as e:
        print(f"⚠️ Tweepy setup failed: {e}")
        client = None
else:
    print("⚠️ No TWITTER_BEARER_TOKEN found — will use snscrape only")

# ----------------------------------------
# 3. Fetch via Twitter API
# ----------------------------------------
def fetch_tweets_api(query="Sangli civic OR traffic OR water OR garbage OR sanitation", max_results=50):
    if not client:
        return pd.DataFrame()

    print("📡 Fetching tweets using Twitter API...")
    tweets_data = []

    try:
        response = client.search_recent_tweets(
            query=f"{query} -is:retweet lang:en",
            tweet_fields=["id", "text", "created_at"],
            max_results=max_results,
        )

        if response.data:
            for tweet in response.data:
                tweets_data.append({
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "source": "TwitterAPI"
                })
            print(f"✅ Collected {len(tweets_data)} tweets using API.")
        else:
            print("⚠️ No tweets found using API.")

    except tweepy.errors.TooManyRequests:
        print("⏳ Rate limit hit — switching to snscrape.")
    except Exception as e:
        print(f"⚠️ API error: {e}")

    return pd.DataFrame(tweets_data)

# ----------------------------------------
# 4. Fetch via SNScrape
# ----------------------------------------
def fetch_tweets_snscrape(query="Sangli civic OR traffic OR water OR garbage OR sanitation", limit=200):
    print("🔍 Fetching tweets using snscrape...")
    since_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    cmd = f'snscrape --jsonl --max-results {limit} twitter-search "{query} since:{since_date}"'

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        tweets_data = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                tweet_json = json.loads(line)
                tweets_data.append({
                    "text": tweet_json.get("content", ""),
                    "created_at": tweet_json.get("date", ""),
                    "source": "snscrape"
                })
            except json.JSONDecodeError:
                continue

        print(f"✅ Collected {len(tweets_data)} tweets using snscrape.")
        return pd.DataFrame(tweets_data)

    except Exception as e:
        print(f"❌ SNScrape error: {e}")
        return pd.DataFrame()

# ----------------------------------------
# 5. Combine and Save
# ----------------------------------------
def collect_all_tweets():
    print("🐦 Starting Twitter data collection...")
    df_api = fetch_tweets_api()
    df_scrape = fetch_tweets_snscrape()

    combined = pd.concat([df_api, df_scrape], ignore_index=True)
    combined.drop_duplicates(subset=["text"], inplace=True)
    print(f"🧹 Total unique tweets: {len(combined)}")

    save_path = "data/raw/twitter_data.csv"
    combined.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved to {save_path}")

    return combined

# ----------------------------------------
# 6. Run
# ----------------------------------------
if __name__ == "__main__":
    collect_all_tweets()
