import tweepy
import pandas as pd
import os
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Authenticate
client = tweepy.Client(bearer_token=BEARER_TOKEN) 


def fetch_tweets(query="Sangli OR Maharashtra OR civic OR traffic OR water OR garbage OR sanitation", max_results=50):
    """Fetch recent tweets (last 7 days)"""
    tweets_data = []

    try:
        response = client.search_recent_tweets(
            query=query,
            tweet_fields=["id", "text", "created_at", "lang"],
            max_results=max_results,
        )
    except Exception as e:
        print(f"❌ Error fetching tweets: {e}")
        return pd.DataFrame()

    if response.data:
        for tweet in response.data:
            tweets_data.append({
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "lang": tweet.lang
            })

    return pd.DataFrame(tweets_data)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)  # ensure folder exists
    df = fetch_tweets(max_results=30)
    if not df.empty:
        df.to_csv("data/raw/twitter_data.csv", index=False)
        print(f"✅ Tweets saved to data/raw/twitter_data.csv ({len(df)} rows)")
    else:
        print("⚠️ No tweets found for the given query.")
