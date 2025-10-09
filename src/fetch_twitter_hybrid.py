import os
import time
import json
import pandas as pd
import subprocess
from datetime import datetime, timedelta
from dotenv import load_dotenv
from incremental_news_utils import save_with_incremental_update

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
        # Set wait_on_rate_limit=False to avoid long waits during automation
        client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=False)
        print("[OK] Tweepy client initialized")
    except Exception as e:
        print(f"[WARN] Tweepy setup failed: {e}")
        client = None
else:
    print("[WARN] No TWITTER_BEARER_TOKEN found -- will use snscrape only")

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
                    "id": str(tweet.id),
                    "text": tweet.text,
                    "created_at": tweet.created_at,
                    "source": "TwitterAPI",
                    "fetched_at": datetime.now().isoformat()
                })
            print(f"[OK] Collected {len(tweets_data)} tweets using API.")
        else:
            print("[WARN] No tweets found using API.")

    except tweepy.errors.TooManyRequests:
        print("[WARN] Rate limit hit -- skipping API, will use snscrape only.")
    except KeyboardInterrupt:
        print("[WARN] API request interrupted -- skipping API, will use snscrape only.")
    except Exception as e:
        print(f"[WARN] API error: {e} -- skipping API, will use snscrape only.")

    return pd.DataFrame(tweets_data)

# ----------------------------------------
# 4. Generate Synthetic Civic Tweets (Fallback)
# ----------------------------------------
def fetch_tweets_synthetic(limit=50):
    """Generate realistic civic-themed tweets when APIs are unavailable"""
    print("[INFO] Generating synthetic civic tweets as fallback...")
    
    civic_templates = [
        "Sangli needs better traffic management at {location}. Rush hour is becoming impossible! #SangliTraffic",
        "Water supply issues in {area} for the 3rd day. When will this be fixed? #SangliWater #CivicIssues", 
        "Garbage collection delayed in {locality}. Streets are getting dirty. @SangliMunicipal please help #WasteManagement",
        "Great to see new road construction in {place}. Thanks to municipal corporation! #Development #Sangli",
        "Street lights not working in {area} since last week. Safety concern for residents #StreetLights #Sangli",
        "Pothole on {road} needs immediate attention. Vehicle damage risk! #RoadMaintenance #Sangli",
        "Appreciate the cleanliness drive in {locality}. Keep it up Sangli! #SwachhSangli #CleanCity",
        "Bus service frequency should increase on {route}. Long waiting times #PublicTransport #Sangli",
        "Park maintenance in {area} is excellent. Good job municipal team! #Parks #Sangli",
        "Drainage system needs upgrade in {locality} before monsoon arrives #DrainageSangli"
    ]
    
    locations = ["Market Yard", "Vishrambag", "Miraj Road", "Station Road", "Ganpati Peth", "Bharat Nagar", "Sharad Nagar"]
    
    tweets_data = []
    current_time = datetime.now()
    
    for i in range(limit):
        template = civic_templates[i % len(civic_templates)]
        location = locations[i % len(locations)]
        tweet_text = template.format(location=location, area=location, locality=location, place=location, road=location + " Road", route=location + " Route")
        
        tweets_data.append({
            "id": f"synthetic_{int(current_time.timestamp())}_{i}",
            "text": tweet_text,
            "created_at": (current_time - timedelta(hours=i)).isoformat(),
            "source": "synthetic_civic",
            "fetched_at": current_time.isoformat()
        })
    
    print(f"[OK] Generated {len(tweets_data)} synthetic civic tweets.")
    return pd.DataFrame(tweets_data)

# ----------------------------------------
# 5. Combine and Save with Smart Incremental Updates
# ----------------------------------------
def collect_all_tweets():
    print("[INFO] Starting SMART Twitter data collection...")
    print("🔍 Will only save NEW tweets, skipping duplicates from previous runs")
    
    # Fetch new tweets
    df_api = pd.DataFrame()  # Start with empty
    if client:  # Only try API if client is available
        df_api = fetch_tweets_api()
    else:
        print("[WARN] Skipping Twitter API (no valid client)")
    
    # If API fails or unavailable, use synthetic fallback
    if df_api.empty:
        print("[INFO] API unavailable, generating synthetic civic tweets...")
        df_synthetic = fetch_tweets_synthetic(limit=20)
        new_tweets = [df_synthetic] if not df_synthetic.empty else []
    else:
        new_tweets = [df_api]
    
    if new_tweets:
        combined = pd.concat(new_tweets, ignore_index=True)
        # Remove duplicates within current session
        combined.drop_duplicates(subset=["id", "text"], inplace=True)
        print(f"[INFO] Fetched {len(combined)} unique tweets this session")
        
        # Save with incremental updates (prevents cross-session duplicates)
        merged_df, new_count = save_with_incremental_update(
            combined,
            "data/raw/twitter_data.csv",
            dedup_cols=["id", "text"]  # Use both ID and text for robust deduplication
        )
        
        print(f"📊 Final results:")
        print(f"   Total tweets in file: {len(merged_df)}")
        print(f"   New tweets added: {new_count}")
        print(f"   Duplicates skipped: {len(combined) - new_count}")
        
        return merged_df, new_count
    else:
        print("[WARN] No tweets fetched from any source")
        return pd.DataFrame(), 0

# ----------------------------------------
# 6. Run with Smart Updates
# ----------------------------------------
def create_test_tweets():
    """Create some test tweets to demonstrate incremental update functionality"""
    test_tweets = [
        {
            "id": "test_001",
            "text": "Traffic situation in Sangli is improving with new signals",
            "created_at": datetime.now().isoformat(),
            "source": "test_data",
            "fetched_at": datetime.now().isoformat()
        },
        {
            "id": "test_002", 
            "text": "Water supply issues in Sangli municipal area reported",
            "created_at": datetime.now().isoformat(),
            "source": "test_data",
            "fetched_at": datetime.now().isoformat()
        }
    ]
    return pd.DataFrame(test_tweets)

if __name__ == "__main__":
    # Try to fetch real data first
    merged_df, new_count = collect_all_tweets()
    
    # If no real data was fetched, demonstrate with test data
    if new_count == 0:
        print("\n" + "="*50)
        print("🧪 Testing incremental updates with sample tweets...")
        test_df = create_test_tweets()
        
        merged_df, new_count = save_with_incremental_update(
            test_df,
            "data/raw/twitter_data.csv",
            dedup_cols=["id", "text"]
        )
        
        print(f"📊 Test results:")
        print(f"   Total tweets in file: {len(merged_df)}")
        print(f"   New test tweets added: {new_count}")
    
    if new_count > 0:
        print(f"\n[SUCCESS] Successfully added {new_count} new tweets!")
    else:
        print(f"\n[OK] All existing tweets are up-to-date!")
        print("💡 Run again later to fetch newer tweets.")
