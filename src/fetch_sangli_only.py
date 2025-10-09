#!/usr/bin/env python3
"""
CivicPulse SANGLI-ONLY News Fetcher
Collects civic data specifically about Sangli city only
Filters out Mumbai, Pune, Delhi and other cities
"""

import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from gnews import GNews
from utils import setup_data_directory, get_date_range
from incremental_news_utils import save_with_incremental_update, create_summary_report

# Load environment variables
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# Initialize APIs
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
gnews = GNews(language='en', country='IN', max_results=50)

# Sangli-specific search terms
SANGLI_KEYWORDS = [
    "Sangli",
    "Sangli city", 
    "Sangli municipal corporation",
    "Sangli civic",
    "Sangli traffic",
    "Sangli water supply",
    "Sangli roads",
    "Miraj Sangli",
    "Sangli district",
    "Sangli Maharashtra"
]

# Cities to exclude (filter out)
EXCLUDE_CITIES = [
    "Mumbai", "Pune", "Delhi", "Bangalore", "Chennai", "Hyderabad", 
    "Kolkata", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur",
    "Nagpur", "Thane", "Nashik", "Aurangabad", "Solapur", "Amravati",
    "Navi Mumbai", "Kalyan", "Vasai", "Virar"
]

def is_sangli_specific(title, description, content):
    """Check if article is specifically about Sangli"""
    text_to_check = f"{title} {description} {content}".lower()
    
    # Must contain Sangli-related keywords
    has_sangli = any(keyword.lower() in text_to_check for keyword in SANGLI_KEYWORDS)
    
    # Must NOT be primarily about other cities
    other_city_mentions = sum(1 for city in EXCLUDE_CITIES if city.lower() in text_to_check)
    
    # Sangli-specific if: has Sangli keywords AND not primarily about other cities
    return has_sangli and other_city_mentions <= 1

def fetch_sangli_newsapi(query, max_results=25):
    """Fetch Sangli-specific news from NewsAPI"""
    if not newsapi:
        return pd.DataFrame()
    
    print(f"📰 Fetching Sangli news: '{query}'")
    from_date, to_date = get_date_range(7)
    
    try:
        # Add Sangli to the query
        sangli_query = f"Sangli AND ({query})"
        
        articles = newsapi.get_everything(
            q=sangli_query,
            language="en", 
            from_param=from_date,
            to=to_date,
            sort_by="publishedAt",
            page_size=max_results
        )
        
        news_data = []
        for article in articles.get("articles", []):
            title = article.get("title", "")
            description = article.get("description", "")  
            content = article.get("content", "")
            
            # Filter: Only include Sangli-specific articles
            if is_sangli_specific(title, description, content):
                news_data.append({
                    "source": article["source"]["name"],
                    "author": article.get("author"),
                    "title": title,
                    "description": description,
                    "content": content,
                    "publishedAt": article.get("publishedAt"),
                    "url": article.get("url"),
                    "fetched_at": datetime.now().isoformat()
                })
        
        print(f"   [OK] Found {len(news_data)} Sangli-specific articles")
        return pd.DataFrame(news_data)
        
    except Exception as e:
        print(f"   [ERROR] NewsAPI error: {e}")
        return pd.DataFrame()

def fetch_sangli_gnews(query, max_results=30):
    """Fetch Sangli-specific news from GNews"""
    print(f"🌐 Fetching Sangli GNews: '{query}'")
    
    try:
        # Add Sangli to the query
        sangli_query = f"Sangli {query}"
        
        articles = gnews.get_news(sangli_query)
        
        news_data = []
        for article in articles[:max_results]:
            title = article.get("title", "")
            description = article.get("description", "")
            
            # Filter: Only include Sangli-specific articles  
            if is_sangli_specific(title, description, ""):
                news_data.append({
                    "source": "GNews",
                    "author": None,
                    "title": title,
                    "description": description,
                    "content": None,
                    "publishedAt": article.get("published date"),
                    "url": article.get("url"),
                    "fetched_at": datetime.now().isoformat()
                })
        
        print(f"   [OK] Found {len(news_data)} Sangli-specific articles")
        return pd.DataFrame(news_data)
        
    except Exception as e:
        print(f"   [ERROR] GNews error: {e}")
        return pd.DataFrame()

def collect_sangli_only_news():
    """Collect ONLY Sangli-specific civic news"""
    print("🏛️ COLLECTING SANGLI-ONLY CIVIC NEWS")
    print("🎯 Filtering out Mumbai, Pune, Delhi and other cities")
    print()
    
    setup_data_directory()
    
    # Sangli-specific civic queries
    civic_queries = [
        "traffic management", 
        "water supply",
        "road repair",
        "municipal corporation", 
        "civic issues",
        "garbage collection",
        "street lights",
        "drainage",
        "public transport",
        "development project"
    ]
    
    all_data = []
    total_collected = 0
    
    for query in civic_queries:
        print(f"🔍 Searching for: Sangli + {query}")
        
        # Collect from NewsAPI  
        df_newsapi = fetch_sangli_newsapi(query, max_results=15)
        if not df_newsapi.empty:
            all_data.append(df_newsapi)
            total_collected += len(df_newsapi)
        
        # Collect from GNews
        df_gnews = fetch_sangli_gnews(query, max_results=20)  
        if not df_gnews.empty:
            all_data.append(df_gnews)
            total_collected += len(df_gnews)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on title
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['title'], keep='first')
        final_count = len(combined_df)
        duplicates_removed = initial_count - final_count
        
        # Save Sangli-only news
        output_file = "data/raw/sangli_only_news.csv"
        combined_df.to_csv(output_file, index=False)
        
        print()
        print("=" * 50)
        print("🎯 SANGLI-ONLY NEWS COLLECTION COMPLETE")
        print(f"📊 Total articles collected: {initial_count}")
        print(f"🔄 Duplicates removed: {duplicates_removed}")  
        print(f"💾 Final unique articles: {final_count}")
        print(f"📁 Saved to: {output_file}")
        print("=" * 50)
        
        return combined_df
    else:
        print("❌ No Sangli-specific articles found")
        return pd.DataFrame()

if __name__ == "__main__":
    collect_sangli_only_news()