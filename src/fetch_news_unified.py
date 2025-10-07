# 📌 Enhanced CivicPulse News Fetcher with Smart Incremental Updates
# Prevents duplicates across sessions and tracks new articles only

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
gnews = GNews(language='en', country='IN', max_results=100)


def fetch_newsapi_data(query, max_results=50):
    """Fetch news from NewsAPI"""
    if not newsapi:
        print("[WARN] NewsAPI key not found - skipping NewsAPI")
        return pd.DataFrame()
    
    print(f"📰 Fetching from NewsAPI: '{query[:50]}...'")
    from_date, to_date = get_date_range(7)

    try:
        articles = newsapi.get_everything(
            q=query,
            language="en",
            from_param=from_date,
            to=to_date,
            sort_by="publishedAt",  # Sort by date to get newest first
            page_size=max_results
        )
        
        news_data = []
        for article in articles.get("articles", []):
            news_data.append({
                "source": article["source"]["name"],
                "author": article.get("author"),
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("content"),
                "publishedAt": article.get("publishedAt"),
                "url": article.get("url"),
                "provider": "NewsAPI",
                "query_type": "general",
                "fetched_at": datetime.now().isoformat()
            })
        
        print(f"   [OK] Fetched {len(news_data)} articles")
        return pd.DataFrame(news_data)
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return pd.DataFrame()


def fetch_gnews_data(query, query_type="general", max_results=80):
    """Fetch news from GNews"""
    print(f"🌐 Fetching from GNews: '{query[:50]}...' ({query_type})")
    
    try:
        gnews.max_results = max_results
        articles = gnews.get_news(query)
        
        news_data = []
        for article in articles:
            news_data.append({
                "source": article.get("publisher", {}).get("title", "Unknown"),
                "author": None,
                "title": article.get("title"),
                "description": article.get("description"),
                "content": article.get("description"),
                "publishedAt": article.get("published date"),
                "url": article.get("url"),
                "provider": "GNews",
                "query_type": query_type,
                "fetched_at": datetime.now().isoformat()
            })
        
        print(f"   [OK] Fetched {len(news_data)} articles")
        return pd.DataFrame(news_data)
        
    except Exception as e:
        print(f"   [ERROR] Error: {e}")
        return pd.DataFrame()


def fetch_sangli_local_news():
    """Fetch Sangli-specific local news with multilingual support"""
    print("🏛️ Fetching Sangli-specific local news...")
    
    queries = [
        {
            "query": "Sangli civic OR traffic OR water OR garbage OR sanitation OR municipality",
            "lang": "en",
            "type": "sangli_civic"
        },
        {
            "query": "सांगली नगरपालिका OR पाणी OR रस्ता OR कचरा OR वाहतूक OR शहर",
            "lang": "mr", 
            "type": "sangli_marathi"
        },
        {
            "query": "Sangli Miraj Kupwad OR Sangli Maharashtra OR Sangli district",
            "lang": "en",
            "type": "sangli_general"
        }
    ]
    
    all_articles = []
    
    for query_info in queries:
        gnews_local = GNews(
            language=query_info["lang"], 
            country='IN', 
            period='14d',
            max_results=60
        )
        
        try:
            results = gnews_local.get_news(query_info["query"])
            query_count = 0
            
            for item in results:
                title = item.get("title", "")
                desc = item.get("description", "")
                
                # Filter strictly for Sangli mentions
                combined_text = (title + " " + desc).lower()
                if "sangli" in combined_text:
                    all_articles.append({
                        "source": item.get("publisher", {}).get("title", "Unknown"),
                        "author": None,
                        "title": title,
                        "description": desc,
                        "content": desc,
                        "publishedAt": item.get("published date"),
                        "url": item.get("url"),
                        "provider": "GNews_Local",
                        "query_type": query_info["type"],
                        "language": query_info["lang"],
                        "fetched_at": datetime.now().isoformat()
                    })
                    query_count += 1
            
            print(f"   📍 {query_info['type']}: {query_count} articles")
            
        except Exception as e:
            print(f"   [ERROR] Error fetching {query_info['type']}: {e}")
    
    return pd.DataFrame(all_articles)


def fetch_general_civic_news():
    """Fetch general civic/governance news"""
    print("🏛️ Fetching general civic news...")
    
    general_queries = [
        "civic OR municipal OR governance OR public services",
        "traffic management OR waste management OR water supply", 
        "Maharashtra civic OR urban planning OR smart city"
    ]
    
    all_data = []
    
    for query in general_queries:
        # Fetch from both NewsAPI and GNews
        if newsapi:
            df_newsapi = fetch_newsapi_data(query, max_results=25)
            if not df_newsapi.empty:
                all_data.append(df_newsapi)
        
        df_gnews = fetch_gnews_data(query, query_type="civic_general", max_results=40)
        if not df_gnews.empty:
            all_data.append(df_gnews)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def collect_all_news(include_local=True, include_general=True):
    """
    Main function to collect news data with smart incremental updates
    Enhanced version that prevents duplicates across sessions
    
    Args:
        include_local: Whether to fetch Sangli-specific local news
        include_general: Whether to fetch general civic news
    """
    print("📰 Starting SMART incremental news collection...")
    print("🔍 Will only save NEW articles, skipping duplicates from previous runs")
    setup_data_directory()
    
    file_updates = {}
    all_collected_data = []  # To combine all news for master file
    
    # Collect Sangli-specific local news
    if include_local:
        print("\n" + "="*50)
        df_local = fetch_sangli_local_news()
        if not df_local.empty:
            # Save to dedicated local news file
            merged_df, new_count = save_with_incremental_update(
                df_local, 
                "data/raw/local_news.csv", 
                dedup_cols=['title', 'url'],
            )
            file_updates['local_news.csv'] = (len(merged_df), new_count)
            # Add to master collection
            all_collected_data.append(df_local)
        else:
            print("[WARN] No local news articles fetched")
            file_updates['local_news.csv'] = (0, 0)
    
    # Collect general civic news  
    if include_general:
        print("\n" + "="*50)
        df_general = fetch_general_civic_news()
        if not df_general.empty:
            # Add to master collection
            all_collected_data.append(df_general)
            
            # Save NewsAPI-only data for compatibility
            newsapi_data = df_general[df_general['provider'] == 'NewsAPI']
            if not newsapi_data.empty:
                merged_api, new_api = save_with_incremental_update(
                    newsapi_data,
                    "data/raw/news_data.csv",
                    dedup_cols=['title', 'url']
                )
                file_updates['news_data.csv'] = (len(merged_api), new_api)
            
            # Save GNews-only data
            gnews_data = df_general[df_general['provider'].str.contains('GNews')]
            if not gnews_data.empty:
                merged_gnews, new_gnews = save_with_incremental_update(
                    gnews_data,
                    "data/raw/gnews_data.csv", 
                    dedup_cols=['title', 'url']
                )
                file_updates['gnews_data.csv'] = (len(merged_gnews), new_gnews)
        else:
            print("[WARN] No general news articles fetched")
    
    # Save master file with ALL collected news (local + general)
    if all_collected_data:
        print("\n" + "="*50)
        print("🔄 Combining ALL news for master file...")
        df_all_news = pd.concat(all_collected_data, ignore_index=True)
        merged_master, new_master = save_with_incremental_update(
            df_all_news,
            "data/raw/all_news_data.csv",
            dedup_cols=['title', 'url']
        )
        file_updates['all_news_data.csv'] = (len(merged_master), new_master)
    else:
        print("[WARN] No news data collected from any source")
        file_updates['all_news_data.csv'] = (0, 0)
    
    # Create summary report
    total_new = create_summary_report(file_updates)
    
    return total_new


if __name__ == "__main__":
    # Run with smart incremental updates
    total_new = collect_all_news(include_local=True, include_general=True)
    
    if total_new > 0:
        print(f"\n[SUCCESS] Successfully added {total_new} new articles!")
    else:
        print(f"\n[INFO] All existing articles are up-to-date!")
        print("💡 Run again later to fetch newer articles.")