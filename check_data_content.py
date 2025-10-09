import pandas as pd

# Check current data content
print("=== CURRENT DATA ANALYSIS ===\n")

# Check raw news data
print("1. RAW NEWS DATA SAMPLE:")
df_news = pd.read_csv('data/raw/all_news_data.csv')
print(f"Total news articles: {len(df_news)}")
print("\nSample news titles:")
for i, title in enumerate(df_news['title'].head(10)):
    print(f"  {i+1}. {title}")

# Check local news data  
print("\n2. LOCAL NEWS DATA SAMPLE:")
df_local = pd.read_csv('data/raw/local_news.csv')
print(f"Total local articles: {len(df_local)}")
print("\nSample local titles:")
for i, title in enumerate(df_local['title'].head(5)):
    print(f"  {i+1}. {title}")

# Check Twitter data
print("\n3. TWITTER DATA SAMPLE:")
df_twitter = pd.read_csv('data/raw/twitter_data.csv')
print(f"Total tweets: {len(df_twitter)}")
print("\nSample tweets:")
for i, text in enumerate(df_twitter['text'].head(5)):
    print(f"  {i+1}. {text[:100]}...")

# Check processed/labeled data
print("\n4. DASHBOARD DATA CONTENT:")
df_labeled = pd.read_csv('data/processed/civic_labeled.csv')
print(f"Total records in dashboard: {len(df_labeled)}")

# Count Sangli-specific vs general content
sangli_count = df_labeled['text'].str.contains('sangli|Sangli', case=False, na=False).sum()
print(f"Sangli-specific records: {sangli_count}")
print(f"General/Other records: {len(df_labeled) - sangli_count}")
print(f"Sangli percentage: {sangli_count/len(df_labeled)*100:.1f}%")

print("\n=== ISSUE IDENTIFIED ===")
if sangli_count < len(df_labeled) * 0.8:
    print("❌ PROBLEM: Too much general data, not enough Sangli-specific content!")
    print("🔧 SOLUTION NEEDED: Filter data collection to focus on Sangli only")
else:
    print("✅ OK: Most data is Sangli-specific")