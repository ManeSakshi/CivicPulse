# 📌 External Dataset Preprocessing Script
# Handles Kaggle datasets (Sentiment140, Tweets.csv) separately from civic data

import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

def clean_external_text(text):
    """Clean external dataset text (similar to civic data cleaning)"""
    if pd.isnull(text) or text == "" or str(text).lower() == "nan":
        return ""
    
    text = str(text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = text.lower()
    
    # Remove URLs, mentions, hashtags (common in tweets)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)  # Remove @mentions and #hashtags
    text = re.sub(r"rt\s+", "", text)  # Remove "RT" (retweet indicators)
    
    # Remove HTML tags and special patterns
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove very short words
    text = " ".join([word for word in text.split() if len(word) >= 2])
    
    return text

def process_sentiment140():
    """Process Sentiment140 dataset"""
    print("🔄 Processing Sentiment140 dataset...")
    
    file_path = "data/external/Sentiment140.csv"
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found. Please ensure the file exists.")
        return None
    
    # Sentiment140 format: target,ids,date,flag,user,text
    # target: 0 = negative, 2 = neutral, 4 = positive
    try:
        df = pd.read_csv(file_path, encoding='latin1', header=None, 
                        names=['target', 'ids', 'date', 'flag', 'user', 'text'])
        
        print(f"   📊 Loaded {len(df)} rows from Sentiment140")
        
        # Convert sentiment labels
        sentiment_map = {0: 'negative', 2: 'neutral', 4: 'positive'}
        df['sentiment'] = df['target'].map(sentiment_map)
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(clean_external_text)
        
        # Filter out empty text
        df = df[df['cleaned_text'].str.len() >= 10].reset_index(drop=True)
        
        # Keep only essential columns
        df_clean = df[['cleaned_text', 'sentiment', 'user', 'date']].copy()
        df_clean.columns = ['text', 'label', 'user', 'date']
        
        print(f"   ✅ Processed {len(df_clean)} valid sentiment records")
        return df_clean
        
    except Exception as e:
        print(f"❌ Error processing Sentiment140: {e}")
        return None

def process_tweets_csv():
    """Process additional Tweets.csv dataset"""
    print("🔄 Processing Tweets.csv dataset...")
    
    file_path = "data/external/Tweets.csv"
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found. Please ensure the file exists.")
        return None
    
    try:
        df = pd.read_csv(file_path)
        print(f"   📊 Loaded {len(df)} rows from Tweets.csv")
        print(f"   📋 Columns: {list(df.columns)}")
        
        # Adapt based on actual column structure - you may need to adjust this
        text_col = None
        sentiment_col = None
        
        # Common column names for text
        for col in ['text', 'tweet', 'content', 'message']:
            if col in df.columns:
                text_col = col
                break
        
        # Common column names for sentiment
        for col in ['sentiment', 'label', 'polarity', 'emotion', 'airline_sentiment']:
            if col in df.columns:
                sentiment_col = col
                break
        
        if text_col and sentiment_col:
            df['cleaned_text'] = df[text_col].apply(clean_external_text)
            df = df[df['cleaned_text'].str.len() >= 10].reset_index(drop=True)
            
            df_clean = df[['cleaned_text', sentiment_col]].copy()
            df_clean.columns = ['text', 'label']
            
            print(f"   ✅ Processed {len(df_clean)} valid records")
            return df_clean
        else:
            print(f"   ⚠️ Could not identify text/sentiment columns. Manual adjustment needed.")
            return None
            
    except Exception as e:
        print(f"❌ Error processing Tweets.csv: {e}")
        return None

def create_training_datasets():
    """Create separate training datasets"""
    print("🚀 Creating Training Datasets")
    print("=" * 50)
    
    # Process external datasets
    sentiment140_df = process_sentiment140()
    tweets_df = process_tweets_csv()
    
    # Combine external datasets if both exist
    external_data = []
    if sentiment140_df is not None:
        external_data.append(sentiment140_df)
    if tweets_df is not None:
        external_data.append(tweets_df)
    
    if external_data:
        combined_external = pd.concat(external_data, ignore_index=True)
        
        # Create train/test split for external data
        X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
            combined_external['text'], combined_external['label'], 
            test_size=0.2, random_state=42, stratify=combined_external['label']
        )
        
        # Save external datasets
        os.makedirs("data/processed/external", exist_ok=True)
        
        train_ext_df = pd.DataFrame({'text': X_train_ext, 'label': y_train_ext})
        test_ext_df = pd.DataFrame({'text': X_test_ext, 'label': y_test_ext})
        
        train_ext_df.to_csv("data/processed/external/train_external.csv", index=False)
        test_ext_df.to_csv("data/processed/external/test_external.csv", index=False)
        
        print(f"📊 External Dataset Summary:")
        print(f"   Total records: {len(combined_external)}")
        print(f"   Training set: {len(train_ext_df)}")
        print(f"   Test set: {len(test_ext_df)}")
        print(f"   Sentiment distribution: {combined_external['label'].value_counts().to_dict()}")
        
    # Load civic data (already processed)
    civic_df = pd.read_csv("data/processed/civicpulse_processed.csv")
    
    print(f"\n📊 Civic Dataset Summary:")
    print(f"   Total records: {len(civic_df)}")
    print(f"   Ready for unsupervised analysis or label generation")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Train BERT on external labeled data: data/processed/external/")
    print(f"   2. Apply trained model to civic data for sentiment prediction")
    print(f"   3. Or generate civic labels using VADER/TextBlob first")

if __name__ == "__main__":
    create_training_datasets()