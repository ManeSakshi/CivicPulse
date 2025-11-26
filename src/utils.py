# 📌 CivicPulse Utility Functions
# Shared utilities for text cleaning, data processing, and common operations

import pandas as pd
import re
import os
from datetime import datetime


def clean_text(text, tweet_mode=False):
    """
    Unified text cleaning function for both civic data and external datasets
    
    Args:
        text: Input text to clean
        tweet_mode: If True, applies tweet-specific cleaning (removes @mentions, #hashtags, RT)
    """
    if pd.isnull(text) or text == "" or str(text).lower() == "nan":
        return ""
    
    text = str(text)  # ensure string type
    
    # Handle encoding issues and special characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and email addresses
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    
    # Tweet-specific cleaning
    if tweet_mode:
        text = re.sub(r"@\w+|#\w+", "", text)  # Remove @mentions and #hashtags
        text = re.sub(r"rt\s+", "", text)  # Remove "RT" (retweet indicators)
    
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    
    # Remove special patterns common in news
    text = re.sub(r"\[\+\d+ chars\]", "", text)  # Remove "[+1234 chars]" patterns
    text = re.sub(r"\.\.\.", " ", text)  # Replace "..." with space
    
    # Keep only alphabetic characters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove very short words (less than 2 characters)
    text = " ".join([word for word in text.split() if len(word) >= 2])
    
    return text


def save_to_csv(dataframe, filepath, message=""):
    """
    Standardized CSV saving with proper encoding and messaging
    
    Args:
        dataframe: pandas DataFrame to save
        filepath: Path to save the CSV file
        message: Custom message to display (optional)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    dataframe.to_csv(filepath, index=False, encoding="utf-8-sig")
    
    if message:
        print(message)
    else:
        print(f"💾 Saved {len(dataframe)} records to {filepath}")


def setup_data_directory():
    """Create necessary data directories if they don't exist"""
    directories = ["data/raw", "data/processed", "data/processed/external"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_date_range(days_back=7):
    """
    Get formatted date range for API queries
    
    Args:
        days_back: Number of days to go back from today
        
    Returns:
        tuple: (from_date, to_date) in YYYY-MM-DD format
    """
    from datetime import timedelta
    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    return from_date, to_date


def remove_duplicates(dataframe, subset_cols=["title"]):
    """
    Remove duplicates from DataFrame based on specified columns
    
    Args:
        dataframe: pandas DataFrame
        subset_cols: List of columns to check for duplicates
        
    Returns:
        pandas DataFrame with duplicates removed
    """
    original_count = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset=subset_cols)
    removed_count = original_count - len(dataframe)
    
    if removed_count > 0:
        print(f"🧹 Removed {removed_count} duplicate records")
    
    return dataframe