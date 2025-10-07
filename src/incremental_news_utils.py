# 📌 Enhanced CivicPulse News Fetcher with Incremental Updates
# Handles duplicates across sessions and tracks new articles

import pandas as pd
import os
from datetime import datetime, timedelta
from utils import save_to_csv, setup_data_directory, get_date_range, remove_duplicates


def load_existing_data(filepath):
    """Load existing data to check for duplicates"""
    if os.path.exists(filepath):
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"[WARN] Error loading {filepath}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def get_last_fetch_time(filepath):
    """Get the timestamp of the last fetch from existing data"""
    existing_df = load_existing_data(filepath)
    if not existing_df.empty and 'fetched_at' in existing_df.columns:
        try:
            # Get the most recent fetch time
            last_time = pd.to_datetime(existing_df['fetched_at']).max()
            return last_time
        except:
            pass
    # Default to 7 days ago if no previous data
    return datetime.now() - timedelta(days=7)


def merge_with_existing(new_df, filepath, dedup_cols=['title', 'url']):
    """
    Merge new data with existing data, removing duplicates
    
    Args:
        new_df: New DataFrame to merge
        filepath: Path to existing CSV file
        dedup_cols: Columns to use for duplicate detection
        
    Returns:
        merged_df: Combined DataFrame with duplicates removed
        new_count: Number of truly new articles
    """
    existing_df = load_existing_data(filepath)
    
    if existing_df.empty:
        print(f"📝 No existing data - all {len(new_df)} articles are new")
        return new_df, len(new_df)
    
    # Combine existing and new data
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Remove duplicates (keeping first occurrence = existing data wins)
    original_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=dedup_cols, keep='first')
    
    # Calculate how many were truly new
    new_count = len(combined_df) - len(existing_df)
    duplicate_count = original_count - len(combined_df)
    
    if new_count > 0:
        print(f"📈 Added {new_count} new articles, skipped {duplicate_count} duplicates")
    else:
        print(f"🔄 No new articles found ({duplicate_count} duplicates skipped)")
    
    return combined_df, new_count


def save_with_incremental_update(new_df, filepath, dedup_cols=['title', 'url'], message=""):
    """
    Save DataFrame with incremental update logic
    
    Args:
        new_df: New DataFrame to save
        filepath: Target file path
        dedup_cols: Columns for duplicate detection
        message: Optional custom message
    """
    merged_df, new_count = merge_with_existing(new_df, filepath, dedup_cols)
    
    # Add timestamp for all records if not present
    if 'fetched_at' not in merged_df.columns:
        merged_df['fetched_at'] = datetime.now().isoformat()
    
    # Update fetched_at for new records only
    if new_count > 0 and len(merged_df) > 0:
        # Mark new records with current timestamp
        existing_count = len(merged_df) - new_count
        merged_df.iloc[existing_count:, merged_df.columns.get_loc('fetched_at')] = datetime.now().isoformat()
    
    # Save the merged data
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    merged_df.to_csv(filepath, index=False, encoding="utf-8-sig")
    
    if message:
        print(message)
    else:
        if new_count > 0:
            print(f"💾 Saved {len(merged_df)} total articles ({new_count} new) to {os.path.basename(filepath)}")
        else:
            print(f"💾 No updates needed for {os.path.basename(filepath)} ({len(merged_df)} existing articles)")
    
    return merged_df, new_count


def create_summary_report(file_updates):
    """Create a summary report of the update session"""
    print("\n" + "="*60)
    print("📋 SESSION SUMMARY REPORT")
    print("="*60)
    
    total_new = 0
    for filename, (total_articles, new_articles) in file_updates.items():
        print(f"{filename:20} | {total_articles:3} total | {new_articles:3} new")
        total_new += new_articles
    
    print(f"\n🎯 TOTAL NEW ARTICLES THIS SESSION: {total_new}")
    
    if total_new == 0:
        print("💡 TIP: Articles are fetched from last 7 days. Try running later for new content.")
    
    return total_new