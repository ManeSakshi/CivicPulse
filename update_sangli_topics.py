#!/usr/bin/env python3
"""
Add topic categories to existing Sangli labeled data
"""

import pandas as pd
import pickle
import os

def update_sangli_data_with_topics():
    """Add topic categories to Sangli labeled data"""
    print("🔄 Adding topic categories to Sangli labeled data...")
    
    # Load Sangli labeled data
    if not os.path.exists('data/processed/sangli_labeled.csv'):
        print("❌ Sangli labeled data not found!")
        return False
    
    df = pd.read_csv('data/processed/sangli_labeled.csv')
    print(f"📊 Loaded {len(df)} Sangli records")
    
    # Load topic results
    if os.path.exists('models/topics/sangli_topic_results.pkl'):
        with open('models/topics/sangli_topic_results.pkl', 'rb') as f:
            topic_data = pickle.load(f)
        
        # Get the dataframe with topic categories from topic analysis
        topic_df = topic_data['dataframe']
        
        if len(topic_df) == len(df):
            # Add topic categories to original data
            df['topic_category'] = topic_df['topic_category']
            df['topic_score'] = topic_df['topic_score']
            
            # Save updated data
            df.to_csv('data/processed/sangli_labeled.csv', index=False)
            
            print("✅ Successfully added topic categories!")
            
            # Show distribution
            category_dist = df['topic_category'].value_counts()
            print("\n📊 Updated Topic Distribution:")
            for category, count in category_dist.items():
                print(f"   📌 {category.replace('_', ' ').title()}: {count}")
            
            return True
        else:
            print(f"❌ Size mismatch: {len(df)} vs {len(topic_df)}")
            return False
    else:
        print("❌ Topic results not found. Run sangli_topic_model.py first")
        return False

if __name__ == "__main__":
    success = update_sangli_data_with_topics()
    if success:
        print("\n🚀 Dashboard ready with topic categories!")
        print("   Launch: python -m streamlit run src/dashboard_simple.py")
    exit(0 if success else 1)