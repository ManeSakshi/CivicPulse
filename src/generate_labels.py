# 📌 Civic Data Label Generation Script
# Generate sentiment labels for civic data using VADER and TextBlob

import pandas as pd
import os
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def install_required_packages():
    """Install required packages if not available"""
    try:
        import vaderSentiment
        import textblob
    except ImportError:
        print("📥 Installing required packages...")
        import subprocess
        subprocess.run(["pip", "install", "vaderSentiment", "textblob"], check=True)
        print("✅ Packages installed")

def generate_labels_civic_data():
    """Generate sentiment labels for civic data"""
    install_required_packages()
    
    # Load processed civic data
    df = pd.read_csv("data/processed/civicpulse_processed.csv")
    print(f"📊 Loaded {len(df)} civic records")
    
    # Initialize sentiment analyzers
    analyzer = SentimentIntensityAnalyzer()
    
    # Generate VADER labels
    print("🔍 Generating VADER sentiment labels...")
    vader_scores = []
    vader_labels = []
    
    for text in df['processed_text']:
        if pd.notna(text) and len(str(text)) > 5:
            scores = analyzer.polarity_scores(str(text))
            vader_scores.append(scores)
            
            # Convert compound score to label
            compound = scores['compound']
            if compound >= 0.05:
                label = 'positive'
            elif compound <= -0.05:
                label = 'negative'
            else:
                label = 'neutral'
            vader_labels.append(label)
        else:
            vader_scores.append({'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0})
            vader_labels.append('neutral')
    
    # Generate TextBlob labels
    print("🔍 Generating TextBlob sentiment labels...")
    textblob_labels = []
    textblob_scores = []
    
    for text in df['processed_text']:
        if pd.notna(text) and len(str(text)) > 5:
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            textblob_scores.append(polarity)
            
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            textblob_labels.append(label)
        else:
            textblob_scores.append(0.0)
            textblob_labels.append('neutral')
    
    # Add labels to dataframe
    df['vader_label'] = vader_labels
    df['vader_compound'] = [score['compound'] for score in vader_scores]
    df['textblob_label'] = textblob_labels
    df['textblob_polarity'] = textblob_scores
    
    # Create consensus label (where both agree)
    consensus_labels = []
    for i in range(len(df)):
        if df.iloc[i]['vader_label'] == df.iloc[i]['textblob_label']:
            consensus_labels.append(df.iloc[i]['vader_label'])
        else:
            # If they disagree, use VADER (generally better for social media text)
            consensus_labels.append(df.iloc[i]['vader_label'])
    
    df['consensus_label'] = consensus_labels
    
    # Create clean dataset for training
    civic_labeled = df[['processed_text', 'consensus_label', 'vader_label', 'textblob_label', 
                      'vader_compound', 'textblob_polarity', 'source', 'title']].copy()
    civic_labeled.columns = ['text', 'label', 'vader_label', 'textblob_label', 
                           'vader_score', 'textblob_score', 'source', 'title']
    
    # Save labeled civic data
    os.makedirs("data/processed", exist_ok=True)
    civic_labeled.to_csv("data/processed/civic_labeled.csv", index=False)
    df.to_csv("data/processed/civicpulse_with_labels.csv", index=False)
    
    # Print statistics
    print(f"\n📈 Label Generation Results:")
    print(f"   Total records: {len(df)}")
    print(f"   VADER distribution: {pd.Series(vader_labels).value_counts().to_dict()}")
    print(f"   TextBlob distribution: {pd.Series(textblob_labels).value_counts().to_dict()}")
    print(f"   Consensus distribution: {pd.Series(consensus_labels).value_counts().to_dict()}")
    
    # Agreement statistics
    agreement = sum(1 for i in range(len(df)) if df.iloc[i]['vader_label'] == df.iloc[i]['textblob_label'])
    print(f"   Agreement rate: {agreement/len(df)*100:.1f}%")
    
    print(f"\n💾 Files saved:")
    print(f"   data/processed/civic_labeled.csv - Clean training format")
    print(f"   data/processed/civicpulse_with_labels.csv - Full data with all labels")
    
    return civic_labeled

if __name__ == "__main__":
    generate_labels_civic_data()