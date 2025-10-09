#!/usr/bin/env python3
"""
Sangli-only sentiment labeling
Generates sentiment labels for Sangli-specific civic data
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import os

def label_sangli_sentiment():
    """Generate sentiment labels for Sangli civic data"""
    print('[INFO] Loading processed Sangli data...')
    
    try:
        df = pd.read_csv('data/processed/sangli_processed.csv')
        print(f'[OK] Loaded {len(df)} Sangli records for labeling')
        
        # Initialize sentiment analyzers
        vader = SentimentIntensityAnalyzer()
        
        print('[INFO] Generating VADER sentiment labels...')
        vader_labels = []
        vader_scores = []
        for i, text in enumerate(df['text']):
            if i % 50 == 0:
                print(f'[PROGRESS] VADER labeling {i+1}/{len(df)}...')
            
            scores = vader.polarity_scores(str(text))
            vader_scores.append(scores['compound'])
            
            if scores['compound'] >= 0.05:
                vader_labels.append('positive')
            elif scores['compound'] <= -0.05:
                vader_labels.append('negative')
            else:
                vader_labels.append('neutral')
        
        print('[INFO] Generating TextBlob sentiment labels...')
        textblob_labels = []
        textblob_scores = []
        for i, text in enumerate(df['text']):
            if i % 50 == 0:
                print(f'[PROGRESS] TextBlob labeling {i+1}/{len(df)}...')
            
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                textblob_scores.append(polarity)
                
                if polarity > 0.1:
                    textblob_labels.append('positive')
                elif polarity < -0.1:
                    textblob_labels.append('negative')
                else:
                    textblob_labels.append('neutral')
            except:
                textblob_labels.append('neutral')
                textblob_scores.append(0.0)
        
        # Add labels to dataframe
        df['vader_label'] = vader_labels
        df['vader_score'] = vader_scores
        df['textblob_label'] = textblob_labels
        df['textblob_score'] = textblob_scores
        df['label'] = vader_labels  # Use VADER as primary label
        
        # Create timestamp column for dashboard
        df['timestamp'] = pd.Timestamp.now()
        
        # Save labeled Sangli data
        os.makedirs('data/processed', exist_ok=True)
        output_file = 'data/processed/sangli_labeled.csv'
        df.to_csv(output_file, index=False)
        
        # Print distribution
        label_dist = df['label'].value_counts()
        print(f'[SUCCESS] Labeled {len(df)} Sangli records')
        print('Sentiment distribution:')
        for label, count in label_dist.items():
            print(f'  {label}: {count} ({count/len(df)*100:.1f}%)')
        
        # Agreement analysis
        agreement = (df['vader_label'] == df['textblob_label']).sum()
        print(f'VADER-TextBlob agreement: {agreement}/{len(df)} ({agreement/len(df)*100:.1f}%)')
        
        print(f'[SAVED] {output_file}')
        
        # Save summary for dashboard
        summary = {
            'total_records': len(df),
            'sentiment_distribution': label_dist.to_dict(),
            'last_updated': pd.Timestamp.now().isoformat(),
            'data_sources': df['source'].value_counts().to_dict()
        }
        
        summary_file = 'data/processed/sangli_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f'[SAVED] Summary: {summary_file}')
        
        return True
        
    except Exception as e:
        print(f'[ERROR] Labeling failed: {e}')
        return False

if __name__ == "__main__":
    success = label_sangli_sentiment()
    exit(0 if success else 1)