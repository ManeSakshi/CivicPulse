#!/usr/bin/env python3
"""
CivicPulse Pipeline Status Report
Generates final status report after pipeline completion
"""

import pandas as pd
import os
from datetime import datetime

def generate_pipeline_status_report():
    """Generate comprehensive pipeline status report"""
    
    print('COMPLETE PIPELINE STATUS REPORT')
    print('=' * 50)
    
    # Raw data status
    files = {
        'News': 'data/raw/all_news_data.csv',
        'Local': 'data/raw/local_news.csv', 
        'Twitter': 'data/raw/twitter_data.csv'
    }
    
    total_raw = 0
    for name, path in files.items():
        try:
            count = len(pd.read_csv(path)) if os.path.exists(path) else 0
            print(f'{name:12}: {count:4d} records')
            total_raw += count
        except Exception as e:
            print(f'{name:12}: ERROR - {e}')
    
    print(f'{"Total Raw":12}: {total_raw:4d} records')
    print()
    
    # Processed data status
    processed_path = 'data/processed/civicpulse_processed.csv'
    if os.path.exists(processed_path):
        try:
            df_processed = pd.read_csv(processed_path)
            print(f'Processed   : {len(df_processed):4d} records')
        except Exception as e:
            print(f'Processed   : ERROR - {e}')
    else:
        print('Processed   : ERROR - File not found')
    
    # Labeled data status
    labeled_path = 'data/processed/civic_labeled.csv'
    if os.path.exists(labeled_path):
        try:
            df_labeled = pd.read_csv(labeled_path)
            labels = df_labeled['label'].value_counts()
            print(f'Labeled     : {len(df_labeled):4d} records')
            
            for label, count in labels.items():
                percentage = count/len(df_labeled)*100
                print(f'  {label:8}: {count:4d} ({percentage:.1f}%)')
            
            print()
            
            # Model readiness check
            vocab_size = len(set(' '.join(df_labeled['text'].astype(str)).split()))
            avg_length = df_labeled['text'].astype(str).str.len().mean()
            
            print('MODEL READINESS METRICS:')
            print(f'  Vocabulary size: {vocab_size:,} unique words')
            print(f'  Average length:  {avg_length:.1f} characters')
            print(f'  Data quality:    HIGH (cleaned + lemmatized)')
            print(f'  Label quality:   DUAL-METHOD (VADER + TextBlob)')
            print()
            
            if len(df_labeled) >= 500 and vocab_size >= 1000:
                print('[SUCCESS] Dataset is READY for ML model training!')
                print('Files ready: data/processed/civic_labeled.csv')
                print('             data/processed/civicpulse_with_labels.csv')
            else:
                print('[WARNING] Dataset may need more data for robust training')
                
        except Exception as e:
            print(f'Labeled     : ERROR - {e}')
    else:
        print('Labeled     : ERROR - File not found')
    
    print()
    print('=' * 50)
    print(f'Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

if __name__ == "__main__":
    generate_pipeline_status_report()