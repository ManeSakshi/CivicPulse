#!/usr/bin/env python3
"""
Sangli-only data preprocessing
Processes only Sangli-specific civic data
"""

import pandas as pd
import spacy
import sys
import os

# Add src directory to path
sys.path.append('src')
from preprocess import preprocess_text

def process_sangli_data():
    """Process ONLY Sangli-specific civic data"""
    print('[INFO] Loading Sangli-only datasets...')
    
    # Load Sangli-only data files
    datasets = []
    file_paths = [
        ('data/raw/sangli_only_news.csv', 'sangli_news'),
        ('data/raw/sangli_only_twitter.csv', 'sangli_twitter'),
        ('data/raw/local_news.csv', 'local_news')  # This is already Sangli-focused
    ]
    
    total_records = 0
    for file_path, source in file_paths:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                # Combine text columns
                df['combined_text'] = df.apply(lambda row: ' '.join([
                    str(row.get('title', '')),
                    str(row.get('description', '')), 
                    str(row.get('content', '')),
                    str(row.get('text', ''))
                ]).strip(), axis=1)
                
                df['source'] = source
                datasets.append(df[['combined_text', 'source']])
                total_records += len(df)
                print(f'[OK] Loaded {len(df)} records from {source}')
        except Exception as e:
            print(f'[WARN] Could not load {file_path}: {e}')
    
    if datasets:
        # Combine all Sangli data
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Filter out empty/short texts
        initial_count = len(combined_df)
        combined_df = combined_df[combined_df['combined_text'].str.len() > 20]
        filtered_count = len(combined_df)
        
        print(f'[INFO] Processing {filtered_count} Sangli civic records (filtered {initial_count - filtered_count} short texts)...')
        
        # Load SpaCy model
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('[ERROR] SpaCy English model not found. Please install with: python -m spacy download en_core_web_sm')
            return False
        
        # Preprocess text
        processed_texts = []
        for i, text in enumerate(combined_df['combined_text']):
            if i % 50 == 0:
                print(f'[PROGRESS] Processing record {i+1}/{len(combined_df)}...')
            processed = preprocess_text(text)
            processed_texts.append(processed)
        
        combined_df['text'] = combined_df['combined_text']  # Original
        combined_df['processed_text'] = processed_texts     # Cleaned
        
        # Save processed Sangli data
        os.makedirs('data/processed', exist_ok=True)
        output_file = 'data/processed/sangli_processed.csv'
        combined_df[['text', 'processed_text', 'source']].to_csv(output_file, index=False)
        
        print(f'[SUCCESS] Processed {len(combined_df)} Sangli records')
        print(f'[SAVED] {output_file}')
        
        # Quality check
        avg_length = combined_df['processed_text'].str.len().mean()
        vocab_size = len(set(' '.join(combined_df['processed_text']).split()))
        
        print('Quality metrics:')
        print(f'  Average length: {avg_length:.1f} characters')
        print(f'  Vocabulary: {vocab_size} unique words')
        print(f'  Sources: {combined_df["source"].value_counts().to_dict()}')
        
        return True
    else:
        print('[ERROR] No Sangli data found!')
        return False

if __name__ == "__main__":
    success = process_sangli_data()
    exit(0 if success else 1)