# 📌 CivicPulse Data Preprocessing Script
# Combines News + Twitter CSVs, cleans text, removes stopwords, lemmatizes

import pandas as pd
import spacy
import glob
import re
import os
from utils import clean_text

# ------------------------------
# 1️⃣ Load SpaCy model for English
# ------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
    print("[OK] SpaCy English model loaded successfully")
except OSError:
    print("[ERROR] SpaCy model 'en_core_web_sm' not found!")
    print("📥 Installing it now...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("[OK] SpaCy model installed and loaded")

# ------------------------------
# 2️⃣ Text cleaning now handled by utils.py
# ------------------------------

# ------------------------------
# 3️⃣ Define enhanced lemmatization + stopword removal
# ------------------------------
def preprocess_text(text):
    if not text or len(text.strip()) < 3:
        return ""
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Extract meaningful tokens
    tokens = []
    for token in doc:
        # Skip if it's a stop word, punctuation, space, or pronoun placeholder
        if (not token.is_stop and 
            not token.is_punct and 
            not token.is_space and 
            token.lemma_ != "-PRON-" and
            len(token.lemma_) >= 2 and
            token.lemma_.isalpha()):
            tokens.append(token.lemma_.lower())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique_tokens.append(token)
    
    return " ".join(unique_tokens)

# ------------------------------
# 3.5️⃣ Intelligent text combination function
# ------------------------------
def combine_text_fields(row):
    """Intelligently combine title, description, content, and text fields"""
    text_parts = []
    
    # Priority order: title -> description -> content -> text
    fields = ['title', 'description', 'content', 'text']
    
    for field in fields:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            cleaned = clean_text(row[field])
            if cleaned and len(cleaned) >= 10:  # Only include meaningful text
                text_parts.append(cleaned)
    
    # Combine all parts, removing duplicates
    combined = " ".join(text_parts)
    
    # If combined text is too short, try individual fields
    if len(combined.strip()) < 20:
        for field in fields:
            if field in row and pd.notna(row[field]):
                cleaned = clean_text(row[field])
                if len(cleaned) >= 5:
                    return cleaned
    
    return combined

# ------------------------------
# 4️⃣ Load all CSV files from 'data/raw' folder
# ------------------------------
file_paths = glob.glob("data/raw/*.csv")  # adjust path if needed
df_list = []

for file in file_paths:
    df = pd.read_csv(file)
    df_list.append(df)

# Merge all CSVs
data = pd.concat(df_list, ignore_index=True)
print(f"📊 Loaded {len(data)} total records from {len(df_list)} CSV files")

# ------------------------------
# 5️⃣ Apply intelligent text processing
# ------------------------------
print("🧹 Step 1: Combining and cleaning text fields...")
data['combined_text'] = data.apply(combine_text_fields, axis=1)

# Filter out rows with insufficient text
before_filter = len(data)
data = data[data['combined_text'].str.len() >= 10].reset_index(drop=True)
after_filter = len(data)
print(f"📝 Filtered out {before_filter - after_filter} rows with insufficient text")

print("🔤 Step 2: Advanced text preprocessing (lemmatization, stopword removal)...")
data['processed_text'] = data['combined_text'].apply(preprocess_text)

# Final cleanup - remove rows where processed text is empty
before_final = len(data)
data = data[data['processed_text'].str.len() >= 5].reset_index(drop=True)
after_final = len(data)
print(f"🎯 Final dataset: {after_final} rows ({before_final - after_final} empty rows removed)")

# ------------------------------
# 6️⃣ Data quality analysis and save
# ------------------------------
print("\n📈 Data Quality Analysis:")
print(f"   • Average text length: {data['combined_text'].str.len().mean():.1f} characters")
print(f"   • Average processed length: {data['processed_text'].str.len().mean():.1f} characters")
print(f"   • Vocabulary size: {len(set(' '.join(data['processed_text']).split()))} unique words")

# Add text statistics columns
data['text_length'] = data['combined_text'].str.len()
data['processed_length'] = data['processed_text'].str.len()
data['word_count'] = data['processed_text'].str.split().str.len()

# Save final cleaned dataset
os.makedirs("data/processed", exist_ok=True)
data.to_csv("data/processed/civicpulse_processed.csv", index=False, encoding='utf-8')

print(f"\n[OK] Preprocessing complete! Total entries: {len(data)}")
print(f"💾 Processed dataset saved at: data/processed/civicpulse_processed.csv")
print(f"🎯 Ready for sentiment analysis and topic modeling!")
