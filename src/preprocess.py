# 📌 CivicPulse Data Preprocessing Script
# Combines News + Twitter CSVs, cleans text, removes stopwords, lemmatizes

import pandas as pd
import spacy
import glob
import re

# ------------------------------
# 1️⃣ Load SpaCy model for English
# ------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy English model loaded successfully")
except OSError:
    print("❌ SpaCy model 'en_core_web_sm' not found!")
    print("📥 Installing it now...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
    print("✅ SpaCy model installed and loaded")

# ------------------------------
# 2️⃣ Define text cleaning function
# ------------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove special characters & numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra whitespace
    return text

# ------------------------------
# 3️⃣ Define lemmatization + stopword removal
# ------------------------------
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ != "-PRON-"]
    return " ".join(tokens)

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

# ------------------------------
# 5️⃣ Apply cleaning & preprocessing
# ------------------------------
data['clean_text'] = data['text'].astype(str).apply(clean_text)
data['processed_text'] = data['clean_text'].apply(preprocess_text)

# ------------------------------
# 6️⃣ Save final cleaned dataset
# ------------------------------
data.to_csv("data/processed/civicpulse_processed.csv", index=False)
print(f"✅ Preprocessing complete! Total entries: {len(data)}")
print("Processed dataset saved at: data/processed/civicpulse_processed.csv")
