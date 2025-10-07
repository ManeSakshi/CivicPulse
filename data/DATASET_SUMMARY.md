# CivicPulse Dataset Processing Summary

## 📊 **Dataset Overview**

### **External Training Data (Kaggle)**

- **Sentiment140**: 1,560,780 labeled tweets (negative: 783,905, positive: 776,875)
- **Airline Tweets**: 14,317 labeled tweets (negative: 9,178, positive: 2,363, neutral: 2,776)
- **Combined**: 1,575,097 total labeled records
- **Split**: 80% train (1,260,077) / 20% test (315,020)

### **Civic Data (Your Collection)**

- **Total**: 544 records from news/Twitter APIs
- **Sources**: NewsAPI, GNews, Local News, Twitter
- **Auto-labeled**: VADER + TextBlob consensus
- **Distribution**: 227 neutral, 183 negative, 134 positive

## 🗂️ **File Structure**

```
data/
├── external/               # Large Kaggle datasets (local only)
│   ├── Sentiment140.csv   # ~239MB - Original dataset
│   └── Tweets.csv          # ~3.4MB - Airline sentiment data
├── processed/
│   ├── external/           # Processed external data (local only)
│   │   ├── train_external.csv  # ~91MB - Training set
│   │   └── test_external.csv   # ~23MB - Test set
│   ├── civicpulse_processed.csv        # Main processed civic data
│   ├── civic_labeled.csv               # Clean civic data with labels
│   └── civicpulse_with_labels.csv     # Full civic data with all labels
└── raw/                    # Raw collected data
    ├── all_news_data.csv
    ├── gnews_data.csv
    └── twitter_data.csv
```

## 🚀 **Next Steps for BERT Training**

### **Option 1: Train on External Data First**

```python
# Use: data/processed/external/train_external.csv
# 1.26M labeled examples - good for BERT fine-tuning
# Then apply to civic data for predictions
```

### **Option 2: Train on Civic Data Only**

```python
# Use: data/processed/civic_labeled.csv
# 544 labeled examples - good for domain-specific model
# May need data augmentation or few-shot learning
```

### **Option 3: Combined Approach (Recommended)**

```python
# 1. Pre-train on external data (general sentiment)
# 2. Fine-tune on civic data (domain adaptation)
# 3. Best of both worlds: scale + domain specificity
```

## 🔧 **Processing Scripts Created**

1. **`src/preprocess.py`** - Main civic data preprocessing
2. **`src/process_external.py`** - External dataset processing
3. **`src/generate_labels.py`** - Auto-label civic data with VADER/TextBlob

## 📈 **Data Quality Metrics**

- **Text Length**: Avg 192.7 → 81.7 chars (processed)
- **Vocabulary**: 2,139 unique words in civic data
- **Label Agreement**: 50.4% VADER-TextBlob consensus
- **Coverage**: 100% of civic data now has meaningful text and labels

Ready for BERT/DistilBERT training! 🎯
