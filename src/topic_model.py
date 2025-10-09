#!/usr/bin/env python3
"""
CivicPulse Topic Modeling & Issue Classification
Identifies civic issue categories using BERTopic and LDA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import os
from datetime import datetime
import pickle

# Advanced topic modeling
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("[INFO] BERTopic not installed. Using LDA only.")
    print("To install: pip install bertopic sentence-transformers")

class CivicTopicModeler:
    def __init__(self):
        self.civic_df = None
        self.topics_lda = None
        self.topics_bert = None
        self.civic_categories = [
            'traffic', 'water', 'sanitation', 'garbage', 'roads', 
            'electricity', 'public_transport', 'healthcare', 'education',
            'development', 'corruption', 'administration'
        ]
        
    def load_data(self):
        """Load processed civic data"""
        print("📊 Loading civic data for topic modeling...")
        
        civic_path = "data/processed/civic_labeled.csv"
        self.civic_df = pd.read_csv(civic_path)
        print(f"   Loaded {len(self.civic_df)} civic records")
        
        # Filter out very short texts for better topic modeling
        self.civic_df = self.civic_df[self.civic_df['text'].str.len() >= 20].reset_index(drop=True)
        print(f"   After filtering: {len(self.civic_df)} records")
        
    def preprocess_for_topics(self):
        """Additional preprocessing specific to topic modeling"""
        print("🧹 Preprocessing text for topic modeling...")
        
        # Additional filtering for topic modeling
        self.texts = self.civic_df['text'].tolist()
        
        # Remove duplicates while preserving order
        seen = set()
        unique_texts = []
        for text in self.texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)
        
        self.texts = unique_texts
        print(f"   Unique texts for topic modeling: {len(self.texts)}")
        
    def run_lda_topic_modeling(self, n_topics=8):
        """Run Latent Dirichlet Allocation topic modeling"""
        print(f"\n📈 LDA Topic Modeling with {n_topics} topics")
        print("=" * 50)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=5,
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Train LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='batch',
            max_iter=25
        )
        
        lda.fit(tfidf_matrix)
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-10:]]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weight': topic.max()
            })
            
        self.topics_lda = topics
        self.lda_model = lda
        self.lda_vectorizer = vectorizer
        
        # Print topics
        print("🔍 Discovered LDA Topics:")
        for i, topic in enumerate(topics):
            print(f"   Topic {i}: {', '.join(topic['top_words'][-5:])}")
            
    def run_bert_topic_modeling(self):
        """Run BERTopic advanced topic modeling"""
        if not BERTOPIC_AVAILABLE:
            print("⏭️  Skipping BERTopic (not installed)")
            return
            
        print("\n🤖 BERTopic Advanced Topic Modeling")
        print("=" * 50)
        
        # Initialize BERTopic with custom settings
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=10,
            nr_topics='auto',
            calculate_probabilities=True
        )
        
        # Fit model and get topics
        topics, probabilities = topic_model.fit_transform(self.texts)
        
        # Extract topic information
        topic_info = topic_model.get_topic_info()
        
        bert_topics = []
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Exclude outlier topic
                topic_words = topic_model.get_topic(row['Topic'])
                top_words = [word for word, _ in topic_words[:10]]
                bert_topics.append({
                    'topic_id': row['Topic'],
                    'top_words': top_words,
                    'count': row['Count'],
                    'name': row['Name']
                })
        
        self.topics_bert = bert_topics
        self.bert_model = topic_model
        
        # Print topics
        print("🔍 Discovered BERT Topics:")
        for topic in bert_topics:
            print(f"   Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])} ({topic['count']} docs)")
            
    def classify_civic_issues(self):
        """Classify topics into civic issue categories"""
        print("\n🏛️ Classifying into Civic Issue Categories")
        print("=" * 50)
        
        # Keyword mapping for civic issues
        civic_keywords = {
            'traffic': ['traffic', 'road', 'vehicle', 'parking', 'congestion', 'signal'],
            'water': ['water', 'supply', 'pipeline', 'shortage', 'quality', 'tap'],
            'sanitation': ['sanitation', 'toilet', 'sewage', 'drain', 'waste', 'clean'],
            'garbage': ['garbage', 'waste', 'collection', 'dump', 'trash', 'bin'],
            'roads': ['road', 'street', 'pothole', 'repair', 'construction', 'infrastructure'],
            'electricity': ['electricity', 'power', 'supply', 'outage', 'line', 'connection'],
            'public_transport': ['bus', 'transport', 'service', 'route', 'frequency', 'station'],
            'healthcare': ['hospital', 'health', 'medical', 'doctor', 'clinic', 'treatment'],
            'education': ['school', 'education', 'teacher', 'student', 'college', 'learning'],
            'development': ['development', 'project', 'construction', 'infrastructure', 'urban'],
            'corruption': ['corruption', 'bribe', 'illegal', 'fraud', 'complaint', 'officer'],
            'administration': ['office', 'government', 'municipal', 'administration', 'service', 'official']
        }
        
        def classify_topic(topic_words):
            """Classify a topic based on its words"""
            scores = {}
            for category, keywords in civic_keywords.items():
                score = sum(1 for word in topic_words if any(keyword in word.lower() for keyword in keywords))
                scores[category] = score
            
            # Return category with highest score, or 'general' if no clear match
            max_category = max(scores.keys(), key=lambda k: scores[k])
            return max_category if scores[max_category] > 0 else 'general'
        
        # Classify LDA topics
        if self.topics_lda:
            for topic in self.topics_lda:
                topic['civic_category'] = classify_topic(topic['top_words'])
                
        # Classify BERT topics
        if self.topics_bert:
            for topic in self.topics_bert:
                topic['civic_category'] = classify_topic(topic['top_words'])
                
        # Print classified results
        print("📋 Civic Issue Classification:")
        if self.topics_lda:
            print("\nLDA Topics:")
            for topic in self.topics_lda:
                print(f"   {topic['civic_category'].upper()}: {', '.join(topic['top_words'][-3:])}")
                
        if self.topics_bert:
            print("\nBERT Topics:")
            for topic in self.topics_bert:
                print(f"   {topic['civic_category'].upper()}: {', '.join(topic['top_words'][:3])} ({topic['count']} docs)")
                
    def analyze_sentiment_by_topic(self):
        """Analyze sentiment distribution across civic issues"""
        print("\n📊 Sentiment Analysis by Civic Issue")
        print("=" * 50)
        
        # Create topic assignments for documents
        if self.lda_model:
            # Get topic assignments from LDA
            tfidf_matrix = self.lda_vectorizer.transform(self.civic_df['text'].tolist())
            topic_assignments = self.lda_model.transform(tfidf_matrix).argmax(axis=1)
            
            # Add to dataframe
            df_analysis = self.civic_df.copy()
            df_analysis['topic_id'] = topic_assignments
            
            # Map topic IDs to civic categories
            topic_category_map = {topic['topic_id']: topic['civic_category'] 
                                for topic in self.topics_lda}
            df_analysis['civic_issue'] = df_analysis['topic_id'].map(topic_category_map)
            
            # Analyze sentiment by civic issue
            sentiment_by_issue = df_analysis.groupby(['civic_issue', 'label']).size().unstack(fill_value=0)
            sentiment_percentages = sentiment_by_issue.div(sentiment_by_issue.sum(axis=1), axis=0) * 100
            
            print("📈 Sentiment Distribution by Civic Issue:")
            for issue in sentiment_percentages.index:
                pos = sentiment_percentages.loc[issue, 'positive']
                neg = sentiment_percentages.loc[issue, 'negative']
                neu = sentiment_percentages.loc[issue, 'neutral']
                print(f"   {issue.upper():15}: {pos:5.1f}% pos, {neu:5.1f}% neu, {neg:5.1f}% neg")
                
            # Save analysis results
            self.sentiment_analysis = {
                'by_issue': sentiment_percentages.to_dict(),
                'raw_counts': sentiment_by_issue.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
            
    def save_results(self):
        """Save topic modeling results"""
        print("\n💾 Saving Results")
        print("=" * 50)
        
        os.makedirs("models/topics", exist_ok=True)
        
        results = {
            'lda_topics': self.topics_lda,
            'bert_topics': self.topics_bert,
            'sentiment_analysis': getattr(self, 'sentiment_analysis', None),
            'civic_categories': self.civic_categories,
            'timestamp': datetime.now().isoformat(),
            'total_documents': len(self.texts)
        }
        
        # Save results
        with open('models/topics/topic_results.pkl', 'wb') as f:
            pickle.dump(results, f)
            
        # Save as CSV for easy viewing
        if self.topics_lda:
            lda_df = pd.DataFrame(self.topics_lda)
            lda_df.to_csv('models/topics/lda_topics.csv', index=False)
            
        if self.topics_bert:
            bert_df = pd.DataFrame(self.topics_bert)
            bert_df.to_csv('models/topics/bert_topics.csv', index=False)
            
        print("✅ Results saved to models/topics/")
        print("   - topic_results.pkl (complete results)")
        print("   - lda_topics.csv (LDA topics)")
        if self.topics_bert:
            print("   - bert_topics.csv (BERT topics)")
            
    def run_complete_analysis(self):
        """Run complete topic modeling pipeline"""
        print("🚀 CIVICPULSE TOPIC MODELING & ISSUE CLASSIFICATION")
        print("=" * 60)
        
        # Step 1: Load and preprocess data
        self.load_data()
        self.preprocess_for_topics()
        
        # Step 2: Run topic modeling algorithms
        self.run_lda_topic_modeling(n_topics=8)
        self.run_bert_topic_modeling()
        
        # Step 3: Classify into civic issue categories
        self.classify_civic_issues()
        
        # Step 4: Analyze sentiment by topic
        self.analyze_sentiment_by_topic()
        
        # Step 5: Save results
        self.save_results()
        
        print("\n🎉 TOPIC MODELING COMPLETE!")
        print("=" * 60)
        print("✅ Civic issues identified and classified")
        print("📊 Sentiment analysis by issue type complete")
        print("💾 Results ready for dashboard visualization")

def main():
    """Main topic modeling function"""
    modeler = CivicTopicModeler()
    modeler.run_complete_analysis()

if __name__ == "__main__":
    main()
