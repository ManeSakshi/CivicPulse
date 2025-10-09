#!/usr/bin/env python3
"""
CivicPulse SANGLI Topic Modeling & Issue Classification
Identifies Sangli civic issue categories: Water, Traffic, Roads, etc.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
import pickle
from datetime import datetime

class SangliCivicTopicModeler:
    def __init__(self):
        self.sangli_df = None
        self.lda_model = None
        self.vectorizer = None
        
        # Sangli-specific civic categories with keywords
        self.sangli_civic_categories = {
            'water_supply': ['water', 'supply', 'shortage', 'pipeline', 'tap', 'drinking', 'quality', 'tank'],
            'traffic_transport': ['traffic', 'road', 'vehicle', 'bus', 'parking', 'signal', 'congestion', 'transport'],
            'roads_infrastructure': ['road', 'pothole', 'repair', 'construction', 'bridge', 'infrastructure', 'maintenance'],
            'waste_sanitation': ['garbage', 'waste', 'clean', 'sanitation', 'dustbin', 'collection', 'drainage'],
            'municipal_services': ['municipal', 'corporation', 'office', 'service', 'complaint', 'application', 'administration'],
            'development_planning': ['development', 'project', 'planning', 'construction', 'new', 'build', 'smart', 'city']
        }
        
    def load_sangli_data(self):
        """Load Sangli-specific processed civic data"""
        print("🏛️ Loading Sangli civic data for topic modeling...")
        
        # Try Sangli-specific data first
        if os.path.exists("data/processed/sangli_labeled.csv"):
            self.sangli_df = pd.read_csv("data/processed/sangli_labeled.csv")
            print(f"   ✅ Loaded {len(self.sangli_df)} Sangli civic records")
        else:
            print("   ❌ Sangli data not found. Run Sangli pipeline first.")
            return False
        
        # Filter for meaningful text
        initial_count = len(self.sangli_df)
        self.sangli_df = self.sangli_df[self.sangli_df['text'].str.len() >= 30].reset_index(drop=True)
        print(f"   📝 After filtering short texts: {len(self.sangli_df)} records")
        
        return True
    
    def classify_by_keywords(self):
        """Classify Sangli civic issues using keyword matching"""
        print("🔍 Classifying Sangli civic issues by keywords...")
        
        topic_assignments = []
        topic_scores = []
        
        for text in self.sangli_df['text']:
            text_lower = text.lower()
            category_scores = {}
            
            # Score each category based on keyword presence
            for category, keywords in self.sangli_civic_categories.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                category_scores[category] = score
            
            # Assign to category with highest score
            if max(category_scores.values()) > 0:
                best_category = max(category_scores, key=category_scores.get)
                topic_assignments.append(best_category)
                topic_scores.append(category_scores[best_category])
            else:
                topic_assignments.append('general_civic')
                topic_scores.append(0)
        
        self.sangli_df['topic_category'] = topic_assignments
        self.sangli_df['topic_score'] = topic_scores
        
        # Print category distribution
        category_dist = self.sangli_df['topic_category'].value_counts()
        print("📊 Sangli Civic Issue Categories:")
        for category, count in category_dist.items():
            percentage = count/len(self.sangli_df)*100
            print(f"   📌 {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        return category_dist
    
    def create_lda_topics(self, n_topics=6):
        """Create LDA topic model for Sangli civic data"""
        print(f"🤖 Creating LDA topic model with {n_topics} topics...")
        
        # Prepare text data
        texts = self.sangli_df['processed_text'] if 'processed_text' in self.sangli_df.columns else self.sangli_df['text']
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Fit and transform the text
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Create LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        
        # Fit LDA model
        lda_matrix = self.lda_model.fit_transform(tfidf_matrix)
        
        # Get topic assignments
        topic_assignments = np.argmax(lda_matrix, axis=1)
        self.sangli_df['lda_topic'] = topic_assignments
        
        # Print top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        print("\n🎯 LDA Topic Keywords:")
        
        topic_keywords = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_keywords[f'topic_{topic_idx}'] = top_words
            print(f"   Topic {topic_idx}: {', '.join(top_words[:5])}")
        
        return topic_keywords
    
    def generate_sangli_topic_report(self):
        """Generate comprehensive Sangli topic analysis report"""
        print("📋 Generating Sangli civic topic analysis report...")
        
        # Combine keyword-based and LDA results
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(self.sangli_df),
            'keyword_categories': self.sangli_df['topic_category'].value_counts().to_dict(),
            'lda_topics': self.sangli_df['lda_topic'].value_counts().to_dict() if 'lda_topic' in self.sangli_df.columns else {},
            'category_examples': {}
        }
        
        # Add examples for each category
        for category in self.sangli_df['topic_category'].unique():
            category_examples = self.sangli_df[
                self.sangli_df['topic_category'] == category
            ]['text'].head(3).tolist()
            report['category_examples'][category] = category_examples
        
        # Save topic results
        os.makedirs('models/topics', exist_ok=True)
        
        # Save keyword-based results
        with open('models/topics/sangli_topic_results.pkl', 'wb') as f:
            pickle.dump({
                'categories': self.sangli_civic_categories,
                'dataframe': self.sangli_df,
                'report': report
            }, f)
        
        # Save for dashboard
        dashboard_topics = {
            'categories': list(report['keyword_categories'].keys()),
            'counts': list(report['keyword_categories'].values()),
            'examples': report['category_examples'],
            'last_updated': report['timestamp']
        }
        
        with open('models/topics/sangli_dashboard_topics.pkl', 'wb') as f:
            pickle.dump(dashboard_topics, f)
        
        print(f"💾 Saved Sangli topic results:")
        print(f"   📁 models/topics/sangli_topic_results.pkl")
        print(f"   📁 models/topics/sangli_dashboard_topics.pkl")
        
        return report
    
    def display_category_samples(self):
        """Display sample texts for each Sangli civic category"""
        print("\n📝 SANGLI CIVIC ISSUE SAMPLES BY CATEGORY:")
        print("=" * 60)
        
        for category in self.sangli_df['topic_category'].unique():
            category_data = self.sangli_df[self.sangli_df['topic_category'] == category]
            print(f"\n🏷️ {category.replace('_', ' ').upper()} ({len(category_data)} records)")
            print("-" * 40)
            
            # Show sentiment distribution for this category
            sentiment_dist = category_data['label'].value_counts()
            print("Sentiment:", dict(sentiment_dist))
            
            # Show sample texts
            for i, text in enumerate(category_data['text'].head(2)):
                print(f"   {i+1}. {text[:80]}...")
    
    def run_sangli_topic_analysis(self):
        """Run complete Sangli topic modeling analysis"""
        print("🏛️ STARTING SANGLI CIVIC TOPIC ANALYSIS")
        print("=" * 50)
        
        # Load data
        if not self.load_sangli_data():
            return False
        
        # Classify by keywords
        category_dist = self.classify_by_keywords()
        
        # Create LDA topics
        topic_keywords = self.create_lda_topics()
        
        # Generate report
        report = self.generate_sangli_topic_report()
        
        # Display samples
        self.display_category_samples()
        
        print("\n" + "=" * 50)
        print("✅ SANGLI TOPIC ANALYSIS COMPLETE!")
        print("📊 Dashboard now has topic-wise Sangli civic data")
        print("🎯 Categories: Water, Traffic, Roads, Municipal Services, etc.")
        
        return True

if __name__ == "__main__":
    modeler = SangliCivicTopicModeler()
    success = modeler.run_sangli_topic_analysis()
    exit(0 if success else 1)