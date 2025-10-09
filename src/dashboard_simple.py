#!/usr/bin/env python3
"""
CivicPulse Simple Dashboard
ASCII-compatible dashboard for Windows
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="CivicPulse Sangli",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleCivicDashboard:
    def __init__(self):
        self.civic_df = None
        self.topic_results = None
        
    def load_data(self):
        """Load all data for dashboard"""
        try:
            # Load civic data
            if os.path.exists("data/processed/civic_labeled.csv"):
                self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")
                if 'timestamp' in self.civic_df.columns:
                    self.civic_df['timestamp'] = pd.to_datetime(self.civic_df['timestamp'])
                
            # Load topic modeling results
            if os.path.exists("models/topics/topic_results.pkl"):
                with open("models/topics/topic_results.pkl", 'rb') as f:
                    self.topic_results = pickle.load(f)
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
    def render_header(self):
        """Render dashboard header"""
        st.title("CivicPulse Sangli - Sentiment Analysis Dashboard")
        st.markdown("**Real-time Civic Sentiment Analysis & Issue Tracking**")
        
        # Quick stats in columns
        if self.civic_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(self.civic_df))
            with col2:
                pos_pct = (self.civic_df['label'] == 'positive').mean() * 100
                st.metric("Positive Sentiment", f"{pos_pct:.1f}%")
            with col3:
                if 'timestamp' in self.civic_df.columns:
                    recent_data = self.civic_df[self.civic_df['timestamp'] > (datetime.now() - timedelta(days=7))]
                    st.metric("This Week", len(recent_data))
                else:
                    st.metric("Available Data", len(self.civic_df))
            with col4:
                neg_pct = (self.civic_df['label'] == 'negative').mean() * 100
                st.metric("Issues Identified", f"{neg_pct:.1f}%")
                
        st.divider()
        
    def render_sentiment_overview(self):
        """Render sentiment analysis overview"""
        st.subheader("Sentiment Analysis Overview")
        
        if self.civic_df is None:
            st.warning("No civic data available")
            return
            
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = self.civic_df['label'].value_counts()
            fig_pie = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Overall Sentiment Distribution",
                color_discrete_map={
                    'positive': '#2E8B57',
                    'neutral': '#FFD700', 
                    'negative': '#DC143C'
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            # Sentiment stats
            st.markdown("**Sentiment Breakdown:**")
            for label, count in sentiment_counts.items():
                pct = (count / len(self.civic_df)) * 100
                if label == 'positive':
                    st.success(f"Positive: {count} ({pct:.1f}%)")
                elif label == 'negative':
                    st.error(f"Negative: {count} ({pct:.1f}%)")
                else:
                    st.info(f"Neutral: {count} ({pct:.1f}%)")
                    
    def render_topic_analysis(self):
        """Render topic modeling results"""
        st.subheader("Civic Issue Categories")
        
        if self.topic_results is None:
            st.warning("No topic modeling results available. Run topic_model.py first.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Topics Discovered:**")
            if 'lda_topics' in self.topic_results and self.topic_results['lda_topics']:
                for i, topic in enumerate(self.topic_results['lda_topics']):
                    category = topic.get('civic_category', 'general')
                    words = ', '.join(topic['top_words'][-5:])
                    st.markdown(f"**Topic {i} ({category.title()})**")
                    st.markdown(f"Keywords: {words}")
                    st.markdown("---")
                    
        with col2:
            st.markdown("**Issue Categories:**")
            categories = {}
            if 'lda_topics' in self.topic_results:
                for topic in self.topic_results['lda_topics']:
                    category = topic.get('civic_category', 'general')
                    if category in categories:
                        categories[category] += 1
                    else:
                        categories[category] = 1
            
            for category, count in categories.items():
                st.markdown(f"- **{category.title()}**: {count} topic(s)")
                
    def render_data_explorer(self):
        """Render data exploration section"""
        st.subheader("Data Explorer")
        
        if self.civic_df is None:
            st.warning("No data available for exploration")
            return
            
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment:",
                options=['All'] + list(self.civic_df['label'].unique())
            )
            
        with col2:
            show_count = st.number_input("Show Records:", min_value=10, max_value=500, value=50)
            
        # Apply filters
        filtered_df = self.civic_df.copy()
        
        if sentiment_filter != 'All':
            filtered_df = filtered_df[filtered_df['label'] == sentiment_filter]
            
        # Display filtered data
        st.markdown(f"**Showing {len(filtered_df)} records** (filtered from {len(self.civic_df)} total)")
        
        # Select columns to display
        display_columns = ['text', 'label']
        if 'timestamp' in filtered_df.columns:
            display_columns.insert(0, 'timestamp')
        if 'vader_sentiment' in filtered_df.columns:
            display_columns.append('vader_sentiment')
        if 'textblob_sentiment' in filtered_df.columns:
            display_columns.append('textblob_sentiment')
            
        display_df = filtered_df[display_columns].head(show_count)
        st.dataframe(display_df, use_container_width=True, height=400)
        
    def render_model_info(self):
        """Render model and data information"""
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Sources:**")
            st.markdown("- Local Sangli news sources")
            st.markdown("- NewsAPI & GNews integration") 
            st.markdown("- Twitter/X civic mentions")
            st.markdown("- Synthetic civic content")
            
            if self.civic_df is not None:
                st.markdown(f"- Total records: {len(self.civic_df)}")
                if 'timestamp' in self.civic_df.columns:
                    st.markdown(f"- Date range: {self.civic_df['timestamp'].min().date()} to {self.civic_df['timestamp'].max().date()}")
                
        with col2:
            st.markdown("**AI Models:**")
            st.markdown("- VADER sentiment analysis")
            st.markdown("- TextBlob sentiment analysis")
            st.markdown("- SpaCy NLP preprocessing")
            
            if self.topic_results:
                st.markdown("- LDA topic modeling")
                    
            # Model file status
            st.markdown("**Model Files:**")
            if os.path.exists("models/sentiment_model.pkl"):
                st.success("Sentiment model trained")
            else:
                st.warning("Sentiment model pending")
                
            if os.path.exists("models/topics/topic_results.pkl"):
                st.success("Topic model ready")
            else:
                st.warning("Topic model pending")

    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # Data refresh button
        if st.sidebar.button("Refresh Data"):
            self.load_data()
            st.rerun()
            
        st.sidebar.divider()
        
        # Model training shortcuts
        st.sidebar.markdown("**Quick Actions:**")
        
        st.sidebar.markdown("Run Sentiment Analysis:")
        st.sidebar.code("python src/sentiment_infer.py")
        
        st.sidebar.markdown("Run Topic Modeling:")
        st.sidebar.code("python src/topic_model.py")
        
        st.sidebar.markdown("Collect New Data:")
        st.sidebar.code("run_complete_pipeline.bat")
        
        st.sidebar.divider()
        
        # System status
        st.sidebar.markdown("**System Status:**")
        
        if self.civic_df is not None:
            st.sidebar.success("Data: Loaded")
        else:
            st.sidebar.error("Data: No Data")
        
        if self.topic_results is not None:
            st.sidebar.success("Topics: Available")
        else:
            st.sidebar.warning("Topics: Pending")
        
    def run_dashboard(self):
        """Run the complete dashboard"""
        # Load data
        self.load_data()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        self.render_header()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview", 
            "Topics", 
            "Explorer",
            "System Info"
        ])
        
        with tab1:
            self.render_sentiment_overview()
            
        with tab2:
            self.render_topic_analysis()
            
        with tab3:
            self.render_data_explorer()
            
        with tab4:
            self.render_model_info()

def main():
    """Main dashboard function"""
    dashboard = SimpleCivicDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()