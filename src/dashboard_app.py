#!/usr/bin/env python3
"""
CivicPulse Interactive Dashboard
Real-time civic sentiment analysis and issue visualization for Sangli
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import sqlite3

# Set page config
st.set_page_config(
    page_title="CivicPulse Sangli",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CivicPulseDashboard:
    def __init__(self):
        self.data_loaded = False
        self.civic_df = None
        self.topic_results = None
        
    def load_data(self):
        """Load all data for dashboard"""
        try:
            # Load civic data
            if os.path.exists("data/processed/civic_labeled.csv"):
                self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")
                self.civic_df['timestamp'] = pd.to_datetime(self.civic_df['timestamp'])
                
            # Load topic modeling results
            if os.path.exists("models/topics/topic_results.pkl"):
                with open("models/topics/topic_results.pkl", 'rb') as f:
                    self.topic_results = pickle.load(f)
                    
            # Load from database if available
            if os.path.exists("data/civic.db"):
                conn = sqlite3.connect("data/civic.db")
                try:
                    db_df = pd.read_sql_query("SELECT * FROM civic_data ORDER BY timestamp DESC LIMIT 1000", conn)
                    if not db_df.empty:
                        db_df['timestamp'] = pd.to_datetime(db_df['timestamp'])
                        # Combine with CSV data if both exist
                        if self.civic_df is not None:
                            self.civic_df = pd.concat([self.civic_df, db_df]).drop_duplicates(subset=['text'], keep='first')
                        else:
                            self.civic_df = db_df
                except:
                    pass
                finally:
                    conn.close()
                    
            self.data_loaded = True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
    def render_header(self):
        """Render dashboard header"""
        st.title("🏛️ CivicPulse Sangli")
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
                recent_data = self.civic_df[self.civic_df['timestamp'] > (datetime.now() - timedelta(days=7))]
                st.metric("This Week", len(recent_data))
            with col4:
                neg_pct = (self.civic_df['label'] == 'negative').mean() * 100
                st.metric("Issues Identified", f"{neg_pct:.1f}%")
                
        st.divider()
        
    def render_sentiment_overview(self):
        """Render sentiment analysis overview"""
        st.subheader("📊 Sentiment Analysis Overview")
        
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
            st.plotly_chart(fig_pie, config={'responsive': True})
            
        with col2:
            # Sentiment stats
            st.markdown("**Sentiment Breakdown:**")
            for label, count in sentiment_counts.items():
                pct = (count / len(self.civic_df)) * 100
                if label == 'positive':
                    st.success(f"✅ Positive: {count} ({pct:.1f}%)")
                elif label == 'negative':
                    st.error(f"❌ Negative: {count} ({pct:.1f}%)")
                else:
                    st.info(f"➖ Neutral: {count} ({pct:.1f}%)")
                    
    def render_time_series(self):
        """Render time series analysis"""
        st.subheader("📈 Sentiment Trends Over Time")
        
        if self.civic_df is None:
            st.warning("No time series data available")
            return
            
        # Group by date and sentiment
        self.civic_df['date'] = self.civic_df['timestamp'].dt.date
        daily_sentiment = self.civic_df.groupby(['date', 'label']).size().unstack(fill_value=0)
        
        fig = px.line(
            x=daily_sentiment.index,
            y=daily_sentiment.values.T,
            title="Daily Sentiment Trends"
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Posts",
            legend_title="Sentiment"
        )
        
    st.plotly_chart(fig, config={'responsive': True})
        
    def render_topic_analysis(self):
        """Render topic modeling results"""
        st.subheader("🏷️ Civic Issue Categories")
        
        if self.topic_results is None:
            st.warning("No topic modeling results available. Run topic_model.py first.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**LDA Topics Discovered:**")
            if 'lda_topics' in self.topic_results and self.topic_results['lda_topics']:
                for topic in self.topic_results['lda_topics']:
                    category = topic.get('civic_category', 'general')
                    words = ', '.join(topic['top_words'][-5:])
                    st.markdown(f"• **{category.title()}**: {words}")
                    
        with col2:
            st.markdown("**BERT Topics Discovered:**")
            if 'bert_topics' in self.topic_results and self.topic_results['bert_topics']:
                for topic in self.topic_results['bert_topics']:
                    category = topic.get('civic_category', 'general')
                    words = ', '.join(topic['top_words'][:5])
                    count = topic.get('count', 0)
                    st.markdown(f"• **{category.title()}** ({count} docs): {words}")
            else:
                st.info("BERT topics not available (requires bertopic installation)")
                
    def render_issue_sentiment_analysis(self):
        """Render sentiment analysis by civic issue"""
        st.subheader("🎯 Issue-Specific Sentiment Analysis")
        
        if self.topic_results is None or 'sentiment_analysis' not in self.topic_results:
            st.info("Issue sentiment analysis not available. Run topic modeling first.")
            return
            
        sentiment_data = self.topic_results['sentiment_analysis']['by_issue']
        
        # Convert to DataFrame for visualization
        sentiment_df = pd.DataFrame(sentiment_data).T
        sentiment_df = sentiment_df.fillna(0)
        
        # Create stacked bar chart
        fig = go.Figure()
        
        colors = {'positive': '#2E8B57', 'neutral': '#FFD700', 'negative': '#DC143C'}
        
        for sentiment in ['positive', 'neutral', 'negative']:
            if sentiment in sentiment_df.columns:
                fig.add_trace(go.Bar(
                    name=sentiment.title(),
                    x=sentiment_df.index,
                    y=sentiment_df[sentiment],
                    marker_color=colors[sentiment]
                ))
        
        fig.update_layout(
            title="Sentiment Distribution by Civic Issue",
            xaxis_title="Civic Issue Category",
            yaxis_title="Percentage",
            barmode='stack',
            height=400
        )
        
    st.plotly_chart(fig, config={'responsive': True})
        
    def render_data_explorer(self):
        """Render data exploration section"""
        st.subheader("🔍 Data Explorer")
        
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
            date_range = st.date_input(
                "Date Range:",
                value=(self.civic_df['timestamp'].min().date(), self.civic_df['timestamp'].max().date()),
                min_value=self.civic_df['timestamp'].min().date(),
                max_value=self.civic_df['timestamp'].max().date()
            )
            
        with col3:
            show_count = st.number_input("Show Records:", min_value=10, max_value=500, value=50)
            
        # Apply filters
        filtered_df = self.civic_df.copy()
        
        if sentiment_filter != 'All':
            filtered_df = filtered_df[filtered_df['label'] == sentiment_filter]
            
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) & 
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
            
        # Display filtered data
        st.markdown(f"**Showing {len(filtered_df)} records** (filtered from {len(self.civic_df)} total)")
        
        display_df = filtered_df[['timestamp', 'text', 'label', 'vader_sentiment', 'textblob_sentiment']].head(show_count)
    st.dataframe(display_df, height=400)
        
    def render_model_info(self):
        """Render model and data information"""
        st.subheader("ℹ️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Sources:**")
            st.markdown("• Local Sangli news sources")
            st.markdown("• NewsAPI & GNews integration") 
            st.markdown("• Twitter/X civic mentions")
            st.markdown("• Synthetic civic content")
            
            if self.civic_df is not None:
                st.markdown(f"• Total records: {len(self.civic_df)}")
                st.markdown(f"• Date range: {self.civic_df['timestamp'].min().date()} to {self.civic_df['timestamp'].max().date()}")
                
        with col2:
            st.markdown("**AI Models:**")
            st.markdown("• VADER sentiment analysis")
            st.markdown("• TextBlob sentiment analysis")
            st.markdown("• SpaCy NLP preprocessing")
            
            if self.topic_results:
                st.markdown("• LDA topic modeling")
                if 'bert_topics' in self.topic_results:
                    st.markdown("• BERTopic analysis")
                    
            # Model file status
            st.markdown("**Model Files:**")
            if os.path.exists("models/sentiment_model.pkl"):
                st.success("✅ Sentiment model trained")
            else:
                st.warning("⏳ Sentiment model pending")
                
            if os.path.exists("models/topics/topic_results.pkl"):
                st.success("✅ Topic model ready")
            else:
                st.warning("⏳ Topic model pending")

    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        
        # Data refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.load_data()
            st.rerun()
            
        st.sidebar.divider()
        
        # Model training shortcuts
        st.sidebar.markdown("**Quick Actions:**")
        
        if st.sidebar.button("🚀 Run Sentiment Analysis"):
            st.sidebar.info("Execute: python src/sentiment_infer.py")
            
        if st.sidebar.button("🏷️ Run Topic Modeling"):
            st.sidebar.info("Execute: python src/topic_model.py")
            
        if st.sidebar.button("📊 Collect New Data"):
            st.sidebar.info("Execute: run_complete_pipeline.bat")
            
        st.sidebar.divider()
        
        # System status
        st.sidebar.markdown("**System Status:**")
        
        data_status = "✅ Loaded" if self.data_loaded else "❌ No Data"
        st.sidebar.markdown(f"Data: {data_status}")
        
        topic_status = "✅ Available" if self.topic_results else "⏳ Pending"
        st.sidebar.markdown(f"Topics: {topic_status}")
        
    def run_dashboard(self):
        """Run the complete dashboard"""
        # Load data
        self.load_data()
        
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        self.render_header()
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Trends", 
            "🏷️ Topics", 
            "🎯 Issues",
            "🔍 Explorer"
        ])
        
        with tab1:
            self.render_sentiment_overview()
            
        with tab2:
            self.render_time_series()
            
        with tab3:
            self.render_topic_analysis()
            
        with tab4:
            self.render_issue_sentiment_analysis()
            
        with tab5:
            self.render_data_explorer()
            
        # Footer
        st.divider()
        self.render_model_info()

def main():
    """Main dashboard function"""
    dashboard = CivicPulseDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
