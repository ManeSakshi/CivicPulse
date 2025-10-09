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
        """Load data for dashboard - prioritize Sangli-only data"""
        try:
            # First try to load Sangli-only data (preferred)
            if os.path.exists("data/processed/sangli_labeled.csv"):
                self.civic_df = pd.read_csv("data/processed/sangli_labeled.csv")
                st.success("✅ Displaying SANGLI-ONLY civic data")
            # Fallback to general civic data if Sangli-only not available
            elif os.path.exists("data/processed/civic_labeled.csv"):
                self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")
                st.warning("⚠️ Displaying mixed data (includes other cities). Run Sangli-only pipeline for pure Sangli data.")
            else:
                st.error("❌ No civic data found. Please run the data collection pipeline first.")
                return
                
            if 'timestamp' in self.civic_df.columns:
                self.civic_df['timestamp'] = pd.to_datetime(self.civic_df['timestamp'])
                
            # Load topic modeling results (prioritize Sangli-specific)
            if os.path.exists("models/topics/sangli_dashboard_topics.pkl"):
                with open("models/topics/sangli_dashboard_topics.pkl", 'rb') as f:
                    self.topic_results = pickle.load(f)
                st.success("✅ Displaying Sangli-specific topic analysis")
            elif os.path.exists("models/topics/topic_results.pkl"):
                with open("models/topics/topic_results.pkl", 'rb') as f:
                    self.topic_results = pickle.load(f)
                st.info("📊 General topic analysis loaded")
                    
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
        """Render Sangli-specific civic issue topic analysis"""
        st.subheader("🏛️ Sangli Civic Issue Categories")
        
        # Check if we have Sangli topic results or civic data with topic categories
        if self.civic_df is not None and 'topic_category' in self.civic_df.columns:
            # Use topic categories from Sangli data
            category_counts = self.civic_df['topic_category'].value_counts()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Topic distribution bar chart
                fig_topics = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Sangli Civic Issue Distribution",
                    labels={'x': 'Issue Category', 'y': 'Number of Records'},
                    color=category_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_topics.update_layout(xaxis_tickangle=45)
                st.plotly_chart(fig_topics, width='stretch')
            
            with col2:
                st.markdown("**📊 Issue Breakdown:**")
                total_records = len(self.civic_df)
                
                for category, count in category_counts.items():
                    percentage = (count / total_records) * 100
                    category_display = category.replace('_', ' ').title()
                    
                    # Get category emoji
                    category_emojis = {
                        'water_supply': '💧',
                        'traffic_transport': '🚦', 
                        'roads_infrastructure': '🛣️',
                        'waste_sanitation': '🗑️',
                        'municipal_services': '🏛️',
                        'development_planning': '🏗️',
                        'general_civic': '📋'
                    }
                    
                    emoji = category_emojis.get(category, '📌')
                    st.markdown(f"{emoji} **{category_display}**: {count} ({percentage:.1f}%)")
            
            # Topic details section
            st.markdown("---")
            st.subheader("📝 Issue Category Details")
            
            # Let user select category to explore
            selected_category = st.selectbox(
                "Select civic issue category to explore:",
                options=category_counts.index,
                format_func=lambda x: f"{category_emojis.get(x, '📌')} {x.replace('_', ' ').title()}"
            )
            
            if selected_category:
                category_data = self.civic_df[self.civic_df['topic_category'] == selected_category]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**📊 {selected_category.replace('_', ' ').title()} - Sentiment Analysis**")
                    sentiment_dist = category_data['label'].value_counts()
                    
                    fig_sentiment = px.pie(
                        values=sentiment_dist.values,
                        names=sentiment_dist.index,
                        title=f"Sentiment for {selected_category.replace('_', ' ').title()}",
                        color_discrete_map={
                            'positive': '#2E8B57',
                            'neutral': '#FFD700',
                            'negative': '#DC143C'
                        }
                    )
                    st.plotly_chart(fig_sentiment, width='stretch')
                
                with col2:
                    st.markdown(f"**📝 Sample {selected_category.replace('_', ' ').title()} Issues:**")
                    
                    sample_texts = category_data['text'].head(3)
                    for i, text in enumerate(sample_texts, 1):
                        with st.expander(f"Example {i}"):
                            st.write(text[:200] + "..." if len(text) > 200 else text)
                            
                            # Show sentiment for this specific text
                            row_sentiment = category_data[category_data['text'] == text]['label'].iloc[0]
                            sentiment_color = {
                                'positive': '🟢',
                                'neutral': '🟡', 
                                'negative': '🔴'
                            }
                            st.markdown(f"Sentiment: {sentiment_color.get(row_sentiment, '⚫')} {row_sentiment.title()}")
                            
        elif self.topic_results is not None:
            # Use old topic results format
            st.markdown("**📊 General Topic Analysis Available**")
            if 'categories' in self.topic_results and 'counts' in self.topic_results:
                categories = self.topic_results['categories']
                counts = self.topic_results['counts']
                
                fig_topics = px.bar(x=categories, y=counts, title="Topic Distribution")
                st.plotly_chart(fig_topics, use_container_width=True)
        
        else:
            st.warning("⚠️ No topic analysis available")
            st.markdown("**To generate Sangli topic analysis:**")
            st.code("python src/sangli_topic_model.py")
            
            if st.button("🔄 Run Topic Analysis Now"):
                with st.spinner("Analyzing Sangli civic issues..."):
                    # This would run the topic analysis
                    st.info("Topic analysis feature - run the command above in terminal")
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