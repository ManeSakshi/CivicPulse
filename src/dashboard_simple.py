#!/usr/bin/env python3
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
import json
import subprocess
import sys
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="CivicPulse Sangli",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple styling for presentation
st.markdown("""
<style>
/* Header tweaks */
.stApp h1 {font-size:28px}
.stApp h2 {color:#2E8B57}
.stMetric {font-weight:600}
.css-1v3fvcr {padding: 8px 12px}


/* Sidebar button sizing: fixed exact height and consistent boxes */
.stSidebar .stButton>button, .stSidebar .stDownloadButton>button, .stButton>button, .stDownloadButton>button {
    height: 64px !important; /* exact fixed height */
    line-height: 1.2 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    padding: 8px 14px !important; /* adjusted padding for content */
    border-radius: 10px !important; /* slightly rounder */
    font-weight: 600 !important;
}

/* Slightly reduce margins in sidebar for compact layout */
.css-1lcbmhc .stMarkdown, .css-1lcbmhc .stText {
    margin-bottom: 6px !important;
}

</style>
""", unsafe_allow_html=True)

class SimpleCivicDashboard:
    def __init__(self):
        self.civic_df = None
        self.topic_results = None
        # presentation mode persisted in session_state; default to False
        self.presentation_mode = False
        self.model_info = None
        self.data_source = None
        self.topic_source = None
        
    def load_data(self):
        """Load data for dashboard - prioritize Sangli-only data"""
        try:
            # First try to load Sangli-only data (preferred)
            if os.path.exists("data/processed/sangli_labeled.csv"):
                self.civic_df = pd.read_csv("data/processed/sangli_labeled.csv")
                self.data_source = "data/processed/sangli_labeled.csv"
            # Fallback to general civic data if Sangli-only not available
            elif os.path.exists("data/processed/civic_labeled.csv"):
                self.civic_df = pd.read_csv("data/processed/civic_labeled.csv")
                self.data_source = "data/processed/civic_labeled.csv"
            else:
                st.error("❌ No civic data found. Please run the data collection pipeline first.")
                return
                
            if 'timestamp' in self.civic_df.columns:
                self.civic_df['timestamp'] = pd.to_datetime(self.civic_df['timestamp'])
                
            # Load topic modeling results (prioritize Sangli-specific)
            if os.path.exists("models/topics/sangli_dashboard_topics.pkl"):
                with open("models/topics/sangli_dashboard_topics.pkl", 'rb') as f:
                    self.topic_results = pickle.load(f)
                self.topic_source = "models/topics/sangli_dashboard_topics.pkl"
            elif os.path.exists("models/topics/topic_results.pkl"):
                with open("models/topics/topic_results.pkl", 'rb') as f:
                    self.topic_results = pickle.load(f)
                self.topic_source = "models/topics/topic_results.pkl"

            # Load model_info.json if present
            if os.path.exists("model_info.json"):
                try:
                    with open("model_info.json", "r") as mf:
                        self.model_info = json.load(mf)
                except Exception:
                    self.model_info = None
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
    def render_header(self):
        """Render dashboard header"""
        if self.presentation_mode:
            st.markdown("<h1 style='font-size:34px;margin:0;padding:0'>CivicPulse Sangli — Sentiment Dashboard</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#666;margin-top:0'>Real-time Civic Sentiment Analysis & Issue Tracking</h3>", unsafe_allow_html=True)
        else:
            st.title("CivicPulse Sangli - Sentiment Analysis Dashboard")
            st.markdown("**Real-time Civic Sentiment Analysis & Issue Tracking**")
        
        # Quick stats in columns (show data source and counts)
        if self.civic_df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(self.civic_df))
                # show which file was used
                # (source caption removed for presentation cleanliness)
                # (removed noisy summary caption to keep header clean)
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
            st.plotly_chart(fig_pie, config={'responsive': True})
            
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

        # Minimal manual inference control: allow the user to trigger the
        # lightweight keyword-based inference on demand. This button intentionally
        # does not show success/error messages (inference runs silently) and it
        # does not use experimental Streamlit APIs.
        try:
            if st.button("Infer topics"):
                if self.civic_df is not None:
                    keywords = {
                        'water_supply': ['water', 'tap', 'pipeline', 'drinking'],
                        'traffic_transport': ['traffic', 'bus', 'auto', 'transport', 'vehicle'],
                        'roads_infrastructure': ['road', 'pothole', 'bridge', 'flyover', 'street'],
                        'waste_sanitation': ['garbage', 'waste', 'sewage', 'sanitation', 'drain'],
                        'municipal_services': ['municipal', 'council', 'ward', 'office', 'service'],
                        'development_planning': ['plan', 'development', 'project', 'construction'],
                        'general_civic': ['civic', 'issue', 'public', 'community']
                    }
                    # default category
                    self.civic_df['topic_category'] = 'general_civic'
                    texts = self.civic_df['text'].astype(str).str.lower()
                    for cat, kws in keywords.items():
                        mask = texts.str.contains('|'.join(kws), na=False)
                        self.civic_df.loc[mask, 'topic_category'] = cat
                    # Persist inferred categories into session_state so they survive
                    # subsequent reruns (load_data restores them from this state).
                    try:
                        st.session_state['topic_category_values'] = list(self.civic_df['topic_category'].astype(str))
                        st.session_state['topic_inferred'] = True
                    except Exception:
                        pass
                    # Subtle UX cue: confirm inference applied (non-intrusive)
                    try:
                        st.caption("Inferred topic categories applied.")
                    except Exception:
                        pass
        except Exception:
            # Fail silently to avoid showing low-level errors in the UI
            pass


        # Check if we have Sangli topic results or civic data with topic categories
        inferred_topics = False
        if self.civic_df is not None and 'topic_category' not in self.civic_df.columns and not getattr(self, 'topic_results', None):
            # Attempt a lightweight inference of topic_category from keywords so the Topics UI is visible
            try:
                # No explicit 'topic_category' found — perform a lightweight
                # keyword-based inference for the UI silently (no success/info messages).
                keywords = {
                    'water_supply': ['water', 'tap', 'pipeline', 'drinking'],
                    'traffic_transport': ['traffic', 'bus', 'auto', 'transport', 'vehicle'],
                    'roads_infrastructure': ['road', 'pothole', 'bridge', 'flyover', 'street'],
                    'waste_sanitation': ['garbage', 'waste', 'sewage', 'sanitation', 'drain'],
                    'municipal_services': ['municipal', 'council', 'ward', 'office', 'service'],
                    'development_planning': ['plan', 'development', 'project', 'construction'],
                    'general_civic': ['civic', 'issue', 'public', 'community']
                }
                # default category
                self.civic_df['topic_category'] = 'general_civic'
                texts = self.civic_df['text'].astype(str).str.lower()
                for cat, kws in keywords.items():
                    mask = texts.str.contains('|'.join(kws), na=False)
                    # only set where default still general_civic or overwrite
                    self.civic_df.loc[mask, 'topic_category'] = cat
                inferred_topics = True
                # Persist inferred categories into session_state so they survive
                # subsequent reruns (load_data restores them from this state).
                try:
                    st.session_state['topic_category_values'] = list(self.civic_df['topic_category'].astype(str))
                    st.session_state['topic_inferred'] = True
                except Exception:
                    pass
            except Exception:
                pass

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
                # use width='stretch' for responsive full-width charts
                st.plotly_chart(fig_topics, config={'responsive': True})

            with col2:
                st.markdown("**📊 Issue Breakdown:**")
                total_records = len(self.civic_df)

                # define emojis once
                category_emojis = {
                    'water_supply': '💧',
                    'traffic_transport': '🚦', 
                    'roads_infrastructure': '🛣️',
                    'waste_sanitation': '🗑️',
                    'municipal_services': '🏛️',
                    'development_planning': '🏗️',
                    'general_civic': '📋'
                }

                for category, count in category_counts.items():
                    percentage = (count / total_records) * 100
                    category_display = category.replace('_', ' ').title()
                    emoji = category_emojis.get(category, '📌')
                    st.markdown(f"{emoji} **{category_display}**: {count} ({percentage:.1f}%)")

            # Topic details section
            st.markdown("---")
            st.subheader("📝 Issue Category Details")

            # Search box for topics (filter by name)
            topic_search = st.text_input("Search topics (name or keyword):", key="topic_search")
            if topic_search:
                filtered = [c for c in category_counts.index if topic_search.lower() in str(c).lower()]
            else:
                filtered = list(category_counts.index)

            st.markdown(f"**Topics matching:** {len(filtered)}")

            # Render topic buttons (simple grid)
            cols = st.columns(4)
            for i, cat in enumerate(filtered):
                col = cols[i % 4]
                label = f"{category_emojis.get(cat, '📌')} {cat.replace('_', ' ').title()} ({category_counts.get(cat,0)})"
                if col.button(label, key=f"topic_button_{cat}"):
                    st.session_state['selected_topic'] = cat

            # Determine selected topic (persist in session state)
            selected_category = st.session_state.get('selected_topic', None)
            # Also allow pick from selectbox if none selected
            if not selected_category:
                selected_category = st.selectbox(
                    "Or choose a topic:",
                    options=filtered,
                    format_func=lambda x: f"{category_emojis.get(x, '📌')} {x.replace('_', ' ').title()}"
                )

            # When a topic is selected, show pie chart left and samples right
            if selected_category:
                category_data = self.civic_df[self.civic_df['topic_category'] == selected_category]
                left, right = st.columns([2, 1])

                with left:
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
                    st.plotly_chart(fig_sentiment, config={'responsive': True})

                with right:
                    st.markdown(f"**📝 Sample {selected_category.replace('_', ' ').title()} Issues:**")
                    # pagination via session state
                    key = f"samples_shown_{selected_category}"
                    if key not in st.session_state:
                        st.session_state[key] = 3
                    samples_to_show = st.session_state[key]
                    sample_texts = category_data['text'].head(samples_to_show)
                    for i, text in enumerate(sample_texts, 1):
                        title_preview = (text[:120] + '...') if len(text) > 120 else text
                        with st.expander(f"Case {i}: {title_preview}"):
                            st.write(text)
                            try:
                                row_sentiment = category_data[category_data['text'] == text]['label'].iloc[0]
                            except Exception:
                                row_sentiment = 'unknown'
                            sentiment_color = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}
                            st.markdown(f"Sentiment: {sentiment_color.get(row_sentiment, '⚫')} {row_sentiment.title() if isinstance(row_sentiment, str) else row_sentiment}")

                    more_col, reset_col = st.columns([1, 1])
                    with more_col:
                        if st.button("Show more", key=f"more_{selected_category}"):
                            st.session_state[key] = min(len(category_data), st.session_state[key] + 5)
                            try:
                                st.rerun()
                            except Exception:
                                pass
                    with reset_col:
                        if st.button("Reset", key=f"reset_{selected_category}"):
                            st.session_state[key] = 3
                            # clear selected topic so UI returns to overview
                            if 'selected_topic' in st.session_state:
                                del st.session_state['selected_topic']
                            try:
                                st.rerun()
                            except Exception:
                                pass

        elif self.topic_results is not None:
            # Use old topic results format
            st.markdown("**📊 General Topic Analysis Available**")
            if 'categories' in self.topic_results and 'counts' in self.topic_results:
                categories = self.topic_results['categories']
                counts = self.topic_results['counts']

                fig_topics = px.bar(x=categories, y=counts, title="Topic Distribution")
                st.plotly_chart(fig_topics, config={'responsive': True})

        else:
            st.warning("⚠️ No topic analysis available")
            st.markdown("**To generate Sangli topic analysis:**")
            st.code("python src/sangli_topic_model.py")
            st.markdown("Run the command above in a terminal to generate Sangli-specific topic results. Once complete, refresh data using 'Refresh Data' in the sidebar.")
                
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
        st.dataframe(display_df, height=400)
        
    def render_model_info(self):
        """Render model and data information"""
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Sources:**")
            st.markdown("- Local Sangli news sources")
            st.markdown("- NewsAPI & GNews integration") 
            st.markdown("- Twitter/X civic mentions")
            st.markdown("- Synthesized civic content (labels/augmentation)")
            
            if self.civic_df is not None:
                st.markdown(f"- Total records: {len(self.civic_df)}")
                if 'timestamp' in self.civic_df.columns and not self.civic_df['timestamp'].isnull().all():
                    st.markdown(f"- Date range: {self.civic_df['timestamp'].min().date()} to {self.civic_df['timestamp'].max().date()}")
                # if a summary file exists, show last updated
                summary_path = 'data/processed/sangli_summary.json'
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as sf:
                            sj = json.load(sf)
                        last_updated = sj.get('last_updated')
                        total = sj.get('total_records')
                        if last_updated:
                            st.markdown(f"- Last summary update: {last_updated}")
                        if total is not None:
                            st.markdown(f"- Summary total records: {total}")
                    except Exception:
                        pass
                
        with col2:
            st.markdown("**AI Models:**")
            st.markdown("- VADER sentiment analysis")
            st.markdown("- TextBlob sentiment analysis")
            st.markdown("- SpaCy NLP preprocessing")
            
            if self.topic_results:
                st.markdown("- LDA topic modeling")
                    
            # Model file status (show only available/ready indicators)
            # Keep the system info clean by not showing 'pending' warnings.
            if os.path.exists("models/sentiment_model.pkl"):
                st.success("Sentiment model trained")

            if os.path.exists("models/topics/topic_results.pkl") or self.topic_results is not None:
                st.success("Topic model ready")

        # Presentation panel (show metrics and visuals when in presentation mode)
        if self.presentation_mode:
            st.markdown("---")
            st.markdown("### Presentation: Model Performance")
            if self.model_info:
                acc = self.model_info.get('accuracy')
                if acc is not None:
                    st.metric("Model Accuracy", f"{acc:.1%}")

                metrics = self.model_info.get('metrics', {})
                # Build per-class metrics chart (precision/recall/f1)
                if metrics:
                    classes = []
                    precisions = []
                    recalls = []
                    f1s = []
                    supports = []
                    for cls in ['negative', 'neutral', 'positive']:
                        m = metrics.get(cls, {})
                        classes.append(cls.title())
                        precisions.append(m.get('precision', 0))
                        recalls.append(m.get('recall', 0))
                        f1s.append(m.get('f1-score', 0))
                        supports.append(int(m.get('support', 0)))

                    df_metrics = pd.DataFrame({
                        'Class': classes,
                        'Precision': precisions,
                        'Recall': recalls,
                        'F1-score': f1s,
                        'Support': supports
                    })

                    # Long format for grouped bar chart
                    df_long = df_metrics.melt(id_vars=['Class', 'Support'], value_vars=['Precision', 'Recall', 'F1-score'], var_name='Metric', value_name='Value')
                    fig_metrics = px.bar(df_long, x='Class', y='Value', color='Metric', barmode='group', title='Per-class metrics (Precision / Recall / F1)')
                    fig_metrics.update_yaxes(range=[0, 1])
                    st.plotly_chart(fig_metrics, config={'responsive': True})

                    # Show support counts as a small table beneath
                    st.markdown("**Support (test samples per class):**")
                    st.table(df_metrics[['Class', 'Support']].set_index('Class'))

            st.markdown("### Visuals")
            reports_dir = "reports"
            # Confusion matrix: if we have an image, display it. (Best-effort fallback.)
            if os.path.exists(os.path.join(reports_dir, 'confusion_matrix.png')):
                st.image(os.path.join(reports_dir, 'confusion_matrix.png'), caption="Confusion matrix")
            else:
                st.info("Confusion matrix image not found; run evaluation to regenerate reports/confusion_matrix.png")

            # Show model_info.json raw summary link / content
            if os.path.exists('model_info.json'):
                try:
                    with open('model_info.json', 'r') as mf:
                        js = json.load(mf)
                    st.download_button('Download model_info.json', data=json.dumps(js, indent=2), file_name='model_info.json', mime='application/json')
                except Exception:
                    pass

    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.title("Navigation")
        # short caption under the title
        st.sidebar.caption("Quick actions, exports and system status")

        # --- Data section ---
        st.sidebar.markdown("### Data")
        # Data refresh button
        if st.sidebar.button("Refresh Data"):
            self.load_data()
            try:
                st.rerun()
            except Exception:
                pass

        st.sidebar.divider()

        # --- Actions section ---
        st.sidebar.markdown("### Project Actions")
        if st.sidebar.button("Run evaluation (background)"):
            try:
                # start evaluation in background
                subprocess.Popen([sys.executable, "src/evaluate_sentiment_model.py"], shell=False)
                st.sidebar.info("Evaluation started in background. Check terminal or reports/ when done.")
            except Exception as e:
                st.sidebar.error(f"Failed to start evaluation: {e}")

        if st.sidebar.button("Refresh topics (background)"):
            try:
                subprocess.Popen([sys.executable, "src/sangli_topic_model.py"], shell=False)
                st.sidebar.info("Topic refresh started in background.")
            except Exception as e:
                st.sidebar.error(f"Failed to start topics refresh: {e}")

        if st.sidebar.button("Open reports folder"):
            reports_dir = os.path.join(os.getcwd(), "reports")
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir, exist_ok=True)
            try:
                # Windows-specific open
                if os.name == 'nt':
                    os.startfile(reports_dir)
                else:
                    subprocess.Popen(["xdg-open", reports_dir])
            except Exception as e:
                st.sidebar.error(f"Can't open reports folder: {e}")
        st.sidebar.divider()
        # --- Export section ---
        st.sidebar.markdown("### Export")
        # Allow exporting current view as CSV
        if self.civic_df is not None:
            csv_bytes = self.civic_df.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button("Download current data (CSV)", data=csv_bytes, file_name="civic_data_export.csv", mime='text/csv')
        
        st.sidebar.divider()
        
        # System status
        st.sidebar.markdown("**System Status:**")

        if self.civic_df is not None:
            st.sidebar.success(f"Data: Loaded ({len(self.civic_df)} records)")
        else:
            st.sidebar.error("Data: No Data")

        if self.topic_results is not None:
            st.sidebar.success("Topics: Available")
        else:
            st.sidebar.warning("Topics: Pending")

        # Presentation toggle (persist in session state)
        if 'presentation_mode' not in st.session_state:
            st.session_state['presentation_mode'] = False
        st.session_state['presentation_mode'] = st.sidebar.checkbox("Presentation mode (large visuals)", value=st.session_state['presentation_mode'])
        self.presentation_mode = st.session_state['presentation_mode']
        
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
