#!/usr/bin/env python3
"""
Test Sangli Topic Dashboard Functionality
"""

import pandas as pd
import plotly.express as px

def test_sangli_topics():
    """Test if Sangli topic data works correctly"""
    print("🧪 Testing Sangli Topic Dashboard Functionality")
    print("=" * 50)
    
    # Load Sangli data
    try:
        df = pd.read_csv('data/processed/sangli_labeled.csv')
        print(f"✅ Loaded {len(df)} Sangli records")
        
        # Check if topic categories exist
        if 'topic_category' in df.columns:
            print("✅ Topic categories found in data")
            
            category_counts = df['topic_category'].value_counts()
            print("\n📊 Topic Distribution:")
            for category, count in category_counts.items():
                percentage = count/len(df)*100
                print(f"   📌 {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
            
            # Test plotly chart creation
            print("\n🔬 Testing Plotly Chart Creation...")
            try:
                fig_topics = px.bar(
                    x=category_counts.index,
                    y=category_counts.values,
                    title="Sangli Civic Issue Distribution",
                    labels={'x': 'Issue Category', 'y': 'Number of Records'},
                    color=category_counts.values,
                    color_continuous_scale='viridis'
                )
                fig_topics.update_layout(xaxis_tickangle=45)
                print("✅ Plotly chart creation successful")
                
                # Test category filtering
                print("\n🔍 Testing Category Filtering...")
                for category in category_counts.index[:2]:  # Test first 2 categories
                    category_data = df[df['topic_category'] == category]
                    sentiment_dist = category_data['label'].value_counts()
                    print(f"   {category}: {len(category_data)} records, sentiments: {dict(sentiment_dist)}")
                
                print("\n✅ All tests passed! Dashboard should work correctly.")
                return True
                
            except Exception as e:
                print(f"❌ Chart creation failed: {e}")
                return False
                
        else:
            print("❌ No topic categories found in data")
            print("   Run: python src/sangli_topic_model.py")
            print("   Then: python update_sangli_topics.py")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load Sangli data: {e}")
        return False

if __name__ == "__main__":
    success = test_sangli_topics()
    
    if success:
        print("\n🚀 Ready to launch dashboard:")
        print("   python -m streamlit run src/dashboard_simple.py")
    else:
        print("\n🔧 Fix the issues above before launching dashboard")
    
    exit(0 if success else 1)