#!/usr/bin/env python3
"""
CivicPulse SANGLI-ONLY Twitter Data Collector
Generates civic tweets specifically about Sangli city issues
"""

import pandas as pd
from datetime import datetime, timedelta
import random
from utils import setup_data_directory

def generate_sangli_civic_tweets(limit=50):
    """Generate realistic Sangli-specific civic tweets"""
    print("🐦 Generating SANGLI-ONLY civic tweets...")
    
    # Sangli-specific locations
    sangli_locations = [
        "Market Yard", "Vishrambag", "Miraj Road", "Station Road", 
        "Ganpati Peth", "Bharat Nagar", "Sharad Nagar", "Sangam Bridge",
        "Sangli Railway Station", "Sangli Bus Stand", "Collector Office",
        "Sangli Court", "Sangli Hospital", "Miraj-Sangli Road"
    ]
    
    # Sangli-specific civic tweet templates
    sangli_civic_templates = [
        "Sangli needs better traffic management at {location}. Rush hour is becoming impossible! #SangliTraffic #CivicIssues",
        "Water supply issues in {area} Sangli for the 3rd day. When will Municipal Corporation fix this? #SangliWater #CivicIssues",
        "Garbage collection delayed in {locality} area of Sangli. Streets getting dirty. @SangliMunicipal please help #SangliClean",
        "Great to see new road construction in {place}, Sangli! Thanks to municipal corporation! #SangliDevelopment #Infrastructure",
        "Street lights not working in {area} Sangli since last week. Safety concern for residents #SangliStreetLights #Safety",
        "Pothole on {road}, Sangli needs immediate attention. Vehicle damage risk! #SangliRoads #RoadMaintenance",
        "Appreciate the cleanliness drive in {locality}, Sangli. Keep it up! #SwachhSangli #CleanSangli",
        "Bus service frequency should increase on Sangli-{route}. Long waiting times #SangliTransport #PublicBus",
        "Park maintenance in {area}, Sangli is excellent. Good job municipal team! #SangliParks #GreenSangli",
        "Drainage system in {locality}, Sangli needs upgrade before monsoon #SangliDrainage #Monsoon2024",
        "Sangli Municipal Corporation should improve waste management in {area} #SangliWaste #MunicipalServices",
        "Traffic signals at {location}, Sangli need proper timing adjustment #SangliTrafficSignals #RoadSafety",
        "Water quality in {area} Sangli has improved! Good work by authorities #SangliWaterQuality #GoodGovernance",
        "Request Sangli MC to install more dustbins in {locality} market area #SangliCleanliness #PublicFacilities",
        "Appreciate quick pothole repair work done in {place}, Sangli! Responsive governance #SangliRoadRepair #ThankYou",
        "Sangli needs better parking facilities near {location}. Traffic congestion increasing #SangliParking #UrbanPlanning",
        "{area} in Sangli requires street vendor regulation for smooth traffic flow #SangliTraffic #StreetVendors",
        "Excellent work by Sangli fire brigade during recent emergency in {locality}! #SangliFireBrigade #EmergencyServices",
        "Sangli hospital services in {area} need improvement. Healthcare is priority #SangliHealthcare #PublicHealth",
        "Happy to see solar street lights installed in {place}, Sangli. Great initiative! #SangliSolar #GreenEnergy"
    ]
    
    # Civic sentiment variations
    sentiments = ['positive', 'neutral', 'negative']
    sentiment_weights = [0.4, 0.3, 0.3]  # Balanced distribution
    
    tweets_data = []
    current_time = datetime.now()
    
    for i in range(limit):
        # Select template and location
        template = sangli_civic_templates[i % len(sangli_civic_templates)]
        location = random.choice(sangli_locations)
        
        # Generate tweet text
        tweet_text = template.format(
            location=location,
            area=location, 
            locality=location,
            place=location,
            road=f"{location} Road",
            route=location
        )
        
        # Assign sentiment
        sentiment = random.choices(sentiments, weights=sentiment_weights)[0]
        
        # Create tweet record
        tweet_time = current_time - timedelta(minutes=random.randint(0, 7*24*60))  # Random time in last week
        
        tweets_data.append({
            "id": f"sangli_civic_{i+1}",
            "text": tweet_text,
            "created_at": tweet_time.isoformat(),
            "source": "Sangli_Synthetic", 
            "fetched_at": datetime.now().isoformat(),
            "sentiment_hint": sentiment  # For validation
        })
    
    print(f"✅ Generated {len(tweets_data)} Sangli-specific civic tweets")
    return pd.DataFrame(tweets_data)

def collect_sangli_only_twitter(limit=100):
    """Collect ONLY Sangli-specific Twitter data"""
    print("🏛️ COLLECTING SANGLI-ONLY TWITTER DATA")
    print("🎯 Focus: Sangli Municipal Corporation civic issues only")
    print()
    
    setup_data_directory()
    
    # Generate Sangli-specific civic tweets
    df_tweets = generate_sangli_civic_tweets(limit)
    
    if not df_tweets.empty:
        # Save Sangli-only Twitter data
        output_file = "data/raw/sangli_only_twitter.csv"
        df_tweets.to_csv(output_file, index=False)
        
        print()
        print("=" * 50)
        print("🎯 SANGLI-ONLY TWITTER COLLECTION COMPLETE")
        print(f"📊 Total tweets generated: {len(df_tweets)}")
        print(f"📁 Saved to: {output_file}")
        
        # Show sentiment distribution
        if 'sentiment_hint' in df_tweets.columns:
            sentiment_dist = df_tweets['sentiment_hint'].value_counts()
            print("📈 Sentiment distribution:")
            for sentiment, count in sentiment_dist.items():
                print(f"   {sentiment}: {count} ({count/len(df_tweets)*100:.1f}%)")
        
        print("=" * 50)
        
        return df_tweets
    else:
        print("❌ No tweets generated")
        return pd.DataFrame()

if __name__ == "__main__":
    collect_sangli_only_twitter()