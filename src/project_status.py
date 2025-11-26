#!/usr/bin/env python3
"""
CivicPulse Project Completion Status Check
Verifies all components are ready for final execution
"""

import os
import pandas as pd
import importlib
from datetime import datetime

def check_package(package_name):
    """Check if a package is installed"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def check_file_exists(filepath, description=""):
    """Check if file exists and get its size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        return True, size
    return False, 0

def main():
    print("🚀 CIVICPULSE PROJECT COMPLETION STATUS")
    print("=" * 50)
    
    # 1. Check Core Data
    print("\n📊 DATA STATUS:")
    print("-" * 30)
    
    civic_exists, civic_size = check_file_exists("data/processed/civic_labeled.csv", "Civic labeled data")
    if civic_exists:
        civic_df = pd.read_csv("data/processed/civic_labeled.csv")
        print(f"✅ Civic data: {len(civic_df)} records ({civic_size:,} bytes)")
    else:
        print("❌ Civic labeled data missing")
        
    external_exists, external_size = check_file_exists("data/processed/external/train_external.csv", "External training data")
    if external_exists:
        external_df = pd.read_csv("data/processed/external/train_external.csv")
        print(f"✅ External data: {len(external_df)} records ({external_size:,} bytes)")
    else:
        print("❌ External training data missing")
    
    # 2. Check Scripts
    print("\n🔧 SCRIPT STATUS:")
    print("-" * 30)
    
    scripts = [
        ("src/sentiment_infer.py", "ML Model Training"),
        ("src/topic_model.py", "Topic Modeling"),
        ("src/dashboard_app.py", "Interactive Dashboard"),
        ("src/preprocess.py", "Data Preprocessing"),
        ("src/generate_labels.py", "Label Generation")
    ]
    
    for script_path, description in scripts:
        exists, size = check_file_exists(script_path, description)
        if exists and size > 1000:  # Non-empty script
            print(f"✅ {description}: Ready ({size:,} bytes)")
        else:
            print(f"❌ {description}: Missing or empty")
    
    # 3. Check Models
    print("\n🤖 MODEL STATUS:")
    print("-" * 30)
    
    sentiment_model_exists, _ = check_file_exists("models/sentiment_model.pkl")
    if sentiment_model_exists:
        print("✅ Sentiment model: Trained and ready")
    else:
        print("⏳ Sentiment model: Needs training")
        
    topic_model_exists, _ = check_file_exists("models/topics/topic_results.pkl")
    if topic_model_exists:
        print("✅ Topic model: Results available")
    else:
        print("⏳ Topic model: Needs execution")
    
    # 4. Check Dependencies
    print("\n📦 DEPENDENCY STATUS:")
    print("-" * 30)
    
    core_packages = [
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning"),
        ("transformers", "BERT models"),
        ("torch", "Deep learning"),
        ("spacy", "NLP processing"),
        ("textblob", "Sentiment analysis"),
        ("vaderSentiment", "Sentiment analysis")
    ]
    
    optional_packages = [
        ("streamlit", "Dashboard"),
        ("plotly", "Visualization"),
        ("bertopic", "Topic modeling"),
        ("sentence_transformers", "Embeddings")
    ]
    
    print("Core packages:")
    missing_core = []
    for package, description in core_packages:
        if check_package(package):
            print(f"✅ {package}: {description}")
        else:
            print(f"❌ {package}: {description} - REQUIRED")
            missing_core.append(package)
    
    print("\nOptional packages:")
    missing_optional = []
    for package, description in optional_packages:
        if check_package(package):
            print(f"✅ {package}: {description}")
        else:
            print(f"⚠️  {package}: {description} - Optional")
            missing_optional.append(package)
    
    # 5. Automation Scripts
    print("\n🔄 AUTOMATION STATUS:")
    print("-" * 30)
    
    batch_files = [
        ("collect_data.bat", "Data Collection"),
        ("run_complete_pipeline.bat", "Complete Pipeline"),
        ("train_models.bat", "ML Training")
    ]
    
    for batch_file, description in batch_files:
        exists, _ = check_file_exists(batch_file)
        if exists:
            print(f"✅ {description}: {batch_file}")
        else:
            print(f"❌ {description}: Missing")
    
    # 6. Project Completion Assessment
    print("\n🎯 PROJECT COMPLETION ASSESSMENT:")
    print("=" * 50)
    
    total_score = 0
    max_score = 0
    
    # Data (30 points)
    max_score += 30
    if civic_exists and external_exists:
        total_score += 30
        print("✅ Data Collection: Complete (30/30)")
    elif civic_exists:
        total_score += 20
        print("⚠️  Data Collection: Partial (20/30) - Missing external data")
    else:
        print("❌ Data Collection: Incomplete (0/30)")
    
    # Scripts (25 points)
    max_score += 25
    script_score = sum(1 for script_path, _ in scripts if check_file_exists(script_path)[0])
    script_points = int((script_score / len(scripts)) * 25)
    total_score += script_points
    print(f"{'✅' if script_score == len(scripts) else '⚠️'} Scripts: {script_score}/{len(scripts)} ready ({script_points}/25)")
    
    # Models (25 points)
    max_score += 25
    model_score = 0
    if sentiment_model_exists:
        model_score += 20
    if topic_model_exists:
        model_score += 5
    total_score += model_score
    print(f"{'✅' if model_score >= 20 else '⏳'} Models: ({model_score}/25)")
    
    # Dependencies (20 points)
    max_score += 20
    if not missing_core:
        total_score += 15
        if not missing_optional:
            total_score += 5
            print("✅ Dependencies: All installed (20/20)")
        else:
            print("⚠️  Dependencies: Core ready (15/20)")
    else:
        print(f"❌ Dependencies: Missing core packages (0/20)")
    
    # Final Assessment
    completion_pct = (total_score / max_score) * 100
    print(f"\n🏆 OVERALL COMPLETION: {completion_pct:.1f}% ({total_score}/{max_score})")
    
    if completion_pct >= 85:
        print("🎉 PROJECT READY FOR FINAL EXECUTION!")
        print("   Run: train_models.bat")
    elif completion_pct >= 70:
        print("⚠️  PROJECT NEARLY COMPLETE")
        if missing_core:
            print(f"   Install missing packages: pip install {' '.join(missing_core)}")
        if not sentiment_model_exists:
            print("   Train sentiment model: python src/sentiment_infer.py")
    else:
        print("❌ PROJECT NEEDS MORE WORK")
        if not civic_exists:
            print("   Collect civic data: run_complete_pipeline.bat")
        if missing_core:
            print(f"   Install core packages: pip install {' '.join(missing_core)}")
    
    print("\n" + "=" * 50)
    print(f"Status check completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()