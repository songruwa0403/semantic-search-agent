"""
🧪 Test Script for Trend Analysis System
Tests temporal pattern analysis on real fitness discussion data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-1-foundation'))

import pandas as pd
import numpy as np
from trend_analyzer import TrendAnalyzer
import json
from pathlib import Path
from datetime import datetime, timedelta

def create_sample_temporal_data():
    """Create sample data with timestamps for trend analysis"""
    # Load real fitness data
    data_path = "../../data/fitness_comments_clean.jsonl"
    if not os.path.exists(data_path):
        print("❌ Data file not found. Creating synthetic data.")
        return create_synthetic_data()
    
    df = pd.read_json(data_path, lines=True)
    
    # Add synthetic timestamps since original may not have them
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 8, 19)
    
    # Create timestamps with realistic patterns
    timestamps = []
    for i in range(len(df)):
        # More activity on weekdays, less on weekends
        base_date = start_date + timedelta(days=i * (end_date - start_date).days / len(df))
        
        # Add some randomness and weekly patterns
        if base_date.weekday() < 5:  # Weekday
            hour_offset = np.random.normal(14, 4)  # Peak around 2 PM
        else:  # Weekend
            hour_offset = np.random.normal(10, 3)  # Later morning
        
        hour_offset = max(0, min(23, hour_offset))
        final_timestamp = base_date + timedelta(hours=hour_offset)
        timestamps.append(final_timestamp.timestamp())
    
    df['created_utc'] = timestamps
    return df

def create_synthetic_data():
    """Create synthetic fitness discussion data for testing"""
    np.random.seed(42)
    
    # Generate 1000 synthetic discussions over 6 months
    start_date = datetime(2024, 1, 1)
    n_discussions = 1000
    
    data = []
    for i in range(n_discussions):
        # Create realistic temporal patterns
        days_offset = np.random.exponential(1) * 180  # 6 months
        days_offset = min(days_offset, 180)
        
        base_date = start_date + timedelta(days=days_offset)
        
        # Weekly patterns (more activity on certain days)
        if base_date.weekday() in [0, 2, 4]:  # Mon, Wed, Fri
            activity_multiplier = 1.5
        elif base_date.weekday() in [5, 6]:  # Weekend
            activity_multiplier = 0.7
        else:
            activity_multiplier = 1.0
        
        # Random discussion topics
        topics = [
            "My knee hurts after squats, any advice?",
            "Best protein powder recommendations?",
            "Home gym setup for beginners",
            "How to stay motivated for workouts?",
            "Deadlift form check please",
            "Weight loss plateau - need help",
            "Running program for fat loss",
            "Yoga for flexibility improvement"
        ]
        
        text = np.random.choice(topics)
        timestamp = base_date.timestamp()
        
        data.append({
            'text': text,
            'created_utc': timestamp
        })
    
    return pd.DataFrame(data)

def simulate_category_assignments():
    """Simulate category assignments for trend analysis"""
    categories = {
        'strength': ['squat', 'deadlift', 'bench', 'powerlifting'],
        'nutrition': ['protein', 'diet', 'supplements', 'cutting'],
        'equipment': ['home gym', 'barbell', 'equipment', 'gear'],
        'motivation': ['motivation', 'consistency', 'discipline'],
        'injury': ['pain', 'hurt', 'injury', 'recovery'],
        'beginner': ['beginner', 'start', 'new', 'first time'],
        'cardio': ['running', 'cardio', 'endurance', 'HIIT'],
        'flexibility': ['yoga', 'stretching', 'flexibility', 'mobility']
    }
    
    return categories

def main():
    """Run trend analysis evaluation"""
    print("📈 Testing Trend Analysis System")
    print("=" * 50)
    
    # Load or create sample data
    print("📂 Loading temporal fitness data...")
    df = create_sample_temporal_data()
    print(f"📊 Loaded {len(df)} discussions with timestamps")
    
    # Check data quality
    if 'created_utc' in df.columns:
        date_range = pd.to_datetime(df['created_utc'], unit='s')
        print(f"📅 Date range: {date_range.min()} to {date_range.max()}")
    
    # Initialize trend analyzer
    print("🔄 Initializing trend analyzer...")
    analyzer = TrendAnalyzer(
        time_column='created_utc',
        text_column='text',
        min_data_points=10
    )
    
    # Create sample category assignments
    categories = simulate_category_assignments()
    
    # Run trend analysis
    print("🔄 Analyzing temporal patterns...")
    cache_path = "trend_analysis_cache.pkl"
    results = analyzer.analyze_trends(
        df=df,
        categories=categories,
        time_granularity='daily',
        cache_path=cache_path
    )
    
    # Print results
    print("\n🎉 Trend Analysis Results:")
    print("=" * 50)
    
    stats = analyzer.get_trend_statistics()
    
    # Temporal patterns
    temporal = stats['temporal_summary']
    print(f"📊 Total Discussions: {temporal['total_discussions']}")
    print(f"📈 Trend Direction: {temporal['trend_direction']}")
    print(f"📊 Daily Average: {temporal['avg_daily_discussions']:.1f}")
    print(f"🔥 Peak Discussions: {temporal['peak_discussions']} on {temporal['peak_date'][:10]}")
    print(f"📈 Growth Rate: {temporal['growth_rate']:.1%}")
    
    # Seasonal patterns
    print(f"\n📅 Seasonal Patterns:")
    seasonal = stats['seasonal_summary']
    if 'day_of_week_pattern' in seasonal:
        dow_pattern = seasonal['day_of_week_pattern']
        best_day = max(dow_pattern, key=dow_pattern.get)
        print(f"   Best Day: {best_day} ({dow_pattern[best_day]:.1f} avg discussions)")
    
    if 'seasonality_strength' in seasonal:
        print(f"   Seasonality Strength: {seasonal['seasonality_strength']:.3f}")
    
    # Anomalies
    print(f"\n⚠️ Anomalies Detected: {stats['anomaly_count']}")
    if stats['high_severity_anomalies'] > 0:
        print(f"   High Severity: {stats['high_severity_anomalies']}")
    
    # Category trends
    print(f"\n🏷️ Category Trends: {stats['category_trends_count']} categories analyzed")
    
    # Key insights
    print(f"\n💡 Key Insights:")
    for i, insight in enumerate(stats['key_insights'], 1):
        print(f"   {i}. {insight}")
    
    # Forecast confidence
    print(f"\n🔮 Forecast Confidence: {stats['forecast_confidence']}")
    
    # Show example anomalies
    if results.anomalies:
        print(f"\n📊 Example Anomalies:")
        for anomaly in results.anomalies[:3]:
            print(f"   {anomaly['date'][:10]}: {anomaly['discussion_count']} discussions "
                  f"({anomaly['type']}, {anomaly['severity']} severity)")
    
    # Show category trends
    if results.category_trends:
        print(f"\n🚀 Growing Categories:")
        growing = [(cat, data['growth_rate']) for cat, data in results.category_trends.items() 
                  if data['trend_direction'] == 'increasing']
        growing.sort(key=lambda x: x[1], reverse=True)
        
        for cat, growth in growing[:5]:
            print(f"   {cat}: {growth:.1%} growth")
    
    # Save detailed results
    results_path = "trend_analysis_results.json"
    analyzer.save_results(results_path)
    print(f"\n💾 Detailed results saved to {results_path}")
    
    # Generate visualization
    print("📊 Generating visualization...")
    try:
        fig = analyzer.visualize_trends(
            save_path="trend_visualization.html",
            show_plot=False
        )
        print("✅ Visualization saved to trend_visualization.html")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    # Test forecasting
    if 'linear_trend' in results.forecasts:
        forecast = results.forecasts['linear_trend']
        if 'forecast_values' in forecast:
            print(f"\n🔮 7-Day Forecast:")
            for i, value in enumerate(forecast['forecast_values'][:3], 1):
                print(f"   Day +{i}: {value:.1f} discussions")
    
    print("\n🎯 Trend analysis evaluation complete!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
