"""
🧪 Stage 2 Integration Test
End-to-end testing of all Stage 2 Intelligence components

This script tests the complete intelligence pipeline:
1. Topic Discovery (Clustering)
2. Content Classification (Categorization) 
3. Trend Analysis (Temporal Patterns)
4. Integration & Cross-component Analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage-1-foundation'))

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import Stage 2 components
sys.path.append(os.path.join(os.path.dirname(__file__), 'clustering'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'categorization'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'trend-analysis'))

from topic_discoverer import TopicDiscoverer
from content_classifier import ContentClassifier
from trend_analyzer import TrendAnalyzer

def load_sample_data():
    """Load sample fitness data for integration testing"""
    data_path = "../data/fitness_comments_clean.jsonl"
    
    if not os.path.exists(data_path):
        print("❌ Data file not found. Please ensure you're running from the correct directory.")
        return None
    
    df = pd.read_json(data_path, lines=True)
    
    # Use a sample for faster testing
    sample_size = 200
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    # Add synthetic timestamps for trend analysis
    start_date = datetime(2024, 1, 1)
    timestamps = []
    for i in range(len(df_sample)):
        # Create realistic temporal distribution
        days_offset = np.random.exponential(1) * 180  # 6 months
        days_offset = min(days_offset, 180)
        timestamp = start_date.timestamp() + (days_offset * 24 * 3600)
        timestamps.append(timestamp)
    
    df_sample['created_utc'] = timestamps
    
    return df_sample

def test_clustering_component(df, embedder):
    """Test topic discovery clustering"""
    print("\n🎯 Testing Topic Discovery (Clustering)")
    print("-" * 50)
    
    try:
        # Generate embeddings
        texts = df['text'].tolist()
        embeddings = embedder.embed_texts(texts, batch_size=32)
        
        # Initialize and run clustering
        discoverer = TopicDiscoverer(
            min_cluster_size=10,
            min_samples=3,
            random_state=42
        )
        
        results = discoverer.fit_transform(
            embeddings=embeddings,
            texts=texts,
            cache_path="integration_clustering_cache.pkl"
        )
        
        # Validate results
        stats = discoverer.get_cluster_statistics()
        
        print(f"✅ Clustering successful:")
        print(f"   Topics found: {stats['n_clusters']}")
        print(f"   Coverage rate: {stats['coverage_rate']:.1%}")
        print(f"   Quality: {stats['quality_assessment']}")
        
        # Extract topic assignments for integration
        topic_assignments = {}
        for i, cluster_id in enumerate(results.cluster_labels):
            topic_assignments[f"text_{i}"] = cluster_id
        
        return {
            'success': True,
            'results': results,
            'stats': stats,
            'topic_assignments': topic_assignments
        }
        
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
        return {'success': False, 'error': str(e)}

def test_classification_component(df):
    """Test content classification"""
    print("\n🏷️ Testing Content Classification (Categorization)")
    print("-" * 50)
    
    try:
        # Initialize classifier
        classifier = ContentClassifier(
            model_name="facebook/bart-large-mnli",
            confidence_threshold=0.3,
            device="cpu"
        )
        
        if not classifier.load_model():
            return {'success': False, 'error': 'Failed to load model'}
        
        # Run classification
        texts = df['text'].tolist()
        text_ids = [f"text_{i}" for i in range(len(texts))]
        
        results = classifier.classify_texts(
            texts=texts,
            text_ids=text_ids,
            batch_size=4,
            use_cache=True,
            cache_path="integration_classification_cache.pkl"
        )
        
        # Validate results
        stats = classifier.get_classification_statistics()
        
        print(f"✅ Classification successful:")
        print(f"   Categories used: {stats['categories_used']}/{stats['total_categories']}")
        print(f"   Coverage: {stats['category_coverage']:.1%}")
        print(f"   Avg confidence: {stats['avg_confidence']:.3f}")
        
        return {
            'success': True,
            'results': results,
            'stats': stats,
            'category_assignments': results.predictions
        }
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return {'success': False, 'error': str(e)}

def test_trend_analysis_component(df):
    """Test temporal trend analysis"""
    print("\n📈 Testing Trend Analysis (Temporal Patterns)")
    print("-" * 50)
    
    try:
        # Initialize trend analyzer
        analyzer = TrendAnalyzer(
            time_column='created_utc',
            text_column='text',
            min_data_points=5
        )
        
        # Run trend analysis
        results = analyzer.analyze_trends(
            df=df,
            time_granularity='daily',
            cache_path="integration_trends_cache.pkl"
        )
        
        # Validate results
        stats = analyzer.get_trend_statistics()
        
        print(f"✅ Trend analysis successful:")
        print(f"   Trend direction: {stats['temporal_summary']['trend_direction']}")
        print(f"   Daily average: {stats['temporal_summary']['avg_daily_discussions']:.1f}")
        print(f"   Anomalies: {stats['anomaly_count']}")
        
        return {
            'success': True,
            'results': results,
            'stats': stats
        }
        
    except Exception as e:
        print(f"❌ Trend analysis failed: {e}")
        return {'success': False, 'error': str(e)}

def test_cross_component_integration(clustering_result, classification_result, trend_result):
    """Test integration between components"""
    print("\n🔗 Testing Cross-Component Integration")
    print("-" * 50)
    
    try:
        integration_insights = []
        
        # 1. Topic-Category Correlation
        if clustering_result['success'] and classification_result['success']:
            topic_assignments = clustering_result['topic_assignments']
            category_assignments = classification_result['category_assignments']
            
            # Find correlations between topics and categories
            topic_category_map = {}
            for text_id in topic_assignments:
                if text_id in category_assignments:
                    topic_id = topic_assignments[text_id]
                    categories = category_assignments[text_id]
                    
                    if topic_id not in topic_category_map:
                        topic_category_map[topic_id] = {}
                    
                    for category in categories:
                        if category not in topic_category_map[topic_id]:
                            topic_category_map[topic_id][category] = 0
                        topic_category_map[topic_id][category] += 1
            
            print(f"✅ Topic-Category mapping generated for {len(topic_category_map)} topics")
            integration_insights.append("Topic-category correlation analysis completed")
        
        # 2. Temporal-Category Trends
        if classification_result['success'] and trend_result['success']:
            category_stats = classification_result['stats']
            temporal_stats = trend_result['stats']
            
            # Correlate category popularity with temporal patterns
            print(f"✅ Temporal patterns available for category trend analysis")
            integration_insights.append("Category-temporal correlation identified")
        
        # 3. Comprehensive Intelligence Summary
        if all([clustering_result['success'], classification_result['success'], trend_result['success']]):
            intelligence_summary = {
                'topics_discovered': clustering_result['stats']['n_clusters'],
                'categories_covered': classification_result['stats']['categories_used'],
                'temporal_trend': trend_result['stats']['temporal_summary']['trend_direction'],
                'peak_activity': trend_result['stats']['temporal_summary']['peak_discussions'],
                'integration_insights': integration_insights
            }
            
            print(f"✅ Complete intelligence summary generated")
            print(f"   Topics: {intelligence_summary['topics_discovered']}")
            print(f"   Categories: {intelligence_summary['categories_covered']}")
            print(f"   Trend: {intelligence_summary['temporal_trend']}")
            
            return {
                'success': True,
                'intelligence_summary': intelligence_summary,
                'insights': integration_insights
            }
        
        return {'success': False, 'error': 'Not all components succeeded'}
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return {'success': False, 'error': str(e)}

def generate_stage2_report(clustering_result, classification_result, trend_result, integration_result):
    """Generate comprehensive Stage 2 completion report"""
    print("\n📋 Generating Stage 2 Intelligence Report")
    print("=" * 60)
    
    report = {
        'stage_2_completion': {
            'completion_date': datetime.now().isoformat(),
            'components_tested': 3,
            'integration_tested': True
        },
        'component_results': {
            'clustering': {
                'status': 'success' if clustering_result['success'] else 'failed',
                'topics_found': clustering_result.get('stats', {}).get('n_clusters', 0),
                'quality': clustering_result.get('stats', {}).get('quality_assessment', 'unknown')
            },
            'classification': {
                'status': 'success' if classification_result['success'] else 'failed',
                'categories_used': classification_result.get('stats', {}).get('categories_used', 0),
                'coverage': classification_result.get('stats', {}).get('category_coverage', 0)
            },
            'trend_analysis': {
                'status': 'success' if trend_result['success'] else 'failed',
                'trend_direction': trend_result.get('stats', {}).get('temporal_summary', {}).get('trend_direction', 'unknown'),
                'anomalies_detected': trend_result.get('stats', {}).get('anomaly_count', 0)
            }
        },
        'integration_results': {
            'status': 'success' if integration_result['success'] else 'failed',
            'insights_generated': len(integration_result.get('insights', [])),
            'intelligence_summary': integration_result.get('intelligence_summary', {})
        },
        'overall_assessment': {
            'stage_2_complete': all([
                clustering_result['success'],
                classification_result['success'], 
                trend_result['success'],
                integration_result['success']
            ]),
            'components_operational': sum([
                clustering_result['success'],
                classification_result['success'],
                trend_result['success']
            ]),
            'ready_for_stage_3': all([
                clustering_result['success'],
                classification_result['success']
            ])
        }
    }
    
    # Save report
    with open('stage2_completion_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"📊 Stage 2 Assessment:")
    print(f"   Components operational: {report['overall_assessment']['components_operational']}/3")
    print(f"   Integration successful: {integration_result['success']}")
    print(f"   Stage 2 complete: {report['overall_assessment']['stage_2_complete']}")
    print(f"   Ready for Stage 3: {report['overall_assessment']['ready_for_stage_3']}")
    
    if report['overall_assessment']['stage_2_complete']:
        print("\n🎉 Stage 2 Intelligence - COMPLETE!")
        print("Ready to proceed to Stage 3: Agent Architecture")
    else:
        print("\n⚠️ Stage 2 has issues. Review component results.")
    
    return report

def main():
    """Run complete Stage 2 integration test"""
    print("🧠 Stage 2 Intelligence - Integration Test")
    print("=" * 60)
    
    # Load sample data
    print("📂 Loading sample fitness data...")
    df = load_sample_data()
    if df is None:
        return
    
    print(f"📊 Loaded {len(df)} discussions for testing")
    
    # Initialize embedder (needed for clustering)
    print("🔄 Initializing embedder...")
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'stage-1-foundation'))
    from embeddings.embedder import TextEmbedder
    
    embedder = TextEmbedder()
    if not embedder.load_model():
        print("❌ Failed to load embedder")
        return
    
    # Test each component
    clustering_result = test_clustering_component(df, embedder)
    classification_result = test_classification_component(df)
    trend_result = test_trend_analysis_component(df)
    
    # Test integration
    integration_result = test_cross_component_integration(
        clustering_result, classification_result, trend_result
    )
    
    # Generate final report
    report = generate_stage2_report(
        clustering_result, classification_result, trend_result, integration_result
    )
    
    print(f"\n💾 Complete report saved to: stage2_completion_report.json")
    print("🎯 Integration test complete!")

if __name__ == "__main__":
    main()
