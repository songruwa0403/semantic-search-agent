"""
Semantic Search Magic Demo - Backend API
Simple Flask app to demonstrate the power of vector search vs keyword search
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import sys
import os
import re
import time
from typing import List, Dict, Any

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-1-foundation', 'embeddings'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-1-foundation', 'vector-search'))
from embedder import TextEmbedder
from vectorstore import VectorStore

# Add Stage 2 Intelligence paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-2-intelligence', 'clustering'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-2-intelligence', 'categorization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-2-intelligence', 'trend-analysis'))

from topic_discoverer import TopicDiscoverer
from content_classifier import ContentClassifier
from trend_analyzer import TrendAnalyzer

app = Flask(__name__)
CORS(app)

# Global variables for loaded systems
embedder = None
vector_store = None
df_data = None

# Stage 2 Intelligence components
topic_discoverer = None
content_classifier = None
trend_analyzer = None
intelligence_cache = {}

def initialize_intelligence_system():
    """Initialize Stage 2 Intelligence components"""
    global topic_discoverer, content_classifier, trend_analyzer, df_data, embedder, intelligence_cache
    
    print("🧠 Initializing Stage 2 Intelligence...")
    
    # Initialize topic discoverer (clustering)
    print("🎯 Loading topic discovery...")
    topic_discoverer = TopicDiscoverer(min_cluster_size=15, min_samples=5, random_state=42)
    
    # Initialize content classifier
    print("🏷️ Loading content classifier...")
    content_classifier = ContentClassifier(
        model_name="facebook/bart-large-mnli",
        confidence_threshold=0.3,
        device="cpu"
    )
    
    # Initialize trend analyzer
    print("📈 Loading trend analyzer...")
    trend_analyzer = TrendAnalyzer(
        time_column='created_utc',
        text_column='text',
        min_data_points=10
    )
    
    # Run initial intelligence analysis (cached)
    cache_file = "../../data/stage2_intelligence_cache.pkl"
    if os.path.exists(cache_file):
        print("📂 Loading cached intelligence results...")
        with open(cache_file, 'rb') as f:
            intelligence_cache = pickle.load(f)
        print("✅ Intelligence cache loaded")
    else:
        print("🔄 Running initial intelligence analysis...")
        try:
            # Sample data for demo (use smaller subset for performance)
            sample_size = 1000
            df_sample = df_data.sample(n=min(sample_size, len(df_data)), random_state=42)
            
            # Generate embeddings for clustering
            texts = df_sample['text'].tolist()
            embeddings = embedder.embed_texts(texts, batch_size=32)
            
            # Run topic discovery
            topic_results = topic_discoverer.fit_transform(embeddings, texts)
            
            # Load classifier model
            if content_classifier.load_model():
                # Run classification
                classification_results = content_classifier.classify_texts(
                    texts=texts,
                    text_ids=[f"text_{i}" for i in range(len(texts))],
                    batch_size=8
                )
            else:
                classification_results = None
            
            # Add synthetic timestamps for trend analysis
            import datetime
            start_date = datetime.datetime(2024, 1, 1)
            timestamps = []
            for i in range(len(df_sample)):
                days_offset = np.random.exponential(1) * 180
                days_offset = min(days_offset, 180)
                timestamp = start_date.timestamp() + (days_offset * 24 * 3600)
                timestamps.append(timestamp)
            
            df_sample['created_utc'] = timestamps
            
            # Run trend analysis
            trend_results = trend_analyzer.analyze_trends(df_sample, time_granularity='daily')
            
            # Cache results
            intelligence_cache = {
                'topics': topic_discoverer.get_cluster_statistics(),
                'topic_results': topic_results,
                'classification': classification_results,
                'classification_stats': content_classifier.get_classification_statistics() if classification_results else None,
                'trends': trend_analyzer.get_trend_statistics(),
                'sample_size': len(df_sample),
                'generated_at': datetime.datetime.now().isoformat()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(intelligence_cache, f)
            
            print("✅ Intelligence analysis complete and cached")
            
        except Exception as e:
            print(f"⚠️ Intelligence initialization failed: {e}")
            intelligence_cache = {'error': str(e)}

def initialize_search_system():
    """Initialize the semantic search system"""
    global embedder, vector_store, df_data
    
    print("🔄 Initializing Semantic Search Magic Demo...")
    
    # Load embedder
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    if not embedder.load_model():
        raise Exception("Failed to load embedding model")
    
    # Load data (go up one directory to find data folder)
    print("🔄 Loading full fitness dataset...")
    df_data = pd.read_json("../../data/fitness_comments_clean.jsonl", lines=True)
    print(f"📊 Total dataset size: {len(df_data)} fitness discussions")
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dim=384, index_type="flat", metric="cosine")
    vector_store.create_index()
    
    # Check for cached embeddings first
    cache_file = "../../data/full_dataset_embeddings.pkl"
    
    if os.path.exists(cache_file):
        print("📂 Loading cached embeddings...")
        try:
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                embeddings = cached_data['embeddings']
                cached_texts = cached_data['text_hashes']
                
            # Verify cache is still valid (same number of texts)
            current_hashes = [hash(text) for text in df_data['text'].tolist()]
            if len(current_hashes) == len(cached_texts):
                print(f"✅ Loaded {len(embeddings)} cached embeddings")
            else:
                print("⚠️ Cache size mismatch, regenerating embeddings...")
                raise ValueError("Cache mismatch")
        except Exception as e:
            print(f"⚠️ Cache loading failed ({e}), regenerating embeddings...")
            embeddings = None
    else:
        embeddings = None
    
    if embeddings is None:
        # Generate embeddings for all texts
        print(f"🔄 Generating embeddings for {len(df_data)} texts...")
        print("⏳ This may take 10-15 minutes for the full dataset...")
        
        texts = df_data['text'].tolist()
        embeddings = embedder.embed_texts(texts)  # Use batch processing
        
        # Cache the embeddings
        print("💾 Caching embeddings for future use...")
        cache_data = {
            'embeddings': embeddings,
            'text_hashes': [hash(text) for text in texts],
            'model_name': embedder.model_name,
            'timestamp': time.time()
        }
        
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ Generated and cached {len(embeddings)} embeddings")
    
    embeddings = np.array(embeddings)
    
    # Add to vector store
    vector_store.add_vectors(
        vectors=embeddings,
        metadata=df_data.to_dict('records'),
        ids=df_data['chunk_id'].tolist()
    )
    
    print("✅ Semantic Search System Ready!")

def keyword_search(query: str, k: int = 5) -> List[Dict]:
    """Simple keyword search for comparison"""
    query_words = set(query.lower().split())
    
    results = []
    for idx, row in df_data.iterrows():
        text = row['text'].lower()
        
        # Count keyword matches
        matches = sum(1 for word in query_words if word in text)
        
        if matches > 0:
            # Simple scoring based on match count and text length
            score = matches / len(query_words)  # Percentage of query words found
            
            results.append({
                'index': idx,
                'score': score,
                'matches': matches,
                'metadata': row.to_dict(),
                'match_explanation': f"Found {matches}/{len(query_words)} keywords"
            })
    
    # Sort by score and return top k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:k]

def explain_semantic_match(query: str, result_text: str, similarity_score: float) -> str:
    """Generate human-readable explanation for semantic match"""
    
    # Extract key concepts from query and result
    query_lower = query.lower()
    text_lower = result_text.lower()
    
    explanations = []
    
    # Concept mapping for common fitness terms
    concept_maps = {
        'pain': ['hurt', 'discomfort', 'ache', 'sore', 'injury'],
        'weight loss': ['lose weight', 'fat loss', 'cutting', 'diet', 'calories'],
        'strength': ['strong', 'power', 'muscle', 'lifting', 'gains'],
        'cardio': ['running', 'endurance', 'aerobic', 'heart rate'],
        'beginner': ['new', 'start', 'first time', 'novice'],
        'knee': ['joint', 'leg', 'lower body'],
        'squat': ['exercise', 'movement', 'lift'],
        'diet': ['nutrition', 'food', 'eating', 'meal'],
        'motivation': ['discipline', 'commitment', 'struggle', 'mindset']
    }
    
    # Find conceptual matches
    for concept, related_terms in concept_maps.items():
        if concept in query_lower:
            for term in related_terms:
                if term in text_lower:
                    explanations.append(f"'{concept}' relates to '{term}' in the discussion")
                    break
    
    # Score-based explanations
    if similarity_score > 0.7:
        confidence = "Very high similarity"
        reason = "The discussion covers very similar topics to your query"
    elif similarity_score > 0.5:
        confidence = "High similarity"  
        reason = "The discussion covers related topics to your query"
    elif similarity_score > 0.3:
        confidence = "Moderate similarity"
        reason = "The discussion has some relevant content to your query"
    else:
        confidence = "Low similarity"
        reason = "The discussion has limited relevance to your query"
    
    if explanations:
        return f"{confidence}: {reason}. " + " | ".join(explanations[:2])
    else:
        return f"{confidence}: {reason} based on semantic understanding of fitness concepts."

@app.route('/')
def index():
    """Main demo page"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests"""
    data = request.json
    query = data.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Please enter a search query'})
    
    try:
        start_time = time.time()
        
        # Semantic search
        semantic_results = vector_store.search(
            query_vector=embedder.embed_texts([query])[0],
            k=5
        )
        
        semantic_time = time.time() - start_time
        
        # Keyword search
        start_time = time.time()
        keyword_results = keyword_search(query, k=5)
        keyword_time = time.time() - start_time
        
        # Format semantic results with explanations
        formatted_semantic = []
        for result in semantic_results:
            formatted_semantic.append({
                'text': result['metadata']['text'],
                'score': result['score'],
                'pain_score': result['metadata'].get('pain_point_score', 0),
                'quality_score': result['metadata'].get('quality_score', 0),
                'word_count': result['metadata'].get('word_count', 0),
                'explanation': explain_semantic_match(
                    query, 
                    result['metadata']['text'], 
                    result['score']
                ),
                'confidence': 'High' if result['score'] > 0.5 else 'Medium' if result['score'] > 0.3 else 'Low'
            })
        
        # Format keyword results
        formatted_keyword = []
        for result in keyword_results:
            formatted_keyword.append({
                'text': result['metadata']['text'],
                'score': result['score'],
                'pain_score': result['metadata'].get('pain_point_score', 0),
                'quality_score': result['metadata'].get('quality_score', 0),
                'word_count': result['metadata'].get('word_count', 0),
                'explanation': result['match_explanation'],
                'confidence': 'High' if result['score'] > 0.7 else 'Medium' if result['score'] > 0.4 else 'Low'
            })
        
        return jsonify({
            'query': query,
            'semantic_results': formatted_semantic,
            'keyword_results': formatted_keyword,
            'semantic_time': f"{semantic_time*1000:.1f}ms",
            'keyword_time': f"{keyword_time*1000:.1f}ms",
            'total_documents': len(df_data)
        })
        
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'})

@app.route('/suggestions')
def suggestions():
    """Get sample search suggestions"""
    suggestions = [
        {
            'query': 'knee pain during squats',
            'description': 'Find discussions about joint pain during exercises'
        },
        {
            'query': 'struggling to lose weight',
            'description': 'Discover weight loss challenges and plateaus'
        },
        {
            'query': 'beginner workout anxiety',
            'description': 'Find stories about gym intimidation and starting out'
        },
        {
            'query': 'muscle building plateau',
            'description': 'Explore discussions about strength training stalls'
        },
        {
            'query': 'post-workout fatigue',
            'description': 'Learn about recovery and energy issues'
        },
        {
            'query': 'diet motivation struggles',
            'description': 'Find discussions about nutrition discipline'
        }
    ]
    
    return jsonify({'suggestions': suggestions})

@app.route('/stats')
def stats():
    """Get system statistics"""
    if vector_store:
        stats = vector_store.get_stats()
        return jsonify({
            'total_documents': stats['total_vectors'],
            'embedding_dimension': stats['embedding_dim'],
            'index_type': stats['index_type'],
            'memory_usage': f"{stats['memory_usage_mb']:.1f} MB"
        })
    return jsonify({'error': 'System not initialized'})

@app.route('/intelligence/topics')
def get_topics():
    """Get discovered fitness topics from clustering"""
    if 'topics' in intelligence_cache and 'topic_results' in intelligence_cache:
        topics_stats = intelligence_cache['topics']
        topic_results = intelligence_cache['topic_results']
        
        # Format topic information
        topics_info = []
        for i, topic_label in enumerate(topic_results.topic_labels):
            cluster_id = i
            keywords = topic_results.topic_keywords.get(cluster_id, [])
            size = topic_results.cluster_sizes.get(cluster_id, 0)
            summary = topic_results.cluster_summaries.get(cluster_id, "")
            
            topics_info.append({
                'id': cluster_id,
                'label': topic_label,
                'keywords': keywords[:5],  # Top 5 keywords
                'size': size,
                'summary': summary,
                'percentage': (size / topics_stats['total_documents'] * 100) if topics_stats['total_documents'] > 0 else 0
            })
        
        return jsonify({
            'topics': topics_info,
            'summary': {
                'total_topics': topics_stats['n_clusters'],
                'coverage_rate': topics_stats['coverage_rate'],
                'quality_assessment': topics_stats['quality_assessment'],
                'sample_size': intelligence_cache['sample_size']
            }
        })
    
    return jsonify({'error': 'Topics not available'})

@app.route('/intelligence/categories')
def get_categories():
    """Get content classification results"""
    if 'classification_stats' in intelligence_cache and intelligence_cache['classification_stats']:
        stats = intelligence_cache['classification_stats']
        
        # Get top categories
        top_categories = []
        if 'top_categories' in stats:
            for category, count in stats['top_categories']:
                percentage = (count / stats['total_texts'] * 100) if stats['total_texts'] > 0 else 0
                top_categories.append({
                    'name': category,
                    'count': count,
                    'percentage': percentage
                })
        
        return jsonify({
            'categories': top_categories,
            'summary': {
                'total_categories': stats['total_categories'],
                'categories_used': stats['categories_used'],
                'coverage': stats['category_coverage'],
                'avg_confidence': stats['avg_confidence'],
                'multi_label_rate': stats['multi_label_rate'],
                'sample_size': intelligence_cache['sample_size']
            }
        })
    
    return jsonify({'error': 'Categories not available'})

@app.route('/intelligence/trends')
def get_trends():
    """Get temporal trend analysis"""
    if 'trends' in intelligence_cache:
        trends_stats = intelligence_cache['trends']
        temporal_summary = trends_stats.get('temporal_summary', {})
        seasonal_summary = trends_stats.get('seasonal_summary', {})
        
        # Format trend information
        trend_info = {
            'temporal_patterns': {
                'trend_direction': temporal_summary.get('trend_direction', 'unknown'),
                'daily_average': temporal_summary.get('avg_daily_discussions', 0),
                'peak_discussions': temporal_summary.get('peak_discussions', 0),
                'peak_date': temporal_summary.get('peak_date', ''),
                'growth_rate': temporal_summary.get('growth_rate', 0)
            },
            'seasonal_patterns': seasonal_summary,
            'insights': trends_stats.get('key_insights', []),
            'anomalies': trends_stats.get('anomaly_count', 0),
            'forecast_confidence': trends_stats.get('forecast_confidence', 'unknown')
        }
        
        return jsonify({
            'trends': trend_info,
            'summary': {
                'sample_size': intelligence_cache['sample_size'],
                'generated_at': intelligence_cache.get('generated_at', ''),
                'insights_count': len(trends_stats.get('key_insights', []))
            }
        })
    
    return jsonify({'error': 'Trends not available'})

@app.route('/intelligence/summary')
def get_intelligence_summary():
    """Get complete intelligence overview"""
    if 'error' in intelligence_cache:
        return jsonify({'error': intelligence_cache['error']})
    
    summary = {}
    
    # Topics summary
    if 'topics' in intelligence_cache:
        topics_stats = intelligence_cache['topics']
        summary['topics'] = {
            'count': topics_stats['n_clusters'],
            'quality': topics_stats['quality_assessment'],
            'coverage': topics_stats['coverage_rate']
        }
    
    # Categories summary
    if 'classification_stats' in intelligence_cache and intelligence_cache['classification_stats']:
        stats = intelligence_cache['classification_stats']
        summary['categories'] = {
            'total_available': stats['total_categories'],
            'used': stats['categories_used'],
            'coverage': stats['category_coverage'],
            'confidence': stats['avg_confidence']
        }
    
    # Trends summary
    if 'trends' in intelligence_cache:
        trends_stats = intelligence_cache['trends']
        temporal = trends_stats.get('temporal_summary', {})
        summary['trends'] = {
            'direction': temporal.get('trend_direction', 'unknown'),
            'daily_avg': temporal.get('avg_daily_discussions', 0),
            'anomalies': trends_stats.get('anomaly_count', 0),
            'insights': len(trends_stats.get('key_insights', []))
        }
    
    summary['meta'] = {
        'sample_size': intelligence_cache.get('sample_size', 0),
        'generated_at': intelligence_cache.get('generated_at', ''),
        'status': 'ready' if summary else 'loading'
    }
    
    return jsonify(summary)

if __name__ == '__main__':
    # Initialize the search system
    initialize_search_system()
    initialize_intelligence_system()
    
    # Run the demo
    print("\n🌟 Semantic Search Magic Demo with AI Intelligence is ready!")
    print("🌐 Open your browser to: http://localhost:5001")
    print("🔍 Try searching for natural language queries like:")
    print("   • 'knee pain during squats'")
    print("   • 'struggling to lose weight'") 
    print("   • 'beginner workout anxiety'")
    print("\n✨ Experience the magic of semantic search!")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
