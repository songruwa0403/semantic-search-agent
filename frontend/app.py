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

from stage3_embeddings.embedder import TextEmbedder
from stage4_vector_search.vectorstore import VectorStore

app = Flask(__name__)
CORS(app)

# Global variables for loaded systems
embedder = None
vector_store = None
df_data = None

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
    df_data = pd.read_json("../data/fitness_comments_clean.jsonl", lines=True)
    print(f"📊 Total dataset size: {len(df_data)} fitness discussions")
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dim=384, index_type="flat", metric="cosine")
    vector_store.create_index()
    
    # Check for cached embeddings first
    cache_file = "../data/full_dataset_embeddings.pkl"
    
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

if __name__ == '__main__':
    # Initialize the search system
    initialize_search_system()
    
    # Run the demo
    print("\n🌟 Semantic Search Magic Demo is ready!")
    print("🌐 Open your browser to: http://localhost:5001")
    print("🔍 Try searching for natural language queries like:")
    print("   • 'knee pain during squats'")
    print("   • 'struggling to lose weight'") 
    print("   • 'beginner workout anxiety'")
    print("\n✨ Experience the magic of semantic search!")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
