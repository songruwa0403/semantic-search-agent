"""
🧪 Test Script for Topic Discovery System
Tests clustering on real fitness discussion data

This script demonstrates the complete clustering pipeline and generates
evaluation results for the README documentation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-1-foundation'))

import pandas as pd
import numpy as np
from topic_discoverer import TopicDiscoverer
from embeddings.embedder import TextEmbedder
import json
from pathlib import Path

def main():
    """Run clustering evaluation on fitness data"""
    print("🎯 Testing Topic Discovery System")
    print("=" * 50)
    
    # Load fitness data
    data_path = "../../data/fitness_comments_clean.jsonl"
    print(f"📂 Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        print("❌ Data file not found. Please ensure you're running from the correct directory.")
        return
    
    df = pd.read_json(data_path, lines=True)
    print(f"📊 Loaded {len(df)} fitness discussions")
    
    # Use a sample for testing (full dataset takes ~30 minutes)
    sample_size = 1000
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    texts = df_sample['text'].tolist()
    print(f"🎯 Using sample of {len(texts)} discussions for clustering")
    
    # Generate embeddings
    print("🔄 Generating embeddings...")
    embedder = TextEmbedder()
    if not embedder.load_model():
        print("❌ Failed to load embedding model")
        return
    
    embeddings = embedder.embed_texts(texts, batch_size=50)
    print(f"✅ Generated {embeddings.shape} embeddings")
    
    # Run clustering
    print("🔄 Discovering topics...")
    discoverer = TopicDiscoverer(
        min_cluster_size=20,  # Larger clusters for demo
        min_samples=5,
        umap_n_neighbors=15,
        random_state=42
    )
    
    cache_path = "clustering_results_cache.pkl"
    results = discoverer.fit_transform(
        embeddings=embeddings,
        texts=texts,
        cache_path=cache_path
    )
    
    # Print results
    print("\n🎉 Topic Discovery Results:")
    print("=" * 50)
    
    stats = discoverer.get_cluster_statistics()
    print(f"📊 Total Documents: {stats['total_documents']}")
    print(f"🎯 Topics Found: {stats['n_clusters']}")
    print(f"📈 Coverage Rate: {stats['coverage_rate']:.1%}")
    print(f"🔍 Silhouette Score: {stats['silhouette_score']:.3f}")
    print(f"📐 Calinski-Harabasz Score: {stats['calinski_harabasz_score']:.1f}")
    print(f"⭐ Quality Assessment: {stats['quality_assessment']}")
    
    print("\n🏷️ Discovered Topics:")
    for i, (cluster_id, topic_label) in enumerate(zip(sorted(results.cluster_sizes.keys()), results.topic_labels)):
        if cluster_id == -1:
            continue
        size = results.cluster_sizes[cluster_id]
        keywords = ", ".join(results.topic_keywords.get(cluster_id, [])[:3])
        print(f"  {i+1}. {topic_label} ({size} discussions)")
        print(f"     Keywords: {keywords}")
    
    # Save detailed results
    results_path = "clustering_evaluation_results.json"
    discoverer.save_results(results_path)
    print(f"\n💾 Detailed results saved to {results_path}")
    
    # Generate visualization
    print("📊 Generating visualization...")
    try:
        fig = discoverer.visualize_clusters(
            save_path="topic_visualization.html",
            show_plot=False
        )
        print("✅ Visualization saved to topic_visualization.html")
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
    
    print("\n🎯 Clustering evaluation complete!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
