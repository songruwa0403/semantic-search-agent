"""
Generate proper keywords for the clustering results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'stage-1-foundation'))

import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def extract_keywords_manual(texts, cluster_labels):
    """Extract keywords manually using TF-IDF"""
    topic_keywords = {}
    
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:  # Skip noise
            continue
            
        # Get texts for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
        
        if len(cluster_texts) < 3:
            topic_keywords[cluster_id] = []
            continue
        
        try:
            # Use TF-IDF to extract keywords
            tfidf = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                max_df=0.8,
                min_df=2
            )
            tfidf_matrix = tfidf.fit_transform(cluster_texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Get top TF-IDF terms
            scores = tfidf_matrix.mean(axis=0).A1
            top_indices = scores.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            topic_keywords[cluster_id] = top_terms[:5]
            
        except Exception as e:
            print(f"Failed to extract keywords for cluster {cluster_id}: {e}")
            # Fallback: use most common words
            from collections import Counter
            import re
            
            # Simple word extraction
            all_words = []
            for text in cluster_texts[:10]:  # Sample first 10 texts
                words = re.findall(r'\b[a-z]{3,}\b', text.lower())
                all_words.extend(words)
            
            common_words = Counter(all_words).most_common(5)
            topic_keywords[cluster_id] = [word for word, count in common_words]
    
    return topic_keywords

def main():
    print("🔄 Extracting keywords for clustering results...")
    
    # Load data
    data_path = "../../data/fitness_comments_clean.jsonl"
    df = pd.read_json(data_path, lines=True)
    df_sample = df.sample(n=1000, random_state=42)
    texts = df_sample['text'].tolist()
    
    # Load cached clustering results
    with open('clustering_results_cache.pkl', 'rb') as f:
        cache_data = pickle.load(f)
    
    cluster_labels = cache_data['cluster_result'].cluster_labels
    
    # Extract keywords
    topic_keywords = extract_keywords_manual(texts, cluster_labels)
    
    # Print results
    print(f"\n🎯 Keywords extracted for {len(topic_keywords)} clusters:")
    for cluster_id, keywords in topic_keywords.items():
        size = (cluster_labels == cluster_id).sum()
        print(f"  Cluster {cluster_id} ({size} docs): {', '.join(keywords[:3])}")
    
    # Update the results file
    try:
        with open('clustering_evaluation_results.json', 'r') as f:
            results = json.load(f)
        
        # Update keywords
        results['topic_keywords'] = {str(k): v for k, v in topic_keywords.items()}
        
        # Generate better topic labels
        topic_labels = []
        for cluster_id in sorted([int(k) for k in results['topic_keywords'].keys()]):
            keywords = topic_keywords.get(cluster_id, [])
            if len(keywords) >= 2:
                label = f"{keywords[0].title()} & {keywords[1].title()}"
            elif len(keywords) == 1:
                label = keywords[0].title()
            else:
                label = f"Topic {cluster_id}"
            topic_labels.append(label)
        
        results['topic_labels'] = topic_labels
        
        # Save updated results
        with open('clustering_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("✅ Updated clustering results with keywords!")
        
    except Exception as e:
        print(f"❌ Failed to update results: {e}")

if __name__ == "__main__":
    main()
