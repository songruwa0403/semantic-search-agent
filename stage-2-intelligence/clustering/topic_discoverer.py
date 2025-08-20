"""
🎯 Topic Discovery System
Advanced clustering pipeline for discovering hidden topics in fitness discussions

This module implements state-of-the-art topic discovery using:
- UMAP for dimensionality reduction and visualization
- HDBSCAN for density-based clustering 
- KeyBERT for extractive keyword labeling
- TF-IDF for cluster characterization

Author: ML Engineering Portfolio Project
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core ML libraries
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from keybert import KeyBERT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClusterResult:
    """Container for clustering results and metadata"""
    cluster_labels: np.ndarray
    topic_labels: List[str]
    topic_keywords: Dict[int, List[str]]
    cluster_sizes: Dict[int, int]
    silhouette_score: float
    calinski_harabasz_score: float
    umap_embeddings: np.ndarray
    cluster_summaries: Dict[int, str]

class TopicDiscoverer:
    """
    Advanced topic discovery system using UMAP + HDBSCAN + KeyBERT
    
    This class provides a complete pipeline for discovering meaningful topics
    in large collections of text documents using state-of-the-art clustering
    and keyword extraction techniques.
    """
    
    def __init__(self, 
                 min_cluster_size: int = 15,
                 min_samples: int = 5,
                 umap_n_neighbors: int = 15,
                 umap_n_components: int = 2,
                 umap_metric: str = 'cosine',
                 random_state: int = 42):
        """
        Initialize topic discovery system
        
        Args:
            min_cluster_size: Minimum size of clusters for HDBSCAN
            min_samples: Minimum samples in a neighborhood for core points
            umap_n_neighbors: Number of neighbors for UMAP
            umap_n_components: Number of components for UMAP reduction
            umap_metric: Distance metric for UMAP
            random_state: Random seed for reproducibility
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_n_components = umap_n_components
        self.umap_metric = umap_metric
        self.random_state = random_state
        
        # Initialize models
        self.umap_model = None
        self.hdbscan_model = None
        self.keybert_model = None
        self.tfidf_vectorizer = None
        
        # Results storage
        self.cluster_result = None
        self.is_fitted = False
    
    def _initialize_models(self):
        """Initialize UMAP, HDBSCAN, and KeyBERT models"""
        logger.info("🔄 Initializing clustering models...")
        
        # UMAP for dimensionality reduction
        self.umap_model = umap.UMAP(
            n_neighbors=self.umap_n_neighbors,
            n_components=self.umap_n_components,
            metric=self.umap_metric,
            random_state=self.random_state,
            verbose=True
        )
        
        # HDBSCAN for clustering
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        # KeyBERT for keyword extraction
        self.keybert_model = KeyBERT()
        
        # TF-IDF for cluster characterization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        
        logger.info("✅ Models initialized successfully")
    
    def fit_transform(self, 
                     embeddings: np.ndarray,
                     texts: List[str],
                     cache_path: Optional[str] = None) -> ClusterResult:
        """
        Discover topics in the provided embeddings and texts
        
        Args:
            embeddings: High-dimensional embeddings (e.g., 384-dim)
            texts: Original text documents
            cache_path: Optional path to save/load cached results
            
        Returns:
            ClusterResult with comprehensive topic information
        """
        logger.info(f"🎯 Starting topic discovery on {len(embeddings)} documents...")
        
        # Check for cached results
        if cache_path and Path(cache_path).exists():
            logger.info("📂 Loading cached clustering results...")
            return self._load_cached_results(cache_path)
        
        # Initialize models if not done
        if not self.umap_model:
            self._initialize_models()
        
        # Step 1: Dimensionality reduction with UMAP
        logger.info("🔄 Reducing dimensionality with UMAP...")
        umap_embeddings = self.umap_model.fit_transform(embeddings)
        logger.info(f"✅ UMAP complete: {embeddings.shape} → {umap_embeddings.shape}")
        
        # Step 2: Clustering with HDBSCAN
        logger.info("🔄 Discovering clusters with HDBSCAN...")
        cluster_labels = self.hdbscan_model.fit_predict(umap_embeddings)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        logger.info(f"✅ Clustering complete: {n_clusters} clusters found, {n_noise} noise points")
        
        # Step 3: Generate cluster statistics and quality metrics
        cluster_sizes = self._calculate_cluster_sizes(cluster_labels)
        silhouette = self._calculate_silhouette_score(umap_embeddings, cluster_labels)
        calinski = self._calculate_calinski_score(umap_embeddings, cluster_labels)
        
        # Step 4: Extract keywords and generate topic labels
        logger.info("🔄 Extracting keywords and generating topic labels...")
        topic_keywords = self._extract_cluster_keywords(texts, cluster_labels)
        topic_labels = self._generate_topic_labels(topic_keywords)
        cluster_summaries = self._generate_cluster_summaries(texts, cluster_labels, topic_keywords)
        
        # Create result object
        self.cluster_result = ClusterResult(
            cluster_labels=cluster_labels,
            topic_labels=topic_labels,
            topic_keywords=topic_keywords,
            cluster_sizes=cluster_sizes,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski,
            umap_embeddings=umap_embeddings,
            cluster_summaries=cluster_summaries
        )
        
        self.is_fitted = True
        
        # Cache results if path provided
        if cache_path:
            self._save_cached_results(cache_path)
        
        logger.info("🎉 Topic discovery complete!")
        return self.cluster_result
    
    def _calculate_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """Calculate the size of each cluster"""
        return dict(Counter(cluster_labels))
    
    def _calculate_silhouette_score(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        if len(set(cluster_labels)) < 2:
            return 0.0
        
        # Remove noise points for silhouette calculation
        mask = cluster_labels != -1
        if mask.sum() < 2:
            return 0.0
            
        return silhouette_score(embeddings[mask], cluster_labels[mask])
    
    def _calculate_calinski_score(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score for clustering quality"""
        if len(set(cluster_labels)) < 2:
            return 0.0
            
        # Remove noise points
        mask = cluster_labels != -1
        if mask.sum() < 2:
            return 0.0
            
        return calinski_harabasz_score(embeddings[mask], cluster_labels[mask])
    
    def _extract_cluster_keywords(self, texts: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
        """Extract representative keywords for each cluster"""
        topic_keywords = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            # Get texts for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
            
            if len(cluster_texts) < 3:  # Need minimum texts for keywords
                topic_keywords[cluster_id] = []
                continue
            
            # Try KeyBERT first, fallback to TF-IDF
            try:
                # Combine all texts in cluster
                combined_text = " ".join(cluster_texts)
                
                # Use KeyBERT with correct API
                keywords = self.keybert_model.extract_keywords(
                    combined_text, 
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_k=10
                )
                topic_keywords[cluster_id] = [kw[0] for kw in keywords]
                
            except Exception as e:
                logger.warning(f"KeyBERT failed for cluster {cluster_id}, using TF-IDF fallback: {e}")
                # Fallback to TF-IDF
                try:
                    # Fit TF-IDF on cluster texts
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
                    
                except Exception as e2:
                    logger.warning(f"Both KeyBERT and TF-IDF failed for cluster {cluster_id}: {e2}")
                    topic_keywords[cluster_id] = [f"topic_{cluster_id}"]
        
        return topic_keywords
    
    def _generate_topic_labels(self, topic_keywords: Dict[int, List[str]]) -> List[str]:
        """Generate human-readable topic labels from keywords"""
        topic_labels = []
        
        for cluster_id in sorted(topic_keywords.keys()):
            keywords = topic_keywords[cluster_id]
            
            if not keywords:
                topic_labels.append(f"Topic {cluster_id}")
                continue
            
            # Create label from top keywords
            if len(keywords) >= 2:
                label = f"{keywords[0].title()} & {keywords[1].title()}"
            else:
                label = keywords[0].title()
            
            topic_labels.append(label)
        
        return topic_labels
    
    def _generate_cluster_summaries(self, 
                                  texts: List[str], 
                                  cluster_labels: np.ndarray,
                                  topic_keywords: Dict[int, List[str]]) -> Dict[int, str]:
        """Generate descriptive summaries for each cluster"""
        summaries = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:
                continue
                
            cluster_mask = cluster_labels == cluster_id
            cluster_size = cluster_mask.sum()
            keywords = topic_keywords.get(cluster_id, [])
            
            if keywords:
                keyword_str = ", ".join(keywords[:5])
                summary = f"Cluster of {cluster_size} discussions about {keyword_str}"
            else:
                summary = f"Cluster of {cluster_size} related discussions"
            
            summaries[cluster_id] = summary
        
        return summaries
    
    def visualize_clusters(self, 
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> go.Figure:
        """
        Create interactive visualization of discovered topics
        
        Args:
            save_path: Optional path to save the plot
            show_plot: Whether to display the plot
            
        Returns:
            Plotly figure object
        """
        if not self.is_fitted:
            raise ValueError("Must fit the model before visualization")
        
        # Create scatter plot with cluster colors
        unique_labels = np.unique(self.cluster_result.cluster_labels)
        colors = px.colors.qualitative.Set3
        
        fig = go.Figure()
        
        for i, cluster_id in enumerate(unique_labels):
            mask = self.cluster_result.cluster_labels == cluster_id
            cluster_embeddings = self.cluster_result.umap_embeddings[mask]
            
            if cluster_id == -1:
                # Noise points in gray
                color = 'lightgray'
                name = 'Noise'
            else:
                color = colors[i % len(colors)]
                name = f"Topic {cluster_id}: {self.cluster_result.topic_labels[cluster_id]}"
            
            fig.add_trace(go.Scatter(
                x=cluster_embeddings[:, 0],
                y=cluster_embeddings[:, 1],
                mode='markers',
                marker=dict(color=color, size=6, opacity=0.7),
                name=name,
                text=[f"Cluster: {cluster_id}"] * len(cluster_embeddings),
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Topic Discovery: {len(unique_labels)-1} Topics Found",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            width=800,
            height=600,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"📊 Visualization saved to {save_path}")
        
        if show_plot:
            fig.show()
        
        return fig
    
    def get_cluster_statistics(self) -> Dict:
        """Get comprehensive statistics about discovered clusters"""
        if not self.is_fitted:
            raise ValueError("Must fit the model before getting statistics")
        
        result = self.cluster_result
        n_clusters = len(set(result.cluster_labels)) - (1 if -1 in result.cluster_labels else 0)
        n_noise = list(result.cluster_labels).count(-1)
        total_docs = len(result.cluster_labels)
        
        stats = {
            "total_documents": total_docs,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "coverage_rate": (total_docs - n_noise) / total_docs,
            "silhouette_score": result.silhouette_score,
            "calinski_harabasz_score": result.calinski_harabasz_score,
            "cluster_sizes": result.cluster_sizes,
            "topic_labels": result.topic_labels,
            "avg_cluster_size": np.mean([size for cluster_id, size in result.cluster_sizes.items() if cluster_id != -1]),
            "quality_assessment": self._assess_clustering_quality()
        }
        
        return stats
    
    def _assess_clustering_quality(self) -> str:
        """Provide qualitative assessment of clustering results"""
        silhouette = self.cluster_result.silhouette_score
        n_clusters = len(set(self.cluster_result.cluster_labels)) - (1 if -1 in self.cluster_result.cluster_labels else 0)
        coverage = (len(self.cluster_result.cluster_labels) - list(self.cluster_result.cluster_labels).count(-1)) / len(self.cluster_result.cluster_labels)
        
        if silhouette > 0.5 and n_clusters >= 5 and coverage > 0.8:
            return "Excellent"
        elif silhouette > 0.3 and n_clusters >= 3 and coverage > 0.6:
            return "Good"
        elif silhouette > 0.2 and n_clusters >= 2 and coverage > 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def save_results(self, save_path: str):
        """Save clustering results to file"""
        if not self.is_fitted:
            raise ValueError("Must fit the model before saving")
        
        # Convert numpy types to native Python types for JSON serialization
        results_dict = {
            "cluster_labels": self.cluster_result.cluster_labels.tolist(),
            "topic_labels": self.cluster_result.topic_labels,
            "topic_keywords": {str(k): v for k, v in self.cluster_result.topic_keywords.items()},
            "cluster_sizes": {str(k): int(v) for k, v in self.cluster_result.cluster_sizes.items()},
            "cluster_summaries": {str(k): v for k, v in self.cluster_result.cluster_summaries.items()},
            "statistics": self._convert_stats_for_json(self.get_cluster_statistics())
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"💾 Results saved to {save_path}")
    
    def _convert_stats_for_json(self, stats: Dict) -> Dict:
        """Convert statistics to JSON-serializable format"""
        json_stats = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                json_stats[key] = {str(k): (int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v) 
                                 for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                json_stats[key] = float(value)
            else:
                json_stats[key] = value
        return json_stats
    
    def _save_cached_results(self, cache_path: str):
        """Save complete clustering state for fast loading"""
        cache_data = {
            'cluster_result': self.cluster_result,
            'umap_model': self.umap_model,
            'hdbscan_model': self.hdbscan_model,
            'is_fitted': self.is_fitted
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"💾 Cached clustering state to {cache_path}")
    
    def _load_cached_results(self, cache_path: str) -> ClusterResult:
        """Load cached clustering results"""
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.cluster_result = cache_data['cluster_result']
        self.umap_model = cache_data['umap_model']
        self.hdbscan_model = cache_data['hdbscan_model']
        self.is_fitted = cache_data['is_fitted']
        
        logger.info("📂 Loaded cached clustering results")
        return self.cluster_result

# Example usage and testing
if __name__ == "__main__":
    # This section will be used for testing
    print("🎯 Topic Discovery System Initialized")
    print("Ready for fitness discussion clustering!")
