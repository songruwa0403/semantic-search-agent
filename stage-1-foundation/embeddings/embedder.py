"""
Chunk 3 — Semantic Embeddings
Create and manage text embeddings for semantic search

Learning Objectives:
- Understand vector representations of text
- Learn about pre-trained embedding models
- Practice computing cosine similarity
- Compare different embedding strategies
- Understand embedding dimensions and their trade-offs

TODO for you to implement:
1. Initialize sentence-transformers model
2. Implement batch embedding generation
3. Add similarity computation functions
4. Implement embedding caching for efficiency
5. Add support for different embedding models (OpenAI, etc.)
6. Create embedding quality evaluation methods
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple, Union
import pickle
import os
import json
from pathlib import Path
import time


class TextEmbedder:
    """Generate and manage text embeddings for semantic search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a specific model
        
        Args:
            model_name: Name of the sentence-transformers model to use
                       Popular options:
                       - "all-MiniLM-L6-v2" (384 dim, fast, good quality)
                       - "all-mpnet-base-v2" (768 dim, slower, better quality)
                       - "paraphrase-multilingual-MiniLM-L12-v2" (384 dim, multilingual)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def load_model(self) -> bool:
        """
        Load the sentence transformer model
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"🔄 Loading embedding model: {self.model_name}")
            
            # Initialize the sentence transformer model
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension by testing with sample text
            sample_embedding = self.model.encode("test")
            self.embedding_dim = len(sample_embedding)
            
            print(f"✅ Model loaded successfully")
            print(f"📏 Embedding dimension: {self.embedding_dim}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def embed_texts(self, 
                   texts: List[str], 
                   batch_size: int = 32,
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (to manage memory)
            show_progress: Whether to show progress bar
            
        Returns:
            np.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return np.array([])
        
        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        start_time = time.time()
        
        try:
            # Generate embeddings using sentence-transformers
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better cosine similarity
            )
            
            elapsed_time = time.time() - start_time
            print(f"✅ Generated {len(embeddings)} embeddings in {elapsed_time:.2f}s")
            print(f"📏 Embedding shape: {embeddings.shape}")
            
            return embeddings
            
        except Exception as e:
            print(f"❌ Error generating embeddings: {e}")
            # Return empty array with correct dimensions
            return np.array([]).reshape(0, self.embedding_dim)
    
    def embed_dataframe(self, 
                       df: pd.DataFrame, 
                       text_column: str = 'text',
                       cache_file: str = None) -> pd.DataFrame:
        """
        Generate embeddings for texts in a DataFrame
        
        Args:
            df: DataFrame containing texts
            text_column: Name of the column containing text
            cache_file: Optional file to cache embeddings
            
        Returns:
            pd.DataFrame: DataFrame with embeddings added
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        # Check for cached embeddings
        if cache_file and os.path.exists(cache_file):
            print(f"📁 Loading cached embeddings from {cache_file}")
            return self._load_cached_embeddings(df, cache_file)
        
        # Generate new embeddings
        texts = df[text_column].tolist()
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to DataFrame
        df_with_embeddings = df.copy()
        df_with_embeddings['embedding'] = embeddings.tolist()
        
        # Cache embeddings if requested
        if cache_file:
            self._cache_embeddings(df_with_embeddings, cache_file)
        
        return df_with_embeddings
    
    def compute_similarity(self, 
                          embedding1: Union[np.ndarray, List[float]], 
                          embedding2: Union[np.ndarray, List[float]]) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        # Convert to numpy arrays if needed
        emb1 = np.array(embedding1).reshape(1, -1)
        emb2 = np.array(embedding2).reshape(1, -1)
        
        # Compute cosine similarity using sklearn
        similarity = cosine_similarity(emb1, emb2)[0, 0]
        
        return float(similarity)
    
    def find_most_similar(self, 
                         query_embedding: Union[np.ndarray, List[float]],
                         candidate_embeddings: np.ndarray,
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            top_k: Number of top similar embeddings to return
            
        Returns:
            List[Tuple[int, float]]: List of (index, similarity_score) tuples
        """
        if len(candidate_embeddings) == 0:
            return []
        
        # Convert query to proper shape
        query = np.array(query_embedding).reshape(1, -1)
        
        # Compute similarities between query and all candidates
        similarities = cosine_similarity(query, candidate_embeddings)[0]
        
        # Get top_k most similar indices
        top_k = min(top_k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return list of (index, similarity_score) tuples
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results
    
    def compare_texts(self, 
                     text1: str, 
                     text2: str) -> float:
        """
        Compare similarity between two texts directly
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score
        """
        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Embed both texts
        embeddings = self.embed_texts([text1, text2])
        
        # Compute and return similarity
        return self.compute_similarity(embeddings[0], embeddings[1])
    
    def semantic_search(self, 
                       query: str,
                       corpus_df: pd.DataFrame,
                       text_column: str = 'text',
                       top_k: int = 10) -> pd.DataFrame:
        """
        Perform semantic search on a corpus
        
        Args:
            query: Search query
            corpus_df: DataFrame with text corpus and embeddings
            text_column: Name of the text column
            top_k: Number of results to return
            
        Returns:
            pd.DataFrame: Top-k most similar texts with similarity scores
        """
        if 'embedding' not in corpus_df.columns:
            raise ValueError("Corpus must have 'embedding' column. Run embed_dataframe() first.")
        
        # Embed the query
        query_embedding = self.embed_texts([query])[0]
        
        # Get corpus embeddings
        corpus_embeddings = np.array(corpus_df['embedding'].tolist())
        
        # Find most similar
        similar_indices = self.find_most_similar(
            query_embedding, 
            corpus_embeddings, 
            top_k
        )
        
        # Create results DataFrame
        results = []
        for idx, score in similar_indices:
            result = corpus_df.iloc[idx].copy()
            result['similarity_score'] = score
            result['rank'] = len(results) + 1
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_embeddings(self, 
                           df: pd.DataFrame,
                           sample_queries: List[str] = None) -> Dict:
        """
        Evaluate embedding quality with various metrics
        
        Args:
            df: DataFrame with embeddings
            sample_queries: Optional list of queries for evaluation
            
        Returns:
            Dict: Evaluation metrics
        """
        if 'embedding' not in df.columns:
            raise ValueError("DataFrame must have 'embedding' column")
        
        print("🔄 Evaluating embedding quality...")
        metrics = {}
        
        embeddings = np.array(df['embedding'].tolist())
        
        # Basic statistics
        metrics['num_embeddings'] = len(embeddings)
        metrics['embedding_dim'] = embeddings.shape[1] if len(embeddings) > 0 else 0
        
        if len(embeddings) > 0:
            # Embedding magnitude statistics
            magnitudes = np.linalg.norm(embeddings, axis=1)
            metrics['mean_magnitude'] = float(np.mean(magnitudes))
            metrics['std_magnitude'] = float(np.std(magnitudes))
            metrics['min_magnitude'] = float(np.min(magnitudes))
            metrics['max_magnitude'] = float(np.max(magnitudes))
            
            # Similarity distribution (sample random pairs)
            if len(embeddings) > 1:
                n_samples = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
                sample_indices = np.random.choice(len(embeddings), size=(n_samples, 2), replace=True)
                
                similarities = []
                for i, j in sample_indices:
                    if i != j:
                        sim = self.compute_similarity(embeddings[i], embeddings[j])
                        similarities.append(sim)
                
                if similarities:
                    metrics['mean_similarity'] = float(np.mean(similarities))
                    metrics['std_similarity'] = float(np.std(similarities))
                    metrics['min_similarity'] = float(np.min(similarities))
                    metrics['max_similarity'] = float(np.max(similarities))
        
        # Query-based evaluation if sample queries provided
        if sample_queries and len(embeddings) > 0:
            query_metrics = []
            
            for query in sample_queries:
                try:
                    results = self.semantic_search(query, df, top_k=5)
                    if not results.empty:
                        # Calculate metrics for this query
                        top_scores = results['similarity_score'].tolist()
                        query_metrics.append({
                            'query': query,
                            'top_score': max(top_scores),
                            'avg_top5_score': np.mean(top_scores),
                            'score_range': max(top_scores) - min(top_scores)
                        })
                except Exception as e:
                    print(f"⚠️ Error evaluating query '{query}': {e}")
            
            if query_metrics:
                metrics['query_evaluation'] = query_metrics
                metrics['avg_top_score'] = float(np.mean([q['top_score'] for q in query_metrics]))
                metrics['avg_score_range'] = float(np.mean([q['score_range'] for q in query_metrics]))
        
        print(f"✅ Evaluation complete. Generated {len(metrics)} metrics.")
        return metrics
    
    def _cache_embeddings(self, df: pd.DataFrame, cache_file: str):
        """Cache embeddings to file with metadata"""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Prepare cache data with comprehensive metadata
        cache_data = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'timestamp': time.time(),
            'num_embeddings': len(df),
            'embeddings': df['embedding'].tolist(),
            'text_hashes': df['text'].apply(lambda x: hash(str(x))).tolist(),
            'metadata': {
                'chunk_ids': df.get('chunk_id', pd.Series()).tolist(),
                'source_ids': df.get('source_id', pd.Series()).tolist(),
                'chunk_types': df.get('chunk_type', pd.Series()).tolist()
            }
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
            print(f"💾 Embeddings cached to {cache_file} ({file_size:.1f} MB)")
            
        except Exception as e:
            print(f"❌ Error caching embeddings: {e}")
    
    def _load_cached_embeddings(self, df: pd.DataFrame, cache_file: str) -> pd.DataFrame:
        """Load cached embeddings with comprehensive validation"""
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache compatibility
            if cache_data['model_name'] != self.model_name:
                print(f"⚠️ Model mismatch in cache. Expected {self.model_name}, got {cache_data['model_name']}")
                return self.embed_dataframe(df, cache_file=None)
            
            if len(cache_data['embeddings']) != len(df):
                print(f"⚠️ Size mismatch in cache. Expected {len(df)}, got {len(cache_data['embeddings'])}")
                return self.embed_dataframe(df, cache_file=None)
            
            # Validate some text hashes for data integrity
            current_hashes = df['text'].apply(lambda x: hash(str(x))).tolist()
            cached_hashes = cache_data['text_hashes']
            
            # Check a sample of hashes for validation
            sample_size = min(100, len(current_hashes))
            sample_indices = np.random.choice(len(current_hashes), sample_size, replace=False)
            
            mismatches = 0
            for idx in sample_indices:
                if current_hashes[idx] != cached_hashes[idx]:
                    mismatches += 1
            
            if mismatches > sample_size * 0.1:  # Allow 10% mismatch tolerance
                print(f"⚠️ Data integrity check failed. {mismatches}/{sample_size} hash mismatches")
                return self.embed_dataframe(df, cache_file=None)
            
            # Cache is valid - add embeddings to DataFrame
            df_cached = df.copy()
            df_cached['embedding'] = cache_data['embeddings']
            
            # Show cache info
            cache_age = time.time() - cache_data.get('timestamp', 0)
            file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
            
            print(f"✅ Loaded {len(cache_data['embeddings'])} cached embeddings")
            print(f"📁 Cache: {file_size:.1f} MB, {cache_age/3600:.1f} hours old")
            
            return df_cached
            
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}. Generating new embeddings...")
            return self.embed_dataframe(df, cache_file=None)


def main():
    """Example usage of TextEmbedder"""
    
    # Initialize embedder
    embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
    
    # Load model
    if not embedder.load_model():
        print("Failed to load embedding model")
        return
    
    # Example with sample posts
    sample_posts = [
        "I'm having trouble with knee pain during squats",
        "Best exercises for building shoulder strength",
        "How to improve running endurance",
        "Lower back pain after deadlifts",
        "Nutrition tips for muscle building"
    ]
    
    print("\n🔄 Testing embedding generation...")
    embeddings = embedder.embed_texts(sample_posts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test similarity computation
    print("\n🔄 Testing similarity computation...")
    query = "knee pain during squats"
    query_embedding = embedder.embed_texts([query])[0]
    
    similarities = []
    for i, post in enumerate(sample_posts):
        similarity = embedder.compute_similarity(query_embedding, embeddings[i])
        similarities.append((post, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nQuery: '{query}'")
    print("Most similar posts:")
    for post, sim in similarities[:3]:
        print(f"  {sim:.3f}: {post}")
    
    # Test with DataFrame (if clean data exists)
    clean_data_file = "data/fitness_posts_clean.jsonl"
    if os.path.exists(clean_data_file):
        print(f"\n🔄 Testing with real data from {clean_data_file}...")
        
        # Load sample of clean data
        df = pd.read_json(clean_data_file, lines=True).head(10)
        
        # Generate embeddings
        df_with_embeddings = embedder.embed_dataframe(
            df, 
            text_column='text',
            cache_file="data/sample_embeddings.pkl"
        )
        
        # Perform semantic search
        results = embedder.semantic_search(
            query="knee pain",
            corpus_df=df_with_embeddings,
            top_k=3
        )
        
        print("\nSemantic search results:")
        for _, row in results.iterrows():
            print(f"  Score: {row['similarity_score']:.3f}")
            print(f"  Text: {row['text'][:100]}...")
            print()


if __name__ == "__main__":
    main()
