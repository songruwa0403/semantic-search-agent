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
            
            # TODO: Initialize the sentence transformer model
            # self.model = SentenceTransformer(self.model_name)
            
            # TODO: Get embedding dimension
            # Test with a sample text to get dimensions
            # sample_embedding = self.model.encode("test")
            # self.embedding_dim = len(sample_embedding)
            
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
        
        # TODO: Implement batch embedding generation
        # Options:
        # 1. Use model.encode() with batch processing
        # 2. Handle memory management for large datasets
        # 3. Add progress tracking
        
        # Placeholder implementation:
        # embeddings = self.model.encode(
        #     texts,
        #     batch_size=batch_size,
        #     show_progress_bar=show_progress,
        #     convert_to_numpy=True
        # )
        
        # For now, return random embeddings as placeholder
        embeddings = np.random.rand(len(texts), 384)  # Replace with actual implementation
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    
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
        # TODO: Implement cosine similarity computation
        # Convert to numpy arrays if needed
        # Use sklearn.metrics.pairwise.cosine_similarity
        # or implement manually: dot(a,b) / (norm(a) * norm(b))
        
        # Placeholder implementation
        return 0.5
    
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
        # TODO: Implement similarity search
        # 1. Compute similarities between query and all candidates
        # 2. Sort by similarity score
        # 3. Return top_k results
        
        # Placeholder implementation
        similarities = np.random.rand(len(candidate_embeddings))
        indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in indices]
    
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
        
        # TODO: Implement direct text comparison
        # 1. Embed both texts
        # 2. Compute similarity
        
        embeddings = self.embed_texts([text1, text2])
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
        
        metrics = {}
        
        # TODO: Implement embedding evaluation
        # 1. Embedding distribution statistics
        # 2. Sample similarity calculations
        # 3. Clustering analysis (optional)
        # 4. Query-based evaluation if sample_queries provided
        
        embeddings = np.array(df['embedding'].tolist())
        
        # Basic statistics
        metrics['num_embeddings'] = len(embeddings)
        metrics['embedding_dim'] = embeddings.shape[1] if len(embeddings) > 0 else 0
        # metrics['mean_magnitude'] = np.mean(np.linalg.norm(embeddings, axis=1))
        # metrics['std_magnitude'] = np.std(np.linalg.norm(embeddings, axis=1))
        
        return metrics
    
    def _cache_embeddings(self, df: pd.DataFrame, cache_file: str):
        """Cache embeddings to file"""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # TODO: Implement caching strategy
        # Options: pickle, HDF5, parquet, etc.
        # Consider storing metadata (model name, version, etc.)
        
        cache_data = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'embeddings': df['embedding'].tolist(),
            'text_hashes': df['text'].apply(hash).tolist()  # For validation
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Embeddings cached to {cache_file}")
    
    def _load_cached_embeddings(self, df: pd.DataFrame, cache_file: str) -> pd.DataFrame:
        """Load cached embeddings"""
        # TODO: Implement cache loading with validation
        # Check model compatibility, text hashes, etc.
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache
            if cache_data['model_name'] != self.model_name:
                print(f"⚠️ Model mismatch in cache. Expected {self.model_name}, got {cache_data['model_name']}")
                return self.embed_dataframe(df, cache_file=None)
            
            # Add embeddings to DataFrame
            df_cached = df.copy()
            df_cached['embedding'] = cache_data['embeddings']
            
            print(f"✅ Loaded {len(cache_data['embeddings'])} cached embeddings")
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
