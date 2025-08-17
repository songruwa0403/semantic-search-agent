"""
Chunk 4 — Vector Database
FAISS-based vector storage and retrieval for semantic search

Learning Objectives:
- Understand vector database concepts and indexing
- Learn FAISS index types and their trade-offs
- Practice efficient similarity search at scale
- Understand metadata storage and retrieval
- Learn about index persistence and loading

TODO for you to implement:
1. Initialize different FAISS index types
2. Implement efficient batch indexing
3. Add metadata storage and retrieval
4. Implement index persistence (save/load)
5. Add search with filtering capabilities
6. Implement index updating and maintenance
7. Add performance monitoring and optimization
"""

import faiss
import numpy as np
import pandas as pd
import pickle
import json
import os
from typing import List, Dict, Optional, Tuple, Any, Union
from pathlib import Path
import time


class VectorStore:
    """FAISS-based vector storage for semantic search"""
    
    def __init__(self, 
                 embedding_dim: int = 384,
                 index_type: str = "flat",
                 metric: str = "cosine"):
        """
        Initialize the vector store
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            metric: Distance metric ("cosine", "l2", "ip")
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.metric = metric
        
        self.index = None
        self.metadata = []  # Store metadata for each vector
        self.id_to_index = {}  # Map document IDs to index positions
        self.index_to_id = {}  # Map index positions to document IDs
        
        self._next_id = 0
    
    def create_index(self) -> bool:
        """
        Create a new FAISS index based on the specified type
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"🔄 Creating FAISS index: {self.index_type} (dim={self.embedding_dim}, metric={self.metric})")
            
            # TODO: Implement different index types
            # FAISS Index Types:
            # 1. IndexFlatL2 - Exact search, L2 distance
            # 2. IndexFlatIP - Exact search, Inner Product (cosine if normalized)
            # 3. IndexIVFFlat - Inverted File index for faster search
            # 4. IndexHNSWFlat - Hierarchical Navigable Small World
            
            if self.index_type == "flat":
                if self.metric == "cosine" or self.metric == "ip":
                    # TODO: Create IndexFlatIP for cosine similarity
                    # self.index = faiss.IndexFlatIP(self.embedding_dim)
                    pass
                else:  # L2
                    # TODO: Create IndexFlatL2
                    # self.index = faiss.IndexFlatL2(self.embedding_dim)
                    pass
                    
            elif self.index_type == "ivf":
                # TODO: Implement IVF index
                # Parameters to consider:
                # - nlist: number of clusters
                # - nprobe: number of clusters to search
                pass
                
            elif self.index_type == "hnsw":
                # TODO: Implement HNSW index
                # Parameters to consider:
                # - M: number of connections
                # - efConstruction: size of dynamic candidate list
                pass
            
            # Placeholder - create a simple flat index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            print(f"✅ Index created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error creating index: {e}")
            return False
    
    def add_vectors(self, 
                   vectors: np.ndarray,
                   metadata: List[Dict] = None,
                   ids: List[str] = None) -> bool:
        """
        Add vectors to the index with optional metadata and IDs
        
        Args:
            vectors: Array of vectors to add (shape: n_vectors, embedding_dim)
            metadata: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.index is None:
            print("❌ Index not created. Call create_index() first.")
            return False
        
        try:
            n_vectors = len(vectors)
            print(f"🔄 Adding {n_vectors} vectors to index...")
            
            # Validate vector dimensions
            if vectors.shape[1] != self.embedding_dim:
                raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.embedding_dim}")
            
            # TODO: Implement vector normalization for cosine similarity
            # if self.metric == "cosine":
            #     vectors = self._normalize_vectors(vectors)
            
            # Generate IDs if not provided
            if ids is None:
                ids = [f"doc_{self._next_id + i}" for i in range(n_vectors)]
            
            # Store mapping between IDs and index positions
            start_idx = self.index.ntotal
            for i, doc_id in enumerate(ids):
                idx = start_idx + i
                self.id_to_index[doc_id] = idx
                self.index_to_id[idx] = doc_id
            
            # Add vectors to index
            # TODO: Implement actual vector addition
            # self.index.add(vectors.astype(np.float32))
            
            # Store metadata
            if metadata is None:
                metadata = [{"id": doc_id} for doc_id in ids]
            
            self.metadata.extend(metadata)
            self._next_id += n_vectors
            
            print(f"✅ Added {n_vectors} vectors. Total: {self.index.ntotal}")
            return True
            
        except Exception as e:
            print(f"❌ Error adding vectors: {e}")
            return False
    
    def search(self, 
               query_vector: np.ndarray,
               k: int = 10,
               filter_func: Optional[callable] = None) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector (shape: embedding_dim)
            k: Number of results to return
            filter_func: Optional function to filter results based on metadata
            
        Returns:
            List[Dict]: Search results with scores and metadata
        """
        if self.index is None or self.index.ntotal == 0:
            print("❌ Index is empty or not created")
            return []
        
        try:
            # Ensure query vector is 2D
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            
            # TODO: Implement normalization for cosine similarity
            # if self.metric == "cosine":
            #     query_vector = self._normalize_vectors(query_vector)
            
            # TODO: Perform search
            # scores, indices = self.index.search(query_vector.astype(np.float32), k)
            
            # Placeholder implementation
            scores = np.random.rand(1, min(k, len(self.metadata)))
            indices = np.random.randint(0, len(self.metadata), size=(1, min(k, len(self.metadata))))
            
            # Process results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                score = float(scores[0][i])
                
                if idx < len(self.metadata):
                    result = {
                        'index': int(idx),
                        'score': score,
                        'metadata': self.metadata[idx].copy(),
                        'id': self.index_to_id.get(idx, f"unknown_{idx}")
                    }
                    
                    # Apply filter if provided
                    if filter_func is None or filter_func(result['metadata']):
                        results.append(result)
            
            # Sort by score (higher is better for IP/cosine)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            return results[:k]
            
        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []
    
    def batch_search(self, 
                    query_vectors: np.ndarray,
                    k: int = 10) -> List[List[Dict]]:
        """
        Search for multiple queries in batch
        
        Args:
            query_vectors: Array of query vectors (shape: n_queries, embedding_dim)
            k: Number of results per query
            
        Returns:
            List[List[Dict]]: List of search results for each query
        """
        # TODO: Implement efficient batch search
        # For now, iterate through individual searches
        results = []
        for query_vector in query_vectors:
            result = self.search(query_vector, k)
            results.append(result)
        
        return results
    
    def get_vector_by_id(self, doc_id: str) -> Optional[np.ndarray]:
        """
        Retrieve a vector by its document ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            np.ndarray: Vector if found, None otherwise
        """
        if doc_id not in self.id_to_index:
            return None
        
        idx = self.id_to_index[doc_id]
        
        # TODO: Implement vector retrieval by index
        # FAISS doesn't directly support this - you might need to store vectors separately
        # or use reconstruct() method for some index types
        
        return None
    
    def update_metadata(self, doc_id: str, new_metadata: Dict) -> bool:
        """
        Update metadata for a document
        
        Args:
            doc_id: Document ID
            new_metadata: New metadata dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        if doc_id not in self.id_to_index:
            print(f"❌ Document ID '{doc_id}' not found")
            return False
        
        idx = self.id_to_index[doc_id]
        if idx < len(self.metadata):
            self.metadata[idx] = new_metadata
            return True
        
        return False
    
    def remove_vectors(self, doc_ids: List[str]) -> bool:
        """
        Remove vectors by their document IDs
        
        Args:
            doc_ids: List of document IDs to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        # TODO: Implement vector removal
        # Note: FAISS doesn't support direct removal from most index types
        # Options:
        # 1. Rebuild index without the removed vectors
        # 2. Use IndexIDMap wrapper for removal support
        # 3. Mark as deleted and filter during search
        
        print("⚠️ Vector removal not implemented. Consider rebuilding index.")
        return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dict: Statistics dictionary
        """
        stats = {
            "index_type": self.index_type,
            "metric": self.metric,
            "embedding_dim": self.embedding_dim,
            "total_vectors": self.index.ntotal if self.index else 0,
            "metadata_count": len(self.metadata),
            "memory_usage_mb": 0  # TODO: Calculate actual memory usage
        }
        
        # TODO: Add more detailed statistics
        # - Index-specific parameters
        # - Memory usage
        # - Search performance metrics
        
        return stats
    
    def save_index(self, file_path: str) -> bool:
        """
        Save the vector store to disk
        
        Args:
            file_path: Path to save the index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # TODO: Implement index saving
            # 1. Save FAISS index: faiss.write_index(self.index, index_file)
            # 2. Save metadata and mappings separately
            
            # Save metadata and mappings
            metadata_file = file_path.replace('.faiss', '_metadata.pkl')
            metadata_data = {
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_id': self._next_id,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'metric': self.metric
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata_data, f)
            
            # TODO: Save FAISS index
            # faiss.write_index(self.index, file_path)
            
            print(f"✅ Index saved to {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving index: {e}")
            return False
    
    def load_index(self, file_path: str) -> bool:
        """
        Load the vector store from disk
        
        Args:
            file_path: Path to the saved index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ Index file not found: {file_path}")
                return False
            
            # Load metadata
            metadata_file = file_path.replace('.faiss', '_metadata.pkl')
            if not os.path.exists(metadata_file):
                print(f"❌ Metadata file not found: {metadata_file}")
                return False
            
            with open(metadata_file, 'rb') as f:
                metadata_data = pickle.load(f)
            
            # Restore metadata and mappings
            self.metadata = metadata_data['metadata']
            self.id_to_index = metadata_data['id_to_index']
            self.index_to_id = metadata_data['index_to_id']
            self._next_id = metadata_data['next_id']
            self.embedding_dim = metadata_data['embedding_dim']
            self.index_type = metadata_data['index_type']
            self.metric = metadata_data['metric']
            
            # TODO: Load FAISS index
            # self.index = faiss.read_index(file_path)
            
            print(f"✅ Index loaded from {file_path}")
            print(f"📊 Loaded {len(self.metadata)} vectors")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        # TODO: Implement L2 normalization
        # norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # return vectors / (norms + 1e-8)  # Add small epsilon to avoid division by zero
        return vectors


def main():
    """Example usage of VectorStore"""
    
    # Initialize vector store
    vector_store = VectorStore(
        embedding_dim=384,
        index_type="flat",
        metric="cosine"
    )
    
    # Create index
    if not vector_store.create_index():
        print("Failed to create index")
        return
    
    # Generate sample vectors and metadata
    n_samples = 100
    sample_vectors = np.random.rand(n_samples, 384).astype(np.float32)
    sample_metadata = [
        {
            "id": f"post_{i}",
            "text": f"Sample post {i} about fitness topic",
            "subreddit": "Fitness",
            "score": np.random.randint(1, 100),
            "category": np.random.choice(["strength", "cardio", "nutrition", "recovery"])
        }
        for i in range(n_samples)
    ]
    sample_ids = [f"post_{i}" for i in range(n_samples)]
    
    # Add vectors to index
    print("\n🔄 Adding sample vectors...")
    success = vector_store.add_vectors(
        vectors=sample_vectors,
        metadata=sample_metadata,
        ids=sample_ids
    )
    
    if not success:
        print("Failed to add vectors")
        return
    
    # Test search
    print("\n🔄 Testing search...")
    query_vector = np.random.rand(384).astype(np.float32)
    results = vector_store.search(query_vector, k=5)
    
    print("Search results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result['score']:.3f}")
        print(f"     ID: {result['id']}")
        print(f"     Category: {result['metadata']['category']}")
        print(f"     Text: {result['metadata']['text']}")
        print()
    
    # Test filtering
    print("🔄 Testing filtered search (strength category)...")
    filter_func = lambda meta: meta['category'] == 'strength'
    filtered_results = vector_store.search(query_vector, k=5, filter_func=filter_func)
    
    print("Filtered search results:")
    for i, result in enumerate(filtered_results, 1):
        print(f"  {i}. Score: {result['score']:.3f}, Category: {result['metadata']['category']}")
    
    # Test index persistence
    print("\n🔄 Testing index save/load...")
    index_file = "data/test_index.faiss"
    if vector_store.save_index(index_file):
        # Create new vector store and load
        new_vector_store = VectorStore()
        if new_vector_store.load_index(index_file):
            print("✅ Index save/load test successful")
            
            # Verify loaded index works
            test_results = new_vector_store.search(query_vector, k=3)
            print(f"Loaded index search returned {len(test_results)} results")
    
    # Show statistics
    print("\n📊 Vector Store Statistics:")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
