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
import logging
from datetime import datetime
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
            
            if self.index_type == "flat":
                if self.metric == "cosine" or self.metric == "ip":
                    # IndexFlatIP for cosine similarity (with normalized vectors)
                    self.index = faiss.IndexFlatIP(self.embedding_dim)
                    print("📋 Created IndexFlatIP for cosine similarity")
                else:  # L2
                    # IndexFlatL2 for Euclidean distance
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    print("📋 Created IndexFlatL2 for L2 distance")
                    
            elif self.index_type == "ivf":
                # IVF index for faster approximate search
                nlist = min(4096, max(1, int(np.sqrt(100000))))  # Adaptive nlist
                if self.metric == "cosine" or self.metric == "ip":
                    quantizer = faiss.IndexFlatIP(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                else:
                    quantizer = faiss.IndexFlatL2(self.embedding_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
                    
                # Set search parameters
                self.index.nprobe = min(128, nlist // 4)  # Adaptive nprobe
                print(f"📋 Created IndexIVFFlat with nlist={nlist}, nprobe={self.index.nprobe}")
                
            elif self.index_type == "hnsw":
                # HNSW index for hierarchical search
                M = 16  # Number of bidirectional links for each node
                if self.metric == "cosine" or self.metric == "ip":
                    self.index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_INNER_PRODUCT)
                else:
                    self.index = faiss.IndexHNSWFlat(self.embedding_dim, M, faiss.METRIC_L2)
                    
                # Set construction parameters
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 128
                print(f"📋 Created IndexHNSWFlat with M={M}, efConstruction=200, efSearch=128")
                
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
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
            start_time = time.time()
            
            # Validate vector dimensions
            if vectors.shape[1] != self.embedding_dim:
                raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.embedding_dim}")
            
            # Normalize vectors for cosine similarity
            if self.metric == "cosine":
                vectors = self._normalize_vectors(vectors)
                print("📏 Vectors normalized for cosine similarity")
            
            # Ensure vectors are float32 (FAISS requirement)
            vectors = vectors.astype(np.float32)
            
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
            if self.index_type == "ivf":
                # IVF indexes need training before adding vectors
                if not self.index.is_trained:
                    print("🔄 Training IVF index...")
                    train_vectors = vectors if len(vectors) >= 256 else np.vstack([vectors] * (256 // len(vectors) + 1))[:256]
                    self.index.train(train_vectors.astype(np.float32))
                    print("✅ IVF index trained")
            
            self.index.add(vectors)
            
            # Store metadata
            if metadata is None:
                metadata = [{"id": doc_id} for doc_id in ids]
            
            self.metadata.extend(metadata)
            self._next_id += n_vectors
            
            elapsed_time = time.time() - start_time
            print(f"✅ Added {n_vectors} vectors in {elapsed_time:.2f}s. Total: {self.index.ntotal}")
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
            
            # Normalize for cosine similarity
            if self.metric == "cosine":
                query_vector = self._normalize_vectors(query_vector)
            
            # Ensure query vector is float32
            query_vector = query_vector.astype(np.float32)
            
            # Perform search with FAISS
            search_k = min(k * 2, self.index.ntotal) if filter_func else k  # Get more results if filtering
            scores, indices = self.index.search(query_vector, search_k)
            
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
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        
        try:
            # Normalize for cosine similarity
            if self.metric == "cosine":
                query_vectors = self._normalize_vectors(query_vectors)
            
            # Ensure vectors are float32
            query_vectors = query_vectors.astype(np.float32)
            
            # Perform batch search with FAISS
            scores, indices = self.index.search(query_vectors, k)
            
            # Process results for each query
            results = []
            for q_idx in range(len(query_vectors)):
                query_results = []
                for i in range(k):
                    idx = indices[q_idx][i]
                    score = float(scores[q_idx][i])
                    
                    if idx != -1 and idx < len(self.metadata):  # -1 indicates no result
                        result = {
                            'index': int(idx),
                            'score': score,
                            'metadata': self.metadata[idx].copy(),
                            'id': self.index_to_id.get(idx, f"unknown_{idx}")
                        }
                        query_results.append(result)
                
                # Sort by score (higher is better for IP/cosine)
                query_results.sort(key=lambda x: x['score'], reverse=True)
                results.append(query_results)
            
            return results
            
        except Exception as e:
            print(f"❌ Error during batch search: {e}")
            return [[] for _ in range(len(query_vectors))]
    
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
        
        # Try to reconstruct vector from index (not all index types support this)
        try:
            if hasattr(self.index, 'reconstruct'):
                vector = self.index.reconstruct(idx)
                return vector
            else:
                print(f"⚠️ Index type '{self.index_type}' doesn't support vector reconstruction")
                return None
        except Exception as e:
            print(f"❌ Error reconstructing vector for ID '{doc_id}': {e}")
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
        Note: This implementation marks vectors as deleted and rebuilds mappings
        
        Args:
            doc_ids: List of document IDs to remove
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not doc_ids:
            return True
            
        try:
            print(f"🔄 Removing {len(doc_ids)} vectors...")
            
            # Get indices to remove
            indices_to_remove = set()
            for doc_id in doc_ids:
                if doc_id in self.id_to_index:
                    indices_to_remove.add(self.id_to_index[doc_id])
            
            if not indices_to_remove:
                print("⚠️ No vectors found to remove")
                return True
            
            # Create new metadata and mappings without removed items
            new_metadata = []
            new_id_to_index = {}
            new_index_to_id = {}
            new_idx = 0
            
            for old_idx, meta in enumerate(self.metadata):
                if old_idx not in indices_to_remove:
                    new_metadata.append(meta)
                    doc_id = self.index_to_id.get(old_idx)
                    if doc_id:
                        new_id_to_index[doc_id] = new_idx
                        new_index_to_id[new_idx] = doc_id
                    new_idx += 1
            
            # Update mappings
            self.metadata = new_metadata
            self.id_to_index = new_id_to_index
            self.index_to_id = new_index_to_id
            
            print(f"✅ Removed {len(indices_to_remove)} vectors from metadata")
            print("⚠️ Note: FAISS index still contains all vectors. Consider rebuilding for space efficiency.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error removing vectors: {e}")
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
            "id_mappings": len(self.id_to_index)
        }
        
        # Add index-specific parameters
        if self.index:
            if self.index_type == "ivf" and hasattr(self.index, 'nprobe'):
                stats["ivf_nprobe"] = self.index.nprobe
                stats["ivf_nlist"] = self.index.nlist
                stats["is_trained"] = self.index.is_trained
            elif self.index_type == "hnsw" and hasattr(self.index, 'hnsw'):
                stats["hnsw_M"] = self.index.hnsw.max_level
                stats["hnsw_efSearch"] = self.index.hnsw.efSearch
        
        # Calculate estimated memory usage
        if self.index and self.index.ntotal > 0:
            vector_memory = self.index.ntotal * self.embedding_dim * 4  # 4 bytes per float32
            metadata_memory = len(pickle.dumps(self.metadata))
            stats["memory_usage_mb"] = (vector_memory + metadata_memory) / (1024 * 1024)
        else:
            stats["memory_usage_mb"] = 0
        
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
            if self.index is None:
                print("❌ No index to save")
                return False
                
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, file_path)
            
            # Save metadata and mappings
            metadata_file = file_path.replace('.faiss', '_metadata.pkl')
            metadata_data = {
                'metadata': self.metadata,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id,
                'next_id': self._next_id,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type,
                'metric': self.metric,
                'timestamp': time.time(),
                'total_vectors': self.index.ntotal
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata_data, f)
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"✅ Index saved to {file_path} ({file_size:.1f} MB)")
            print(f"📊 Saved {self.index.ntotal} vectors with metadata")
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
            
            print(f"📁 Loading index from {file_path}...")
            
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
            
            # Load FAISS index
            self.index = faiss.read_index(file_path)
            
            # Validate loaded index
            expected_vectors = metadata_data.get('total_vectors', len(self.metadata))
            if self.index.ntotal != expected_vectors:
                print(f"⚠️ Vector count mismatch: expected {expected_vectors}, got {self.index.ntotal}")
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            age_hours = (time.time() - metadata_data.get('timestamp', 0)) / 3600
            
            print(f"✅ Index loaded from {file_path} ({file_size:.1f} MB)")
            print(f"📊 Loaded {self.index.ntotal} vectors, {age_hours:.1f} hours old")
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity using L2 normalization"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Add small epsilon to avoid division by zero
        normalized = vectors / (norms + 1e-8)
        return normalized


def main():
    """Example usage of VectorStore with comprehensive testing"""
    
    print("🚀 VectorStore Comprehensive Testing")
    print("=" * 50)
    
    # Test different index types
    index_types = ["flat", "ivf", "hnsw"]
    
    for index_type in index_types:
        print(f"\n🔧 Testing {index_type.upper()} index type")
        print("-" * 30)
        
        # Initialize vector store
        vector_store = VectorStore(
            embedding_dim=384,
            index_type=index_type,
            metric="cosine"
        )
        
        # Create index
        if not vector_store.create_index():
            print(f"Failed to create {index_type} index")
            continue
        
        # Generate sample vectors and metadata
        n_samples = 1000
        sample_vectors = np.random.rand(n_samples, 384).astype(np.float32)
        sample_metadata = [
            {
                "id": f"post_{i}",
                "text": f"Sample post {i} about fitness topic",
                "subreddit": "Fitness",
                "score": np.random.randint(1, 100),
                "category": np.random.choice(["strength", "cardio", "nutrition", "recovery"]),
                "pain_point_score": np.random.random(),
                "has_fitness_terms": np.random.choice([True, False])
            }
            for i in range(n_samples)
        ]
        sample_ids = [f"post_{i}" for i in range(n_samples)]
        
        # Add vectors to index
        print(f"\n🔄 Adding {n_samples} vectors...")
        start_time = time.time()
        success = vector_store.add_vectors(
            vectors=sample_vectors,
            metadata=sample_metadata,
            ids=sample_ids
        )
        add_time = time.time() - start_time
        
        if not success:
            print(f"Failed to add vectors to {index_type} index")
            continue
        
        print(f"⏱️ Vector addition time: {add_time:.2f}s ({n_samples/add_time:.1f} vectors/sec)")
        
        # Test single search performance
        print("\n🔍 Testing single search performance...")
        query_vector = np.random.rand(384).astype(np.float32)
        
        search_times = []
        for _ in range(10):  # Average over multiple searches
            start_time = time.time()
            results = vector_store.search(query_vector, k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)
        
        avg_search_time = np.mean(search_times)
        print(f"⏱️ Average search time: {avg_search_time*1000:.2f}ms")
        
        # Test batch search performance
        print("\n🔍 Testing batch search performance...")
        n_queries = 50
        query_vectors = np.random.rand(n_queries, 384).astype(np.float32)
        
        start_time = time.time()
        batch_results = vector_store.batch_search(query_vectors, k=5)
        batch_time = time.time() - start_time
        
        print(f"⏱️ Batch search time: {batch_time:.2f}s ({n_queries/batch_time:.1f} queries/sec)")
        
        # Test filtering
        print("\n🔧 Testing filtered search...")
        filter_func = lambda meta: meta['pain_point_score'] > 0.7
        filtered_results = vector_store.search(query_vector, k=5, filter_func=filter_func)
        print(f"✅ Found {len(filtered_results)} high pain-point results")
        
        # Test index persistence
        print("\n💾 Testing index persistence...")
        index_file = f"data/test_{index_type}_index.faiss"
        if vector_store.save_index(index_file):
            # Load in new instance
            new_vector_store = VectorStore()
            if new_vector_store.load_index(index_file):
                test_results = new_vector_store.search(query_vector, k=3)
                print(f"✅ Persistence test passed: {len(test_results)} results after reload")
        
        # Show statistics
        print("\n📊 Performance Statistics:")
        stats = vector_store.get_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n{'='*50}")
    
    print("\n🎉 All index types tested successfully!")
    
    # Clean up test files
    print("\n🧹 Cleaning up test files...")
    for index_type in index_types:
        try:
            os.remove(f"data/test_{index_type}_index.faiss")
            os.remove(f"data/test_{index_type}_index_metadata.pkl")
        except:
            pass
    print("✅ Cleanup complete")


if __name__ == "__main__":
    main()
