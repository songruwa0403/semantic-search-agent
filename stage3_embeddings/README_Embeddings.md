# Stage 3: Semantic Embeddings

## Overview
This stage transforms cleaned text chunks into vector representations using sentence-transformers for semantic search capabilities.

## Components

### `embedder.py`
**Purpose**: Generate and manage text embeddings for semantic search

**Key Features** (To Be Implemented):
- 🔄 Sentence-transformers model integration
- 🔄 Batch embedding generation
- 🔄 Similarity computation functions  
- 🔄 Embedding caching for efficiency
- 🔄 Support for different embedding models
- 🔄 Embedding quality evaluation methods

## Planned Implementation

### Model Selection
```python
# Recommended models for fitness domain:
"all-MiniLM-L6-v2"     # 384 dim, fast, good quality
"all-mpnet-base-v2"    # 768 dim, slower, better quality
```

### Expected Usage
```python
from embedder import TextEmbedder

# Initialize embedder
embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
embedder.load_model()

# Process cleaned data
df_with_embeddings = embedder.embed_dataframe(
    clean_data,
    text_column='text',
    cache_file="data/embeddings_cache.pkl"
)

# Perform semantic search
results = embedder.semantic_search(
    query="knee pain during squats",
    corpus_df=df_with_embeddings,
    top_k=10
)
```

## Input Data (From Stage 2)
- **18,130 cleaned comment chunks** with rich context
- **429 cleaned post chunks** for topic coverage
- **Rich metadata**: pain scores, content types, quality metrics

## Expected Output
- Vector representations (384 or 768 dimensions)
- Cached embeddings for efficiency
- Similarity search functionality
- Quality evaluation metrics

## Dependencies
- `sentence-transformers` - Pre-trained embedding models
- `scikit-learn` - Similarity computation
- `numpy` - Vector operations
- `pandas` - Data manipulation

## Next Stage
Generated embeddings flow to **Stage 4: Vector Search** for FAISS indexing and fast similarity search.

## TODO
This stage is currently a skeleton and needs full implementation of:
1. Model loading and initialization
2. Batch embedding generation
3. Similarity computation
4. Caching mechanisms
5. Quality evaluation tools
