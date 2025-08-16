# Semantic Search Agent - Stage 1: Foundations

A portfolio project building an AI-powered research assistant that autonomously mines, clusters, and summarizes fitness-related user pain points from Reddit posts.

## Project Overview

The Semantic Search Agent operates as an intelligent agent that reasons over user goals, selects appropriate tools (semantic search, clustering, summarization, trend analysis), and executes multi-step workflows to return actionable insights grounded in real user discussions.

**Stage 1 Focus**: Building the core technical foundation with a hands-on learning approach.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Reddit API    │───▶│ Text Preprocessing│───▶│   Embeddings    │
│  (Data Source)  │    │  (Clean & Parse)  │    │ (Semantic Rep.) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐             │
│  BM25 Baseline  │    │  Vector Database │◀────────────┘
│ (Keyword Search)│    │     (FAISS)      │
└─────────────────┘    └──────────────────┘
         │                        │
         └────────┬─────────────────┘
                  ▼
        ┌──────────────────┐
        │ Search Interface │
        │  & Comparison    │
        └──────────────────┘
```

## Stage 1 Components

### Chunk 1: Data Collection (`reddit_collector.py`)
**Learning Focus**: API integration, data extraction, rate limiting
- Pull posts from r/Fitness using PRAW (Python Reddit API Wrapper)
- Handle authentication and rate limits
- Save data in JSONL/CSV format for processing
- **Deliverable**: Clean dataset of Reddit posts

### Chunk 2: Text Preprocessing (`clean_posts.py`)
**Learning Focus**: Text cleaning, data quality, pandas manipulation
- Clean titles and post bodies (remove URLs, normalize text)
- Filter out low-quality posts (too short, deleted, etc.)
- Optional: sentence splitting for chunked embeddings
- **Deliverable**: Preprocessed, clean text dataset

### Chunk 3: Semantic Embeddings (`embedder.py`)
**Learning Focus**: Vector representations, similarity computation
- Generate embeddings using sentence-transformers
- Compute cosine similarity between posts manually
- Compare different embedding models and dimensions
- **Deliverable**: Embedding generation and similarity system

### Chunk 4: Vector Database (`vectorstore.py`)
**Learning Focus**: Vector indexing, efficient search, FAISS
- Build FAISS index for fast similarity search
- Store and retrieve embeddings with metadata
- Implement semantic search functionality
- **Deliverable**: Scalable vector search system

### Chunk 5: Keyword Baseline (`bm25_baseline.py`)
**Learning Focus**: Traditional search, evaluation, comparison
- Implement BM25 keyword search using rank_bm25
- Compare results with semantic search qualitatively
- Understand trade-offs between approaches
- **Deliverable**: Baseline search system for comparison

## Setup Instructions

### 1. Environment Setup
```bash
# Clone/navigate to project directory
cd "Semantic Search Agent"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reddit API Setup
1. Go to https://www.reddit.com/prefs/apps
2. Create a new application (script type)
3. Copy the example environment file:
   ```bash
   cp env_example.txt .env
   ```
4. Fill in your Reddit API credentials in `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=SemanticSearchAgent/1.0 by your_username
   ```

### 3. Directory Structure
```
Semantic Search Agent/
├── data/                    # Data storage directory
├── reddit_collector.py      # Chunk 1: Data collection
├── clean_posts.py          # Chunk 2: Text preprocessing  
├── embedder.py             # Chunk 3: Semantic embeddings
├── vectorstore.py          # Chunk 4: Vector database
├── bm25_baseline.py        # Chunk 5: Keyword search baseline
├── requirements.txt        # Python dependencies
├── env_example.txt         # Environment variables template
└── README.md              # This file
```

## Learning Path & Implementation Guide

### Phase 1: Data Pipeline (Chunks 1-2)
1. **Start with `reddit_collector.py`**:
   - Implement Reddit API authentication
   - Build post collection logic with proper error handling
   - Add data filtering and quality checks
   - Test with small datasets first

2. **Move to `clean_posts.py`**:
   - Implement comprehensive text cleaning functions
   - Add data quality filtering
   - Experiment with different preprocessing strategies
   - Create chunking strategies for embeddings

### Phase 2: Search Systems (Chunks 3-5)
3. **Implement `embedder.py`**:
   - Start with sentence-transformers model loading
   - Implement batch embedding generation
   - Add similarity computation functions
   - Test with sample data

4. **Build `vectorstore.py`**:
   - Initialize FAISS indices (start with IndexFlatIP)
   - Implement vector addition and search
   - Add metadata storage and retrieval
   - Test with your generated embeddings

5. **Create `bm25_baseline.py`**:
   - Implement BM25 indexing and search
   - Add text preprocessing specific to keyword search
   - Compare results with semantic search
   - Implement evaluation metrics

### Phase 3: Integration & Evaluation
6. **Test the complete pipeline**:
   - Run data collection → preprocessing → embedding → search
   - Compare BM25 vs semantic search results
   - Analyze strengths and weaknesses of each approach

## Key Learning Objectives

### Technical Skills
- **API Integration**: Reddit API, authentication, rate limiting
- **Text Processing**: Regex, normalization, tokenization, filtering
- **Vector Operations**: Embeddings, similarity computation, normalization
- **Vector Databases**: FAISS indexing, search optimization
- **Search Algorithms**: BM25, semantic search, ranking

### ML/AI Concepts
- **Text Embeddings**: Understanding vector representations of text
- **Similarity Metrics**: Cosine similarity, inner product, L2 distance
- **Search Paradigms**: Keyword vs semantic search trade-offs
- **Evaluation**: Precision, recall, ranking quality

### Software Engineering
- **Data Pipelines**: ETL processes, data quality, error handling
- **Performance**: Batch processing, caching, optimization
- **Evaluation**: A/B testing, metrics, comparison frameworks

## Running the Components

### Individual Component Testing
```bash
# Test data collection
python reddit_collector.py

# Test text preprocessing  
python clean_posts.py

# Test embedding generation
python embedder.py

# Test vector search
python vectorstore.py

# Test BM25 baseline
python bm25_baseline.py
```

### End-to-End Pipeline Example
```python
from reddit_collector import RedditCollector
from clean_posts import TextCleaner
from embedder import TextEmbedder
from vectorstore import VectorStore
from bm25_baseline import BM25Baseline

# 1. Collect data
collector = RedditCollector()
collector.setup_reddit_client()
raw_data = collector.collect_posts("Fitness", limit=1000)

# 2. Clean data
cleaner = TextCleaner()
clean_data = cleaner.process_posts(raw_data)

# 3. Generate embeddings
embedder = TextEmbedder()
embedder.load_model()
df_with_embeddings = embedder.embed_dataframe(clean_data)

# 4. Build vector index
vector_store = VectorStore()
vector_store.create_index()
vector_store.add_vectors(embeddings, metadata)

# 5. Build BM25 baseline
bm25 = BM25Baseline()
bm25.build_index(texts, metadata)

# 6. Compare search results
query = "knee pain during squats"
semantic_results = vector_store.search(query)
keyword_results = bm25.search(query)
```

## Next Steps (Future Stages)

- **Stage 2**: Advanced features (clustering, summarization, trend analysis)
- **Stage 3**: Agent reasoning and tool selection
- **Stage 4**: User interface and deployment

## Implementation Tips

1. **Start Small**: Test each component with small datasets before scaling
2. **Iterate**: Implement basic versions first, then add complexity
3. **Compare**: Always compare your implementations with expected behavior
4. **Document**: Keep notes on what works and what doesn't
5. **Measure**: Add timing and quality metrics to understand performance

## Common Issues & Solutions

### Reddit API
- **Rate Limiting**: Implement proper delays and respect API limits
- **Authentication**: Double-check credentials and user agent string
- **Data Quality**: Some posts may be deleted or have missing content

### Text Processing
- **Encoding**: Handle different text encodings properly
- **Memory**: Large datasets may require batch processing
- **Edge Cases**: Handle empty texts, special characters, very long posts

### Embeddings
- **Model Loading**: First-time download may take time
- **Memory Usage**: Monitor RAM usage with large datasets
- **Batch Size**: Tune batch size for optimal performance

### FAISS
- **Index Types**: Start with flat indices, then experiment with approximate methods
- **Normalization**: Ensure vectors are normalized for cosine similarity
- **Persistence**: Implement proper save/load functionality

This foundation stage will give you solid understanding of the core technologies before moving to more advanced agent capabilities in later stages.
