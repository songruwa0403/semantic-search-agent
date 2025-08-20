# Data Cleaning Pipeline - Stage 2

## Overview

The `clean_posts.py` module implements a comprehensive data cleaning and preprocessing pipeline specifically designed for Reddit fitness data. It transforms raw Reddit posts and comments into clean, structured text chunks ready for semantic search and embedding generation.

## Key Features

### 🧹 **MVP-Focused Cleaning Strategy**
- **Preserves authenticity**: Keeps real user language, slang, and fitness terminology
- **Minimal over-processing**: Only removes content that breaks embeddings/search
- **Context preservation**: Maintains the semantic relationship between posts and comments

### 🔄 **Multiple Processing Modes**
- **Comments processing**: Uses pre-created `semantic_text` (post context + comment)
- **Posts processing**: Combines title and body with quality filtering
- **Mixed processing**: Creates both post and comment chunks

### 📊 **Rich Feature Engineering**
- Text statistics (word count, character count, sentence count)
- Content type detection (questions, fitness terms, pain indicators)
- Quality scoring for prioritizing high-value content
- Pain point analysis integration

## How It Works

### 1. **Text Cleaning Process**

```python
def clean_text(self, text: str) -> str:
```

**What gets cleaned:**
- ✅ URLs → `[URL]` placeholder
- ✅ Reddit usernames (`u/username` → `[USER]`)
- ✅ Subreddit references (`r/subreddit` → `[SUBREDDIT]`)
- ✅ HTML tags and entities
- ✅ Excessive whitespace
- ✅ Deleted/removed content markers

**What gets preserved:**
- ❌ Emojis (meaningful in fitness context: 💪, 🔥)
- ❌ Slang and informal language
- ❌ Fitness terminology and abbreviations
- ❌ Natural typos and user patterns
- ❌ Profanity (authentic user expression)

### 2. **Content Filtering**

```python
def filter_posts(self, df: pd.DataFrame) -> pd.DataFrame:
```

**Filtering criteria:**
- Remove deleted/removed content (`[deleted]`, `[removed]`)
- Remove very short content (< 10 characters total)
- Remove URL-heavy posts (> 80% URLs)
- Remove obvious spam (score < -10)

**Preservation philosophy:**
- Keep low-score content (might contain valuable pain points)
- Keep short but meaningful responses
- Keep informal/casual language

### 3. **Chunking Strategies**

#### **A. Semantic Text Chunks (Primary)**
```python
chunk_strategy="semantic_text"
```
- Uses pre-created `semantic_text` field
- Format: `"Post: [TITLE] | Context: [POST_PREVIEW] | Comment: [COMMENT]"`
- **Best for**: Comment-level semantic search with context
- **Output**: ~18K chunks from comments data

#### **B. Post Chunks**
```python
chunk_strategy="post"  
```
- Combines post title and body
- **Best for**: Topic-level search and broad content discovery
- **Output**: ~429 chunks from posts data

#### **C. Comment-Only Chunks**
```python
chunk_strategy="comment"
```
- Just the comment text without context
- **Use case**: When post context is unnecessary

#### **D. Mixed Chunks**
```python
chunk_strategy="mixed"
```
- Creates both post and semantic comment chunks
- **Use case**: Comprehensive coverage for diverse search needs

### 4. **Feature Engineering**

The pipeline automatically adds rich metadata to each chunk:

#### **Basic Text Statistics**
```json
{
  "word_count": 107,
  "char_count": 596,
  "sentence_count": 3,
  "avg_words_per_sentence": 16.33
}
```

#### **Content Type Detection**
```json
{
  "contains_question": true,
  "contains_url": false,
  "contains_numbers": true,
  "has_fitness_terms": true,
  "has_pain_indicators": false
}
```

#### **Quality Assessment**
```json
{
  "quality_score": 5,
  "pain_point_score": 0.9
}
```

## Usage Examples

### Basic Usage
```python
from clean_posts import TextCleaner

# Initialize cleaner
cleaner = TextCleaner()

# Process comments data (primary use case)
output_file = cleaner.process_posts(
    input_file="data/fitness_comments.jsonl",
    output_file="data/fitness_comments_clean.jsonl",
    chunk_strategy="semantic_text"
)
```

### Advanced Usage
```python
# Process with different strategies
cleaner = TextCleaner()

# 1. Process comments with context (recommended)
comments_clean = cleaner.process_posts(
    input_file="data/fitness_comments.jsonl",
    chunk_strategy="semantic_text"
)

# 2. Process posts for topic-level search
posts_clean = cleaner.process_posts(
    input_file="data/fitness_posts_with_comments.jsonl", 
    chunk_strategy="post"
)

# 3. Create mixed chunks for comprehensive coverage
mixed_clean = cleaner.process_posts(
    input_file="data/fitness_comments.jsonl",
    chunk_strategy="mixed"
)
```

## Output Structure

### Cleaned Data Format
Each chunk in the output JSONL contains:

```json
{
  "chunk_id": "semantic_lt05kku",
  "text": "Post: 2,000 Workouts Without a Rest Day | Context: ... | Comment: I wish I had the self-control...",
  "chunk_type": "semantic_comment",
  "source_id": "lt05kku",
  "post_id": "1g54ik2", 
  "score": 3,
  "pain_point_score": 0.9,
  "comment_type": "pain_point",
  "word_count": 213,
  "char_count": 1118,
  "sentence_count": 13,
  "contains_question": true,
  "has_fitness_terms": true,
  "has_pain_indicators": true,
  "quality_score": 6
}
```

### Statistics Output
Automatically generated stats file (`*_stats.json`):

```json
{
  "total_chunks": 18130,
  "chunk_types": {
    "semantic_comment": 18130
  },
  "text_stats": {
    "avg_word_count": 107.76,
    "avg_char_count": 596.67,
    "avg_quality_score": 5.22
  },
  "content_analysis": {
    "has_fitness_terms": 16176,
    "has_pain_indicators": 3732,
    "contains_questions": 15972
  },
  "pain_point_analysis": {
    "high_pain_points": 421,
    "medium_pain_points": 2484,
    "avg_pain_score": 0.20
  }
}
```

## Pipeline Results Summary

### Comments Dataset (Primary)
- **18,130 cleaned chunks** ready for embeddings
- **89% fitness-relevant** content (16,176/18,130)
- **88% questions** (15,972/18,130) - excellent for Q&A search
- **421 high pain points** (score > 0.5) for targeted analysis
- **Average 108 words per chunk** - optimal for embeddings

### Posts Dataset (Supplementary)  
- **429 cleaned chunks** for broader topic coverage
- **90% fitness-relevant** content (384/429)
- **Average 178 words per chunk** - substantial content

## Design Philosophy

### MVP Approach
1. **Start simple**: Clean only what's necessary for embeddings
2. **Preserve authenticity**: Keep real user language patterns
3. **Focus on quantity**: Better to have more diverse data than perfect data
4. **Test early**: Generate embeddings with minimally cleaned data first

### Why This Works for Semantic Search
- **Context preservation**: Comments retain their conversational context
- **Domain authenticity**: Real fitness discussions with natural language
- **Rich metadata**: Multiple ways to filter and prioritize content
- **Scalable processing**: Handles large datasets efficiently

## Next Steps

The cleaned data is ready for **Stage 3: Semantic Embeddings**:
1. Load cleaned chunks into embedding pipeline
2. Generate vector representations using sentence-transformers
3. Build FAISS index for fast similarity search
4. Test semantic search quality with real queries

## Files Generated

- `fitness_comments_clean.jsonl` - Primary dataset for semantic search
- `fitness_comments_clean_stats.json` - Detailed statistics
- `fitness_posts_clean.jsonl` - Posts dataset for topic search
- `fitness_posts_clean_stats.json` - Posts statistics

This cleaning pipeline ensures your semantic search system has high-quality, contextually-rich data while maintaining the authentic voice of fitness community discussions.
