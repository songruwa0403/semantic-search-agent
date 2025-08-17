# Stage 1: Data Collection

## Overview
This stage implements the Reddit data collection system using PRAW (Python Reddit API Wrapper) to gather fitness-related posts and comments with sophisticated pain point detection.

## Components

### `reddit_collector.py`
**Purpose**: Comprehensive Reddit data collection with pain point analysis

**Key Features**:
- ✅ Reddit API Authentication (read-only access)
- ✅ Subreddit Post Collection (configurable time periods)
- ✅ Hierarchical Comment Extraction (all thread levels)
- ✅ Advanced Pain Point Detection System (0.0-1.0 scoring)
- ✅ Comment Classification (pain_point, question, advice, detailed_response, general)
- ✅ Semantic Text Creation (combines post context with comment)
- ✅ Rate Limiting & Progress Tracking
- ✅ Multiple Export Formats (JSONL, CSV, beautified JSON)

## Usage

```bash
# Basic collection
python reddit_collector.py

# The script is configured to collect:
# - Subreddit: r/Fitness
# - Time period: Past year (for maximum data volume)
# - Limit: 1000 posts
# - Comments per post: 50 (high-quality comments)
```

## Configuration
Edit the main function parameters in `reddit_collector.py`:

```python
output_file = collector.collect_posts(
    subreddit_name="Fitness",
    time_period="year",  # year, month, week, day, all
    limit=1000,  # Maximum posts to collect
    collect_comments=True,
    max_comments_per_post=50  # Comments per post
)
```

## Output Data Structure

### Posts with Comments
```json
{
  "id": "1g54ik2",
  "title": "2,000 Workouts Without a Rest Day",
  "body": "This is another update about training...",
  "author": "gzcl",
  "score": 584,
  "comments": [...]
}
```

### Comments with Pain Point Analysis
```json
{
  "comment_id": "lt05kku",
  "body": "I wish I had the self-control...",
  "semantic_text": "Post: 2,000 Workouts Without a Rest Day | Context: ... | Comment: ...",
  "pain_point_score": 0.9,
  "comment_type": "pain_point",
  "post_context": {
    "post_title": "2,000 Workouts Without a Rest Day",
    "subreddit": "Fitness"
  }
}
```

## Pain Point Detection Algorithm

The system uses sophisticated keyword analysis and context understanding:

**High-value pain keywords** (with weights):
- 'pain': 0.3, 'injury': 0.3, 'hurt': 0.25
- 'problem': 0.15, 'struggle': 0.2
- 'can\'t': 0.1, 'unable': 0.1

**Scoring factors**:
- First-person language: +0.1 bonus
- Comment length: longer = higher score
- Contextual relevance within fitness domain

## Recent Collection Results
- **429 posts** from r/Fitness (past year)
- **18,130 comments** with full analysis
- **1,544 pain point comments** identified
- **89% fitness-relevant content**

## Dependencies
- `praw` - Reddit API access
- `pandas` - Data manipulation
- `tqdm` - Progress tracking
- `python-dotenv` - Environment management

## Setup Requirements
1. Reddit API credentials in `.env` file
2. Virtual environment with dependencies
3. Rate limiting compliance (built-in)

## Next Stage
Output data flows to **Stage 2: Data Cleaning** for preprocessing and chunking before embedding generation.
