"""
Chunk 2 — Text Preprocessing
Clean and preprocess Reddit posts for better text analysis

Learning Objectives:
- Master regex patterns for text cleaning
- Understand text normalization techniques
- Practice pandas data manipulation
- Learn about text chunking strategies for embeddings

TODO for you to implement:
1. Implement comprehensive text cleaning functions
2. Add data quality filtering
3. Implement sentence splitting for chunked embeddings
4. Add text statistics and quality metrics
5. Handle edge cases (deleted posts, empty content, etc.)
"""

import pandas as pd
import re
import json
import emoji
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path


class TextCleaner:
    """Clean and preprocess Reddit posts for analysis"""
    
    def __init__(self):
        """Initialize the text cleaner with regex patterns"""
        # Define regex patterns for cleaning
        import re
        
        # URL patterns (http/https/www)
        self.url_pattern = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+', re.IGNORECASE)
        
        # Reddit-specific patterns
        self.reddit_user_pattern = re.compile(r'\bu/([A-Za-z0-9_-]+)', re.IGNORECASE)
        self.reddit_sub_pattern = re.compile(r'\br/([A-Za-z0-9_-]+)', re.IGNORECASE)
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # HTML patterns
        self.html_pattern = re.compile(r'<[^<>]*>')
        self.html_entity_pattern = re.compile(r'&[a-zA-Z0-9#]+;')
        
        # Excessive whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Deleted/removed content patterns
        self.deleted_patterns = [
            '[deleted]', '[removed]', '[deleted by user]', 
            '[removed by moderator]', '[Comment deleted by user]'
        ]
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load Reddit posts from JSONL or CSV file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            pd.DataFrame: Loaded posts data
        """
        try:
            if file_path.endswith('.jsonl'):
                # Read JSONL line by line and parse JSON
                posts = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:  # Skip empty lines
                            try:
                                posts.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Error parsing JSON at line {line_num}: {e}")
                                continue
                
                print(f"✅ Loaded {len(posts)} records from {file_path}")
                return pd.DataFrame(posts)
                
            elif file_path.endswith('.csv'):
                # Load CSV file
                df = pd.read_csv(file_path)
                print(f"✅ Loaded {len(df)} records from {file_path}")
                return df
                
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single piece of text
        
        Args:
            text: Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Check for deleted/removed content
        if any(pattern.lower() in text.lower() for pattern in self.deleted_patterns):
            return ""
        
        cleaned = text
        
        # Apply cleaning steps
        cleaned = self._remove_urls(cleaned)
        cleaned = self._remove_reddit_formatting(cleaned)
        cleaned = self._remove_html(cleaned)
        cleaned = self._normalize_whitespace(cleaned)
        
        return cleaned.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # Replace URLs with [URL] placeholder to preserve context
        text = self.url_pattern.sub('[URL]', text)
        # Remove email addresses
        text = self.email_pattern.sub('[EMAIL]', text)
        return text
    
    def _remove_reddit_formatting(self, text: str) -> str:
        """Remove Reddit-specific formatting"""
        # Replace u/username with [USER]
        text = self.reddit_user_pattern.sub(r'[USER]', text)
        # Replace r/subreddit with [SUBREDDIT]  
        text = self.reddit_sub_pattern.sub(r'[SUBREDDIT]', text)
        return text
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and entities"""
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Decode common HTML entities
        html_entities = {
            '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"',
            '&#x27;': "'", '&#39;': "'", '&nbsp;': ' ', '&ndash;': '-',
            '&mdash;': '-', '&hellip;': '...', '&rsquo;': "'", '&lsquo;': "'"
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Generic HTML entity pattern as fallback
        text = self.html_entity_pattern.sub('', text)
        
        return text
    
    def _handle_emojis(self, text: str, strategy: str = "keep") -> str:
        """
        Handle emojis in text
        
        Args:
            text: Input text
            strategy: "remove", "keep", or "convert_to_text"
        """
        # For MVP, we keep emojis as they can be meaningful in fitness context
        # (💪, 🔥, etc. often convey important sentiment)
        if strategy == "keep":
            return text
        elif strategy == "remove":
            try:
                import emoji
                # Remove emojis completely
                return emoji.replace_emoji(text, replace='')
            except ImportError:
                # If emoji library not available, just return original
                return text
        elif strategy == "convert_to_text":
            try:
                import emoji
                # Convert emojis to text descriptions
                return emoji.demojize(text)
            except ImportError:
                return text
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # Replace multiple whitespace characters with single space
        text = self.whitespace_pattern.sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def filter_posts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter posts based on quality criteria
        
        Args:
            df: DataFrame with posts
            
        Returns:
            pd.DataFrame: Filtered posts
        """
        if df.empty:
            return df
        
        print(f"📊 Starting with {len(df)} posts")
        original_count = len(df)
        
        # 1. Remove posts with deleted/removed content
        def is_valid_content(text):
            if pd.isna(text) or not isinstance(text, str):
                return False
            text_lower = text.lower().strip()
            return not any(pattern.lower() in text_lower for pattern in self.deleted_patterns)
        
        # Filter by title and body content
        if 'title' in df.columns:
            df = df[df['title'].apply(is_valid_content)]
            print(f"   After title filter: {len(df)} posts")
        
        if 'body' in df.columns:
            df = df[df['body'].apply(is_valid_content)]
            print(f"   After body filter: {len(df)} posts")
        
        # 2. Remove very short content (< 10 characters total)
        def has_sufficient_content(row):
            title_len = len(str(row.get('title', '')).strip())
            body_len = len(str(row.get('body', '')).strip())
            return (title_len + body_len) >= 10
        
        df = df[df.apply(has_sufficient_content, axis=1)]
        print(f"   After length filter: {len(df)} posts")
        
        # 3. Remove posts that are mostly URLs (>80% of content)
        def not_mostly_urls(text):
            if pd.isna(text) or not isinstance(text, str):
                return True
            if len(text.strip()) < 20:  # Too short to matter
                return True
            
            # Count URL-like patterns
            url_matches = len(self.url_pattern.findall(text))
            words = len(text.split())
            
            # If more than 80% URLs or very few words with URLs, filter out
            return not (url_matches > 0 and (url_matches / max(words, 1)) > 0.8)
        
        if 'body' in df.columns:
            df = df[df['body'].apply(not_mostly_urls)]
            print(f"   After URL filter: {len(df)} posts")
        
        # 4. Keep posts with reasonable engagement (score > -10 to filter obvious spam)
        if 'score' in df.columns:
            df = df[df['score'] > -10]
            print(f"   After score filter: {len(df)} posts")
        
        # Reset index after filtering
        df = df.reset_index(drop=True)
        
        filtered_count = len(df)
        print(f"📊 After filtering: {filtered_count} posts ({original_count - filtered_count} removed)")
        return df
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences for chunked embeddings
        
        Args:
            text: Input text
            
        Returns:
            List[str]: List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting using regex
        # Split on sentence endings but preserve some context
        import re
        
        # Split on sentence-ending punctuation followed by whitespace and capital letter
        sentence_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_pattern.split(text)
        
        # Clean and filter sentences
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences that are meaningful (at least 10 characters)
            if len(sentence) >= 10:
                clean_sentences.append(sentence)
        
        # If no sentences found or text is short, return the whole text
        if not clean_sentences:
            return [text] if len(text) >= 10 else []
        
        return clean_sentences
    
    def create_chunks(self, df: pd.DataFrame, chunk_strategy: str = "semantic_text") -> pd.DataFrame:
        """
        Create text chunks for embedding
        
        Args:
            df: DataFrame with cleaned posts or comments
            chunk_strategy: "post", "semantic_text", "comment", or "mixed"
            
        Returns:
            pd.DataFrame: DataFrame with text chunks
        """
        chunks = []
        print(f"🔄 Creating chunks using strategy: {chunk_strategy}")
        
        for idx, row in df.iterrows():
            try:
                if chunk_strategy == "post":
                    # Use entire post (title + body)
                    title = str(row.get('title', '')).strip()
                    body = str(row.get('body', '')).strip()
                    text = f"{title} {body}".strip()
                    
                    if len(text) >= 10:  # Minimum meaningful length
                        chunks.append({
                            'chunk_id': f"post_{row.get('id', idx)}",
                            'text': self.clean_text(text),
                            'chunk_type': 'post',
                            'source_id': row.get('id', ''),
                            'score': row.get('score', 0),
                            'pain_point_score': 0.0,  # Posts don't have pain point scores
                            'original_index': idx
                        })
                
                elif chunk_strategy == "semantic_text":
                    # Use the pre-created semantic_text (post context + comment)
                    semantic_text = str(row.get('semantic_text', '')).strip()
                    
                    if len(semantic_text) >= 10:
                        chunks.append({
                            'chunk_id': f"semantic_{row.get('comment_id', idx)}",
                            'text': self.clean_text(semantic_text),
                            'chunk_type': 'semantic_comment',
                            'source_id': row.get('comment_id', ''),
                            'post_id': row.get('post_id', ''),
                            'score': row.get('score', 0),
                            'pain_point_score': row.get('pain_point_score', 0.0),
                            'comment_type': row.get('comment_type', ''),
                            'original_index': idx
                        })
                
                elif chunk_strategy == "comment":
                    # Use just the comment body
                    comment_body = str(row.get('body', '')).strip()
                    
                    if len(comment_body) >= 10:
                        chunks.append({
                            'chunk_id': f"comment_{row.get('comment_id', idx)}",
                            'text': self.clean_text(comment_body),
                            'chunk_type': 'comment',
                            'source_id': row.get('comment_id', ''),
                            'post_id': row.get('post_id', ''),
                            'score': row.get('score', 0),
                            'pain_point_score': row.get('pain_point_score', 0.0),
                            'comment_type': row.get('comment_type', ''),
                            'original_index': idx
                        })
                
                elif chunk_strategy == "mixed":
                    # Create both post and comment chunks
                    # First, try post chunk
                    if 'title' in row and 'body' in row:
                        title = str(row.get('title', '')).strip()
                        body = str(row.get('body', '')).strip()
                        text = f"{title} {body}".strip()
                        
                        if len(text) >= 10:
                            chunks.append({
                                'chunk_id': f"post_{row.get('id', idx)}",
                                'text': self.clean_text(text),
                                'chunk_type': 'post',
                                'source_id': row.get('id', ''),
                                'score': row.get('score', 0),
                                'pain_point_score': 0.0,
                                'original_index': idx
                            })
                    
                    # Then, try semantic text chunk
                    if 'semantic_text' in row:
                        semantic_text = str(row.get('semantic_text', '')).strip()
                        
                        if len(semantic_text) >= 10:
                            chunks.append({
                                'chunk_id': f"semantic_{row.get('comment_id', idx)}",
                                'text': self.clean_text(semantic_text),
                                'chunk_type': 'semantic_comment',
                                'source_id': row.get('comment_id', ''),
                                'post_id': row.get('post_id', ''),
                                'score': row.get('score', 0),
                                'pain_point_score': row.get('pain_point_score', 0.0),
                                'comment_type': row.get('comment_type', ''),
                                'original_index': idx
                            })
                            
            except Exception as e:
                print(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        chunks_df = pd.DataFrame(chunks)
        print(f"✅ Created {len(chunks_df)} text chunks")
        
        # Remove chunks with empty text after cleaning
        chunks_df = chunks_df[chunks_df['text'].str.len() >= 10]
        print(f"📊 After final filtering: {len(chunks_df)} chunks")
        
        return chunks_df
    
    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add text statistics and features
        
        Args:
            df: DataFrame with text data
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        if 'text' not in df.columns:
            return df
        
        print("🔄 Adding text features...")
        
        # Basic text statistics
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        df['char_count'] = df['text'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df['sentence_count'] = df['text'].apply(lambda x: len(self.split_into_sentences(str(x))) if pd.notna(x) else 0)
        
        # Content type indicators
        df['contains_question'] = df['text'].str.contains(r'\?|how|what|why|when|where|which|should i|can i', case=False, na=False)
        df['contains_url'] = df['text'].str.contains(r'http|www|\[URL\]', case=False, na=False)
        df['contains_numbers'] = df['text'].str.contains(r'\d+', na=False)
        
        # Fitness-specific terms
        fitness_terms = [
            'workout', 'exercise', 'gym', 'lift', 'squat', 'bench', 'deadlift',
            'cardio', 'muscle', 'strength', 'weight', 'rep', 'set', 'pr', 'max',
            'protein', 'diet', 'nutrition', 'calorie', 'bulk', 'cut', 'form'
        ]
        fitness_pattern = '|'.join(fitness_terms)
        df['has_fitness_terms'] = df['text'].str.contains(fitness_pattern, case=False, na=False)
        
        # Pain/problem indicators
        pain_terms = [
            'pain', 'hurt', 'ache', 'injury', 'problem', 'issue', 'struggle',
            'difficult', 'can\'t', 'unable', 'fail', 'wrong', 'mistake'
        ]
        pain_pattern = '|'.join(pain_terms)
        df['has_pain_indicators'] = df['text'].str.contains(pain_pattern, case=False, na=False)
        
        # Simple readability (average words per sentence)
        df['avg_words_per_sentence'] = df.apply(
            lambda row: row['word_count'] / max(row['sentence_count'], 1), axis=1
        )
        
        # Text quality score (simple heuristic)
        def calculate_quality_score(row):
            score = 0
            
            # Length factors
            if 20 <= row['word_count'] <= 500:  # Good length range
                score += 2
            elif row['word_count'] < 5:  # Too short
                score -= 2
            
            # Content factors
            if row['has_fitness_terms']:
                score += 2
            if row['contains_question']:
                score += 1
            if row['contains_url']:
                score -= 1  # URLs often less valuable for embeddings
            
            # Readability factors
            if 5 <= row['avg_words_per_sentence'] <= 25:  # Good sentence length
                score += 1
            
            return max(score, 0)  # Don't go below 0
        
        df['quality_score'] = df.apply(calculate_quality_score, axis=1)
        
        print("✅ Text features added")
        return df
    
    def process_posts(self, 
                     input_file: str,
                     output_file: str = None,
                     chunk_strategy: str = "semantic_text") -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            input_file: Path to raw posts file
            output_file: Path to save cleaned posts
            chunk_strategy: Strategy for creating text chunks
            
        Returns:
            str: Path to output file
        """
        if output_file is None:
            base_name = input_file.replace('.jsonl', '').replace('.csv', '')
            output_file = f"{base_name}_clean.jsonl"
        
        print("🧹 Starting text preprocessing pipeline...")
        print(f"   Input: {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Strategy: {chunk_strategy}")
        
        # Load data
        df = self.load_data(input_file)
        if df.empty:
            print("❌ No data loaded")
            return None
        
        # For comments data, we'll process semantic_text primarily
        # For posts data, we'll process title and body
        
        # Clean text fields based on available columns
        if 'semantic_text' in df.columns:
            print("🔄 Processing comments data...")
            # Don't clean semantic_text here - we'll clean it during chunking
        else:
            print("🔄 Processing posts data...")
            if 'title' in df.columns:
                print("   Cleaning titles...")
                df['title'] = df['title'].apply(self.clean_text)
            if 'body' in df.columns:
                print("   Cleaning bodies...")
                df['body'] = df['body'].apply(self.clean_text)
            
            # Filter posts (for posts data only)
            df = self.filter_posts(df)
        
        # Create chunks
        chunks_df = self.create_chunks(df, chunk_strategy)
        
        if chunks_df.empty:
            print("❌ No valid chunks created")
            return None
        
        # Add text features
        chunks_df = self.add_text_features(chunks_df)
        
        # Save processed data
        self._save_data(chunks_df, output_file)
        
        # Save statistics
        stats_file = output_file.replace('.jsonl', '_stats.json')
        self._save_statistics(chunks_df, stats_file)
        
        print(f"✅ Preprocessing complete!")
        print(f"   Output: {output_file}")
        print(f"   Stats: {stats_file}")
        print(f"📊 Final dataset: {len(chunks_df)} text chunks")
        
        return output_file
    
    def _save_data(self, df: pd.DataFrame, output_file: str):
        """Save processed data to file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if output_file.endswith('.jsonl'):
            df.to_json(output_file, orient='records', lines=True, force_ascii=False)
        elif output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        else:
            # Default to JSONL for embedding pipeline
            df.to_json(output_file, orient='records', lines=True, force_ascii=False)
    
    def _save_statistics(self, df: pd.DataFrame, stats_file: str):
        """Save processing statistics"""
        try:
            stats = {
                'total_chunks': len(df),
                'chunk_types': df['chunk_type'].value_counts().to_dict() if 'chunk_type' in df.columns else {},
                'text_stats': {
                    'avg_word_count': float(df['word_count'].mean()) if 'word_count' in df.columns else 0,
                    'avg_char_count': float(df['char_count'].mean()) if 'char_count' in df.columns else 0,
                    'avg_quality_score': float(df['quality_score'].mean()) if 'quality_score' in df.columns else 0
                },
                'content_analysis': {
                    'has_fitness_terms': int(df['has_fitness_terms'].sum()) if 'has_fitness_terms' in df.columns else 0,
                    'has_pain_indicators': int(df['has_pain_indicators'].sum()) if 'has_pain_indicators' in df.columns else 0,
                    'contains_questions': int(df['contains_question'].sum()) if 'contains_question' in df.columns else 0
                },
                'pain_point_analysis': {
                    'high_pain_points': int((df['pain_point_score'] > 0.5).sum()) if 'pain_point_score' in df.columns else 0,
                    'medium_pain_points': int(((df['pain_point_score'] > 0.2) & (df['pain_point_score'] <= 0.5)).sum()) if 'pain_point_score' in df.columns else 0,
                    'avg_pain_score': float(df['pain_point_score'].mean()) if 'pain_point_score' in df.columns else 0
                }
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
                
            print(f"📊 Statistics saved to {stats_file}")
            
        except Exception as e:
            print(f"⚠️ Error saving statistics: {e}")


def main():
    """Example usage of TextCleaner"""
    
    print("🚀 Starting data cleaning pipeline...")
    
    # Initialize cleaner
    cleaner = TextCleaner()
    
    # Process the comments data (primary for semantic search)
    comments_file = "data/fitness_comments.jsonl"
    if os.path.exists(comments_file):
        print("\n📝 Processing comments data...")
        output_file = cleaner.process_posts(
            input_file=comments_file,
            output_file="data/fitness_comments_clean.jsonl",
            chunk_strategy="semantic_text"
        )
        
        if output_file:
            print(f"✅ Clean comments data saved to: {output_file}")
    
    # Also process posts data  
    posts_file = "data/fitness_posts_with_comments.jsonl"
    if os.path.exists(posts_file):
        print("\n📰 Processing posts data...")
        posts_output = cleaner.process_posts(
            input_file=posts_file,
            output_file="data/fitness_posts_clean.jsonl",
            chunk_strategy="post"
        )
        
        if posts_output:
            print(f"✅ Clean posts data saved to: {posts_output}")
    
    print("\n🎉 Data cleaning pipeline complete!")
    print("📂 Check the data/ directory for cleaned files and statistics")


if __name__ == "__main__":
    main()
