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
        # TODO: Define regex patterns for cleaning
        # Common patterns to clean:
        # - URLs, email addresses
        # - Reddit-specific formatting (u/username, r/subreddit)
        # - Special characters, excessive whitespace
        # - HTML entities
        
        self.url_pattern = None  # TODO: regex for URLs
        self.reddit_user_pattern = None  # TODO: regex for u/username
        self.reddit_sub_pattern = None  # TODO: regex for r/subreddit
        self.email_pattern = None  # TODO: regex for emails
        self.html_pattern = None  # TODO: regex for HTML tags
        
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
                # TODO: Implement JSONL loading
                # Read line by line and parse JSON
                posts = []
                # with open(file_path, 'r', encoding='utf-8') as f:
                #     for line in f:
                #         posts.append(json.loads(line.strip()))
                return pd.DataFrame(posts)
                
            elif file_path.endswith('.csv'):
                # TODO: Load CSV file
                return pd.read_csv(file_path)
                
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
        
        # TODO: Implement comprehensive text cleaning
        # Steps:
        # 1. Remove URLs
        # 2. Remove Reddit-specific formatting (u/user, r/sub)
        # 3. Remove HTML tags and entities
        # 4. Handle emojis (remove or convert to text)
        # 5. Normalize whitespace
        # 6. Remove excessive punctuation
        # 7. Convert to lowercase (optional - consider keeping case for some analyses)
        
        cleaned = text
        
        # Placeholder cleaning steps - implement these!
        # cleaned = self._remove_urls(cleaned)
        # cleaned = self._remove_reddit_formatting(cleaned)
        # cleaned = self._remove_html(cleaned)
        # cleaned = self._handle_emojis(cleaned)
        # cleaned = self._normalize_whitespace(cleaned)
        
        return cleaned.strip()
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # TODO: Implement URL removal
        return text
    
    def _remove_reddit_formatting(self, text: str) -> str:
        """Remove Reddit-specific formatting"""
        # TODO: Remove u/username and r/subreddit references
        return text
    
    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and entities"""
        # TODO: Remove HTML tags and decode entities
        return text
    
    def _handle_emojis(self, text: str, strategy: str = "remove") -> str:
        """
        Handle emojis in text
        
        Args:
            text: Input text
            strategy: "remove", "keep", or "convert_to_text"
        """
        # TODO: Implement emoji handling
        # Options:
        # - Remove: emoji.demojize() then remove
        # - Keep: keep as-is
        # - Convert: emoji.demojize() to convert to :emoji_name:
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text"""
        # TODO: Remove excessive whitespace, normalize line breaks
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
        
        # TODO: Implement filtering criteria
        # 1. Remove deleted/removed posts
        # 2. Remove very short posts (< 10 words)
        # 3. Remove very long posts (> 1000 words) - might be spam
        # 4. Remove posts with excessive special characters
        # 5. Remove posts that are mostly URLs
        # 6. Filter by score/engagement (optional)
        
        # Example filters (implement these):
        # df = df[df['title'].str.len() > 10]  # Title length
        # df = df[df['body'].str.len() > 20]   # Body length
        # df = df[df['score'] > 0]             # Positive score
        
        print(f"📊 After filtering: {len(df)} posts")
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
        
        # TODO: Implement sentence splitting
        # Options:
        # 1. Simple regex splitting on .!?
        # 2. Use nltk.sent_tokenize (more accurate)
        # 3. Use spaCy for more advanced splitting
        
        # Simple implementation to start:
        # sentences = re.split(r'[.!?]+', text)
        # return [s.strip() for s in sentences if s.strip()]
        
        return [text]  # Placeholder - return full text as single "sentence"
    
    def create_chunks(self, df: pd.DataFrame, chunk_strategy: str = "post") -> pd.DataFrame:
        """
        Create text chunks for embedding
        
        Args:
            df: DataFrame with cleaned posts
            chunk_strategy: "post", "sentence", or "paragraph"
            
        Returns:
            pd.DataFrame: DataFrame with text chunks
        """
        # TODO: Implement chunking strategies
        # 1. "post": Use entire post (title + body)
        # 2. "sentence": Split into sentences
        # 3. "paragraph": Split by paragraphs
        
        chunks = []
        
        for idx, row in df.iterrows():
            if chunk_strategy == "post":
                # Combine title and body
                text = f"{row['title']} {row['body']}".strip()
                if text:
                    chunks.append({
                        'post_id': row['id'],
                        'chunk_id': f"{row['id']}_0",
                        'text': text,
                        'chunk_type': 'full_post',
                        'original_index': idx
                    })
            
            elif chunk_strategy == "sentence":
                # TODO: Split into sentences
                pass
            
            elif chunk_strategy == "paragraph":
                # TODO: Split into paragraphs
                pass
        
        return pd.DataFrame(chunks)
    
    def add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add text statistics and features
        
        Args:
            df: DataFrame with text data
            
        Returns:
            pd.DataFrame: DataFrame with additional features
        """
        # TODO: Add text features
        # - word_count, char_count, sentence_count
        # - readability_score (optional)
        # - sentiment_polarity (optional)
        # - contains_question, contains_urls, etc.
        
        if 'text' in df.columns:
            # df['word_count'] = df['text'].str.split().str.len()
            # df['char_count'] = df['text'].str.len()
            # df['sentence_count'] = df['text'].apply(lambda x: len(self.split_into_sentences(x)))
            pass
        
        return df
    
    def process_posts(self, 
                     input_file: str,
                     output_file: str = None,
                     chunk_strategy: str = "post") -> str:
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
            output_file = input_file.replace('.jsonl', '_clean.jsonl').replace('.csv', '_clean.csv')
        
        print("🧹 Starting text preprocessing pipeline...")
        
        # Load data
        df = self.load_data(input_file)
        if df.empty:
            print("❌ No data loaded")
            return None
        
        # Clean text fields
        if 'title' in df.columns:
            df['title'] = df['title'].apply(self.clean_text)
        if 'body' in df.columns:
            df['body'] = df['body'].apply(self.clean_text)
        
        # Filter posts
        df = self.filter_posts(df)
        
        # Create chunks
        chunks_df = self.create_chunks(df, chunk_strategy)
        
        # Add text features
        chunks_df = self.add_text_features(chunks_df)
        
        # Save processed data
        self._save_data(chunks_df, output_file)
        
        print(f"✅ Preprocessing complete. Output: {output_file}")
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
            # Default to CSV
            df.to_csv(output_file, index=False)


def main():
    """Example usage of TextCleaner"""
    
    # Initialize cleaner
    cleaner = TextCleaner()
    
    # Process the data
    input_file = "data/fitness_posts.jsonl"
    output_file = cleaner.process_posts(
        input_file=input_file,
        output_file="data/fitness_posts_clean.jsonl",
        chunk_strategy="post"
    )
    
    if output_file:
        print(f"✅ Clean data saved to: {output_file}")
        
        # Show some statistics
        df = cleaner.load_data(output_file)
        if not df.empty and 'text' in df.columns:
            print("\n📊 Dataset Statistics:")
            print(f"Total chunks: {len(df)}")
            if 'word_count' in df.columns:
                print(f"Avg words per chunk: {df['word_count'].mean():.1f}")
            print(f"Sample text: {df['text'].iloc[0][:100]}...")


if __name__ == "__main__":
    main()
