"""
Chunk 1 — Data Collection
Reddit data collection using PRAW (Python Reddit API Wrapper)

Learning Objectives:
- Understand Reddit API structure and rate limits
- Practice API authentication and data extraction
- Learn about data serialization (JSONL/CSV)
- Handle API errors and pagination

TODO for you to implement:
1. Set up Reddit API credentials
2. Implement post collection logic
3. Add proper error handling and rate limiting
4. Implement data filtering (date range, post quality)
5. Add progress tracking with tqdm
"""

import praw
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Load environment variables
load_dotenv()


class RedditCollector:
    """Collects posts from Reddit using PRAW"""
    
    def __init__(self):
        """Initialize Reddit API client"""
        self.reddit = None

        

    def setup_reddit_client(self) -> bool:
        """
        Set up Reddit API client with credentials
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Initialize self.reddit with proper credentials for read-only access
            # Get credentials from environment variables
            client_id = os.getenv("REDDIT_CLIENT_ID")
            client_secret = os.getenv("REDDIT_CLIENT_SECRET")
            user_agent = os.getenv("REDDIT_USER_AGENT")
            
            if not all([client_id, client_secret, user_agent]):
                print("❌ Missing required environment variables. Check your .env file.")
                return False

            # For read-only access, we don't need refresh_token
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test the connection by accessing a public subreddit
            test_subreddit = self.reddit.subreddit("test")
            test_subreddit.display_name  # This will raise an exception if auth fails
            
            print("✅ Reddit API client initialized successfully")
            print(f"📊 Reddit instance read-only: {self.reddit.read_only}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Reddit client: {e}")
            return False
    
    def collect_posts(self, 
                     subreddit_name: str = "Fitness",
                     time_period: str = "month", 
                     limit: int = 1000,
                     output_file: str = "data/reddit_posts.jsonl",
                     collect_comments: bool = False,
                     max_comments_per_post: int = 20) -> Optional[str]:
        """
        Collect posts from a specific subreddit
        
        Args:
            subreddit_name: Name of the subreddit (without r/)
            time_period: Time period for collection ("week", "month", "year", "all")
            limit: Maximum number of posts to collect
            output_file: Path to save the collected data
            collect_comments: Whether to collect comments for each post
            max_comments_per_post: Maximum number of comments to collect per post
            
        Returns:
            str: Path to output file if successful, None otherwise
        """
        
        if not self.reddit:
            print("❌ Reddit client not initialized. Call setup_reddit_client() first.")
            return None
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            print(f"🔄 Collecting posts from r/{subreddit_name}...")
            print(f"📅 Time period: {time_period}, Limit: {limit}")
            
            # Get subreddit object
            subreddit = self.reddit.subreddit(subreddit_name)
            
            # Get posts based on time period
            if time_period == "week":
                posts = subreddit.top(time_filter="week", limit=limit)
            elif time_period == "month":
                posts = subreddit.top(time_filter="month", limit=limit)
            elif time_period == "year":
                posts = subreddit.top(time_filter="year", limit=limit)
            elif time_period == "all":
                posts = subreddit.top(time_filter="all", limit=limit)
            else:
                # Default to hot posts if time_period not recognized
                posts = subreddit.hot(limit=limit)
            
            posts_collected = []
            
            # Iterate through posts with progress bar
            for post in tqdm(posts, desc="Collecting posts", unit="post"):
                try:
                    # Extract data from each post
                    post_data = self._extract_post_data(post)
                    
                    # Collect comments if requested
                    if collect_comments:
                        post_data['comments'] = self._collect_post_comments(
                            post, max_comments_per_post
                        )
                    
                    posts_collected.append(post_data)
                    
                    # Add delay to respect rate limits (longer if collecting comments)
                    time.sleep(0.2 if collect_comments else 0.1)
                    
                except Exception as e:
                    print(f"⚠️ Error processing post {getattr(post, 'id', 'unknown')}: {e}")
                    continue
            
            # Save to JSONL
            self._save_to_jsonl(posts_collected, output_file)
            
            print(f"✅ Collected {len(posts_collected)} posts from r/{subreddit_name}")
            print(f"💾 Data saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error collecting posts: {e}")
            return None
    
    def _save_to_jsonl(self, posts: List[Dict], output_file: str):
        """Save posts to JSONL format"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for post in posts:
                json.dump(post, f, ensure_ascii=False)
                f.write('\n')
    
    def _extract_post_data(self, post) -> Dict:
        """
        Extract relevant data from a Reddit post object
        
        Args:
            post: PRAW Submission object
            
        Returns:
            Dict: Cleaned post data
        """
        try:
            # Extract author name safely
            author_name = "[deleted]"
            if hasattr(post, 'author') and post.author:
                try:
                    author_name = str(post.author)
                except:
                    author_name = "[unavailable]"
            
            # Extract flair safely
            flair = ""
            if hasattr(post, 'link_flair_text') and post.link_flair_text:
                flair = str(post.link_flair_text)
            
            # Convert UTC timestamp to human-readable format
            created_time = datetime.fromtimestamp(post.created_utc)
            
            return {
                "id": post.id,
                "title": post.title,
                "body": post.selftext if post.selftext else "",
                "author": author_name,
                # score is the number of upvotes minus the number of downvotes 
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
                "created_datetime": created_time.isoformat(),
                "url": post.url,
                "subreddit": str(post.subreddit),
                # User-assigned or moderator-assigned category tags for posts
                    # Examples in r/Fitness:
                    # "Form Check" - Posts asking for exercise form feedback
                    # "Nutrition" - Posts about diet and nutrition
                    # "Equipment" - Posts about gym equipment
                    # "Injury Recovery" - Posts about dealing with injuries
                    # "Success Story" - Posts celebrating achievements
                    # "Rant" - Posts venting frustrations
                "flair": flair,
                "is_self": post.is_self,
                "over_18": post.over_18,
                "spoiler": post.spoiler,
                # Posts pinned to the top of the subreddit by moderators
                "stickied": post.stickied,
                "upvote_ratio": getattr(post, 'upvote_ratio', 0.0),
                "permalink": f"https://reddit.com{post.permalink}"
            }
        except Exception as e:
            print(f"⚠️ Error extracting data from post: {e}")
            # Return minimal data if extraction fails
            return {
                "id": getattr(post, 'id', 'unknown'),
                "title": getattr(post, 'title', 'Error extracting title'),
                "body": "",
                "author": "[error]",
                "score": 0,
                "num_comments": 0,
                "created_utc": 0,
                "created_datetime": "",
                "url": "",
                "subreddit": "",
                "flair": "",
                "is_self": False,
                "over_18": False,
                "spoiler": False,
                "stickied": False,
                "upvote_ratio": 0.0,
                "permalink": ""
            }
    
    def _collect_post_comments(self, post, max_comments: int = 20) -> List[Dict]:
        """
        Collect comments for a specific post, focusing on pain points
        
        Args:
            post: PRAW Submission object
            max_comments: Maximum number of comments to collect
            
        Returns:
            List[Dict]: List of comment data optimized for pain point detection
        """
        try:
            # Configure comment collection to expand "MoreComments" objects
            post.comments.replace_more(limit=0)
            
            comments_collected = []
            comment_count = 0
            
            # Collect top-level comments first (often contain main pain points)
            for comment in post.comments.list()[:max_comments]:
                if comment_count >= max_comments:
                    break
                    
                comment_data = self._extract_comment_data(comment, post)
                if comment_data:  # Only add if extraction successful
                    comments_collected.append(comment_data)
                    comment_count += 1
            
            # Sort comments by pain point likelihood (score, length, keywords)
            comments_collected = self._prioritize_pain_point_comments(comments_collected)
            
            return comments_collected[:max_comments]
            
        except Exception as e:
            print(f"⚠️ Error collecting comments for post {post.id}: {e}")
            return []
    
    def _extract_comment_data(self, comment, parent_post) -> Optional[Dict]:
        """
        Extract comment data with focus on semantic search and pain point detection
        
        Args:
            comment: PRAW Comment object
            parent_post: PRAW Submission object (for context)
            
        Returns:
            Dict: Comment data optimized for pain point analysis
        """
        try:
            # Skip deleted, removed, or very short comments
            if (hasattr(comment, 'body') and 
                comment.body in ['[deleted]', '[removed]', None] or
                len(getattr(comment, 'body', '')) < 10):
                return None
            
            # Extract author safely
            author_name = "[deleted]"
            if hasattr(comment, 'author') and comment.author:
                try:
                    author_name = str(comment.author)
                except:
                    author_name = "[unavailable]"
            
            # Determine comment type for pain point analysis
            comment_type = self._classify_comment_type(comment.body)
            
            # Create semantic search optimized text
            semantic_text = self._create_semantic_text(comment, parent_post)
            
            return {
                "comment_id": comment.id,
                "post_id": parent_post.id,
                "parent_id": getattr(comment.parent(), 'id', None) if hasattr(comment, 'parent') else None,
                "author": author_name,
                "body": comment.body,
                "semantic_text": semantic_text,  # Combined text for embeddings
                "score": comment.score,
                "created_utc": comment.created_utc,
                "created_datetime": datetime.fromtimestamp(comment.created_utc).isoformat(),
                "depth": getattr(comment, 'depth', 0),
                "is_submitter": comment.is_submitter,
                "controversiality": getattr(comment, 'controversiality', 0),
                "comment_type": comment_type,  # pain_point, advice, question, etc.
                "pain_point_score": self._calculate_pain_point_score(comment.body),
                "permalink": f"https://reddit.com{comment.permalink}",
                "post_context": {
                    "post_title": parent_post.title,
                    "post_flair": getattr(parent_post, 'link_flair_text', '') or '',
                    "subreddit": str(parent_post.subreddit)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error extracting comment data: {e}")
            return None
    
    def _classify_comment_type(self, comment_body: str) -> str:
        """
        Classify comment type to prioritize pain points for semantic search
        
        Args:
            comment_body: Comment text
            
        Returns:
            str: Comment classification
        """
        text_lower = comment_body.lower()
        
        # Pain point indicators (these are most valuable for your agent)
        pain_indicators = [
            'pain', 'hurt', 'ache', 'injury', 'problem', 'issue', 'struggle',
            'difficult', 'can\'t', 'unable', 'fail', 'wrong', 'mistake',
            'frustrated', 'annoying', 'bothering', 'uncomfortable'
        ]
        
        # Question indicators
        question_indicators = [
            'how', 'what', 'why', 'when', 'where', 'which', 'should i',
            'can i', 'is it', 'does', 'would', '?'
        ]
        
        # Advice indicators
        advice_indicators = [
            'try', 'suggest', 'recommend', 'should', 'better', 'instead',
            'helped me', 'worked for me', 'tip', 'advice'
        ]
        
        # Count indicators
        pain_count = sum(1 for indicator in pain_indicators if indicator in text_lower)
        question_count = sum(1 for indicator in question_indicators if indicator in text_lower)
        advice_count = sum(1 for indicator in advice_indicators if indicator in text_lower)
        
        # Classify based on strongest signal
        if pain_count >= 2 or any(strong in text_lower for strong in ['pain', 'hurt', 'injury']):
            return 'pain_point'
        elif question_count >= 2 or text_lower.strip().endswith('?'):
            return 'question'
        elif advice_count >= 2:
            return 'advice'
        elif len(comment_body) > 200:
            return 'detailed_response'
        else:
            return 'general'
    
    def _calculate_pain_point_score(self, comment_body: str) -> float:
        """
        Calculate likelihood that comment contains pain points (0.0 to 1.0)
        
        Args:
            comment_body: Comment text
            
        Returns:
            float: Pain point likelihood score
        """
        text_lower = comment_body.lower()
        score = 0.0
        
        # High-value pain point keywords
        high_value_keywords = {
            'pain': 0.3, 'injury': 0.3, 'hurt': 0.25, 'ache': 0.2,
            'problem': 0.15, 'issue': 0.15, 'struggle': 0.2,
            'can\'t': 0.1, 'unable': 0.1, 'difficult': 0.1
        }
        
        # Add score for each keyword found
        for keyword, value in high_value_keywords.items():
            if keyword in text_lower:
                score += value
        
        # Bonus for first-person pain descriptions
        first_person_indicators = ['i have', 'i get', 'my', 'me', 'i feel']
        if any(indicator in text_lower for indicator in first_person_indicators):
            score += 0.1
        
        # Penalty for very short comments
        if len(comment_body) < 30:
            score *= 0.5
        
        # Bonus for longer, detailed descriptions
        if len(comment_body) > 100:
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _create_semantic_text(self, comment, parent_post) -> str:
        """
        Create optimized text for semantic search embeddings
        
        Args:
            comment: PRAW Comment object
            parent_post: PRAW Submission object
            
        Returns:
            str: Combined text optimized for semantic understanding
        """
        # Combine post context with comment for better semantic understanding
        semantic_parts = []
        
        # Add post title for context
        semantic_parts.append(f"Post: {parent_post.title}")
        
        # Add post flair if available (often indicates topic area)
        if hasattr(parent_post, 'link_flair_text') and parent_post.link_flair_text:
            semantic_parts.append(f"Topic: {parent_post.link_flair_text}")
        
        # Add brief post context if available
        if hasattr(parent_post, 'selftext') and parent_post.selftext:
            post_preview = parent_post.selftext[:150] + "..." if len(parent_post.selftext) > 150 else parent_post.selftext
            semantic_parts.append(f"Context: {post_preview}")
        
        # Add the comment itself
        semantic_parts.append(f"Comment: {comment.body}")
        
        return " | ".join(semantic_parts)
    
    def _prioritize_pain_point_comments(self, comments: List[Dict]) -> List[Dict]:
        """
        Sort comments by pain point likelihood for semantic search optimization
        
        Args:
            comments: List of comment data dictionaries
            
        Returns:
            List[Dict]: Comments sorted by pain point relevance
        """
        def sort_key(comment):
            # Primary: pain point score (higher is better)
            pain_score = comment.get('pain_point_score', 0)
            
            # Secondary: comment score (higher is better, but normalized)
            reddit_score = max(0, comment.get('score', 0)) / 100.0
            
            # Tertiary: length (moderate length preferred)
            length_score = min(len(comment.get('body', '')), 500) / 500.0
            
            # Combined score
            return (pain_score * 0.6) + (reddit_score * 0.3) + (length_score * 0.1)
        
        return sorted(comments, key=sort_key, reverse=True)
    
    def convert_to_csv(self, jsonl_file: str, csv_file: str = None) -> str:
        """
        Convert JSONL file to CSV format
        
        Args:
            jsonl_file: Path to input JSONL file
            csv_file: Path to output CSV file (optional)
            
        Returns:
            str: Path to CSV file
        """
        if csv_file is None:
            csv_file = jsonl_file.replace('.jsonl', '.csv')
        
        try:
            # Read JSONL file and convert to pandas DataFrame
            posts = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            post = json.loads(line.strip())
                            posts.append(post)
                        except json.JSONDecodeError as e:
                            print(f"⚠️ Error parsing JSON line: {e}")
                            continue
            
            if not posts:
                print("❌ No valid posts found in JSONL file")
                return None
            
            # Convert to DataFrame and save as CSV
            df = pd.DataFrame(posts)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            print(f"✅ Converted {jsonl_file} to {csv_file}")
            print(f"📊 Converted {len(posts)} posts to CSV format")
            return csv_file
            
        except Exception as e:
            print(f"❌ Error converting to CSV: {e}")
            return None
    
    def save_comments_separately(self, posts_data: List[Dict], comments_file: str = "data/comments.jsonl") -> str:
        """
        Extract and save all comments from posts to a separate file
        
        Args:
            posts_data: List of post data dictionaries containing comments
            comments_file: Path to save comments file
            
        Returns:
            str: Path to comments file
        """
        try:
            all_comments = []
            
            for post in posts_data:
                if 'comments' in post and post['comments']:
                    all_comments.extend(post['comments'])
            
            if not all_comments:
                print("⚠️ No comments found in posts data")
                return None
            
            # Save comments to separate JSONL file
            os.makedirs(os.path.dirname(comments_file), exist_ok=True)
            with open(comments_file, 'w', encoding='utf-8') as f:
                for comment in all_comments:
                    json.dump(comment, f, ensure_ascii=False)
                    f.write('\n')
            
            print(f"💬 Saved {len(all_comments)} comments to {comments_file}")
            
            # Create CSV version for easy inspection
            comments_csv = comments_file.replace('.jsonl', '.csv')
            df_comments = pd.DataFrame(all_comments)
            df_comments.to_csv(comments_csv, index=False, encoding='utf-8')
            
            print(f"📊 Comments analysis:")
            print(f"   Total comments: {len(all_comments)}")
            print(f"   Pain point comments: {sum(1 for c in all_comments if c.get('comment_type') == 'pain_point')}")
            print(f"   Question comments: {sum(1 for c in all_comments if c.get('comment_type') == 'question')}")
            print(f"   Advice comments: {sum(1 for c in all_comments if c.get('comment_type') == 'advice')}")
            print(f"   Average pain point score: {sum(c.get('pain_point_score', 0) for c in all_comments) / len(all_comments):.3f}")
            
            return comments_file
            
        except Exception as e:
            print(f"❌ Error saving comments: {e}")
            return None


def main():
    """Example usage of RedditCollector"""
    
    print("🚀 Starting Reddit data collection...")
    
    # Initialize collector
    collector = RedditCollector()
    
    # Setup Reddit client
    if not collector.setup_reddit_client():
        print("Failed to setup Reddit client. Check your credentials.")
        return
    
    # Collect posts with comments for pain point analysis
    output_file = collector.collect_posts(
        subreddit_name="Fitness",
        time_period="week",  # Start with a smaller dataset
        limit=5,  # Small number for testing comments
        output_file="data/fitness_posts_with_comments.jsonl",
        collect_comments=True,  # Enable comment collection
        max_comments_per_post=10  # Limit comments per post for testing
    )
    
    if output_file:
        # Convert to CSV for easier inspection
        csv_file = collector.convert_to_csv(output_file)
        if csv_file:
            print(f"📁 Data saved to: {output_file} and {csv_file}")
            
            # Load the raw posts data to extract comments
            try:
                posts_data = []
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            posts_data.append(json.loads(line.strip()))
                
                # Save comments separately for semantic search analysis
                comments_file = collector.save_comments_separately(
                    posts_data, "data/fitness_comments.jsonl"
                )
                
                # Show basic statistics
                df = pd.read_csv(csv_file)
                print(f"\n📊 Collection Summary:")
                print(f"   Total posts: {len(df)}")
                print(f"   Date range: {df['created_datetime'].min()} to {df['created_datetime'].max()}")
                print(f"   Average score: {df['score'].mean():.1f}")
                print(f"   Average comments per post: {df['num_comments'].mean():.1f}")
                print(f"   Self posts: {df['is_self'].sum()}")
                
                # Show pain point analysis if comments were collected
                if comments_file:
                    print(f"\n🎯 Pain Point Analysis Complete!")
                    print(f"   Comments saved to: {comments_file}")
                    print(f"   Comments CSV: {comments_file.replace('.jsonl', '.csv')}")
                
                print(f"\n📋 Sample titles:")
                for title in df['title'].head(3):
                    print(f"   - {title[:80]}...")
                    
            except Exception as e:
                print(f"⚠️ Error reading summary: {e}")
    else:
        print("❌ Data collection failed")


if __name__ == "__main__":
    main()
