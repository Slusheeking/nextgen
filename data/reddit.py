"""
Reddit API integration for financial sentiment analysis and discussions.

A production-ready client for accessing Reddit API to fetch posts, comments,
and track ticker mentions for algorithmic trading systems.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import praw
from dotenv import load_dotenv
from loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

# Load environment variables
load_dotenv()

# Setup traditional logging as fallback
logger = logging.getLogger(__name__)

class RedditClient:
    """
    Production client for the Reddit API focused on financial market sentiment.
    """

    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None, user_agent: Optional[str] = None):
        """
        Initialize the Reddit API client.
        
        Args:
            client_id: Reddit API client ID (defaults to REDDIT_CLIENT_ID environment variable)
            client_secret: Reddit API client secret (defaults to REDDIT_CLIENT_SECRET environment variable)
            user_agent: Reddit API user agent (defaults to REDDIT_USER_AGENT environment variable)
        """
        # Initialize observability tools
        self.loki = LokiManager(service_name="data-reddit")
        self.prom = PrometheusManager(service_name="data-reddit")
        
        # Create metrics
        self.request_counter = self.prom.create_counter(
            "reddit_api_requests_total", 
            "Total count of Reddit API requests",
            ["endpoint", "subreddit", "status"]
        )
        
        self.request_latency = self.prom.create_histogram(
            "reddit_api_request_duration_seconds",
            "Reddit API request duration in seconds",
            ["endpoint", "subreddit"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.posts_retrieved_gauge = self.prom.create_gauge(
            "reddit_posts_retrieved",
            "Number of posts retrieved from Reddit",
            ["endpoint", "subreddit"]
        )
        
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent or os.environ.get("REDDIT_USER_AGENT", "financial-sentiment-bot")
        
        if not self.client_id or not self.client_secret:
            log_msg = "Missing Reddit API credentials - API calls will fail"
            logger.warning(log_msg)
            self.loki.warning(log_msg, component="reddit")
        
        # Define recommended financial subreddits
        self.recommended_subreddits = [
            "wallstreetbets", 
            "investing", 
            "stocks", 
            "options",
            "SecurityAnalysis",
            "finance",
            "StockMarket"
        ]
        
        # Initialize the Reddit API client
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        logger.info("RedditClient initialized")
        self.loki.info("RedditClient initialized", component="reddit")

    def get_recent_posts(self, subreddit: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent posts from a subreddit.
        
        Args:
            subreddit: The subreddit to fetch posts from
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of posts with their details
        """
        logger.info(f"Fetching {limit} recent posts from r/{subreddit}")
        self.loki.info(f"Fetching {limit} recent posts from r/{subreddit}", 
                     component="reddit", 
                     endpoint="recent_posts", 
                     subreddit=subreddit)
        
        start_time = time.time()
        posts = []
        
        try:
            for submission in self.reddit.subreddit(subreddit).new(limit=limit):
                post = {
                    "id": submission.id,
                    "title": submission.title,
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "subreddit": subreddit
                }
                posts.append(post)
                
            elapsed = time.time() - start_time
            post_count = len(posts)
            
            # Record metrics
            self.prom.observe_histogram(
                "reddit_api_request_duration_seconds",
                elapsed,
                endpoint="recent_posts",
                subreddit=subreddit
            )
            
            self.prom.set_gauge(
                "reddit_posts_retrieved",
                post_count,
                endpoint="recent_posts",
                subreddit=subreddit
            )
            
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="recent_posts",
                subreddit=subreddit,
                status="success"
            )
            
            logger.info(f"Retrieved {post_count} posts from r/{subreddit}")
            self.loki.info(f"Retrieved {post_count} posts from r/{subreddit}", 
                         component="reddit", 
                         endpoint="recent_posts", 
                         subreddit=subreddit,
                         duration=f"{elapsed:.2f}",
                         post_count=post_count)
            
            return posts
            
        except Exception as e:
            error_msg = f"Error fetching posts from r/{subreddit}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="reddit", 
                          endpoint="recent_posts", 
                          subreddit=subreddit,
                          error_type="exception")
            
            # Record failure
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="recent_posts",
                subreddit=subreddit,
                status="error"
            )
            
            return []
    
    def search_ticker_mentions(self, ticker: str, subreddit: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for mentions of a specific ticker symbol.
        
        Args:
            ticker: Stock ticker symbol to search for
            subreddit: Specific subreddit to search (defaults to 'all')
            limit: Maximum number of results to return
            
        Returns:
            List of posts mentioning the ticker
        """
        sub = self.reddit.subreddit(subreddit) if subreddit else self.reddit.subreddit("all")
        target_sub = subreddit or "all"
        
        logger.info(f"Searching for {ticker} mentions in r/{target_sub}, limit={limit}")
        self.loki.info(f"Searching for {ticker} mentions in r/{target_sub}", 
                     component="reddit", 
                     endpoint="ticker_search", 
                     subreddit=target_sub,
                     ticker=ticker)
        
        start_time = time.time()
        
        mentions = []
        try:
            for submission in sub.search(ticker, limit=limit):
                mention = {
                    "id": submission.id,
                    "title": submission.title,
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "subreddit": submission.subreddit.display_name
                }
                mentions.append(mention)
                
            elapsed = time.time() - start_time
            mention_count = len(mentions)
            
            # Record metrics
            self.prom.observe_histogram(
                "reddit_api_request_duration_seconds",
                elapsed,
                endpoint="ticker_search",
                subreddit=target_sub
            )
            
            self.prom.set_gauge(
                "reddit_posts_retrieved",
                mention_count,
                endpoint="ticker_search",
                subreddit=target_sub
            )
            
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="ticker_search",
                subreddit=target_sub,
                status="success"
            )
            
            logger.info(f"Found {mention_count} mentions of {ticker} in r/{target_sub}")
            self.loki.info(f"Found {mention_count} mentions of {ticker} in r/{target_sub}", 
                         component="reddit", 
                         endpoint="ticker_search", 
                         subreddit=target_sub,
                         ticker=ticker,
                         duration=f"{elapsed:.2f}",
                         mention_count=mention_count)
            
            return mentions
            
        except Exception as e:
            error_msg = f"Error searching for {ticker} in r/{target_sub}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="reddit", 
                          endpoint="ticker_search", 
                          subreddit=target_sub,
                          ticker=ticker,
                          error_type="exception")
            
            # Record failure
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="ticker_search",
                subreddit=target_sub,
                status="error"
            )
            
            return []
    
    def get_hot_posts(self, subreddit: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get hot (trending) posts from a subreddit.
        
        Args:
            subreddit: The subreddit to fetch posts from
            limit: Maximum number of posts to retrieve
            
        Returns:
            List of hot posts with their details
        """
        logger.info(f"Fetching {limit} hot posts from r/{subreddit}")
        self.loki.info(f"Fetching {limit} hot posts from r/{subreddit}", 
                     component="reddit", 
                     endpoint="hot_posts", 
                     subreddit=subreddit)
        
        start_time = time.time()
        posts = []
        
        try:
            for submission in self.reddit.subreddit(subreddit).hot(limit=limit):
                post = {
                    "id": submission.id,
                    "title": submission.title,
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "subreddit": subreddit
                }
                posts.append(post)
                
            elapsed = time.time() - start_time
            post_count = len(posts)
            
            # Record metrics
            self.prom.observe_histogram(
                "reddit_api_request_duration_seconds",
                elapsed,
                endpoint="hot_posts",
                subreddit=subreddit
            )
            
            self.prom.set_gauge(
                "reddit_posts_retrieved",
                post_count,
                endpoint="hot_posts",
                subreddit=subreddit
            )
            
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="hot_posts",
                subreddit=subreddit,
                status="success"
            )
            
            logger.info(f"Retrieved {post_count} hot posts from r/{subreddit}")
            self.loki.info(f"Retrieved {post_count} hot posts from r/{subreddit}", 
                         component="reddit", 
                         endpoint="hot_posts", 
                         subreddit=subreddit,
                         duration=f"{elapsed:.2f}",
                         post_count=post_count)
            
            return posts
            
        except Exception as e:
            error_msg = f"Error fetching hot posts from r/{subreddit}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="reddit", 
                          endpoint="hot_posts", 
                          subreddit=subreddit,
                          error_type="exception")
            
            # Record failure
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="hot_posts",
                subreddit=subreddit,
                status="error"
            )
            
            return []
    
    def get_comments(self, submission_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get comments from a specific Reddit submission.
        
        Args:
            submission_id: The ID of the submission to fetch comments from
            limit: Maximum number of comments to retrieve
            
        Returns:
            List of comments with their details
        """
        logger.info(f"Fetching up to {limit} comments from submission {submission_id}")
        self.loki.info(f"Fetching up to {limit} comments from submission {submission_id}", 
                     component="reddit", 
                     endpoint="comments", 
                     submission_id=submission_id)
        
        start_time = time.time()
        comments = []
        
        try:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Only get top-level comments
            
            for comment in submission.comments[:limit]:
                comment_data = {
                    "id": comment.id,
                    "body": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "submission_id": submission_id
                }
                comments.append(comment_data)
                
            elapsed = time.time() - start_time
            comment_count = len(comments)
            
            # Record metrics
            self.prom.observe_histogram(
                "reddit_api_request_duration_seconds",
                elapsed,
                endpoint="comments",
                subreddit="[by_submission_id]"
            )
            
            # Create a special gauge for comments
            self.prom.set_gauge(
                "reddit_comments_retrieved",
                comment_count,
                endpoint="comments",
                submission_id=submission_id
            )
            
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="comments",
                subreddit="[by_submission_id]",
                status="success"
            )
            
            logger.info(f"Retrieved {comment_count} comments from submission {submission_id}")
            self.loki.info(f"Retrieved {comment_count} comments from submission {submission_id}", 
                         component="reddit", 
                         endpoint="comments", 
                         submission_id=submission_id,
                         duration=f"{elapsed:.2f}",
                         comment_count=comment_count)
            
            return comments
            
        except Exception as e:
            error_msg = f"Error fetching comments from submission {submission_id}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="reddit", 
                          endpoint="comments", 
                          submission_id=submission_id,
                          error_type="exception")
            
            # Record failure
            self.prom.increment_counter(
                "reddit_api_requests_total",
                1,
                endpoint="comments",
                subreddit="[by_submission_id]",
                status="error"
            )
            
            return []
    
    def calculate_sentiment_metrics(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate basic sentiment metrics from a list of posts.
        
        Args:
            posts: List of Reddit posts
            
        Returns:
            Dictionary containing sentiment metrics
        """
        if not posts:
            return {
                "post_count": 0,
                "avg_score": 0,
                "avg_comments": 0,
                "total_engagement": 0
            }
        
        post_count = len(posts)
        total_score = sum(post.get("score", 0) for post in posts)
        total_comments = sum(post.get("num_comments", 0) for post in posts)
        avg_score = total_score / post_count
        avg_comments = total_comments / post_count
        total_engagement = total_score + total_comments
        
        return {
            "post_count": post_count,
            "avg_score": avg_score,
            "avg_comments": avg_comments,
            "total_engagement": total_engagement
        }
    
    def get_wsb_sentiment(self, ticker: str = None, limit: int = 100) -> Dict[str, Any]:
        """
        Get WallStreetBets sentiment, either general or for a specific ticker.
        
        Args:
            ticker: Optional stock ticker to filter by
            limit: Maximum number of posts to analyze
            
        Returns:
            Dictionary containing sentiment metrics and posts
        """
        logger.info(f"Fetching WallStreetBets sentiment{' for ' + ticker if ticker else ''}")
        self.loki.info(f"Fetching WallStreetBets sentiment{' for ' + ticker if ticker else ''}", 
                     component="reddit", 
                     endpoint="wsb_sentiment",
                     ticker=ticker if ticker else "general")
        
        if ticker:
            posts = self.search_ticker_mentions(ticker, subreddit="wallstreetbets", limit=limit)
        else:
            posts = self.get_hot_posts("wallstreetbets", limit=limit)
            
        metrics = self.calculate_sentiment_metrics(posts)
        
        # Record a separate metric for sentiment analysis
        if ticker:
            self.prom.set_gauge(
                "reddit_wsb_sentiment",
                metrics.get("total_engagement", 0),
                ticker=ticker
            )
            
        result = {
            "sentiment_metrics": metrics,
            "posts": posts[:10]  # Return only up to 10 posts to keep the response manageable
        }
        
        self.loki.info(f"Calculated WSB sentiment metrics", 
                     component="reddit", 
                     endpoint="wsb_sentiment",
                     ticker=ticker if ticker else "general",
                     post_count=metrics.get("post_count", 0),
                     engagement=metrics.get("total_engagement", 0))
        
        return result
