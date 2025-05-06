
"""
Reddit MCP Server

This module implements a Model Context Protocol (MCP) server for the Reddit API,
providing access to posts, comments, and sentiment analysis for financial subreddits.
"""

import os
import time

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
from typing import Dict, List, Any, Optional

# Direct imports with graceful error handling
import importlib

# Try to import required dependencies
try:
    import praw
except ImportError:
    praw = None

try:
    import dotenv
except ImportError:
    dotenv = None

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP

# Load environment variables


class RedditMCP(BaseDataMCP):
    """
    MCP server for Reddit API.

    This server provides access to Reddit data including posts, comments,
    and sentiment analysis for financial subreddits.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Reddit MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - client_id: Reddit API client ID (overrides environment variable)
                - client_secret: Reddit API client secret (overrides environment variable)
                - user_agent: Reddit API user agent (overrides environment variable)
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 300)
                - default_limit: Default number of posts to retrieve
        """
        super().__init__(name="reddit_mcp", config=config)

        # Initialize monitoring/logger
        self.logger = NetdataLogger(component_name="reddit-mcp")
        self.logger.info("RedditMCP initialized")

        # Initialize client configuration
        self.reddit_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Initialize recommended financial subreddits
        self.financial_subreddits = [
            "wallstreetbets",
            "investing",
            "stocks",
            "options",
            "SecurityAnalysis",
            "finance",
            "StockMarket",
        ]

        # Initialize sentiment analyzer
        self._initialize_sentiment_analyzer()

        # Register specific tools
        self._register_specific_tools()

        self.logger.info(f"RedditMCP initialized with {len(self.endpoints)} endpoints")

    def _initialize_client(self) -> Dict[str, Any]:
        """
        Initialize Reddit client configuration.

        Returns:
            Client configuration or None if initialization fails
        """
        try:
            # Prioritize environment variables for credentials
            client_id = os.environ.get("REDDIT_CLIENT_ID") or os.environ.get("CLIENT_ID") or self.config.get("client_id")
            client_secret = os.environ.get("REDDIT_CLIENT_SECRET") or os.environ.get("CLIENT_SECRET") or self.config.get("client_secret")
            user_agent = os.environ.get("REDDIT_USER_AGENT") or os.environ.get("USER_AGENT") or self.config.get("user_agent", "fingpt-mcp-reddit")

            if not client_id or not client_secret:
                self.logger.error("Missing Reddit API credentials - API calls will fail")
                return None

            # Initialize PRAW client
            reddit = praw.Reddit(
                client_id=client_id, client_secret=client_secret, user_agent=user_agent
            )

            self.logger.info(f"Loaded Reddit API credentials: client_id={client_id[:4]}...{client_id[-4:]}")

            return {
                "client_id": client_id,
                "client_secret": client_secret,
                "user_agent": user_agent,
                "reddit": reddit,
                "default_limit": self.config.get("default_limit", 100),
            }

        except Exception as e:
            self.logger.error(f"Failed to initialize Reddit client: {e}")
            self.logger.counter("error_count", 1)
            return None

    def _initialize_sentiment_analyzer(self):
        """Initialize the sentiment analyzer for Reddit content."""
        try:
            # Initialize VADER sentiment analysis if available
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer for Reddit content")
        except ImportError:
            self.logger.warning(
                "NLTK VADER not available, using simple keyword-based sentiment analysis"
            )
            #
            # Initialize a dictionary of positive and negative terms for basic
            # sentiment analysis
            self.positive_terms = set(
                [
                    "bullish",
                    "buy",
                    "calls",
                    "moon",
                    "rocket",
                    "up",
                    "gains",
                    "long",
                    "good",
                    "great",
                    "growth",
                    "positive",
                    "profit",
                    "strong",
                    "success",
                    "undervalued",
                    "winner",
                    "winning",
                    "beat",
                    "beats",
                    "breakout",
                    "catalyst",
                    "climb",
                    "dividend",
                    "double",
                    "hold",
                    "holding",
                    "jump",
                    "opportunity",
                    "potential",
                    "rally",
                    "recovery",
                    "rise",
                    "rising",
                    "soar",
                    "surge",
                    "target",
                    "upside",
                    "uptrend",
                    "yolo",
                ]
            )
            self.negative_terms = set(
                [
                    "bearish",
                    "sell",
                    "puts",
                    "short",
                    "down",
                    "crash",
                    "dump",
                    "losses",
                    "bad",
                    "debt",
                    "decline",
                    "decrease",
                    "disappointing",
                    "drop",
                    "fail",
                    "falling",
                    "fear",
                    "loss",
                    "negative",
                    "overvalued",
                    "plummet",
                    "poor",
                    "risk",
                    "risky",
                    "scam",
                    "sell",
                    "selling",
                    "sink",
                    "slump",
                    "tank",
                    "trouble",
                    "warning",
                    "weak",
                    "worse",
                    "worst",
                    "worthless",
                ]
            )
            self.sentiment_analyzer = None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Reddit API.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "recent_posts": {
                "description": "Get recent posts from a subreddit",
                "category": "posts",
                "required_params": ["subreddit"],
                "optional_params": ["limit", "sort"],
                "default_values": {
                    "limit": str(
                        self.reddit_client.get("default_limit", 100)
                        if self.reddit_client
                        else 100
                    ),
                    "sort": "new",  # new, hot, top, rising
                },
                "handler": self._handle_recent_posts,
            },
            "hot_posts": {
                "description": "Get hot (trending) posts from a subreddit",
                "category": "posts",
                "required_params": ["subreddit"],
                "optional_params": ["limit"],
                "default_values": {
                    "limit": str(
                        self.reddit_client.get("default_limit", 100)
                        if self.reddit_client
                        else 100
                    )
                },
                "handler": self._handle_hot_posts,
            },
            "ticker_mentions": {
                "description": "Search for mentions of a specific ticker symbol",
                "category": "search",
                "required_params": ["ticker"],
                "optional_params": ["subreddit", "limit"],
                "default_values": {
                    "limit": str(
                        self.reddit_client.get("default_limit", 100)
                        if self.reddit_client
                        else 100
                    )
                },
                "handler": self._handle_ticker_mentions,
            },
            "post_comments": {
                "description": "Get comments from a specific Reddit post",
                "category": "comments",
                "required_params": ["post_id"],
                "optional_params": ["limit"],
                "default_values": {"limit": "100"},
                "handler": self._handle_post_comments,
            },
            "sentiment_analysis": {
                "description": "Analyze sentiment for a subreddit or ticker",
                "category": "analysis",
                "required_params": [],
                "optional_params": ["subreddit", "ticker", "limit", "time_period"],
                "default_values": {
                    "limit": "100",
                    "time_period": "day",  # hour, day, week, month
                },
                "handler": self._handle_sentiment_analysis,
            },
            "wsb_sentiment": {
                "description": "Get WallStreetBets sentiment for a ticker",
                "category": "analysis",
                "required_params": [],
                "optional_params": ["ticker", "limit"],
                "default_values": {"limit": "100"},
                "handler": self._handle_wsb_sentiment,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Reddit API."""
        self.register_tool(self.get_subreddit_posts)
        self.register_tool(self.search_ticker_mentions)
        self.register_tool(self.get_post_comments)
        self.register_tool(self.get_sentiment)
        self.register_tool(self.get_wsb_sentiment)
        self.register_tool(self.get_financial_subreddits)

    def _execute_endpoint_fetch(
        self, endpoint_config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the fetch operation based on endpoint configuration.

        Args:
            endpoint_config: Configuration for the endpoint
            params: Parameters for the request

        Returns:
            Fetched data

        Raises:
            Exception: If the fetch fails
        """
        # Merge provided params with default values
        merged_params = {**endpoint_config.get("default_values", {}), **params}

        # Get the handler function for this endpoint
        handler = endpoint_config.get("handler")
        if handler and callable(handler):
            return handler(merged_params)
        else:
            raise ValueError("No handler defined for endpoint")

    # Handler methods for specific endpoints

    def _handle_recent_posts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recent posts endpoint."""
        subreddit = params.get("subreddit")
        if not subreddit:
            raise ValueError("Missing required parameter: subreddit")

        limit = int(params.get("limit", 100))
        sort = params.get("sort", "new")

        try:
            start_time = time.time()
            if not self.reddit_client or not self.reddit_client.get("reddit"):
                return {"error": "Reddit client not initialized"}

            reddit = self.reddit_client["reddit"]
            subreddit_obj = reddit.subreddit(subreddit)

            posts = []

            # Get posts based on sort method
            if sort == "new":
                submissions = subreddit_obj.new(limit=limit)
            elif sort == "hot":
                submissions = subreddit_obj.hot(limit=limit)
            elif sort == "top":
                submissions = subreddit_obj.top(limit=limit)
            elif sort == "rising":
                submissions = subreddit_obj.rising(limit=limit)
            else:
                return {"error": f"Invalid sort method: {sort}"}

            # Process submissions
            for submission in submissions:
                post = {
                    "id": submission.id,
                    "title": submission.title,
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "permalink": submission.permalink,
                    "author": str(submission.author)
                    if submission.author
                    else "[deleted]",
                    "subreddit": subreddit,
                }

                # Add sentiment analysis
                post["sentiment"] = self._analyze_sentiment(
                    submission.title + " " + submission.selftext
                )

                posts.append(post)

            elapsed = (time.time() - start_time) * 1000  # ms
            self.logger.counter("external_api_call_count", 1)
            self.logger.timing("data_fetch_time_ms", elapsed)
            self.logger.gauge("data_volume_posts", len(posts))

            return {
                "subreddit": subreddit,
                "sort": sort,
                "posts": posts,
                "count": len(posts),
            }

        except Exception as e:
            self.logger.error(f"Error fetching posts from r/{subreddit}: {e}")
            self.logger.counter("error_count", 1)
            return {"error": f"Failed to fetch posts: {str(e)}"}

    def _handle_hot_posts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hot posts endpoint."""
        # Set sort to "hot" and use recent posts handler
        params["sort"] = "hot"
        return self._handle_recent_posts(params)

    def _handle_ticker_mentions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ticker mentions endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        subreddit = params.get("subreddit")  # Optional
        limit = int(params.get("limit", 100))

        try:
            if not self.reddit_client or not self.reddit_client.get("reddit"):
                return {"error": "Reddit client not initialized"}

            reddit = self.reddit_client["reddit"]

            # Determine which subreddit to search
            if subreddit:
                subreddit_obj = reddit.subreddit(subreddit)
            else:
                # Search across all subreddits
                subreddit_obj = reddit.subreddit("all")
                subreddit = "all"

            # Search for the ticker
            query = f"{ticker}"  # Search for exact ticker

            mentions = []
            for submission in subreddit_obj.search(query, limit=limit):
                mention = {
                    "id": submission.id,
                    "title": submission.title,
                    "created_utc": submission.created_utc,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "selftext": submission.selftext,
                    "url": submission.url,
                    "permalink": submission.permalink,
                    "author": str(submission.author)
                    if submission.author
                    else "[deleted]",
                    "subreddit": submission.subreddit.display_name,
                }

                # Add sentiment analysis
                mention["sentiment"] = self._analyze_sentiment(
                    submission.title + " " + submission.selftext
                )

                mentions.append(mention)

            return {
                "ticker": ticker,
                "subreddit": subreddit,
                "mentions": mentions,
                "count": len(mentions),
            }

        except Exception as e:
            self.logger.error(f"Error searching for {ticker} in r/{subreddit}: {e}")
            self.logger.counter("error_count", 1)
            return {"error": f"Failed to search for ticker: {str(e)}"}

    def _handle_post_comments(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle post comments endpoint."""
        post_id = params.get("post_id")
        if not post_id:
            raise ValueError("Missing required parameter: post_id")

        limit = int(params.get("limit", 100))

        try:
            if not self.reddit_client or not self.reddit_client.get("reddit"):
                return {"error": "Reddit client not initialized"}

            reddit = self.reddit_client["reddit"]

            # Get the submission
            submission = reddit.submission(id=post_id)

            # Replace MoreComments objects with actual comments
            submission.comments.replace_more(limit=0)

            comments = []
            for comment in submission.comments[:limit]:
                comment_data = {
                    "id": comment.id,
                    "body": comment.body,
                    "created_utc": comment.created_utc,
                    "score": comment.score,
                    "author": str(comment.author) if comment.author else "[deleted]",
                    "permalink": comment.permalink,
                }

                # Add sentiment analysis
                comment_data["sentiment"] = self._analyze_sentiment(comment.body)

                comments.append(comment_data)

            # Get basic submission info
            submission_data = {
                "id": submission.id,
                "title": submission.title,
                "created_utc": submission.created_utc,
                "score": submission.score,
                "num_comments": submission.num_comments,
                "selftext": submission.selftext,
                "url": submission.url,
                "permalink": submission.permalink,
                "author": str(submission.author) if submission.author else "[deleted]",
                "subreddit": submission.subreddit.display_name,
            }

            return {
                "submission": submission_data,
                "comments": comments,
                "count": len(comments),
            }

        except Exception as e:
            self.logger.error(f"Error fetching comments for post {post_id}: {e}")
            self.logger.counter("error_count", 1)
            return {"error": f"Failed to fetch comments: {str(e)}"}

    def _handle_sentiment_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sentiment analysis endpoint."""
        subreddit = params.get("subreddit")
        ticker = params.get("ticker")
        limit = int(params.get("limit", 100))
        time_period = params.get("time_period", "day")

        # Need at least one of subreddit or ticker
        if not subreddit and not ticker:
            return {"error": "Either subreddit or ticker parameter is required"}

        try:
            # If ticker is provided, get ticker mentions
            if ticker:
                if subreddit:
                    # Get ticker mentions in specific subreddit
                    data = self._handle_ticker_mentions(
                        {"ticker": ticker, "subreddit": subreddit, "limit": limit}
                    )
                    posts = data.get("mentions", [])
                else:
                    # Get ticker mentions across all subreddits
                    data = self._handle_ticker_mentions(
                        {"ticker": ticker, "limit": limit}
                    )
                    posts = data.get("mentions", [])
            else:
                # Get posts from subreddit
                data = self._handle_recent_posts(
                    {"subreddit": subreddit, "limit": limit, "sort": "hot"}
                )
                posts = data.get("posts", [])

            # Filter by time period if specified
            if time_period:
                now = time.time()
                if time_period == "hour":
                    cutoff = now - 3600
                elif time_period == "day":
                    cutoff = now - 86400
                elif time_period == "week":
                    cutoff = now - 604800
                elif time_period == "month":
                    cutoff = now - 2592000
                else:
                    cutoff = 0

                posts = [p for p in posts if p.get("created_utc", 0) > cutoff]

            # Calculate sentiment metrics
            if not posts:
                return {
                    "subreddit": subreddit,
                    "ticker": ticker,
                    "time_period": time_period,
                    "error": "No posts found for sentiment analysis",
                }

            # Extract sentiment scores
            sentiment_scores = [
                p.get("sentiment", {}).get("score", 0)
                for p in posts
                if "sentiment" in p
            ]

            if not sentiment_scores:
                return {
                    "subreddit": subreddit,
                    "ticker": ticker,
                    "time_period": time_period,
                    "error": "No sentiment data available",
                }

            # Calculate metrics
            avg_score = sum(sentiment_scores) / len(sentiment_scores)
            positive_count = sum(1 for s in sentiment_scores if s > 0.1)
            negative_count = sum(1 for s in sentiment_scores if s < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count

            # Determine overall sentiment
            if avg_score > 0.2:
                overall = "bullish"
            elif avg_score > 0.05:
                overall = "slightly_bullish"
            elif avg_score < -0.2:
                overall = "bearish"
            elif avg_score < -0.05:
                overall = "slightly_bearish"
            else:
                overall = "neutral"

            return {
                "subreddit": subreddit,
                "ticker": ticker,
                "time_period": time_period,
                "post_count": len(posts),
                "sentiment": {
                    "average_score": avg_score,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                    "overall": overall,
                },
                "top_posts": posts[:5],  # Include top 5 posts for reference
            }

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            self.logger.counter("error_count", 1)
            return {"error": f"Failed to analyze sentiment: {str(e)}"}

    def _handle_wsb_sentiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle WallStreetBets sentiment endpoint."""
        ticker = params.get("ticker")
        limit = int(params.get("limit", 100))

        # If ticker is provided, get WSB sentiment for that ticker
        if ticker:
            params["subreddit"] = "wallstreetbets"
            return self._handle_sentiment_analysis(params)

        # Otherwise, get general WSB sentiment
        return self._handle_sentiment_analysis(
            {"subreddit": "wallstreetbets", "limit": limit, "time_period": "day"}
        )

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis results
        """
        if not text:
            return {"score": 0, "magnitude": 0, "label": "neutral"}

        if self.sentiment_analyzer:
            try:
                # Use VADER if available
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                score = sentiment["compound"]

                # Determine label based on score
                if score >= 0.25:
                    label = "bullish"
                elif score >= 0.05:
                    label = "slightly_bullish"
                elif score <= -0.25:
                    label = "bearish"
                elif score <= -0.05:
                    label = "slightly_bearish"
                else:
                    label = "neutral"

                return {
                    "score": score,
                    "magnitude": abs(score),
                    "label": label,
                    "positive": sentiment["pos"],
                    "negative": sentiment["neg"],
                    "neutral": sentiment["neu"],
                }
            except Exception as e:
                self.logger.warning(f"Error using VADER sentiment analyzer: {e}")
                # Fall through to simple analysis

        # Simple keyword-based sentiment analysis
        text = text.lower()
        words = set(text.split())

        positive_count = len(words.intersection(self.positive_terms))
        negative_count = len(words.intersection(self.negative_terms))

        # Calculate simple sentiment score
        if positive_count + negative_count > 0:
            score = (positive_count - negative_count) / (
                positive_count + negative_count
            )
        else:
            score = 0

        # Determine label based on score
        if score >= 0.25:
            label = "bullish"
        elif score >= 0.05:
            label = "slightly_bullish"
        elif score <= -0.25:
            label = "bearish"
        elif score <= -0.05:
            label = "slightly_bearish"
        else:
            label = "neutral"

        return {"score": score, "magnitude": abs(score), "label": label}

    # Public API methods for models to use directly

    def get_subreddit_posts(
        self, subreddit: str, sort: str = "hot", limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get posts from a subreddit.

        Args:
            subreddit: Subreddit name
            sort: Sort method ("hot", "new", "top", "rising")
            limit: Maximum number of posts to return

        Returns:
            Dictionary with posts
        """
        return self.fetch_data(
            "recent_posts", {"subreddit": subreddit, "sort": sort, "limit": str(limit)}
        )

    def search_ticker_mentions(
        self, ticker: str, subreddit: str = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Search for mentions of a ticker.

        Args:
            ticker: Stock ticker symbol
            subreddit: Optional subreddit to search in (defaults to all)
            limit: Maximum number of mentions to return

        Returns:
            Dictionary with ticker mentions
        """
        params = {"ticker": ticker, "limit": str(limit)}

        if subreddit:
            params["subreddit"] = subreddit

        return self.fetch_data("ticker_mentions", params)

    def get_post_comments(self, post_id: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get comments from a Reddit post.

        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to return

        Returns:
            Dictionary with comments
        """
        return self.fetch_data(
            "post_comments", {"post_id": post_id, "limit": str(limit)}
        )

    def get_sentiment(
        self, ticker: str = None, subreddit: str = None, time_period: str = "day"
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for a ticker or subreddit.

        Args:
            ticker: Optional stock ticker symbol
            subreddit: Optional subreddit name
            time_period: Time period for analysis ("hour", "day", "week", "month")

        Returns:
            Dictionary with sentiment analysis
        """
        params = {"time_period": time_period}

        if ticker:
            params["ticker"] = ticker

        if subreddit:
            params["subreddit"] = subreddit

        return self.fetch_data("sentiment_analysis", params)

    def get_wsb_sentiment(self, ticker: str = None) -> Dict[str, Any]:
        """
        Get WallStreetBets sentiment.

        Args:
            ticker: Optional stock ticker symbol

        Returns:
            Dictionary with WSB sentiment analysis
        """
        params = {}

        if ticker:
            params["ticker"] = ticker

        return self.fetch_data("wsb_sentiment", params)

    def get_financial_subreddits(self) -> List[str]:
        """
        Get list of recommended financial subreddits.

        Returns:
            List of financial subreddit names
        """
        return self.financial_subreddits
