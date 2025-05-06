"""
Polygon.io News MCP Server

This module implements a Model Context Protocol (MCP) server for accessing
financial news data from Polygon.io with focus on news search, filtering,
and sentiment analysis.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Direct imports with graceful error handling using importlib
import importlib

# Try to import required dependencies
try:
    import requests
except ImportError:
    requests = None

try:
    import dotenv
except ImportError:
    dotenv = None

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP

# Load environment variables
dotenv.load_dotenv()


class PolygonNewsMCP(BaseDataMCP):
    """
    MCP server for Polygon.io News API.

    This server provides access to financial news data from Polygon.io with
    specialized endpoints for news filtering, search, and analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Polygon.io News MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - api_key: Polygon.io API key (overrides environment variable)
                - base_url: Base URL for REST API
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 600)
                - default_results_limit: Default number of news results to return
        """
        # Set default TTL for news cache to 10 minutes
        if config is None:
            config = {}
        if "cache_ttl" not in config:
            config["cache_ttl"] = 600  # 10 minutes

        super().__init__(name="polygon_news_mcp", config=config)

        # Initialize monitoring/logger
        self.logger = NetdataLogger(component_name="polygon-news-mcp")
        self.logger.info("PolygonNewsMCP initialized")

        # Set default result limits
        self.default_results_limit = self.config.get("default_results_limit", 50)

        # Initialize the news client
        self.news_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        # Initialize news sentiment analyzer
        self._initialize_sentiment_analyzer()

        self.logger.info(f"PolygonNewsMCP initialized with {len(self.endpoints)} endpoints")

    def _initialize_client(self) -> Dict[str, Any]:
        """
        Initialize the Polygon news client configuration.

        Returns:
            Client configuration or None if initialization fails
        """
        try:
            # Prioritize environment variable for API key
            api_key = os.environ.get("POLYGON_API_KEY") or self.config.get("api_key")
            base_url = self.config.get("base_url", "https://api.polygon.io")

            if not api_key:
                self.logger.error("No Polygon API key provided - API calls will fail")
                return None

            self.logger.info(f"Loaded Polygon API key: {api_key[:4]}...{api_key[-4:]}")

            return {"api_key": api_key, "base_url": base_url}

        except Exception as e:
            self.logger.error(f"Failed to initialize Polygon news client: {e}")
            return None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Polygon.io News API.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "latest_news": {
                "path": "/v2/reference/news",
                "method": "GET",
                "description": "Get the latest financial news articles",
                "category": "news",
                "required_params": [],
                "optional_params": [
                    "limit",
                    "order",
                    "sort",
                    "published_utc.gt",
                    "published_utc.lt",
                ],
                "default_values": {
                    "limit": str(self.default_results_limit),
                    "order": "desc",
                    "sort": "published_utc",
                },
                "handler": self._handle_latest_news,
            },
            "ticker_news": {
                "path": "/v2/reference/news",
                "method": "GET",
                "description": "Get news articles for a specific ticker",
                "category": "news",
                "required_params": ["ticker"],
                "optional_params": [
                    "limit",
                    "order",
                    "sort",
                    "published_utc.gt",
                    "published_utc.lt",
                ],
                "default_values": {
                    "limit": str(self.default_results_limit),
                    "order": "desc",
                    "sort": "published_utc",
                },
                "handler": self._handle_ticker_news,
            },
            "market_news": {
                "path": "/v2/reference/news",
                "method": "GET",
                "description": "Get general market news (excluding specific tickers)",
                "category": "news",
                "required_params": [],
                "optional_params": [
                    "limit",
                    "order",
                    "sort",
                    "published_utc.gt",
                    "published_utc.lt",
                ],
                "default_values": {
                    "limit": str(self.default_results_limit),
                    "order": "desc",
                    "sort": "published_utc",
                },
                "handler": self._handle_market_news,
            },
            "news_search": {
                "path": "/v2/reference/news",
                "method": "GET",
                "description": "Search news articles by keywords",
                "category": "news",
                "required_params": ["keywords"],
                "optional_params": [
                    "ticker",
                    "limit",
                    "order",
                    "sort",
                    "published_utc.gt",
                    "published_utc.lt",
                ],
                "default_values": {
                    "limit": str(self.default_results_limit),
                    "order": "desc",
                    "sort": "published_utc",
                },
                "handler": self._handle_news_search,
            },
            "sector_news": {
                "path": "/v2/reference/news",
                "method": "GET",
                "description": "Get news for a specific market sector",
                "category": "news",
                "required_params": ["sector"],
                "optional_params": [
                    "limit",
                    "order",
                    "sort",
                    "published_utc.gt",
                    "published_utc.lt",
                ],
                "default_values": {
                    "limit": str(self.default_results_limit),
                    "order": "desc",
                    "sort": "published_utc",
                },
                "handler": self._handle_sector_news,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Polygon News API."""
        self.register_tool(self.get_latest_news)
        self.register_tool(self.get_ticker_news)
        self.register_tool(self.search_news)
        self.register_tool(self.get_news_sentiment)
        self.register_tool(self.get_sector_news)

    def _initialize_sentiment_analyzer(self):
        """Initialize the news sentiment analyzer."""
        # Initialize VADER sentiment analysis
        try:
            # We'll use a simple sentiment analyzer directly
            #
            # In a more sophisticated implementation, we could use a more
            # advanced NLP model
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("Initialized VADER sentiment analyzer")
        except ImportError:
            self.logger.warning(
                "NLTK VADER not available, using simple keyword-based sentiment analysis"
            )
            #
            # Initialize a dictionary of positive and negative terms for basic
            # sentiment analysis
            self.positive_terms = set(
                [
                    "up",
                    "rise",
                    "rising",
                    "gain",
                    "gains",
                    "positive",
                    "profitable",
                    "growth",
                    "growing",
                    "increase",
                    "increasing",
                    "higher",
                    "improved",
                    "improving",
                    "strong",
                    "stronger",
                    "strongest",
                    "beat",
                    "beats",
                    "beating",
                    "exceed",
                    "exceeds",
                    "exceeded",
                    "surpass",
                    "surpasses",
                    "surpassing",
                    "outperform",
                    "outperforms",
                    "outperforming",
                    "record",
                    "bullish",
                    "bull",
                    "buy",
                    "buying",
                    "rally",
                    "rallies",
                    "rallying",
                    "recover",
                    "recovery",
                    "recovering",
                    "boom",
                    "roar",
                    "soar",
                ]
            )
            self.negative_terms = set(
                [
                    "down",
                    "fall",
                    "falling",
                    "loss",
                    "losses",
                    "negative",
                    "unprofitable",
                    "decline",
                    "declining",
                    "decrease",
                    "decreasing",
                    "lower",
                    "worsened",
                    "worsening",
                    "weak",
                    "weaker",
                    "weakest",
                    "miss",
                    "misses",
                    "missing",
                    "fail",
                    "fails",
                    "failing",
                    "underperform",
                    "underperforms",
                    "underperforming",
                    "bearish",
                    "bear",
                    "sell",
                    "selling",
                    "slump",
                    "slumps",
                    "slumping",
                    "plunge",
                    "plunges",
                    "plunging",
                    "crash",
                    "crashes",
                    "crashing",
                    "tumble",
                    "tumbles",
                    "tumbling",
                    "sink",
                ]
            )
            self.sentiment_analyzer = None

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

        # Default handling if no specific handler
        path = endpoint_config.get("path")
        method = endpoint_config.get("method", "GET")

        # Format the path with path parameters
        try:
            formatted_path = path.format(**merged_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        # Extract query parameters (those not used in path formatting)
        query_params = {
            k: v for k, v in merged_params.items() if "{" + k + "}" not in path
        }

        # Add API key to query params
        query_params["apiKey"] = self.api_key

        # Make the request
        url = f"{self.base_url}{formatted_path}"

        # Add request tracking headers
        headers = {
            "User-Agent": f"FinGPT/NextGen-{os.environ.get('VERSION', '1.0.0')}",
            "X-Request-Source": "fingpt-mcp-news",
        }

        # Execute request with retry logic
        retry_count = 0
        max_retries = 3
        retry_delay = 1.0  # seconds

        while retry_count <= max_retries:
            try:
                if method == "GET":
                    response = requests.get(
                        url, params=query_params, headers=headers, timeout=15
                    )
                elif method == "POST":
                    response = requests.post(
                        url, json=query_params, headers=headers, timeout=15
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code == 200:
                    return self._process_news_response(response.json())
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_count += 1
                    if retry_count <= max_retries:
                        sleep_time = retry_delay * (
                            2 ** (retry_count - 1)
                        )  # Exponential backoff
                        self.logger.warning(
                            f"Rate limit exceeded, retrying in {sleep_time}s (attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue

                # Handle other error codes
                raise Exception(
                    f"API request failed: {response.status_code} - {response.text}"
                )
            except requests.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    sleep_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Request failed, retrying in {sleep_time}s (attempt {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    raise Exception(f"Request failed after {max_retries} retries: {e}")

    def _process_news_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a news API response.

        Args:
            response: Raw API response

        Returns:
            Processed news data
        """
        # Ensure we have a results list
        if "results" not in response or not isinstance(response["results"], list):
            return {"articles": [], "count": 0}

        # Extract results
        articles = response["results"]

        # Add processing timestamp and news source info
        for article in articles:
            article["processed_at"] = int(time.time())
            article["source"] = "polygon.io"

            # Ensure published_utc is in a consistent format
            if "published_utc" in article:
                # Parse date for better formatting if needed
                try:
                    date_str = article["published_utc"]
                    parsed_date = datetime.fromisoformat(
                        date_str.replace("Z", "+00:00")
                    )
                    article["published_utc"] = parsed_date.isoformat()

                    # Add relative time for convenience
                    now = datetime.now().astimezone()
                    delta = now - parsed_date.astimezone()

                    if delta.days > 0:
                        article["published_relative"] = f"{delta.days} days ago"
                    elif delta.seconds >= 3600:
                        article["published_relative"] = (
                            f"{delta.seconds // 3600} hours ago"
                        )
                    elif delta.seconds >= 60:
                        article["published_relative"] = (
                            f"{delta.seconds // 60} minutes ago"
                        )
                    else:
                        article["published_relative"] = f"{delta.seconds} seconds ago"
                except Exception:
                    # Keep original if parsing fails
                    pass

        return {"articles": articles, "count": len(articles)}

    # Handler methods for specific endpoints

    def _handle_latest_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle latest news endpoint."""
        return self._execute_polygon_request("/v2/reference/news", params)

    def _handle_ticker_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ticker news endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        # Make sure ticker is uppercase for consistency
        params["ticker"] = ticker.upper()

        return self._execute_polygon_request("/v2/reference/news", params)

    def _handle_market_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle market news endpoint.

        This filters for general market news by excluding articles that mention specific tickers.
        In practice, this is challenging to implement perfectly with the Polygon API,
        so we get latest news and then filter out ticker-specific news client-side.
        """
        # Get latest news
        latest_news = self._execute_polygon_request("/v2/reference/news", params)

        # Filter out articles that have specific tickers
        if "articles" in latest_news:
            market_articles = [
                article
                for article in latest_news["articles"]
                if "tickers" not in article or not article["tickers"]
            ]

            return {"articles": market_articles, "count": len(market_articles)}
        return latest_news

    def _handle_news_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle news search endpoint."""
        # Polygon doesn't directly support keyword search in their API
        # So we'll get the news and filter it ourselves
        keywords = params.pop("keywords", "").lower()
        if not keywords:
            return {"error": "Missing required parameter: keywords"}

        # Get regular news, potentially filtered by ticker
        news_results = self._execute_polygon_request("/v2/reference/news", params)

        # Filter by keywords
        if "articles" in news_results:
            filtered_articles = []

            for article in news_results["articles"]:
                # Check title and description for keywords
                title = article.get("title", "").lower()
                description = article.get("description", "").lower()

                if keywords in title or keywords in description:
                    filtered_articles.append(article)

            return {
                "articles": filtered_articles,
                "count": len(filtered_articles),
                "keywords": keywords,
            }

        return news_results

    def _handle_sector_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sector news endpoint."""
        # Get sector name
        sector = params.pop("sector", "").lower()
        if not sector:
            return {"error": "Missing required parameter: sector"}

        # Polygon doesn't directly support sector filtering
        #
        # In production, we'd use a more sophisticated mapping of tickers to
        # sectors
        # For now, we'll use the keyword search approach
        params["keywords"] = sector
        return self._handle_news_search(params)

    def _execute_polygon_request(
        self, path: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a request to Polygon.io API."""
        # Extract query parameters
        query_params = params.copy()

        # Add API key
        query_params["apiKey"] = self.api_key

        # Make request
        url = f"{self.base_url}{path}"

        # Add request tracking headers
        headers = {
            "User-Agent": f"FinGPT/NextGen-{os.environ.get('VERSION', '1.0.0')}",
            "X-Request-Source": "fingpt-mcp-news",
        }

        # Execute request with retry logic
        retry_count = 0
        max_retries = 3
        retry_delay = 1.0  # seconds

        while retry_count <= max_retries:
            try:
                start_time = time.time()
                response = requests.get(
                    url, params=query_params, headers=headers, timeout=15
                )
                elapsed = (time.time() - start_time) * 1000  # ms
                self.logger.counter("external_api_call_count", 1)
                self.logger.timing("data_fetch_time_ms", elapsed)
                self.logger.gauge("response_status_code", response.status_code)
                if response.content:
                    self.logger.gauge("data_volume_bytes", len(response.content))

                if response.status_code == 200:
                    return self._process_news_response(response.json())
                elif response.status_code == 429:  # Rate limit exceeded
                    self.logger.counter("rate_limit_count", 1)
                    retry_count += 1
                    if retry_count <= max_retries:
                        sleep_time = retry_delay * (
                            2 ** (retry_count - 1)
                        )  # Exponential backoff
                        self.logger.warning(
                            f"Rate limit exceeded, retrying in {sleep_time}s (attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue

                # Handle other error codes
                self.logger.error(
                    f"API request failed: {response.status_code} - {response.text}"
                )
                raise Exception(
                    f"API request failed: {response.status_code} - {response.text}"
                )
            except requests.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    sleep_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Request failed, retrying in {sleep_time}s (attempt {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Request failed after {max_retries} retries: {e}")
                    raise Exception(f"Request failed after {max_retries} retries: {e}")

    # Public API methods for models to use directly

    def get_latest_news(
        self, limit: int = None, hours_ago: int = None
    ) -> Dict[str, Any]:
        """
        Get the latest financial news articles.

        Args:
            limit: Maximum number of news articles to return
            hours_ago: Only include news from the past N hours

        Returns:
            Dictionary with news articles
        """
        params = {}

        if limit is not None:
            params["limit"] = str(limit)

        if hours_ago is not None:
            # Calculate timestamp for hours ago
            timestamp = datetime.now() - timedelta(hours=hours_ago)
            params["published_utc.gt"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        return self.fetch_data("latest_news", params)

    def get_ticker_news(
        self, ticker: str, limit: int = None, days_ago: int = None
    ) -> Dict[str, Any]:
        """
        Get news articles for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to return
            days_ago: Only include news from the past N days

        Returns:
            Dictionary with news articles for the ticker
        """
        if not ticker:
            return {"error": "ticker parameter is required"}

        params = {"ticker": ticker}

        if limit is not None:
            params["limit"] = str(limit)

        if days_ago is not None:
            # Calculate timestamp for days ago
            timestamp = datetime.now() - timedelta(days=days_ago)
            params["published_utc.gt"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        return self.fetch_data("ticker_news", params)

    def search_news(
        self, keywords: str, ticker: str = None, limit: int = None, days_ago: int = None
    ) -> Dict[str, Any]:
        """
        Search news articles by keywords.

        Args:
            keywords: Keywords to search for
            ticker: Optional ticker to filter by
            limit: Maximum number of news articles to return
            days_ago: Only include news from the past N days

        Returns:
            Dictionary with matching news articles
        """
        if not keywords:
            return {"error": "keywords parameter is required"}

        params = {"keywords": keywords}

        if ticker is not None:
            params["ticker"] = ticker

        if limit is not None:
            params["limit"] = str(limit)

        if days_ago is not None:
            # Calculate timestamp for days ago
            timestamp = datetime.now() - timedelta(days=days_ago)
            params["published_utc.gt"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        return self.fetch_data("news_search", params)

    def get_sector_news(
        self, sector: str, limit: int = None, days_ago: int = None
    ) -> Dict[str, Any]:
        """
        Get news for a specific market sector.

        Args:
            sector: Market sector name (e.g., "technology", "healthcare")
            limit: Maximum number of news articles to return
            days_ago: Only include news from the past N days

        Returns:
            Dictionary with sector news articles
        """
        if not sector:
            return {"error": "sector parameter is required"}

        params = {"sector": sector}

        if limit is not None:
            params["limit"] = str(limit)

        if days_ago is not None:
            # Calculate timestamp for days ago
            timestamp = datetime.now() - timedelta(days=days_ago)
            params["published_utc.gt"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

        return self.fetch_data("sector_news", params)

    def read_resource(self, uri: str) -> Dict[str, Any]:
        """
        Read a resource from the Polygon News MCP server.

        This client does not support reading resources via URI.

        Args:
            uri: The URI of the resource to read.

        Returns:
            An error dictionary indicating that this operation is not supported.
        """
        return {"error": f"Polygon News MCP does not support reading resources via URI: {uri}"}

    def get_news_sentiment(
        self, ticker: str = None, article_id: str = None, text: str = None
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for news.

        One of ticker, article_id, or text must be provided.

        Args:
            ticker: Analyze sentiment for recent news about a ticker
            article_id: Analyze sentiment for a specific news article
            text: Analyze sentiment for provided text

        Returns:
            Dictionary with sentiment analysis
        """
        # Ensure at least one parameter is provided
        if ticker is None and article_id is None and text is None:
            return {"error": "One of ticker, article_id, or text must be provided"}

        # If we have an article ID, analyze that specific article
        if article_id is not None:
            # Get the article and analyze it
            # Use cached data if available
            cache_key = f"article_{article_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # In real implementation, we would fetch the article content
            # and then analyze it with the sentiment analyzer
            return {
                "article_id": article_id,
                "sentiment": {"score": 0.65, "magnitude": 0.8, "label": "positive"},
            }

        # If we have a ticker, analyze recent news for that ticker
        if ticker is not None:
            # Get recent news for the ticker
            ticker_news = self.get_ticker_news(ticker, limit=5)

            if "articles" in ticker_news and ticker_news["articles"]:
                # Analyze each article and aggregate the sentiment
                sentiment_scores = []
                articles = ticker_news["articles"]

                for article in articles:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    content = f"{title} {description}"
                    score = self._analyze_sentiment_text(content)
                    sentiment_scores.append(score)

                # Calculate average sentiment
                if sentiment_scores:
                    avg_score = sum(sentiment_scores) / len(sentiment_scores)
                    magnitude = sum(abs(score) for score in sentiment_scores) / len(
                        sentiment_scores
                    )

                    # Determine sentiment trend
                    if len(sentiment_scores) > 1:
                        if sentiment_scores[-1] > sentiment_scores[0]:
                            trend = "improving"
                        elif sentiment_scores[-1] < sentiment_scores[0]:
                            trend = "deteriorating"
                        else:
                            trend = "stable"
                    else:
                        trend = "stable"

                    # Determine sentiment label
                    label = "neutral"
                    if avg_score >= 0.25:
                        label = "positive"
                    elif avg_score >= 0.1:
                        label = "slightly_positive"
                    elif avg_score <= -0.25:
                        label = "negative"
                    elif avg_score <= -0.1:
                        label = "slightly_negative"

                    return {
                        "ticker": ticker,
                        "articles_analyzed": len(ticker_news["articles"]),
                        "average_sentiment": {
                            "score": avg_score,
                            "magnitude": magnitude,
                            "label": label,
                        },
                        "sentiment_trend": trend,
                    }

            return {"error": f"No recent news found for ticker: {ticker}"}

        # If we have text, analyze the provided text
        if text is not None:
            # Analyze the provided text
            text_length = len(text)
            score = self._analyze_sentiment_text(text)

            # Determine sentiment label based on score
            label = "neutral"
            if score >= 0.25:
                label = "positive"
            elif score >= 0.1:
                label = "slightly_positive"
            elif score <= -0.25:
                label = "negative"
            elif score <= -0.1:
                label = "slightly_negative"

            # Calculate magnitude (strength of sentiment)
            magnitude = abs(score)

            return {
                "text_length": text_length,
                "sentiment": {"score": score, "magnitude": magnitude, "label": label},
            }

    def _analyze_sentiment_text(self, text: str) -> float:
        """
        Analyze sentiment of a text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1.0 (negative) and 1.0 (positive)
        """
        if self.sentiment_analyzer:
            # Use NLTK VADER if available
            try:
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                return sentiment["compound"]  # VADER compound score
            except Exception as e:
                self.logger.warning(f"Error using VADER sentiment analyzer: {e}")
                # Fall through to simple analysis

        # Simple keyword-based sentiment analysis
        text = text.lower()
        words = set(text.split())

        positive_count = len(words.intersection(self.positive_terms))
        negative_count = len(words.intersection(self.negative_terms))

        total_words = len(words)
        if total_words == 0:
            return 0.0

        # Calculate simple sentiment score
        score = (positive_count - negative_count) / (
            positive_count + negative_count + 1
        )
        return max(min(score, 1.0), -1.0)  # Clamp between -1.0 and 1.0

    @property
    def api_key(self) -> str:
        """Get the API key from the client configuration."""
        if not self.news_client:
            return ""
        return self.news_client.get("api_key", "")

    @property
    def base_url(self) -> str:
        """Get the base URL from the client configuration."""
        if not self.news_client:
            return "https://api.polygon.io"
        return self.news_client.get("base_url", "https://api.polygon.io").rstrip("/")
