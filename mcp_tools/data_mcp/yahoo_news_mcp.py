"""
Yahoo News MCP Server

This module implements a Model Context Protocol (MCP) server for Yahoo News
API, providing access to financial news and articles.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import yfinance as yf
from monitoring.system_monitor import MonitoringManager

from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP


class YahooNewsMCP(BaseDataMCP):
    """
    MCP server for Yahoo News API.

    This server provides access to Yahoo Finance news including articles,
    company-specific news, and market news.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Yahoo News MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 600)
                - default_news_limit: Default number of news articles to return
        """
        # Set default TTL for news cache to 10 minutes
        if config is None:
            config = {}
        if "cache_ttl" not in config:
            config["cache_ttl"] = 600  # 10 minutes

        super().__init__(name="yahoo_news_mcp", config=config)

        # Initialize monitoring
        self.monitor = MonitoringManager(
            service_name="yahoo-news-mcp"
        )
        self.monitor.log_info(
            "YahooNewsMCP initialized",
            component="yahoo_news_mcp",
            action="initialization",
        )

        # Set default result limits
        self.default_news_limit = self.config.get("default_news_limit", 10)

        # Initialize client configuration
        self.news_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        # Initialize news sentiment analyzer
        self._initialize_sentiment_analyzer()

        self.logger.info(
            "YahooNewsMCP initialized with %d endpoints", len(self.endpoints)
        )
        self.monitor.log_info(
            f"YahooNewsMCP initialized with {len(self.endpoints)} endpoints",
            component="yahoo_news_mcp",
            action="init_endpoints",
        )

    def _initialize_client(self) -> Dict[str, Any]:
        """
        Initialize client configuration.

        Yahoo Finance API doesn't require an API key as we're using yfinance.

        Returns:
            Client configuration or None if initialization fails
        """
        try:
            # Setup default configuration
            config = {"default_news_limit": self.default_news_limit}

            return config

        except Exception as e:
            self.logger.error(f"Failed to initialize Yahoo News client: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Failed to initialize Yahoo News client: {e}",
                    component="yahoo_news_mcp",
                    action="client_init_error",
                    error=str(e),
                )
            return None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Yahoo News.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "ticker_news": {
                "description": "Get news for a specific ticker",
                "category": "news",
                "required_params": ["ticker"],
                "optional_params": ["limit"],
                "default_values": {"limit": str(self.default_news_limit)},
                "handler": self._handle_ticker_news,
            },
            "market_news": {
                "description": "Get general market news",
                "category": "news",
                "required_params": [],
                "optional_params": ["limit"],
                "default_values": {"limit": str(self.default_news_limit)},
                "handler": self._handle_market_news,
            },
            "news_search": {
                "description": "Search for news by keywords",
                "category": "news",
                "required_params": ["keywords"],
                "optional_params": ["limit"],
                "default_values": {"limit": str(self.default_news_limit)},
                "handler": self._handle_news_search,
            },
            "news_sentiment": {
                "description": "Get sentiment analysis for news",
                "category": "analysis",
                "required_params": ["ticker"],
                "optional_params": ["limit"],
                "default_values": {"limit": str(self.default_news_limit)},
                "handler": self._handle_news_sentiment,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Yahoo News API."""
        self.register_tool(self.get_ticker_news)
        self.register_tool(self.get_market_news)
        self.register_tool(self.search_news)
        self.register_tool(self.get_news_sentiment)

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
        else:
            raise ValueError("No handler defined for endpoint")

    # Handler methods for specific endpoints

    def _handle_ticker_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ticker news endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        limit = int(params.get("limit", self.default_news_limit))

        try:
            ticker_obj = yf.Ticker(ticker)
            news = ticker_obj.news

            if not news:
                return {"error": f"No news found for {ticker}"}

            # Limit the number of news items
            news = news[:limit]

            #
            # Clean up the news items to ensure all values are JSON
            # serializable
            clean_news = []
            for item in news:
                clean_item = {}
                for k, v in item.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_item[k] = v
                    elif k == "datetime" and isinstance(v, int):
                        # Convert Unix timestamp to ISO format
                        clean_item[k] = v
                        clean_item["datetime_iso"] = datetime.fromtimestamp(
                            v
                        ).isoformat()
                    else:
                        clean_item[k] = str(v)
                clean_news.append(clean_item)

            return {
                "ticker": ticker,
                "news_items": clean_news,
                "count": len(clean_news),
                "source": "yahoo_finance",
            }

        except Exception as e:
            self.logger.error(f"Error fetching news for {ticker}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error fetching news for {ticker}: {e}",
                    component="yahoo_news_mcp",
                    action="ticker_news_error",
                    error=str(e),
                    ticker=ticker,
                )
            return {"error": f"Failed to fetch news: {str(e)}"}

    def _handle_market_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market news endpoint."""
        limit = int(params.get("limit", self.default_news_limit))

        try:
            # Use a market index as a proxy for market news
            ticker_obj = yf.Ticker("^GSPC")  # S&P 500
            news = ticker_obj.news

            if not news:
                return {"error": "No market news found"}

            # Limit the number of news items
            news = news[:limit]

            #
            # Clean up the news items to ensure all values are JSON
            # serializable
            clean_news = []
            for item in news:
                clean_item = {}
                for k, v in item.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        clean_item[k] = v
                    elif k == "datetime" and isinstance(v, int):
                        # Convert Unix timestamp to ISO format
                        clean_item[k] = v
                        clean_item["datetime_iso"] = datetime.fromtimestamp(
                            v
                        ).isoformat()
                    else:
                        clean_item[k] = str(v)
                clean_news.append(clean_item)

            return {
                "market": "general",
                "news_items": clean_news,
                "count": len(clean_news),
                "source": "yahoo_finance",
            }

        except Exception as e:
            self.logger.error(f"Error fetching market news: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error fetching market news: {e}",
                    component="yahoo_news_mcp",
                    action="market_news_error",
                    error=str(e),
                )
            return {"error": f"Failed to fetch market news: {str(e)}"}

    def _handle_news_search(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle news search endpoint."""
        keywords = params.get("keywords")
        if not keywords:
            raise ValueError("Missing required parameter: keywords")

        limit = int(params.get("limit", self.default_news_limit))

        try:
            # Yahoo Finance doesn't provide a direct news search API
            # We'll get market news and filter by keywords
            market_news = self._handle_market_news(
                {"limit": limit * 2}
            )  # Get more to filter

            if "error" in market_news:
                return market_news

            # Filter news by keywords
            filtered_news = []
            for item in market_news.get("news_items", []):
                title = item.get("title", "").lower()
                summary = item.get("summary", "").lower()

                if keywords.lower() in title or keywords.lower() in summary:
                    filtered_news.append(item)

                if len(filtered_news) >= limit:
                    break

            return {
                "keywords": keywords,
                "news_items": filtered_news,
                "count": len(filtered_news),
                "source": "yahoo_finance",
            }

        except Exception as e:
            self.logger.error(f"Error searching news for {keywords}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error searching news for {keywords}: {e}",
                    component="yahoo_news_mcp",
                    action="news_search_error",
                    error=str(e),
                    keywords=keywords,
                )
            return {"error": f"Failed to search news: {str(e)}"}

    def _handle_news_sentiment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle news sentiment endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        limit = int(params.get("limit", self.default_news_limit))

        try:
            # Get news for the ticker
            news_data = self._handle_ticker_news({"ticker": ticker, "limit": limit})

            if "error" in news_data:
                return news_data

            # Analyze sentiment for each news item
            news_items = news_data.get("news_items", [])
            sentiment_results = []

            for item in news_items:
                title = item.get("title", "")
                summary = item.get("summary", "")
                content = f"{title} {summary}"

                sentiment_score = self._analyze_sentiment_text(content)

                # Determine sentiment label
                label = "neutral"
                if sentiment_score >= 0.25:
                    label = "positive"
                elif sentiment_score >= 0.1:
                    label = "slightly_positive"
                elif sentiment_score <= -0.25:
                    label = "negative"
                elif sentiment_score <= -0.1:
                    label = "slightly_negative"

                sentiment_results.append(
                    {
                        "title": title,
                        "sentiment_score": sentiment_score,
                        "sentiment_label": label,
                        "datetime": item.get("datetime_iso", item.get("datetime", "")),
                    }
                )

            # Calculate average sentiment
            if sentiment_results:
                avg_score = sum(
                    item["sentiment_score"] for item in sentiment_results
                ) / len(sentiment_results)

                # Determine overall sentiment label
                overall_label = "neutral"
                if avg_score >= 0.25:
                    overall_label = "positive"
                elif avg_score >= 0.1:
                    overall_label = "slightly_positive"
                elif avg_score <= -0.25:
                    overall_label = "negative"
                elif avg_score <= -0.1:
                    overall_label = "slightly_negative"

                return {
                    "ticker": ticker,
                    "average_sentiment_score": avg_score,
                    "overall_sentiment": overall_label,
                    "news_count": len(sentiment_results),
                    "sentiment_details": sentiment_results,
                }
            else:
                return {"error": f"No news found for sentiment analysis for {ticker}"}

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {ticker}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error analyzing sentiment for {ticker}: {e}",
                    component="yahoo_news_mcp",
                    action="news_sentiment_error",
                    error=str(e),
                    ticker=ticker,
                )
            return {"error": f"Failed to analyze sentiment: {str(e)}"}

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

    # Public API methods for models to use directly

    def get_ticker_news(self, ticker: str, limit: int = None) -> Dict[str, Any]:
        """
        Get news articles for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to return

        Returns:
            Dictionary with news articles for the ticker
        """
        params = {"ticker": ticker}

        if limit is not None:
            params["limit"] = str(limit)

        return self.fetch_data("ticker_news", params)

    def get_market_news(self, limit: int = None) -> Dict[str, Any]:
        """
        Get general market news.

        Args:
            limit: Maximum number of news articles to return

        Returns:
            Dictionary with market news articles
        """
        params = {}

        if limit is not None:
            params["limit"] = str(limit)

        return self.fetch_data("market_news", params)

    def search_news(self, keywords: str, limit: int = None) -> Dict[str, Any]:
        """
        Search for news by keywords.

        Args:
            keywords: Keywords to search for
            limit: Maximum number of news articles to return

        Returns:
            Dictionary with matching news articles
        """
        params = {"keywords": keywords}

        if limit is not None:
            params["limit"] = str(limit)

        return self.fetch_data("news_search", params)

    def get_news_sentiment(self, ticker: str, limit: int = None) -> Dict[str, Any]:
        """
        Get sentiment analysis for news about a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to analyze

        Returns:
            Dictionary with sentiment analysis results
        """
        params = {"ticker": ticker}

        if limit is not None:
            params["limit"] = str(limit)

        return self.fetch_data("news_sentiment", params)
