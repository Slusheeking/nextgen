#!/usr/bin/env python3
"""
Test suite for the Polygon News MCP Tool.

This module tests the functionality of the PolygonNewsMCP class, including:
- Initialization and configuration
- News data processing
- Endpoint handling
- Public API methods
- Sentiment analysis
- Error handling and retry mechanisms
- Data throughput integrity
"""

import os
import unittest
import json
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

# Create proper mocks for setup_monitoring
mock_monitor = MagicMock()
mock_metrics = MagicMock()
mock_setup_monitoring = MagicMock(return_value=(mock_monitor, mock_metrics))

# Mock all required modules in correct order
sys.modules['utils'] = MagicMock()
sys.modules['utils.env_loader'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.sentiment'] = MagicMock()
sys.modules['nltk.sentiment.vader'] = MagicMock()

# Set up monitoring mock chain
monitoring_mock = MagicMock()
monitoring_mock.setup_monitoring = mock_setup_monitoring
sys.modules['monitoring'] = monitoring_mock
sys.modules['monitoring.prometheus_loki_utils'] = MagicMock()
sys.modules['monitoring.prometheus_loki_utils'].setup_monitoring = mock_setup_monitoring
sys.modules['monitoring.loki'] = MagicMock()
sys.modules['monitoring.loki.loki_manager'] = MagicMock()

# Now import the module with proper patch
with patch('mcp_tools.base_mcp_server.BaseMCPServer'), \
     patch('monitoring.prometheus_loki_utils.setup_monitoring', mock_setup_monitoring), \
     patch('monitoring.setup_monitoring', mock_setup_monitoring):
    from mcp_tools.data_mcp.polygon_news_mcp import PolygonNewsMCP


class TestPolygonNewsMCP(unittest.TestCase):
    """Test cases for the PolygonNewsMCP class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure the MCP server for testing
        self.test_config = {
            "api_key": "test_api_key",
            "base_url": "https://test.polygon.io",
            "cache_enabled": True,
            "cache_ttl": 60,
            "default_results_limit": 25,
        }
        
        # Create a mock for PolygonNewsMCP to avoid external API calls
        with patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._register_specific_tools'), \
             patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._initialize_sentiment_analyzer'):
            self.news_mcp = PolygonNewsMCP(self.test_config)
            
        # Mock logger to avoid actual logging
        self.news_mcp.logger = MagicMock()
        
        # Sample news data for testing
        self.sample_news_response = {
            "results": [
                {
                    "id": "news1",
                    "title": "Test Company Reports Strong Earnings",
                    "author": "Financial Reporter",
                    "published_utc": "2025-03-15T14:30:00Z",
                    "article_url": "https://example.com/news1",
                    "tickers": ["TEST", "SMPL"],
                    "description": "Test Company's Q1 earnings exceed expectations.",
                    "keywords": ["earnings", "growth", "exceed"]
                },
                {
                    "id": "news2",
                    "title": "Market Overview: Indices Rise",
                    "author": "Market Analyst",
                    "published_utc": "2025-03-15T09:45:00Z",
                    "article_url": "https://example.com/news2",
                    "tickers": [],
                    "description": "Major indices showed strong performance today.",
                    "keywords": ["market", "indices", "rise"]
                }
            ],
            "status": "OK",
            "request_id": "request123"
        }
        
        # Sample for sentiment analysis
        self.positive_text = "The company reported strong growth, exceeding expectations and raising outlook."
        self.negative_text = "Shares tumbled after the company missed estimates and lowered guidance."
        self.neutral_text = "The company announced a new board member effective next month."
        
        # Mock the sentiment analyzer
        self.news_mcp.sentiment_analyzer = MagicMock()
        self.news_mcp.sentiment_analyzer.polarity_scores.return_value = {"compound": 0.65}
        
        # Set up positive and negative terms for keyword-based sentiment analysis
        self.news_mcp.positive_terms = set(["strong", "growth", "exceed", "rise"])
        self.news_mcp.negative_terms = set(["tumble", "miss", "lower"])

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        pass

    def test_initialization(self):
        """Test initialization with various configurations."""
        # Test default initialization
        with patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._register_specific_tools'), \
             patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._initialize_sentiment_analyzer'):
            default_mcp = PolygonNewsMCP()
            
        # Verify default values
        self.assertEqual(default_mcp.cache_ttl, 600)  # Default 10 minutes
        self.assertEqual(default_mcp.default_results_limit, 50)  # Default limit
        
        # Test custom initialization
        custom_config = {
            "cache_ttl": 300,
            "default_results_limit": 100,
            "api_key": "custom_key"
        }
        
        with patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._register_specific_tools'), \
             patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._initialize_sentiment_analyzer'):
            custom_mcp = PolygonNewsMCP(custom_config)
        
        # Verify custom values
        self.assertEqual(custom_mcp.cache_ttl, 300)
        self.assertEqual(custom_mcp.default_results_limit, 100)
        self.assertEqual(custom_mcp.news_client.get("api_key"), "custom_key")

    def test_initialize_client(self):
        """Test client initialization."""
        # Test with API key in config
        client = self.news_mcp._initialize_client()
        self.assertEqual(client["api_key"], "test_api_key")
        self.assertEqual(client["base_url"], "https://test.polygon.io")
        
        # Test with API key in environment variable
        with patch.dict(os.environ, {"POLYGON_API_KEY": "env_api_key"}):
            with patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._register_specific_tools'), \
                 patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._initialize_sentiment_analyzer'):
                env_mcp = PolygonNewsMCP({})
                client = env_mcp._initialize_client()
                self.assertEqual(client["api_key"], "env_api_key")
        
        # Test with no API key
        with patch.dict(os.environ, {}, clear=True):
            with patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._register_specific_tools'), \
                 patch('mcp_tools.data_mcp.polygon_news_mcp.PolygonNewsMCP._initialize_sentiment_analyzer'):
                no_key_mcp = PolygonNewsMCP({})
                client = no_key_mcp._initialize_client()
                self.assertIsNone(client["api_key"])
                # Verify warning log
                no_key_mcp.logger.warning.assert_called_with("No Polygon API key provided - API calls will fail")

    def test_initialize_endpoints(self):
        """Test endpoint initialization."""
        endpoints = self.news_mcp._initialize_endpoints()
        
        # Check that all expected endpoints are present
        required_endpoints = [
            "latest_news", "ticker_news", "market_news", "news_search", "sector_news"
        ]
        
        for endpoint in required_endpoints:
            self.assertIn(endpoint, endpoints)
            
        # Check endpoint structure
        for name, config in endpoints.items():
            self.assertIn("path", config)
            self.assertIn("method", config)
            self.assertIn("description", config)
            self.assertIn("category", config)
            self.assertIn("required_params", config)
            self.assertIn("optional_params", config)
            self.assertIn("default_values", config)
            self.assertIn("handler", config)
            
        # Verify specific endpoint configuration
        ticker_news = endpoints["ticker_news"]
        self.assertEqual(ticker_news["path"], "/v2/reference/news")
        self.assertEqual(ticker_news["method"], "GET")
        self.assertIn("ticker", ticker_news["required_params"])
        self.assertEqual(ticker_news["default_values"]["limit"], "25")  # From test config

    def test_process_news_response(self):
        """Test news response processing."""
        processed = self.news_mcp._process_news_response(self.sample_news_response)
        
        # Check structure
        self.assertIn("articles", processed)
        self.assertIn("count", processed)
        
        # Check count
        self.assertEqual(processed["count"], 2)
        
        # Check that articles are processed correctly
        articles = processed["articles"]
        for article in articles:
            # Check that processing timestamp was added
            self.assertIn("processed_at", article)
            self.assertIsInstance(article["processed_at"], int)
            
            # Check that source was added
            self.assertEqual(article["source"], "polygon.io")
            
            # Check that published_utc is properly formatted
            self.assertIn("published_utc", article)
            
            # Check that relative time was added
            self.assertIn("published_relative", article)
            self.assertIsInstance(article["published_relative"], str)

    def test_handle_latest_news(self):
        """Test handling latest news requests."""
        with patch.object(self.news_mcp, '_execute_polygon_request') as mock_execute:
            mock_execute.return_value = {"articles": [], "count": 0}
            
            params = {"limit": "10"}
            result = self.news_mcp._handle_latest_news(params)
            
            # Verify the correct endpoint was called
            mock_execute.assert_called_once_with("/v2/reference/news", params)
            
            # Check result structure
            self.assertEqual(result, {"articles": [], "count": 0})

    def test_handle_ticker_news(self):
        """Test handling ticker news requests."""
        with patch.object(self.news_mcp, '_execute_polygon_request') as mock_execute:
            mock_execute.return_value = {"articles": [], "count": 0}
            
            # Test with lowercase ticker
            params = {"ticker": "aapl", "limit": "10"}
            # Call method but we're only testing that it doesn't raise an exception
            self.news_mcp._handle_ticker_news(params)
            
            # Verify ticker was uppercased
            mock_execute.assert_called_once_with("/v2/reference/news", {"ticker": "AAPL", "limit": "10"})
            
            # Test with missing ticker
            params = {"limit": "10"}
            with self.assertRaises(ValueError):
                self.news_mcp._handle_ticker_news(params)

    def test_handle_market_news(self):
        """Test handling market news requests."""
        with patch.object(self.news_mcp, '_execute_polygon_request') as mock_execute:
            # Set up mock to return sample data with both ticker-specific and market news
            mock_execute.return_value = {
                "articles": [
                    {"id": "1", "title": "Market Overview", "tickers": []},
                    {"id": "2", "title": "AAPL News", "tickers": ["AAPL"]},
                    {"id": "3", "title": "Economic Report", "tickers": None}
                ],
                "count": 3
            }
            
            params = {"limit": "10"}
            result = self.news_mcp._handle_market_news(params)
            
            # Verify correct endpoint was called
            mock_execute.assert_called_once_with("/v2/reference/news", params)
            
            # Verify filtering
            self.assertEqual(len(result["articles"]), 2)  # Should exclude the AAPL news
            self.assertEqual(result["articles"][0]["id"], "1")
            self.assertEqual(result["articles"][1]["id"], "3")

    def test_handle_news_search(self):
        """Test handling news search requests."""
        with patch.object(self.news_mcp, '_execute_polygon_request') as mock_execute:
            # Set up mock to return sample news
            mock_execute.return_value = {
                "articles": [
                    {"id": "1", "title": "Technology Trends", "description": "AI adoption increasing"},
                    {"id": "2", "title": "Market Analysis", "description": "AI impact on stocks"},
                    {"id": "3", "title": "Financial Report", "description": "Quarterly earnings"}
                ],
                "count": 3
            }
            
            # Test with keyword that should match 2 articles
            params = {"keywords": "ai", "limit": "10"}
            result = self.news_mcp._handle_news_search(params)
            
            # Verify keywords were removed from params for API call
            mock_execute.assert_called_once()
            call_args = mock_execute.call_args[0][1]
            self.assertNotIn("keywords", call_args)
            
            # Verify correct filtering
            self.assertEqual(len(result["articles"]), 2)  # Should match the two AI articles
            self.assertEqual(result["keywords"], "ai")
            
            # Test with empty keywords
            mock_execute.reset_mock()
            result = self.news_mcp._handle_news_search({})
            self.assertIn("error", result)
            self.assertEqual(result["error"], "Missing required parameter: keywords")

    def test_handle_sector_news(self):
        """Test handling sector news requests."""
        with patch.object(self.news_mcp, '_handle_news_search') as mock_search:
            mock_search.return_value = {"articles": [], "count": 0}
            
            # Test with valid sector
            params = {"sector": "technology", "limit": "10"}
            self.news_mcp._handle_sector_news(params)
            
            # Verify sector was converted to keywords
            mock_search.assert_called_once()
            call_args = mock_search.call_args[0][0]
            self.assertEqual(call_args["keywords"], "technology")
            
            # Test with empty sector
            mock_search.reset_mock()
            result = self.news_mcp._handle_sector_news({})
            self.assertIn("error", result)
            self.assertEqual(result["error"], "Missing required parameter: sector")

    def test_execute_polygon_request(self):
        """Test executing Polygon API requests with retry logic."""
        # Test successful request
        with patch('requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": []}
            mock_get.return_value = mock_response
            
            result = self.news_mcp._execute_polygon_request("/test/path", {"param": "value"})
            
            # Verify request was made with correct parameters
            mock_get.assert_called_once()
            args, kwargs = mock_get.call_args
            self.assertEqual(kwargs["params"]["param"], "value")
            self.assertEqual(kwargs["params"]["apiKey"], "test_api_key")
            
            # Verify response processing
            self.assertEqual(result, {"articles": [], "count": 0})
            
        # Test rate limit with retry
        with patch('requests.get') as mock_get, patch('time.sleep') as mock_sleep:
            # First response is rate limit, second is success
            mock_rate_limit = MagicMock()
            mock_rate_limit.status_code = 429
            mock_rate_limit.text = "Too many requests"
            
            mock_success = MagicMock()
            mock_success.status_code = 200
            mock_success.json.return_value = {"results": []}
            
            mock_get.side_effect = [mock_rate_limit, mock_success]
            
            result = self.news_mcp._execute_polygon_request("/test/path", {})
            
            # Verify retry occurred
            self.assertEqual(mock_get.call_count, 2)
            mock_sleep.assert_called_once()
            
            # Verify final result
            self.assertEqual(result, {"articles": [], "count": 0})
            
        # Test max retries exceeded
        with patch('requests.get') as mock_get, patch('time.sleep') as mock_sleep:
            # All responses are rate limits
            mock_rate_limit = MagicMock()
            mock_rate_limit.status_code = 429
            mock_rate_limit.text = "Too many requests"
            
            mock_get.side_effect = [mock_rate_limit, mock_rate_limit, mock_rate_limit, mock_rate_limit]
            
            with self.assertRaises(Exception) as context:
                self.news_mcp._execute_polygon_request("/test/path", {})
                
            self.assertIn("API request failed", str(context.exception))
            self.assertEqual(mock_get.call_count, 4)  # Initial + 3 retries

    def test_get_latest_news(self):
        """Test get_latest_news public method."""
        with patch.object(self.news_mcp, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = {"articles": [], "count": 0}
            
            # Test with defaults
            self.news_mcp.get_latest_news()
            mock_fetch.assert_called_with("latest_news", {})
            
            # Test with custom limit
            mock_fetch.reset_mock()
            self.news_mcp.get_latest_news(limit=10)
            mock_fetch.assert_called_with("latest_news", {"limit": "10"})
            
            # Test with hours_ago
            mock_fetch.reset_mock()
            with patch('datetime.datetime') as mock_dt:
                mock_now = datetime(2025, 4, 1, 12, 0, 0)
                mock_dt.now.return_value = mock_now
                
                self.news_mcp.get_latest_news(hours_ago=24)
                
                # Verify timestamp was formatted correctly
                args = mock_fetch.call_args[0][1]
                self.assertIn("published_utc.gt", args)
                self.assertEqual(args["published_utc.gt"], "2025-03-31T12:00:00Z")

    def test_get_ticker_news(self):
        """Test get_ticker_news public method."""
        with patch.object(self.news_mcp, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = {"articles": [], "count": 0}
            
            # Test with required params
            self.news_mcp.get_ticker_news("AAPL")
            mock_fetch.assert_called_with("ticker_news", {"ticker": "AAPL"})
            
            # Test with optional params
            mock_fetch.reset_mock()
            self.news_mcp.get_ticker_news("TSLA", limit=5, days_ago=7)
            
            # Verify params
            args = mock_fetch.call_args[0][1]
            self.assertEqual(args["ticker"], "TSLA")
            self.assertEqual(args["limit"], "5")
            self.assertIn("published_utc.gt", args)
            
            # Test with missing ticker
            mock_fetch.reset_mock()
            result = self.news_mcp.get_ticker_news("")
            self.assertIn("error", result)
            mock_fetch.assert_not_called()

    def test_search_news(self):
        """Test search_news public method."""
        with patch.object(self.news_mcp, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = {"articles": [], "count": 0}
            
            # Test with required params
            self.news_mcp.search_news("earnings")
            mock_fetch.assert_called_with("news_search", {"keywords": "earnings"})
            
            # Test with all optional params
            mock_fetch.reset_mock()
            self.news_mcp.search_news("guidance", ticker="AAPL", limit=10, days_ago=30)
            
            # Verify params
            args = mock_fetch.call_args[0][1]
            self.assertEqual(args["keywords"], "guidance")
            self.assertEqual(args["ticker"], "AAPL")
            self.assertEqual(args["limit"], "10")
            self.assertIn("published_utc.gt", args)
            
            # Test with missing keywords
            mock_fetch.reset_mock()
            result = self.news_mcp.search_news("")
            self.assertIn("error", result)
            mock_fetch.assert_not_called()

    def test_get_sector_news(self):
        """Test get_sector_news public method."""
        with patch.object(self.news_mcp, 'fetch_data') as mock_fetch:
            mock_fetch.return_value = {"articles": [], "count": 0}
            
            # Test with required params
            self.news_mcp.get_sector_news("technology")
            mock_fetch.assert_called_with("sector_news", {"sector": "technology"})
            
            # Test with optional params
            mock_fetch.reset_mock()
            self.news_mcp.get_sector_news("healthcare", limit=15, days_ago=5)
            
            # Verify params
            args = mock_fetch.call_args[0][1]
            self.assertEqual(args["sector"], "healthcare")
            self.assertEqual(args["limit"], "15")
            self.assertIn("published_utc.gt", args)
            
            # Test with missing sector
            mock_fetch.reset_mock()
            result = self.news_mcp.get_sector_news("")
            self.assertIn("error", result)
            mock_fetch.assert_not_called()

    def test_analyze_sentiment_text_with_vader(self):
        """Test sentiment analysis using NLTK VADER."""
        # Test with sentiment analyzer available
        self.news_mcp.sentiment_analyzer = MagicMock()
        self.news_mcp.sentiment_analyzer.polarity_scores.return_value = {"compound": 0.75}
        
        score = self.news_mcp._analyze_sentiment_text("Test text")
        self.assertEqual(score, 0.75)
        
        # Test error handling
        self.news_mcp.sentiment_analyzer.polarity_scores.side_effect = Exception("VADER error")
        score = self.news_mcp._analyze_sentiment_text("Test text")
        
        # Should still return a score using keyword-based fallback
        self.assertIsInstance(score, float)
        
        # Verify warning was logged
        self.news_mcp.logger.warning.assert_called()

    def test_analyze_sentiment_text_with_keywords(self):
        """Test sentiment analysis using keyword-based approach."""
        # Set sentiment_analyzer to None to force keyword approach
        self.news_mcp.sentiment_analyzer = None
        
        # Test positive sentiment
        score = self.news_mcp._analyze_sentiment_text(self.positive_text)
        self.assertGreater(score, 0)
        
        # Test negative sentiment
        score = self.news_mcp._analyze_sentiment_text(self.negative_text)
        self.assertLess(score, 0)
        
        # Test neutral sentiment
        score = self.news_mcp._analyze_sentiment_text(self.neutral_text)
        self.assertAlmostEqual(score, 0.0, delta=0.3)  # Should be close to neutral
        
        # Test empty text
        score = self.news_mcp._analyze_sentiment_text("")
        self.assertEqual(score, 0.0)

    def test_get_news_sentiment_with_ticker(self):
        """Test get_news_sentiment with ticker parameter."""
        with patch.object(self.news_mcp, 'get_ticker_news') as mock_get_news:
            # Set up mock to return sample news
            mock_get_news.return_value = {
                "articles": [
                    {"title": "Good news", "description": "Positive development"},
                    {"title": "Bad news", "description": "Negative development"}
                ]
            }
            
            # Setup sentiment analysis to return different scores
            self.news_mcp._analyze_sentiment_text = MagicMock()
            self.news_mcp._analyze_sentiment_text.side_effect = [0.5, -0.3]
            
            result = self.news_mcp.get_news_sentiment(ticker="AAPL")
            
            # Verify structure
            self.assertEqual(result["ticker"], "AAPL")
            self.assertEqual(result["articles_analyzed"], 2)
            self.assertIn("average_sentiment", result)
            self.assertIn("sentiment_trend", result)
            
            # Verify sentiment calculation
            avg_sentiment = result["average_sentiment"]
            self.assertGreater(avg_sentiment["score"], 0)  # Average of 0.5 and -0.3
            self.assertIn(avg_sentiment["label"], ["slightly_positive", "neutral"])
            
            # Test with no articles
            mock_get_news.return_value = {"articles": []}
            result = self.news_mcp.get_news_sentiment(ticker="EMPTY")
            self.assertIn("error", result)

    def test_get_news_sentiment_with_text(self):
        """Test get_news_sentiment with text parameter."""
        # Test positive sentiment
        with patch.object(self.news_mcp, '_analyze_sentiment_text') as mock_analyze:
            mock_analyze.return_value = 0.6
            
            result = self.news_mcp.get_news_sentiment(text=self.positive_text)
            
            # Verify structure
            self.assertIn("text_length", result)
            self.assertIn("sentiment", result)
            self.assertEqual(result["text_length"], len(self.positive_text))
            
            # Verify sentiment
            sentiment = result["sentiment"]
            self.assertEqual(sentiment["score"], 0.6)
            self.assertEqual(sentiment["label"], "positive")
            
        # Test negative sentiment
        with patch.object(self.news_mcp, '_analyze_sentiment_text') as mock_analyze:
            mock_analyze.return_value = -0.7
            
            result = self.news_mcp.get_news_sentiment(text=self.negative_text)
            
            # Verify sentiment
            sentiment = result["sentiment"]
            self.assertEqual(sentiment["score"], -0.7)
            self.assertEqual(sentiment["label"], "negative")

    def test_get_news_sentiment_with_article_id(self):
        """Test get_news_sentiment with article_id parameter."""
        # Set up mock cache
        self.news_mcp.cache = {}
        article_id = "test_article_123"
        
        result = self.news_mcp.get_news_sentiment(article_id=article_id)
        
        # Verify structure
        self.assertEqual(result["article_id"], article_id)
        self.assertIn("sentiment", result)

    def test_get_news_sentiment_missing_params(self):
        """Test get_news_sentiment with no parameters."""
        result = self.news_mcp.get_news_sentiment()
        self.assertIn("error", result)
        self.assertEqual(result["error"], "One of ticker, article_id, or text must be provided")

    def test_property_accessors(self):
        """Test property accessors."""
        # Test api_key property
        self.assertEqual(self.news_mcp.api_key, "test_api_key")
        
        # Test base_url property
        self.assertEqual(self.news_mcp.base_url, "https://test.polygon.io")
        
        # Test with no client
        self.news_mcp.news_client = None
        self.assertEqual(self.news_mcp.api_key, "")
        self.assertEqual(self.news_mcp.base_url, "https://api.polygon.io")

    def test_data_throughput_integrity(self):
        """Test that data integrity is maintained throughout processing."""
        # Make a copy of original test data
        original_data = json.dumps(self.sample_news_response)
        
        # Process the data
        processed = self.news_mcp._process_news_response(self.sample_news_response)
        
        # Verify original data was not modified
        self.assertEqual(json.dumps(self.sample_news_response), original_data)
        
        # Verify all original article fields are preserved
        for i, original_article in enumerate(self.sample_news_response["results"]):
            processed_article = processed["articles"][i]
            
            # Check all original fields are present
            for key, value in original_article.items():
                if key == "published_utc":  # This gets reformatted
                    self.assertIn(key, processed_article)
                else:
                    self.assertEqual(processed_article[key], value)


if __name__ == "__main__":
    unittest.main()