#!/usr/bin/env python3
"""
Integrated Financial Analytics System

This module combines FinBERT for financial text analysis with XGBoost for predictive modeling,
creating a powerful system for financial market prediction that leverages both textual and
numerical data.

Key features:
- Sentiment analysis using pre-trained FinBERT
- Market data processing and feature engineering
- XGBoost-based prediction models
- Integrated caching and performance monitoring
- Comprehensive backtesting framework
- Configuration via /home/ubuntu/nextgen/config/financial_data_mcp/financial_data_mcp_config.json
"""

import os
import json
import time
import numpy as np
import pandas as pd
import threading
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

# ML imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV

# Financial data handling imports
from mcp_tools.base_mcp_server import BaseMCPServer

# Monitoring imports
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector
from monitoring.stock_charts import StockChartGenerator


class FinancialDataMCP(BaseMCPServer):
    """
    Integrated system combining FinBERT for financial sentiment analysis with XGBoost
    for market prediction models. This class serves as the primary interface for financial data operations.
    """
    
    def __init__(self, config_path: str = "/home/ubuntu/nextgen/config/financial_data_mcp/financial_data_mcp_config.json"):
        """
        Initialize the integrated FinBERT-XGBoost system.

        Args:
            config_path: Path to configuration file
        """
        # Initialize logger first for proper error handling
        self.logger = NetdataLogger(component_name="finbert-xgboost-integration")
        
        # Load configuration from the specified path
        self.config = self._load_config(config_path)
        
        # Initialize base MCP server with name from config if available
        super().__init__(
            name=self.config.get("component_name", "finbert_xgboost_integration"),
            config=self.config
        )
        
        # Initialize locks for thread safety
        self.model_lock = threading.RLock()
        self.cache_lock = threading.RLock()
        self.rate_limit_lock = threading.RLock()
        
        # Initialize cache
        self.cache = {}
        self.cache_timestamps = {}
        
        # Initialize rate limiting data structures
        self.request_timestamps = []
        
        # Extract configuration
        self._configure_from_config()
        
        # Initialize component MCPs with their respective configurations
        data_mcp_config = self.config.get("financial_data_mcp", {})
        text_mcp_config = self.config.get("financial_text_mcp", {})
        
        # Add API keys from config to the respective MCP configs
        if "api_keys" in self.config:
            # Add API keys to data MCP config
            if "polygon" in self.config["api_keys"]:
                data_mcp_config.setdefault("sources", {}).setdefault("polygon_rest", {})["api_key"] = self._resolve_env_var(self.config["api_keys"]["polygon"])
            if "unusual_whales" in self.config["api_keys"]:
                data_mcp_config.setdefault("sources", {}).setdefault("unusual_whales", {})["api_key"] = self._resolve_env_var(self.config["api_keys"]["unusual_whales"])
            if "yahoo_finance" in self.config["api_keys"]:
                data_mcp_config.setdefault("sources", {}).setdefault("yahoo_finance", {})["api_key"] = self._resolve_env_var(self.config["api_keys"]["yahoo_finance"])
        
        # Enable/disable data sources based on config
        if "data_sources" in self.config:
            for source in data_mcp_config.get("sources", {}):
                data_mcp_config["sources"][source]["enabled"] = source in self.config["data_sources"]
        
        # Initialize data_mcp with the appropriate configuration
        # Since FinancialDataMCP is now this class, use a different approach if needed
        # For now, we'll use self as the data_mcp or initialize a sub-component if required
        self.data_mcp = self  # Self-reference as this class is FinancialDataMCP
        
        # Initialize models
        self._initialize_models()
        
        # Initialize metrics collector and chart generator
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.metrics_collector.start()
        self.chart_generator = StockChartGenerator()
        
        # Register tools
        self._register_tools()
        
        # Start health check thread if enabled
        if self.config.get("monitoring", {}).get("health_check_interval_mins"):
            self._start_health_check_thread()
        
        self.logger.info("FinBERT-XGBoost Integration initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file with error handling.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "api_keys": {},
            "data_sources": ["polygon"],
            "cache_settings": {
                "enabled": True,
                "ttl": 300,
                "max_items": 1000
            },
            "rate_limiting": {
                "max_requests_per_minute": 60,
                "backoff_factor": 2,
                "max_retries": 3
            },
            "finbert_settings": {
                "model_path": "ProsusAI/finbert",
                "sentiment_weight": 0.3
            },
            "xgboost_settings": {
                "params": {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "eta": 0.1,
                    "max_depth": 6
                },
                "prediction_threshold": 0.5
            },
            "feature_engineering": {
                "lookback_periods": [1, 3, 5, 10, 20]
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found at {config_path}. Using default configuration.")
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading configuration from {config_path}: {e}")
            return default_config
            
    def _resolve_env_var(self, value: str) -> str:
        """
        Resolve environment variable references in string values.
        
        Args:
            value: String potentially containing ${ENV_VAR} references
            
        Returns:
            String with environment variables resolved
        """
        if not isinstance(value, str):
            return value
            
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, "")
            
        return value
        
    def _configure_from_config(self):
        """Extract configuration values from loaded config."""
        # Cache settings
        cache_config = self.config.get("cache_settings", {})
        self.enable_cache = cache_config.get("enabled", True)
        self.cache_ttl = cache_config.get("ttl", 300)  # 5 minutes default
        self.max_cache_items = cache_config.get("max_items", 1000)
        self.cache_cleanup_threshold = cache_config.get("cleanup_threshold", 0.2)
        
        # Rate limiting settings
        rate_config = self.config.get("rate_limiting", {})
        self.max_requests_per_minute = rate_config.get("max_requests_per_minute", 60)
        self.backoff_factor = rate_config.get("backoff_factor", 2)
        self.max_retries = rate_config.get("max_retries", 3)
        
        # FinBERT settings
        finbert_config = self.config.get("finbert_settings", {})
        self.finbert_model_path = finbert_config.get("model_path", "ProsusAI/finbert")
        self.max_sequence_length = finbert_config.get("max_sequence_length", 512)
        self.use_gpu = finbert_config.get("use_gpu", True) and torch.cuda.is_available()
        self.sentiment_weight = finbert_config.get("sentiment_weight", 0.3)
        self.batch_size = finbert_config.get("batch_size", 16)
        
        # XGBoost settings
        xgboost_config = self.config.get("xgboost_settings", {})
        self.xgboost_params = xgboost_config.get("params", {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "eta": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1
        })
        self.early_stopping_rounds = xgboost_config.get("early_stopping_rounds", 10)
        self.num_boost_round = xgboost_config.get("num_boost_round", 100)
        self.prediction_threshold = xgboost_config.get("prediction_threshold", 0.5)
        
        # Convert hours to seconds for model update interval
        update_hours = xgboost_config.get("model_update_interval_hours", 24)
        self.model_update_interval = update_hours * 3600
        
        # Feature engineering settings
        feature_config = self.config.get("feature_engineering", {})
        self.lookback_periods = feature_config.get("lookback_periods", [1, 3, 5, 10, 20])
        self.include_ta_features = feature_config.get("include_technical_indicators", True)
        self.include_volatility = feature_config.get("include_volatility", True)
        self.include_sentiment = feature_config.get("include_sentiment", True)
        
        # Advanced features
        self.advanced_features = feature_config.get("advanced_features", {
            "use_macd": True,
            "use_bollinger_bands": True,
            "use_rsi": True,
            "use_stochastic": False
        })
        
        # Backtesting settings
        backtest_config = self.config.get("backtesting", {})
        self.default_train_size = backtest_config.get("default_train_size", 0.7)
        self.evaluation_metrics = backtest_config.get("evaluation_metrics", 
                                                    ["accuracy", "f1"])
        
        # Trading simulation settings
        trading_config = self.config.get("trading_simulation", {})
        self.initial_capital = trading_config.get("initial_capital", 10000)
        self.position_size_pct = trading_config.get("position_size_pct", 0.1)
        
        # Sentiment analysis settings
        sentiment_config = self.config.get("sentiment_analysis", {})
        self.sentiment_weighting = sentiment_config.get("weighting_scheme", "exponential_decay")
        self.title_multiplier = sentiment_config.get("title_multiplier", 2.0)
        
        # Logging settings
        logging_config = self.config.get("logging", {})
        self.log_level = logging_config.get("level", "INFO")
        
        # Log loaded configuration
        self.logger.info("Configuration loaded", 
                        finbert_model=self.finbert_model_path,
                        prediction_threshold=self.prediction_threshold,
                        sentiment_weight=self.sentiment_weight,
                        cache_enabled=self.enable_cache,
                        max_requests_per_minute=self.max_requests_per_minute)
        
    def _initialize_models(self):
        """Initialize FinBERT and XGBoost models based on configuration."""
        # Initialize model containers
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.xgboost_models = {}  # Dictionary of XGBoost models by symbol
        self.model_last_updated = {}  # Timestamp of last model update by symbol
        
        # Load FinBERT if sentiment analysis is enabled
        if self.include_sentiment:
            try:
                self.logger.info(f"Loading FinBERT model: {self.finbert_model_path}")
                start_time = time.time()
                
                self.finbert_tokenizer = AutoTokenizer.from_pretrained(self.finbert_model_path)
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained(self.finbert_model_path)
                
                # Set to evaluation mode and move to GPU if available and enabled
                self.finbert_model.eval()
                if self.use_gpu:
                    self.finbert_model = self.finbert_model.to('cuda')
                    self.logger.info("FinBERT model moved to GPU")
                    
                load_time = time.time() - start_time
                self.logger.timing("model_load_time_ms.finbert", load_time * 1000)
                self.logger.info(f"FinBERT model loaded in {load_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Error loading FinBERT model: {e}", exc_info=True)
        else:
            self.logger.info("Sentiment analysis disabled in config, skipping FinBERT model loading")
            
        # XGBoost models will be created on-demand per symbol
        
    def _start_health_check_thread(self):
        """Start a background thread for health monitoring."""
        def health_check_loop():
            check_interval = self.config.get("monitoring", {}).get("health_check_interval_mins", 15) * 60
            self.logger.info(f"Starting health check thread with interval {check_interval} seconds")
            
            while True:
                try:
                    # Get system health
                    health_report = self.get_system_health()
                    
                    # Log health status
                    self.logger.info(f"Health check: {health_report['overall_status']}")
                    
                    # Check for model degradation if configured
                    if self.config.get("monitoring", {}).get("alert_on_model_degradation", False):
                        degradation_threshold = self.config.get("monitoring", {}).get("degradation_threshold", 0.1)
                        
                        # Check models with performance metrics
                        for symbol, model in health_report.get("components", {}).get("models", {}).get("symbol_metrics", {}).items():
                            if model.get("degradation", 0) > degradation_threshold:
                                self.logger.warning(f"Model degradation detected for {symbol}: {model['degradation']:.2f}")
                                
                                # Trigger model update if degradation is severe
                                if model.get("degradation", 0) > degradation_threshold * 2:
                                    self.logger.info(f"Severe degradation detected, triggering model update for {symbol}")
                                    try:
                                        # Run model update in a separate thread to avoid blocking health check
                                        update_thread = threading.Thread(
                                            target=self.train_prediction_model,
                                            args=(symbol,)
                                        )
                                        update_thread.daemon = True
                                        update_thread.start()
                                    except Exception as e:
                                        self.logger.error(f"Failed to trigger model update: {e}")
                                    
                except Exception as e:
                    self.logger.error(f"Error in health check thread: {e}", exc_info=True)
                    
                # Sleep until next check
                time.sleep(check_interval)
        
        # Start the health check thread
        health_thread = threading.Thread(target=health_check_loop)
        health_thread.daemon = True
        health_thread.start()
        
    def _register_tools(self):
        """Register all available tools based on configuration."""
        # Register market prediction tools
        self.register_tool(
            self.predict_price_movement,
            "predict_price_movement",
            "Predict price movement direction (up/down) for a given symbol"
        )
        self.register_tool(
            self.analyze_symbol_with_news,
            "analyze_symbol_with_news",
            "Combined analysis of market data and news sentiment for a symbol"
        )
        self.register_tool(
            self.train_prediction_model,
            "train_prediction_model",
            "Train or update a prediction model for a specific symbol"
        )
        self.register_tool(
            self.backtest_model,
            "backtest_model",
            "Backtest a prediction model over historical data"
        )
        self.register_tool(
            self.get_model_performance,
            "get_model_performance",
            "Get performance metrics for a prediction model"
        )
        
        # Register monitoring and maintenance tools
        self.register_tool(
            self.get_system_health,
            "get_system_health",
            "Get system health metrics"
        )
        self.register_tool(
            self.clear_cache,
            "clear_cache",
            "Clear the cache for a specific symbol or all symbols"
        )
        
        # Register configuration tools
        self.register_tool(
            self.get_current_config,
            "get_current_config",
            "Get the current configuration settings"
        )
        self.register_tool(
            self.update_config,
            "update_config",
            "Update configuration settings at runtime"
        )
    
    def _check_rate_limit(self) -> bool:
        """
        Check if we are within rate limits.
        
        Returns:
            True if request can proceed, False if rate limited
        """
        with self.rate_limit_lock:
            current_time = time.time()
            
            # Remove timestamps older than 1 minute
            self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
            
            # Check if we are within rate limits
            if len(self.request_timestamps) >= self.max_requests_per_minute:
                return False
                
            # Add current timestamp
            self.request_timestamps.append(current_time)
            return True
    
    def analyze_symbol_with_news(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of a symbol using both market data and news sentiment.
        
        Args:
            symbol: The stock symbol to analyze
            days: Number of days of historical data to include
            
        Returns:
            Dict with combined market and sentiment analysis
        """
        start_time = time.time()
        self.logger.info(f"Analyzing symbol {symbol} with news for past {days} days")
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded. Try again later.")
            return {"error": "Rate limit exceeded. Try again later."}
            
        # Check cache first
        cache_key = f"analyze_symbol_with_news:{symbol}:{days}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self.logger.info(f"Using cached analysis for {symbol}")
            return cached_result
        
        try:
            # Get market data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            market_data_result = self.data_mcp.get_market_data(
                symbols=[symbol],
                timeframe="1d",
                start_date=start_date,
                end_date=end_date,
                data_type="bars"
            )
            
            if "error" in market_data_result:
                return {"error": f"Failed to get market data: {market_data_result['error']}"}
                
            # Get news for the symbol
            news_result = self.data_mcp.get_news(symbols=[symbol], limit=30)
            
            if "error" in news_result:
                self.logger.warning(f"Failed to get news for {symbol}: {news_result['error']}")
                news_articles = []
            else:
                news_articles = news_result.get("news", [])
            
            # Analyze sentiment for each news article
            sentiment_scores = []
            
            for article in news_articles:
                title = article.get("title", "")
                content = article.get("summary", "")
                
                # Skip articles with missing content
                if not title and not content:
                    continue
                
                # Apply title multiplier if configured
                if self.title_multiplier > 1.0:
                    combined_text = f"{title} " * int(self.title_multiplier) + content
                else:
                    combined_text = f"{title} {content}"
                
                # Analyze sentiment
                sentiment_result = self.text_mcp.analyze_sentiment(combined_text)
                
                if "error" not in sentiment_result:
                    # Get published date for weighting if using exponential decay
                    published_date = article.get("published_utc", "")
                    published_timestamp = None
                    
                    if published_date and self.sentiment_weighting == "exponential_decay":
                        try:
                            if isinstance(published_date, str):
                                published_timestamp = datetime.fromisoformat(published_date).timestamp()
                            else:
                                published_timestamp = published_date
                        except ValueError:
                            published_timestamp = None
                    
                    # Add to sentiment scores with metadata
                    sentiment_scores.append({
                        "date": article.get("published_utc", ""),
                        "title": title,
                        "sentiment": sentiment_result.get("sentiment", "neutral"),
                        "score": sentiment_result.get("score", 0.5),
                        "timestamp": published_timestamp
                    })
            
            # Process market data
            price_data = self._process_price_data(market_data_result.get("data", {}).get(symbol, []))
            
            # Calculate summary metrics
            price_change = 0
            if price_data and len(price_data) > 1:
                first_price = price_data[0].get("c", 0)
                last_price = price_data[-1].get("c", 0)
                price_change = (last_price - first_price) / first_price if first_price > 0 else 0
            
            # Calculate sentiment summary with configured weighting scheme
            avg_sentiment_score = self._calculate_weighted_sentiment(sentiment_scores)
            
            # Make prediction if we have a model
            prediction = None
            with self.model_lock:
                if symbol in self.xgboost_models:
                    # Get features for prediction
                    features = self._extract_features(
                        price_data, 
                        avg_sentiment_score, 
                        include_sentiment=self.include_sentiment
                    )
                    
                    if features is not None:
                        # Predict using XGBoost model
                        xgb_data = xgb.DMatrix([features])
                        prediction_prob = float(self.xgboost_models[symbol].predict(xgb_data)[0])
                        prediction = {
                            "direction": "up" if prediction_prob > self.prediction_threshold else "down",
                            "probability": prediction_prob,
                            "confidence": abs(prediction_prob - 0.5) * 2  # Scale to 0-1
                        }
            
            # Create the result
            result = {
                "symbol": symbol,
                "current_price": price_data[-1].get("c") if price_data else None,
                "price_change_pct": price_change * 100,
                "period_days": days,
                "news_count": len(news_articles),
                "avg_sentiment": avg_sentiment_score,
                "sentiment_summary": "positive" if avg_sentiment_score > 0.1 else "negative" if avg_sentiment_score < -0.1 else "neutral",
                "recent_news": sentiment_scores[:5],  # Include 5 most recent news articles
                "prediction": prediction,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol} with news: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}
    
    def _calculate_weighted_sentiment(self, sentiment_scores: List[Dict[str, Any]]) -> float:
        """
        Calculate weighted sentiment score using the configured weighting scheme.
        
        Args:
            sentiment_scores: List of sentiment score dicts
            
        Returns:
            Weighted average sentiment score
        """
        if not sentiment_scores:
            return 0
            
        # Apply weighting scheme
        if self.sentiment_weighting == "exponential_decay":
            # Weight recent articles more heavily
            current_time = time.time()
            weighted_sum = 0
            total_weight = 0
            
            # Get half-life in seconds (default 48 hours)
            half_life_hours = self.config.get("sentiment_analysis", {}).get("recency_half_life_hours", 48)
            half_life = half_life_hours * 3600
            
            for score in sentiment_scores:
                # Get sentiment value (-1 to 1 scale)
                value = score.get("score", 0.5)
                if score.get("sentiment") == "negative":
                    value = -value
                elif score.get("sentiment") == "neutral":
                    value = 0
                
                # Calculate time-based weight
                timestamp = score.get("timestamp")
                if timestamp:
                    age = current_time - timestamp
                    weight = 2 ** (-age / half_life)  # Exponential decay
                else:
                    weight = 1.0  # Default weight if no timestamp
                
                weighted_sum += value * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0
            
        elif self.sentiment_weighting == "source_based":
            # Weight by news source importance
            # This would require source information in the sentiment scores
            # Use simple average as fallback
            values = []
            for score in sentiment_scores:
                value = score.get("score", 0.5)
                if score.get("sentiment") == "negative":
                    value = -value
                elif score.get("sentiment") == "neutral":
                    value = 0
                
                values.append(value)
                
            return sum(values) / len(values) if values else 0
            
        else:
            # Simple average (default)
            values = []
            for score in sentiment_scores:
                value = score.get("score", 0.5)
                if score.get("sentiment") == "negative":
                    value = -value
                elif score.get("sentiment") == "neutral":
                    value = 0
                
                values.append(value)
                
            return sum(values) / len(values) if values else 0
    
    def predict_price_movement(self, symbol: str, include_sentiment: bool = None) -> Dict[str, Any]:
        """
        Predict price movement direction for a given symbol.
        
        Args:
            symbol: The stock symbol to predict
            include_sentiment: Whether to include sentiment data in prediction (defaults to config setting)
            
        Returns:
            Dict with prediction results
        """
        start_time = time.time()
        self.logger.info(f"Predicting price movement for {symbol}")
        
        # Use config default if include_sentiment is not specified
        if include_sentiment is None:
            include_sentiment = self.include_sentiment
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded. Try again later.")
            return {"error": "Rate limit exceeded. Try again later."}
            
        # Check cache first
        cache_key = f"predict_price_movement:{symbol}:{include_sentiment}"
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            self.logger.info(f"Using cached prediction for {symbol}")
            return cached_result
        
        try:
            # Check if we need to train/update the model
            self._ensure_model_is_current(symbol)
            
            # Get latest market data
            end_date = datetime.now().strftime("%Y-%m-%d")
            lookback_days = max(self.lookback_periods) * 2  # Need enough data for features
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            market_data_result = self.data_mcp.get_market_data(
                symbols=[symbol],
                timeframe="1d",
                start_date=start_date,
                end_date=end_date,
                data_type="bars"
            )
            
            if "error" in market_data_result:
                return {"error": f"Failed to get market data: {market_data_result['error']}"}
                
            # Process price data
            price_data = self._process_price_data(market_data_result.get("data", {}).get(symbol, []))
            
            if not price_data or len(price_data) < max(self.lookback_periods):
                return {"error": f"Insufficient price data for {symbol}"}
            
            # Get sentiment if needed
            sentiment_score = 0
            if include_sentiment:
                # Get recent news
                news_result = self.data_mcp.get_news(symbols=[symbol], limit=10)
                
                news_articles = []
                if "error" not in news_result:
                    news_articles = news_result.get("news", [])
                
                # Analyze sentiment
                sentiment_scores = []
                for article in news_articles:
                    title = article.get("title", "")
                    content = article.get("summary", "")
                    
                    if not title and not content:
                        continue
                    
                    # Apply title multiplier if configured
                    if self.title_multiplier > 1.0:
                        combined_text = f"{title} " * int(self.title_multiplier) + content
                    else:
                        combined_text = f"{title} {content}"
                    
                    sentiment_result = self.text_mcp.analyze_sentiment(combined_text)
                    
                    if "error" not in sentiment_result:
                        # Get published date for weighting
                        published_date = article.get("published_utc", "")
                        published_timestamp = None
                        
                        if published_date and self.sentiment_weighting == "exponential_decay":
                            try:
                                if isinstance(published_date, str):
                                    published_timestamp = datetime.fromisoformat(published_date).timestamp()
                                else:
                                    published_timestamp = published_date
                            except ValueError:
                                published_timestamp = None
                        
                        # Add to sentiment scores with metadata
                        sentiment_scores.append({
                            "date": article.get("published_utc", ""),
                            "title": title,
                            "sentiment": sentiment_result.get("sentiment", "neutral"),
                            "score": sentiment_result.get("score", 0.5),
                            "timestamp": published_timestamp
                        })
                
                # Calculate weighted sentiment
                sentiment_score = self._calculate_weighted_sentiment(sentiment_scores)
            
            # Extract features
            features = self._extract_features(price_data, sentiment_score, include_sentiment)
            
            if features is None:
                return {"error": "Failed to extract features for prediction"}
            
            # Make prediction
            with self.model_lock:
                if symbol not in self.xgboost_models:
                    return {"error": f"No prediction model available for {symbol}"}
                
                # Convert features to DMatrix
                xgb_data = xgb.DMatrix([features])
                
                # Predict
                prediction_prob = float(self.xgboost_models[symbol].predict(xgb_data)[0])
                
                # Create result
                result = {
                    "symbol": symbol,
                    "prediction": "up" if prediction_prob > self.prediction_threshold else "down",
                    "probability": prediction_prob,
                    "confidence": abs(prediction_prob - 0.5) * 2,  # Scale to 0-1
                    "current_price": price_data[-1].get("c") if price_data else None,
                    "sentiment_included": include_sentiment,
                    "sentiment_score": sentiment_score if include_sentiment else None,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
                
                # Add model information
                result["model_info"] = {
                    "last_updated": self.model_last_updated.get(symbol, "never"),
                    "features_used": len(features)
                }
                
                # Cache the result
                self._add_to_cache(cache_key, result)
                
                return result
        except Exception as e:
            self.logger.error(f"Error predicting price movement for {symbol}: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}
    
    def train_prediction_model(self, symbol: str, lookback_days: int = 365, 
                              include_sentiment: bool = None) -> Dict[str, Any]:
        """
        Train or update a prediction model for a specific symbol.
        
        Args:
            symbol: The stock symbol to train for
            lookback_days: Number of days of historical data to use
            include_sentiment: Whether to include sentiment features (defaults to config setting)
            
        Returns:
            Dict with training results
        """
        start_time = time.time()
        self.logger.info(f"Training prediction model for {symbol}")
        
        # Use config default if include_sentiment is not specified
        if include_sentiment is None:
            include_sentiment = self.include_sentiment
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded. Try again later.")
            return {"error": "Rate limit exceeded. Try again later."}
        
        try:
            # Get historical market data
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            market_data_result = self.data_mcp.get_market_data(
                symbols=[symbol],
                timeframe="1d",
                start_date=start_date,
                end_date=end_date,
                data_type="bars"
            )
            
            if "error" in market_data_result:
                return {"error": f"Failed to get market data: {market_data_result['error']}"}
                
            # Process price data
            price_data = self._process_price_data(market_data_result.get("data", {}).get(symbol, []))
            
            if not price_data or len(price_data) < 60:  # Need enough data for training
                return {"error": f"Insufficient price data for {symbol}"}
            
            # Get sentiment data if needed
            sentiment_data = None
            if include_sentiment:
                sentiment_data = self._get_historical_sentiment(symbol, start_date, end_date)
            
            # Prepare training data
            X, y = self._prepare_training_data(price_data, sentiment_data, include_sentiment)
            
            if len(X) < 50 or len(y) < 50:
                return {"error": f"Insufficient processed data for training (need at least 50 samples)"}
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create DMatrix objects
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            # Set up parameters from config
            params = self.xgboost_params.copy()
            
            # Train the model
            evals = [(dtrain, 'train'), (dval, 'val')]
            
            model = xgb.train(
                params,
                dtrain,
                self.num_boost_round,
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )
            
            # Evaluate model
            y_pred_proba = model.predict(dval)
            y_pred = (y_pred_proba > self.prediction_threshold).astype(int)
            
            # Calculate metrics
            metrics = {}
            if "accuracy" in self.evaluation_metrics:
                metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
            if "precision" in self.evaluation_metrics:
                metrics["precision"] = float(precision_score(y_val, y_pred))
            if "recall" in self.evaluation_metrics:
                metrics["recall"] = float(recall_score(y_val, y_pred))
            if "f1" in self.evaluation_metrics:
                metrics["f1_score"] = float(f1_score(y_val, y_pred))
            
            # Save the model
            with self.model_lock:
                self.xgboost_models[symbol] = model
                self.model_last_updated[symbol] = datetime.now().isoformat()
            
            # Create result
            result = {
                "symbol": symbol,
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "metrics": metrics,
                "model_info": {
                    "feature_count": X.shape[1],
                    "includes_sentiment": include_sentiment,
                    "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else None,
                    "training_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            }
            
            # Log top features
            if "feature_importance" in self.evaluation_metrics:
                feature_importance = model.get_score(importance_type='gain')
                result["feature_importance"] = {f"feature_{k}": v for k, v in feature_importance.items()}
            
            self.logger.info(f"Model for {symbol} trained successfully. Metrics: {metrics}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error training model for {symbol}: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}
    
    def backtest_model(self, symbol: str, start_date: str, end_date: str = None,
                      include_sentiment: bool = None, sliding_window: bool = None) -> Dict[str, Any]:
        """
        Backtest a prediction model over historical data.
        
        Args:
            symbol: The stock symbol to backtest
            start_date: Start date for backtesting in YYYY-MM-DD format
            end_date: End date for backtesting (defaults to today)
            include_sentiment: Whether to include sentiment in backtesting (defaults to config setting)
            sliding_window: Whether to use sliding window backtesting (defaults to config setting)
            
        Returns:
            Dict with backtesting results
        """
        start_time = time.time()
        
        # Use config defaults if parameters not specified
        if include_sentiment is None:
            include_sentiment = self.include_sentiment
            
        if sliding_window is None:
            sliding_window = self.config.get("backtesting", {}).get("sliding_window", {}).get("enabled", False)
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        self.logger.info(f"Backtesting model for {symbol} from {start_date} to {end_date}")
        
        # Check rate limit
        if not self._check_rate_limit():
            self.logger.warning(f"Rate limit exceeded. Try again later.")
            return {"error": "Rate limit exceeded. Try again later."}
        
        try:
            # Get historical market data
            market_data_result = self.data_mcp.get_market_data(
                symbols=[symbol],
                timeframe="1d",
                start_date=start_date,
                end_date=end_date,
                data_type="bars"
            )
            
            if "error" in market_data_result:
                return {"error": f"Failed to get market data: {market_data_result['error']}"}
                
            # Process price data
            price_data = self._process_price_data(market_data_result.get("data", {}).get(symbol, []))
            
            if not price_data or len(price_data) < 60:  # Need enough data for meaningful backtest
                return {"error": f"Insufficient price data for {symbol}"}
            
            # Get sentiment data if needed
            sentiment_data = None
            if include_sentiment:
                sentiment_data = self._get_historical_sentiment(symbol, start_date, end_date)
            
            # Use sliding window approach if enabled
            if sliding_window:
                return self._sliding_window_backtest(symbol, price_data, sentiment_data, include_sentiment)
            else:
                # Use single train/test split
                return self._standard_backtest(symbol, price_data, sentiment_data, include_sentiment)
            
        except Exception as e:
            self.logger.error(f"Error backtesting model for {symbol}: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}
    
    def _standard_backtest(self, symbol: str, price_data: List[Dict[str, Any]], 
                          sentiment_data: Optional[Dict[str, float]], 
                          include_sentiment: bool) -> Dict[str, Any]:
        """
        Perform standard backtesting with a single train/test split.
        
        Args:
            symbol: The stock symbol
            price_data: Processed price data
            sentiment_data: Sentiment data dictionary
            include_sentiment: Whether to include sentiment
            
        Returns:
            Dict with backtesting results
        """
        start_time = time.time()
        
        # Prepare training and testing data
        train_size = self.default_train_size
        split_idx = int(len(price_data) * train_size)
        
        train_price_data = price_data[:split_idx]
        test_price_data = price_data[split_idx:]
        
        # Prepare sentiment data if available
        train_sentiment_data = None
        test_sentiment_data = None
        if sentiment_data:
            # Match sentiment data to price data dates
            price_dates = [self._get_date_str(pd.get("t")) for pd in price_data]
            aligned_sentiment = {}
            
            for date, sentiment in sentiment_data.items():
                if date in price_dates:
                    aligned_sentiment[date] = sentiment
            
            # Split sentiment data
            train_date_strs = [self._get_date_str(pd.get("t")) for pd in train_price_data]
            test_date_strs = [self._get_date_str(pd.get("t")) for pd in test_price_data]
            
            train_sentiment_data = {k: v for k, v in aligned_sentiment.items() if k in train_date_strs}
            test_sentiment_data = {k: v for k, v in aligned_sentiment.items() if k in test_date_strs}
        
        # Prepare training data
        X_train, y_train = self._prepare_training_data(train_price_data, train_sentiment_data, include_sentiment)
        
        if len(X_train) < 30:
            return {"error": f"Insufficient training data after preprocessing"}
        
        # Train a model specifically for this backtest
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = self.xgboost_params.copy()
        backtest_model = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)
        
        # Prepare test data
        X_test, y_test = self._prepare_training_data(test_price_data, test_sentiment_data, include_sentiment)
        
        if len(X_test) < 10:
            return {"error": f"Insufficient test data after preprocessing"}
        
        # Make predictions
        dtest = xgb.DMatrix(X_test)
        y_pred_proba = backtest_model.predict(dtest)
        y_pred = (y_pred_proba > self.prediction_threshold).astype(int)
        
        # Calculate metrics
        metrics = {}
        if "accuracy" in self.evaluation_metrics:
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        if "precision" in self.evaluation_metrics:
            metrics["precision"] = float(precision_score(y_test, y_pred))
        if "recall" in self.evaluation_metrics:
            metrics["recall"] = float(recall_score(y_test, y_pred))
        if "f1" in self.evaluation_metrics:
            metrics["f1_score"] = float(f1_score(y_test, y_pred))
        
        # Calculate returns for trading simulation
        test_returns, cumulative_returns = self._calculate_trading_returns(
            symbol, test_price_data, y_test, y_pred
        )
        
        # Calculate trading performance metrics
        win_rate = sum(r > 0 for r in test_returns) / len(test_returns) if test_returns else 0
        
        # Calculate profit factor if in evaluation metrics
        profit_factor = 0
        if "profit_factor" in self.evaluation_metrics and test_returns:
            gains = sum(r for r in test_returns if r > 0)
            losses = abs(sum(r for r in test_returns if r < 0))
            profit_factor = gains / losses if losses > 0 else float('inf')
            metrics["profit_factor"] = profit_factor
        
        # Create result
        result = {
            "symbol": symbol,
            "period": {
                "start_date": train_price_data[0].get("t") if train_price_data else None,
                "end_date": test_price_data[-1].get("t") if test_price_data else None,
                "train_samples": len(X_train),
                "test_samples": len(X_test)
            },
            "metrics": metrics,
            "trading_performance": {
                "win_rate": win_rate,
                "cumulative_return": cumulative_returns[-1] if cumulative_returns else 0,
                "average_return": sum(test_returns) / len(test_returns) if test_returns else 0
            },
            "includes_sentiment": include_sentiment,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Generate performance visualization if possible
        try:
            chart_path = self._generate_backtest_chart(symbol, y_test, y_pred, test_returns, cumulative_returns)
            if chart_path:
                result["chart_path"] = chart_path
        except Exception as chart_e:
            self.logger.warning(f"Failed to generate backtest chart: {chart_e}")
        
        return result
    
    def _sliding_window_backtest(self, symbol: str, price_data: List[Dict[str, Any]], 
                                sentiment_data: Optional[Dict[str, float]], 
                                include_sentiment: bool) -> Dict[str, Any]:
        """
        Perform sliding window backtesting for more robust results.
        
        Args:
            symbol: The stock symbol
            price_data: Processed price data
            sentiment_data: Sentiment data dictionary
            include_sentiment: Whether to include sentiment
            
        Returns:
            Dict with backtesting results
        """
        start_time = time.time()
        
        # Get sliding window parameters from config
        window_size_days = self.config.get("backtesting", {}).get("sliding_window", {}).get("window_size_days", 90)
        step_size_days = self.config.get("backtesting", {}).get("sliding_window", {}).get("step_size_days", 30)
        
        # Convert days to data points (assuming daily data)
        window_size = window_size_days
        step_size = step_size_days
        
        # Initialize result containers
        all_metrics = []
        all_returns = []
        
        # Calculate number of windows
        num_windows = max(1, (len(price_data) - window_size) // step_size)
        
        self.logger.info(f"Performing sliding window backtest with {num_windows} windows")
        
        # Process each window
        for i in range(num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            # Get window data
            window_data = price_data[start_idx:end_idx]
            
            # Skip windows with insufficient data
            if len(window_data) < 60:
                continue
                
            # Split window data
            train_size = self.default_train_size
            split_idx = int(len(window_data) * train_size)
            
            train_price_data = window_data[:split_idx]
            test_price_data = window_data[split_idx:]
            
            # Prepare sentiment data if available
            train_sentiment_data = None
            test_sentiment_data = None
            if sentiment_data:
                # Match sentiment data to window price data dates
                train_date_strs = [self._get_date_str(pd.get("t")) for pd in train_price_data]
                test_date_strs = [self._get_date_str(pd.get("t")) for pd in test_price_data]
                
                train_sentiment_data = {k: v for k, v in sentiment_data.items() if k in train_date_strs}
                test_sentiment_data = {k: v for k, v in sentiment_data.items() if k in test_date_strs}
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(train_price_data, train_sentiment_data, include_sentiment)
            
            if len(X_train) < 20:
                continue
            
            # Train a model for this window
            dtrain = xgb.DMatrix(X_train, label=y_train)
            params = self.xgboost_params.copy()
            window_model = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)
            
            # Prepare test data
            X_test, y_test = self._prepare_training_data(test_price_data, test_sentiment_data, include_sentiment)
            
            if len(X_test) < 5:
                continue
            
            # Make predictions
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = window_model.predict(dtest)
            y_pred = (y_pred_proba > self.prediction_threshold).astype(int)
            
            # Calculate metrics
            window_metrics = {}
            if "accuracy" in self.evaluation_metrics:
                window_metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            if "precision" in self.evaluation_metrics:
                window_metrics["precision"] = float(precision_score(y_test, y_pred))
            if "recall" in self.evaluation_metrics:
                window_metrics["recall"] = float(recall_score(y_test, y_pred))
            if "f1" in self.evaluation_metrics:
                window_metrics["f1_score"] = float(f1_score(y_test, y_pred))
            
            # Calculate window returns
            window_returns, _ = self._calculate_trading_returns(
                symbol, test_price_data, y_test, y_pred
            )
            
            # Store window results
            all_metrics.append(window_metrics)
            all_returns.extend(window_returns)
        
        # Calculate average metrics across all windows
        avg_metrics = {}
        for metric in self.evaluation_metrics:
            if metric in all_metrics[0]:
                values = [m.get(metric, 0) for m in all_metrics]
                avg_metrics[metric] = sum(values) / len(values)
        
        # Calculate cumulative returns
        cumulative_returns = []
        cumulative = 0
        for r in all_returns:
            cumulative += r
            cumulative_returns.append(cumulative)
        
        # Calculate trading performance metrics
        win_rate = sum(r > 0 for r in all_returns) / len(all_returns) if all_returns else 0
        
        # Calculate profit factor if in evaluation metrics
        if "profit_factor" in self.evaluation_metrics and all_returns:
            gains = sum(r for r in all_returns if r > 0)
            losses = abs(sum(r for r in all_returns if r < 0))
            profit_factor = gains / losses if losses > 0 else float('inf')
            avg_metrics["profit_factor"] = profit_factor
        
        # Create result
        result = {
            "symbol": symbol,
            "period": {
                "start_date": price_data[0].get("t") if price_data else None,
                "end_date": price_data[-1].get("t") if price_data else None,
                "windows": num_windows,
                "window_size_days": window_size_days,
                "step_size_days": step_size_days
            },
            "metrics": avg_metrics,
            "trading_performance": {
                "win_rate": win_rate,
                "cumulative_return": cumulative_returns[-1] if cumulative_returns else 0,
                "average_return": sum(all_returns) / len(all_returns) if all_returns else 0
            },
            "includes_sentiment": include_sentiment,
            "method": "sliding_window",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Generate performance visualization if possible
        try:
            # Use last window predictions for visualization
            if len(y_test) > 0 and len(y_pred) > 0:
                chart_path = self._generate_backtest_chart(symbol, y_test, y_pred, all_returns, cumulative_returns)
                if chart_path:
                    result["chart_path"] = chart_path
        except Exception as chart_e:
            self.logger.warning(f"Failed to generate backtest chart: {chart_e}")
        
        return result
    
    def _calculate_trading_returns(self, symbol: str, price_data: List[Dict[str, Any]], 
                                y_actual: List[int], y_pred: List[int]) -> Tuple[List[float], List[float]]:
        """
        Calculate returns based on trading simulation.
        
        Args:
            symbol: The stock symbol
            price_data: Price data for the test period
            y_actual: Actual price movements
            y_pred: Predicted price movements
            
        Returns:
            Tuple of (returns, cumulative_returns)
        """
        # Check lengths
        if len(y_pred) > len(price_data) - 1:
            y_pred = y_pred[:(len(price_data) - 1)]
        if len(y_actual) > len(price_data) - 1:
            y_actual = y_actual[:(len(price_data) - 1)]
            
        # Get trading simulation parameters
        position_size = self.position_size_pct
        include_costs = self.config.get("trading_simulation", {}).get("include_transaction_costs", True)
        commission_rate = self.config.get("trading_simulation", {}).get("commission_per_trade", 0.001)
        
        # Calculate returns
        returns = []
        
        for i in range(len(y_pred)):
            # Get prices
            current_price = price_data[i].get("c", 0)
            next_price = price_data[i+1].get("c", 0)
            
            if current_price <= 0 or next_price <= 0:
                continue
                
            # Calculate actual return
            actual_return = (next_price - current_price) / current_price
            
            # Determine position
            position = 1 if y_pred[i] == 1 else -1  # Long or short
            
            # Calculate strategy return based on position
            strategy_return = position * actual_return
            
            # Apply position sizing
            strategy_return *= position_size
            
            # Apply transaction costs if enabled
            if include_costs:
                strategy_return -= commission_rate
            
            returns.append(strategy_return)
        
        # Calculate cumulative returns
        cumulative_returns = []
        cumulative = 0
        for r in returns:
            cumulative += r
            cumulative_returns.append(cumulative)
        
        return returns, cumulative_returns
    
    def get_model_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance metrics for a prediction model.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Dict with model performance metrics
        """
        start_time = time.time()
        
        # Check if model exists
        with self.model_lock:
            if symbol not in self.xgboost_models:
                return {"error": f"No model found for {symbol}"}
            
            model = self.xgboost_models[symbol]
            last_updated = self.model_last_updated.get(symbol, "unknown")
        
        # Get feature importance
        feature_importance = {f"feature_{i}": float(importance) 
                             for i, importance in enumerate(model.get_score(importance_type='gain').values())}
        
        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), 
                                        key=lambda item: item[1], reverse=True))
        
        # Get recent prediction accuracy
        # This would require storing past predictions and outcomes
        # Here we'll simulate this with a simple 30-day backtest
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        backtest_result = self.backtest_model(symbol, start_date, end_date)
        
        recent_metrics = {}
        if "error" not in backtest_result:
            recent_metrics = backtest_result.get("metrics", {})
            
            # Check for model degradation
            if hasattr(self, "baseline_metrics") and symbol in self.baseline_metrics:
                baseline = self.baseline_metrics[symbol]
                
                # Calculate degradation as percentage decrease in accuracy
                if "accuracy" in recent_metrics and "accuracy" in baseline:
                    degradation = (baseline["accuracy"] - recent_metrics["accuracy"]) / baseline["accuracy"]
                    recent_metrics["degradation"] = degradation
        
        # Create result
        result = {
            "symbol": symbol,
            "last_updated": last_updated,
            "feature_importance": feature_importance,
            "recent_metrics": recent_metrics,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return result
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get system health metrics.
        
        Returns:
            Dict with system health information
        """
        start_time = time.time()
        
        # Get data MCP health
        data_mcp_health = self.data_mcp.get_health_status() if hasattr(self.data_mcp, "get_health_status") else {"status": "unknown"}
        
        # Get text MCP health
        text_mcp_health = self.text_mcp.get_health_status() if hasattr(self.text_mcp, "get_health_status") else {"status": "unknown"}
        
        # Get model information
        model_info = {}
        symbol_metrics = {}
        with self.model_lock:
            model_count = len(self.xgboost_models)
            model_symbols = list(self.xgboost_models.keys())
            
            # Get last update time for each model
            model_ages = {}
            for symbol, update_time in self.model_last_updated.items():
                if update_time != "never":
                    try:
                        last_update = datetime.fromisoformat(update_time)
                        age_days = (datetime.now() - last_update).days
                        model_ages[symbol] = age_days
                        
                        # Check model performance if we have baseline metrics
                        if hasattr(self, "baseline_metrics") and symbol in self.baseline_metrics:
                            # Get latest performance
                            end_date = datetime.now().strftime("%Y-%m-%d")
                            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
                            
                            try:
                                backtest_result = self.backtest_model(symbol, start_date, end_date)
                                
                                if "error" not in backtest_result:
                                    recent_metrics = backtest_result.get("metrics", {})
                                    baseline = self.baseline_metrics[symbol]
                                    
                                    # Calculate degradation
                                    if "accuracy" in recent_metrics and "accuracy" in baseline:
                                        degradation = (baseline["accuracy"] - recent_metrics["accuracy"]) / baseline["accuracy"]
                                        
                                        # Add to symbol metrics
                                        symbol_metrics[symbol] = {
                                            "recent_accuracy": recent_metrics["accuracy"],
                                            "baseline_accuracy": baseline["accuracy"],
                                            "degradation": degradation,
                                            "age_days": age_days
                                        }
                            except Exception as e:
                                self.logger.warning(f"Failed to check model performance for {symbol}: {e}")
                    except ValueError:
                        model_ages[symbol] = "unknown"
                else:
                    model_ages[symbol] = "never"
            
            model_info = {
                "model_count": model_count,
                "symbols_with_models": model_symbols,
                "model_ages_days": model_ages,
                "symbol_metrics": symbol_metrics
            }
        
        # Get cache stats
        # Get cache stats
        cache_stats = {}
        with self.cache_lock:
            cache_stats = {
                "size": len(self.cache),
                "enabled": self.enable_cache,
                "ttl_seconds": self.cache_ttl,
                "keys_by_type": {}
            }
            
            # Count keys by type
            for key in self.cache.keys():
                key_type = key.split(":")[0] if ":" in key else "unknown"
                if key_type not in cache_stats["keys_by_type"]:
                    cache_stats["keys_by_type"][key_type] = 0
                cache_stats["keys_by_type"][key_type] += 1
        
        # Get rate limiting stats
        rate_limit_stats = {}
        with self.rate_limit_lock:
            current_time = time.time()
            recent_requests = [t for t in self.request_timestamps if current_time - t < 60]
            rate_limit_stats = {
                "max_requests_per_minute": self.max_requests_per_minute,
                "current_usage": len(recent_requests),
                "available": self.max_requests_per_minute - len(recent_requests)
            }
        
        # Get system metrics from the metrics collector
        system_metrics = {}
        if hasattr(self, "metrics_collector"):
            system_metrics = {
                "cpu_usage": self.metrics_collector.get_cpu_usage(),
                "memory_usage": self.metrics_collector.get_memory_usage(),
                "disk_usage": self.metrics_collector.get_disk_usage()
            }
        
        # Create result
        result = {
            "overall_status": "healthy",  # Default to healthy
            "components": {
                "data_mcp": data_mcp_health,
                "text_mcp": text_mcp_health,
                "models": model_info,
                "cache": cache_stats,
                "rate_limiting": rate_limit_stats
            },
            "system_metrics": system_metrics,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Determine overall status based on component statuses
        if data_mcp_health.get("status") == "critical" or text_mcp_health.get("status") == "critical":
            result["overall_status"] = "critical"
        elif data_mcp_health.get("status") == "degraded" or text_mcp_health.get("status") == "degraded":
            result["overall_status"] = "degraded"
        elif data_mcp_health.get("status") == "warning" or text_mcp_health.get("status") == "warning":
            result["overall_status"] = "warning"
        
        return result
    
    def clear_cache(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the cache for a specific symbol or all symbols.
        
        Args:
            symbol: The stock symbol to clear cache for. If None, clear all.
            
        Returns:
            Dict with operation result
        """
        start_time = time.time()
        
        with self.cache_lock:
            if symbol:
                # Clear cache for specific symbol
                keys_to_delete = [k for k in self.cache.keys() if f":{symbol}:" in k]
                for key in keys_to_delete:
                    del self.cache[key]
                    if key in self.cache_timestamps:
                        del self.cache_timestamps[key]
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "keys_deleted": len(keys_to_delete),
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
            else:
                # Clear all cache
                key_count = len(self.cache)
                self.cache.clear()
                self.cache_timestamps.clear()
                
                return {
                    "status": "success",
                    "keys_deleted": key_count,
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2)
                }
    
    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration settings.
        
        Returns:
            Dict with current configuration
        """
        # Create a sanitized version of the config (removing API keys)
        safe_config = {}
        
        # Copy all non-sensitive config
        if hasattr(self, "config"):
            safe_config = self.config.copy()
            
            # Remove API keys
            if "api_keys" in safe_config:
                safe_config["api_keys"] = {k: "***" for k in safe_config["api_keys"]}
        
        # Add runtime configuration
        runtime_config = {
            "finbert": {
                "model_loaded": self.finbert_model is not None,
                "using_gpu": self.use_gpu and torch.cuda.is_available(),
                "sentiment_weight": self.sentiment_weight
            },
            "xgboost": {
                "models_loaded": len(self.xgboost_models),
                "params": self.xgboost_params,
                "prediction_threshold": self.prediction_threshold
            },
            "cache": {
                "enabled": self.enable_cache,
                "ttl_seconds": self.cache_ttl,
                "size": len(self.cache) if hasattr(self, "cache") else 0
            },
            "feature_engineering": {
                "lookback_periods": self.lookback_periods,
                "include_ta_features": self.include_ta_features,
                "include_volatility": self.include_volatility,
                "include_sentiment": self.include_sentiment
            }
        }
        
        # Combine the configs
        result = {
            "file_config": safe_config,
            "runtime_config": runtime_config
        }
        
        return result
    
    def update_config(self, config_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration settings at runtime.
        
        Args:
            config_updates: Dictionary of configuration updates
            
        Returns:
            Dict with update status
        """
        start_time = time.time()
        updated_params = []
        failed_updates = []
        
        try:
            # Update XGBoost parameters if specified
            if "xgboost_params" in config_updates:
                try:
                    new_params = config_updates["xgboost_params"]
                    # Validate params (basic check)
                    if isinstance(new_params, dict):
                        with self.model_lock:
                            self.xgboost_params.update(new_params)
                        updated_params.append("xgboost_params")
                    else:
                        failed_updates.append("xgboost_params")
                except Exception as e:
                    self.logger.error(f"Failed to update xgboost_params: {e}")
                    failed_updates.append("xgboost_params")
            
            # Update prediction threshold if specified
            if "prediction_threshold" in config_updates:
                try:
                    new_threshold = float(config_updates["prediction_threshold"])
                    if 0.0 <= new_threshold <= 1.0:
                        with self.model_lock:
                            self.prediction_threshold = new_threshold
                        updated_params.append("prediction_threshold")
                    else:
                        failed_updates.append("prediction_threshold")
                except Exception as e:
                    self.logger.error(f"Failed to update prediction_threshold: {e}")
                    failed_updates.append("prediction_threshold")
            
            # Update sentiment weight if specified
            if "sentiment_weight" in config_updates:
                try:
                    new_weight = float(config_updates["sentiment_weight"])
                    if 0.0 <= new_weight <= 1.0:
                        self.sentiment_weight = new_weight
                        updated_params.append("sentiment_weight")
                    else:
                        failed_updates.append("sentiment_weight")
                except Exception as e:
                    self.logger.error(f"Failed to update sentiment_weight: {e}")
                    failed_updates.append("sentiment_weight")
            
            # Update cache settings if specified
            if "cache_enabled" in config_updates:
                try:
                    self.enable_cache = bool(config_updates["cache_enabled"])
                    updated_params.append("cache_enabled")
                except Exception as e:
                    self.logger.error(f"Failed to update cache_enabled: {e}")
                    failed_updates.append("cache_enabled")
            
            if "cache_ttl" in config_updates:
                try:
                    new_ttl = int(config_updates["cache_ttl"])
                    if new_ttl > 0:
                        with self.cache_lock:
                            self.cache_ttl = new_ttl
                        updated_params.append("cache_ttl")
                    else:
                        failed_updates.append("cache_ttl")
                except Exception as e:
                    self.logger.error(f"Failed to update cache_ttl: {e}")
                    failed_updates.append("cache_ttl")
            
            # Update feature engineering settings if specified
            if "include_ta_features" in config_updates:
                try:
                    self.include_ta_features = bool(config_updates["include_ta_features"])
                    updated_params.append("include_ta_features")
                except Exception as e:
                    self.logger.error(f"Failed to update include_ta_features: {e}")
                    failed_updates.append("include_ta_features")
            
            if "include_volatility" in config_updates:
                try:
                    self.include_volatility = bool(config_updates["include_volatility"])
                    updated_params.append("include_volatility")
                except Exception as e:
                    self.logger.error(f"Failed to update include_volatility: {e}")
                    failed_updates.append("include_volatility")
            
            if "include_sentiment" in config_updates:
                try:
                    self.include_sentiment = bool(config_updates["include_sentiment"])
                    updated_params.append("include_sentiment")
                except Exception as e:
                    self.logger.error(f"Failed to update include_sentiment: {e}")
                    failed_updates.append("include_sentiment")
            
            # Create result
            result = {
                "status": "success" if not failed_updates else "partial",
                "updated_params": updated_params,
                "failed_updates": failed_updates,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            # Log updates
            self.logger.info(f"Configuration updated", 
                           updated=updated_params, 
                           failed=failed_updates)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "updated_params": updated_params,
                "failed_updates": failed_updates,
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
    
    def _ensure_model_is_current(self, symbol: str) -> None:
        """
        Check if model needs to be trained/updated and do so if needed.
        
        Args:
            symbol: The stock symbol
        """
        with self.model_lock:
            current_time = time.time()
            
            # Check if model exists
            if symbol not in self.xgboost_models:
                self.logger.info(f"No model found for {symbol}, training new model")
                # Train model outside the lock to avoid blocking
                self.model_lock.release()
                try:
                    self.train_prediction_model(symbol)
                finally:
                    self.model_lock.acquire()
                return
            
            # Check if model needs update
            if symbol in self.model_last_updated:
                # Parse the timestamp
                try:
                    last_updated = datetime.fromisoformat(self.model_last_updated[symbol])
                    last_updated_time = last_updated.timestamp()
                    
                    if current_time - last_updated_time > self.model_update_interval:
                        self.logger.info(f"Model for {symbol} is outdated, updating")
                        # Update model outside the lock
                        self.model_lock.release()
                        try:
                            self.train_prediction_model(symbol)
                        finally:
                            self.model_lock.acquire()
                except ValueError:
                    # Invalid timestamp format
                    self.logger.warning(f"Invalid timestamp format for {symbol}: {self.model_last_updated[symbol]}")
    
    def _process_price_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process raw price data into a standard format.
        
        Args:
            raw_data: Raw price data from the data MCP
            
        Returns:
            List of standardized price data dicts
        """
        if not raw_data:
            return []
            
        # Check if we have standard Polygon-style data
        if isinstance(raw_data, list) and len(raw_data) > 0 and "c" in raw_data[0]:
            # Data is already in standard format
            # Sort by timestamp to ensure chronological order
            return sorted(raw_data, key=lambda x: x.get("t", 0))
        
        # Handle other data formats
        processed_data = []
        
        for item in raw_data:
            # Check if this is a Yahoo-style format
            if "timestamp" in item and "close" in item:
                processed_item = {
                    "t": item["timestamp"],
                    "o": item["open"],
                    "h": item["high"],
                    "l": item["low"],
                    "c": item["close"],
                    "v": item["volume"] if "volume" in item else 0
                }
                processed_data.append(processed_item)
            elif "date" in item and "close" in item:
                # Another common format
                try:
                    # Convert date string to timestamp if needed
                    if isinstance(item["date"], str):
                        timestamp = int(datetime.strptime(item["date"], "%Y-%m-%d").timestamp() * 1000)
                    else:
                        timestamp = item["date"]
                        
                    processed_item = {
                        "t": timestamp,
                        "o": item["open"],
                        "h": item["high"],
                        "l": item["low"],
                        "c": item["close"],
                        "v": item["volume"] if "volume" in item else 0
                    }
                    processed_data.append(processed_item)
                except Exception as e:
                    self.logger.warning(f"Error processing price data item: {e}")
                    continue
            else:
                # Unknown format, try to adapt by guessing fields
                try:
                    # Look for keys containing common field names
                    open_key = next((k for k in item.keys() if "open" in k.lower()), None)
                    high_key = next((k for k in item.keys() if "high" in k.lower()), None)
                    low_key = next((k for k in item.keys() if "low" in k.lower()), None)
                    close_key = next((k for k in item.keys() if "close" in k.lower()), None)
                    volume_key = next((k for k in item.keys() if "volume" in k.lower()), None)
                    time_key = next((k for k in item.keys() if any(x in k.lower() for x in ["time", "date", "timestamp"])), None)
                    
                    if close_key and time_key:
                        processed_item = {
                            "t": item[time_key],
                            "o": item.get(open_key, item[close_key]),
                            "h": item.get(high_key, item[close_key]),
                            "l": item.get(low_key, item[close_key]),
                            "c": item[close_key],
                            "v": item.get(volume_key, 0)
                        }
                        processed_data.append(processed_item)
                except Exception as e:
                    self.logger.warning(f"Failed to process unknown price data format: {e}")
                    continue
        
        # Sort by timestamp to ensure chronological order
        return sorted(processed_data, key=lambda x: x.get("t", 0))
    
    def _get_date_str(self, timestamp) -> str:
        """
        Convert timestamp to YYYY-MM-DD date string.
        
        Args:
            timestamp: Unix timestamp in milliseconds or seconds
            
        Returns:
            Date string in YYYY-MM-DD format
        """
        if not timestamp:
            return ""
            
        try:
            # Check if timestamp is in milliseconds (13 digits) or seconds (10 digits)
            if timestamp > 1000000000000:  # milliseconds
                dt = datetime.fromtimestamp(timestamp / 1000)
            else:  # seconds
                dt = datetime.fromtimestamp(timestamp)
                
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            self.logger.warning(f"Error converting timestamp {timestamp} to date string: {e}")
            return ""
    
    def _get_historical_sentiment(self, symbol: str, start_date: str, end_date: str) -> Dict[str, float]:
        """
        Get historical sentiment data for a symbol.
        
        Args:
            symbol: The stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dict mapping dates to sentiment scores
        """
        try:
            # Get news for the symbol
            news_result = self.data_mcp.get_news(symbols=[symbol], limit=100)
            
            if "error" in news_result:
                self.logger.warning(f"Failed to get news for {symbol}: {news_result['error']}")
                return {}
                
            news_articles = news_result.get("news", [])
            
            # Filter articles by date
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86400  # Add one day to include end date
            
            filtered_articles = []
            for article in news_articles:
                # Check if published date is available
                published = article.get("published_utc")
                if not published:
                    continue
                    
                # Convert to timestamp if it's a string
                if isinstance(published, str):
                    try:
                        published_timestamp = int(datetime.fromisoformat(published).timestamp())
                    except ValueError:
                        # Try other date formats
                        try:
                            published_timestamp = int(datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").timestamp())
                        except ValueError:
                            continue
                else:
                    published_timestamp = published
                
                # Check if within date range
                if start_timestamp <= published_timestamp <= end_timestamp:
                    filtered_articles.append(article)
            
            # Analyze sentiment for each article
            daily_sentiments = {}
            for article in filtered_articles:
                title = article.get("title", "")
                content = article.get("summary", "")
                
                if not title and not content:
                    continue
                
                # Get date string in YYYY-MM-DD format
                published = article.get("published_utc")
                if isinstance(published, str):
                    try:
                        date_str = datetime.fromisoformat(published).strftime("%Y-%m-%d")
                    except ValueError:
                        try:
                            date_str = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")
                        except ValueError:
                            continue
                else:
                    date_str = datetime.fromtimestamp(published).strftime("%Y-%m-%d")
                
                # Apply title multiplier if configured
                if self.title_multiplier > 1.0:
                    combined_text = f"{title} " * int(self.title_multiplier) + content
                else:
                    combined_text = f"{title} {content}"
                
                # Analyze sentiment
                sentiment_result = self.text_mcp.analyze_sentiment(combined_text)
                
                if "error" not in sentiment_result:
                    # Calculate sentiment score (-1 to 1 scale)
                    sentiment = sentiment_result.get("sentiment")
                    score = sentiment_result.get("score", 0.5)
                    
                    value = score
                    if sentiment == "negative":
                        value = -score
                    elif sentiment == "neutral":
                        value = 0
                    
                    # Add to daily sentiments
                    if date_str not in daily_sentiments:
                        daily_sentiments[date_str] = []
                    
                    daily_sentiments[date_str].append(value)
            
            # Calculate average sentiment per day
            result = {}
            for date_str, values in daily_sentiments.items():
                result[date_str] = sum(values) / len(values) if values else 0
                
            return result
                
        except Exception as e:
            self.logger.error(f"Error getting historical sentiment for {symbol}: {e}", exc_info=True)
            return {}
    
    def _extract_features(self, price_data: List[Dict[str, Any]], 
                          sentiment_score: float = 0.0,
                          include_sentiment: bool = True) -> Optional[List[float]]:
        """
        Extract features from price data for prediction.
        
        Args:
            price_data: Processed price data
            sentiment_score: Sentiment score to include
            include_sentiment: Whether to include sentiment
            
        Returns:
            List of feature values or None if feature extraction fails
        """
        if not price_data or len(price_data) < max(self.lookback_periods) + 1:
            return None
            
        try:
            # Get latest price data point
            current = price_data[-1]
            
            # Basic price features
            features = []
            
            # Price changes over different periods
            for period in self.lookback_periods:
                if len(price_data) > period:
                    past = price_data[-(period+1)]
                    
                    # Percent change in close price
                    if past["c"] > 0:
                        pct_change = (current["c"] - past["c"]) / past["c"]
                        features.append(pct_change)
                    else:
                        features.append(0)
                    
                    # Volume change
                    if past["v"] > 0:
                        vol_change = (current["v"] - past["v"]) / past["v"]
                        features.append(vol_change)
                    else:
                        features.append(0)
                        
                    # Range as percent of close
                    current_range = (current["h"] - current["l"]) / current["c"] if current["c"] > 0 else 0
                    features.append(current_range)
                    
                    # Open-close relation
                    open_close = (current["c"] - current["o"]) / current["o"] if current["o"] > 0 else 0
                    features.append(open_close)
            
            # Add technical indicators if enabled
            if self.include_ta_features:
                # Moving averages
                for period in [5, 10, 20, 50]:
                    if len(price_data) >= period:
                        ma = sum(item["c"] for item in price_data[-period:]) / period
                        # Relation to current price
                        if ma > 0:
                            ma_diff = (current["c"] - ma) / ma
                            features.append(ma_diff)
                        else:
                            features.append(0)
                
                # RSI (simplified 14-period)
                if len(price_data) >= 15 and self.advanced_features.get("use_rsi", True):  # Need 14 periods + 1 for calculation
                    gains = []
                    losses = []
                    
                    for i in range(-14, 0):
                        change = price_data[i]["c"] - price_data[i-1]["c"]
                        if change >= 0:
                            gains.append(change)
                            losses.append(0)
                        else:
                            gains.append(0)
                            losses.append(abs(change))
                    
                    avg_gain = sum(gains) / 14 if gains else 0
                    avg_loss = sum(losses) / 14 if losses else 0
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 100
                    
                    # Normalize to 0-1
                    features.append(rsi / 100)
                
                # Add MACD if enabled
                if self.advanced_features.get("use_macd", True) and len(price_data) >= 26:
                    # Calculate EMA-12
                    ema12 = self._calculate_ema([p["c"] for p in price_data[-26:]], 12)
                    
                    # Calculate EMA-26
                    ema26 = self._calculate_ema([p["c"] for p in price_data[-26:]], 26)
                    
                    # MACD Line
                    macd_line = ema12 - ema26
                    
                    # Add to features
                    features.append(macd_line / current["c"] if current["c"] > 0 else 0)
                
                # Add Bollinger Bands if enabled
                if self.advanced_features.get("use_bollinger_bands", True) and len(price_data) >= 20:
                    # Calculate 20-day SMA
                    sma20 = sum(item["c"] for item in price_data[-20:]) / 20
                    
                    # Calculate standard deviation
                    std_dev = np.std([item["c"] for item in price_data[-20:]])
                    
                    # Calculate upper and lower bands
                    upper_band = sma20 + (2 * std_dev)
                    lower_band = sma20 - (2 * std_dev)
                    
                    # Calculate %B (current price relative to bands)
                    if upper_band > lower_band:
                        percent_b = (current["c"] - lower_band) / (upper_band - lower_band)
                        features.append(percent_b)
                    else:
                        features.append(0.5)  # Default to middle
                
                # Add Stochastic Oscillator if enabled
                if self.advanced_features.get("use_stochastic", False) and len(price_data) >= 14:
                    # Get highest high and lowest low of last 14 periods
                    highest_high = max(item["h"] for item in price_data[-14:])
                    lowest_low = min(item["l"] for item in price_data[-14:])
                    
                    # Calculate %K
                    if highest_high > lowest_low:
                        percent_k = 100 * (current["c"] - lowest_low) / (highest_high - lowest_low)
                        features.append(percent_k / 100)  # Normalize to 0-1
                    else:
                        features.append(0.5)  # Default to middle
            
            # Add volatility features if enabled
            if self.include_volatility:
                # Historical volatility
                for period in [5, 10, 20]:
                    if len(price_data) >= period:
                        returns = []
                        for i in range(-period, 0):
                            if price_data[i-1]["c"] > 0:
                                daily_return = (price_data[i]["c"] - price_data[i-1]["c"]) / price_data[i-1]["c"]
                                returns.append(daily_return)
                        
                        if returns:
                            volatility = np.std(returns) if returns else 0
                            features.append(volatility)
                        else:
                            features.append(0)
            
            # Add sentiment feature if enabled
            if include_sentiment:
                # Sentiment score (-1 to 1)
                features.append(sentiment_score * self.sentiment_weight)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}", exc_info=True)
            return None
    
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """
        Calculate Exponential Moving Average for a list of prices.
        
        Args:
            prices: List of price values
            period: EMA period
            
        Returns:
            EMA value
        """
        if not prices or len(prices) < period:
            return 0
            
        # Calculate multiplier
        multiplier = 2 / (period + 1)
        
        # Calculate initial SMA
        sma = sum(prices[:period]) / period
        
        # Calculate EMA
        ema = sma
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
            
        return ema
    
    def _prepare_training_data(self, price_data: List[Dict[str, Any]], 
                              sentiment_data: Optional[Dict[str, float]] = None,
                              include_sentiment: bool = True) -> Tuple[List[List[float]], List[int]]:
        """
        Prepare training data from price and sentiment data.
        
        Args:
            price_data: Processed price data
            sentiment_data: Dict mapping dates to sentiment scores
            include_sentiment: Whether to include sentiment features
            
        Returns:
            Tuple of (features, labels) for training
        """
        X = []
        y = []
        
        if not price_data or len(price_data) < max(self.lookback_periods) + 2:
            return [], []
            
        # Need at least one day for the target label
        for i in range(max(self.lookback_periods), len(price_data) - 1):
            # Data up to current day for features
            current_slice = price_data[:i+1]
            current = current_slice[-1]
            
            # Next day for label
            next_day = price_data[i+1]
            
            # Extract features
            sentiment_score = 0
            if include_sentiment and sentiment_data:
                # Get sentiment for the current day
                date_str = self._get_date_str(current["t"])
                
                if date_str in sentiment_data:
                    sentiment_score = sentiment_data[date_str]
            
            features = self._extract_features(current_slice, sentiment_score, include_sentiment)
            
            if features:
                # Create binary target: 1 if price went up, 0 if it went down
                target = 1 if next_day["c"] > current["c"] else 0
                
                X.append(features)
                y.append(target)
        
        return X, y
    
    def _generate_backtest_chart(self, symbol: str, y_test: List[int], y_pred: List[int], 
                                returns: List[float], cumulative_returns: List[float]) -> Optional[str]:
        """
        Generate a chart for backtest results.
        
        Args:
            symbol: The stock symbol
            y_test: Actual price movements
            y_pred: Predicted price movements
            returns: Strategy returns
            cumulative_returns: Cumulative strategy returns
            
        Returns:
            Path to the generated chart file or None if generation fails
        """
        if not hasattr(self, "chart_generator"):
            return None
            
        try:
            # Create data for the chart
            chart_data = {
                "Cumulative Return": cumulative_returns,
                "Accuracy": [sum(1 for a, p in zip(y_test[:i+1], y_pred[:i+1]) if a == p) / (i+1) 
                            if i > 0 else 0 for i in range(len(y_test))]
            }
            
            # Generate the chart
            chart_path = self.chart_generator.create_backtest_chart(
                chart_data,
                title=f"{symbol} Backtest Results",
                include_timestamps=True
            )
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error generating backtest chart: {e}", exc_info=True)
            return None
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get item from cache if available and not expired."""
        if not self.enable_cache:
            return None
            
        with self.cache_lock:
            if key in self.cache and key in self.cache_timestamps:
                timestamp = self.cache_timestamps[key]
                if time.time() - timestamp <= self.cache_ttl:
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.cache_timestamps[key]
            
            return None
    
    def _add_to_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Add item to cache."""
        if not self.enable_cache:
            return
            
        with self.cache_lock:
            self.cache[key] = value
            self.cache_timestamps[key] = time.time()
            
            # Clean cache if too large
            if len(self.cache) > self.max_cache_items:
                # Remove oldest items
                cleanup_count = int(len(self.cache) * self.cache_cleanup_threshold)
                
                # Sort by timestamp (oldest first)
                sorted_keys = sorted(
                    [(k, t) for k, t in self.cache_timestamps.items()],
                    key=lambda x: x[1]
                )
                
                # Remove oldest items
                for k, _ in sorted_keys[:cleanup_count]:
                    if k in self.cache:
                        del self.cache[k]
                    if k in self.cache_timestamps:
                        del self.cache_timestamps[k]
                        
                self.logger.info(f"Cache cleanup: removed {cleanup_count} oldest items")

# If running directly as a script, initialize and test the system
if __name__ == "__main__":
    # Create the integrated system
    config_path = "/home/ubuntu/nextgen/config/financial_data_mcp/financial_data_mcp_config.json"
    integration = FinBERTXGBoostIntegration(config_path)
    
    # Test the system with a sample prediction
    test_symbol = "AAPL"
    print(f"Testing price movement prediction for {test_symbol}")
    result = integration.predict_price_movement(test_symbol)
    print(json.dumps(result, indent=2))
    
    # Test system health
    print("\nSystem health:")
    health = integration.get_system_health()
    print(json.dumps(health, indent=2))