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
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[str] = "/home/ubuntu/nextgen/config/financial_data_mcp/financial_data_mcp_config.json"):
        """
        Initialize the integrated FinBERT-XGBoost system.

        Args:
            config: Optional configuration dictionary. If provided, overrides loading from config_path.
            config_path: Path to configuration file (used if config is None).
        """
        # Initialize logger first for proper error handling
        self.logger = NetdataLogger(component_name="finbert-xgboost-integration")

        # Load configuration
        if config is not None:
            self.config = config
            self.logger.info("Configuration provided directly.")
        else:
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
        
        # Load API keys directly from environment variables
        polygon_api_key = os.environ.get("POLYGON_API_KEY", "")
        unusual_whales_api_key = os.environ.get("UNUSUAL_WHALES_API_KEY", "")
        yahoo_finance_api_key = os.environ.get("YAHOO_FINANCE_API_KEY", "")

        # Add API keys to data MCP config if they exist in environment variables
        if polygon_api_key:
            data_mcp_config.setdefault("sources", {}).setdefault("polygon_rest", {})["api_key"] = polygon_api_key
        if unusual_whales_api_key:
            data_mcp_config.setdefault("sources", {}).setdefault("unusual_whales", {})["api_key"] = unusual_whales_api_key
        if yahoo_finance_api_key:
            data_mcp_config.setdefault("sources", {}).setdefault("yahoo_finance", {})["api_key"] = yahoo_finance_api_key

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

        if not config_path:
             self.logger.warning("Config path is None or empty. Using default configuration.")
             return default_config

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
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of a given text using the FinBERT model.

        Args:
            text: The text string to analyze.

        Returns:
            Dict with sentiment analysis results (e.g., sentiment, score).
        """
        if not self.include_sentiment or self.finbert_model is None or self.finbert_tokenizer is None:
            self.logger.warning("Sentiment analysis is disabled or FinBERT model not loaded.")
            return {"sentiment": "neutral", "score": 0.5, "error": "Sentiment analysis disabled or model not loaded"}

        try:
            device = torch.device("cuda" if self.use_gpu else "cpu")
            inputs = self.finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_sequence_length).to(device)

            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)[0]

            # FinBERT output classes: positive, negative, neutral
            sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}
            predicted_class_id = scores.argmax().item()
            sentiment = sentiment_map.get(predicted_class_id, "neutral")
            score = scores[predicted_class_id].item()

            # Adjust score for negative sentiment to be on a -1 to 1 scale (approx)
            # Positive: score (0 to 1)
            # Negative: -score (0 to -1)
            # Neutral: 0
            adjusted_score = 0
            if sentiment == "positive":
                adjusted_score = score
            elif sentiment == "negative":
                adjusted_score = -score
            # Neutral remains 0

            return {"sentiment": sentiment, "score": score, "adjusted_score": adjusted_score}

        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}", exc_info=True)
            return {"sentiment": "neutral", "score": 0.5, "error": str(e)}

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get the current health status of the MCP server and its components.

        Returns:
            Dict with health status information.
        """
        health_report = {
            "overall_status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "models": {
                    "status": "ok",
                    "details": f"{len(self.xgboost_models)} models loaded",
                    "symbol_metrics": {} # Placeholder for detailed model metrics
                },
                "cache": {
                    "status": "ok" if self.enable_cache else "disabled",
                    "details": f"{len(self.cache)} items in cache" if self.enable_cache else "Cache is disabled"
                },
                "rate_limiting": {
                    "status": "ok", # Basic status, could add more detailed checks
                    "details": f"Max requests per minute: {self.max_requests_per_minute}"
                }
                # Add other components as needed
            }
            # Add system metrics from metrics_collector if available
        }

        # Check if metrics collector is running and add its data
        if self.metrics_collector and self.metrics_collector.is_running():
             health_report["system_metrics"] = self.metrics_collector.get_latest_metrics()
             # Check system metrics for potential issues (e.g., high CPU/memory usage)
             if health_report["system_metrics"].get("cpu_percent", 0) > 80 or \
                health_report["system_metrics"].get("memory_percent", 0) > 80:
                 health_report["overall_status"] = "warning"
                 health_report["components"]["system_metrics"]["status"] = "warning"
                 health_report["components"]["system_metrics"]["details"] = "High CPU or memory usage detected"


        # Check model health (placeholder - actual degradation check would be more complex)
        for symbol, model in self.xgboost_models.items():
             # In a real scenario, you would evaluate model performance on recent data
             # and calculate a degradation metric. For now, just report presence.
             health_report["components"]["models"]["symbol_metrics"][symbol] = {
                 "status": "loaded",
                 "last_updated": self.model_last_updated.get(symbol, "unknown")
             }


        # Check if FinBERT model is loaded if sentiment is enabled
        if self.include_sentiment:
             if self.finbert_model is None or self.finbert_tokenizer is None:
                 health_report["overall_status"] = "warning"
                 health_report["components"]["finbert_model"] = {
                     "status": "error",
                     "details": "FinBERT model not loaded despite sentiment being enabled"
                 }
             else:
                 health_report["components"]["finbert_model"] = {
                     "status": "ok",
                     "details": "FinBERT model loaded"
                 }


        # Aggregate overall status
        if any(comp.get("status") == "error" for comp in health_report["components"].values()):
             health_report["overall_status"] = "error"
        elif any(comp.get("status") == "warning" for comp in health_report["components"].values()):
             if health_report["overall_status"] != "error":
                 health_report["overall_status"] = "warning"


        return health_report

    def _start_health_check_thread(self):
        """Start a background thread for health monitoring."""
        def health_check_loop(instance):
            check_interval = instance.config.get("monitoring", {}).get("health_check_interval_mins", 15) * 60
            instance.logger.info(f"Starting health check thread with interval {check_interval} seconds")

            while True:
                try:
                    # Get system health
                    health_report = instance.get_system_health()

                    # Log health status
                    instance.logger.info(f"Health check: {health_report['overall_status']}")

                    # Check for model degradation if configured
                    if instance.config.get("monitoring", {}).get("alert_on_model_degradation", False):
                        degradation_threshold = instance.config.get("monitoring", {}).get("degradation_threshold", 0.1)

                        # Check models with performance metrics
                        for symbol, model in health_report.get("components", {}).get("models", {}).get("symbol_metrics", {}).items():
                            if model.get("degradation", 0) > degradation_threshold:
                                instance.logger.warning(f"Model degradation detected for {symbol}: {model['degradation']:.2f}")

                                # Trigger model update if degradation is severe
                                if model.get("degradation", 0) > degradation_threshold * 2:
                                    instance.logger.info(f"Severe degradation detected, triggering model update for {symbol}")
                                    try:
                                        # Run model update in a separate thread to avoid blocking health check
                                        update_thread = threading.Thread(
                                            target=instance.train_prediction_model,
                                            args=(symbol,)
                                        )
                                        update_thread.daemon = True
                                        update_thread.start()
                                    except Exception as e:
                                        instance.logger.error(f"Failed to trigger model update: {e}")

                except Exception as e:
                    instance.logger.error(f"Error in health check thread: {e}", exc_info=True)

                # Sleep until next check
                time.sleep(check_interval)

        # Start the health check thread, passing the instance
        health_thread = threading.Thread(target=health_check_loop, args=(self,))
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
        
    def clear_cache(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the cache for a specific symbol or all symbols.
        
        Args:
            symbol: Optional symbol to clear cache for. If None, clears all cache.
            
        Returns:
            Dict with result status and details.
        """
        with self.cache_lock:
            items_before = len(self.cache)
            
            if symbol:
                # Clear cache for specific symbol
                keys_to_remove = [
                    k for k in list(self.cache.keys())
                    if k.startswith(f"analyze_symbol_with_news:{symbol}") or
                       k.startswith(f"predict_price_movement:{symbol}")
                ]
                
                for k in keys_to_remove:
                    if k in self.cache:
                        del self.cache[k]
                    if k in self.cache_timestamps:
                        del self.cache_timestamps[k]
                        
                self.logger.info(f"Cleared cache for symbol {symbol}",
                                items_removed=len(keys_to_remove))
                
                return {
                    "status": "success",
                    "symbol": symbol,
                    "items_removed": len(keys_to_remove),
                    "items_remaining": len(self.cache)
                }
            else:
                # Clear all cache
                self.cache = {}
                self.cache_timestamps = {}
                
                self.logger.info("Cleared entire cache", items_removed=items_before)
                
                return {
                    "status": "success",
                    "items_removed": items_before,
                    "items_remaining": 0
                }
    
    def get_current_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current configuration settings.
        
        Args:
            section: Optional section of config to return (e.g., "finbert_settings", "cache_settings").
                    If None, returns the entire config.
            
        Returns:
            Dict with configuration settings.
        """
        with self.model_lock:  # Use model_lock since we're accessing config
            if section:
                if section in self.config:
                    return {section: self.config[section]}
                else:
                    return {"error": f"Configuration section '{section}' not found"}
            else:
                # Return a copy of entire config (without sensitive info)
                config_copy = self.config.copy()
                # Remove any sensitive API keys or credentials that might be in the config
                if "api_keys" in config_copy:
                    for key in config_copy["api_keys"]:
                        if config_copy["api_keys"][key]:
                            config_copy["api_keys"][key] = "[REDACTED]"
                            
                return config_copy
    
    def update_config(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update configuration settings at runtime.
        
        Args:
            updates: Dict containing configuration updates to apply
            
        Returns:
            Dict with update status and applied changes.
        """
        if not updates or not isinstance(updates, dict):
            return {"error": "Invalid update format. Expected a dictionary of config updates."}
            
        with self.model_lock:  # Use model_lock for config updates
            applied_updates = {}
            rejected_updates = {}
            
            # Process each update
            for section, values in updates.items():
                if section in self.config and isinstance(values, dict):
                    # For dict sections, update individual keys
                    if isinstance(self.config[section], dict):
                        for key, value in values.items():
                            # Validate sensitive sections
                            if section == "api_keys":
                                # Don't allow API key updates via this method for security
                                rejected_updates[f"{section}.{key}"] = "API keys cannot be updated via this method"
                                continue
                                
                            # Apply the update
                            if key in self.config[section] or section in ["advanced_features"]:
                                old_value = self.config[section].get(key, None)
                                self.config[section][key] = value
                                applied_updates[f"{section}.{key}"] = {"old": old_value, "new": value}
                            else:
                                rejected_updates[f"{section}.{key}"] = "Key not found in configuration"
                    else:
                        # For non-dict sections, replace the entire value
                        old_value = self.config[section]
                        self.config[section] = values
                        applied_updates[section] = {"old": old_value, "new": values}
                else:
                    rejected_updates[section] = "Section not found in configuration"
            
            # If we've updated anything that affects the runtime configuration, reconfigure
            if any(section in ["cache_settings", "rate_limiting", "finbert_settings",
                             "xgboost_settings", "feature_engineering", "backtesting",
                             "trading_simulation", "sentiment_analysis"]
                   for section in updates.keys()):
                self._configure_from_config()
                applied_updates["reconfigured"] = True
                
            # Log the changes
            if applied_updates:
                self.logger.info(f"Configuration updated with {len(applied_updates)} changes")
            if rejected_updates:
                self.logger.warning(f"{len(rejected_updates)} configuration updates were rejected")
                
            return {
                "status": "success" if applied_updates else "no_changes",
                "applied": applied_updates,
                "rejected": rejected_updates
            }
    
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
            # Calculate start and end dates based on lookback_days
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

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
