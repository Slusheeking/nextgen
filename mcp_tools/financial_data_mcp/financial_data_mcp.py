#!/usr/bin/env python3
"""
Financial Data MCP Tool

This module implements a consolidated Model Context Protocol (MCP) server that provides
a unified interface for retrieving financial data from various external sources like
Polygon.io, Yahoo Finance, Reddit, and Unusual Whales.

Features:
- Unified API for market data, news, social sentiment, and options flow
- Source prioritization and fallback
- Performance monitoring and metrics
- Detailed health reporting
- Visualization of financial data and performance
"""

import os
import json
import time
import threading
import importlib
from typing import Dict, List, Any, Optional
from datetime import datetime

# Direct imports instead of dynamic loading
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer

# Standard imports for MCP data clients with proper error handling
try:
    from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
    HAVE_POLYGON_REST = True
except ImportError:
    HAVE_POLYGON_REST = False
    print("Warning: PolygonRestMCP not found. Historical market data from Polygon will be unavailable.")

try:
    from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWsMCP
    HAVE_POLYGON_WS = True
except ImportError:
    HAVE_POLYGON_WS = False
    print("Warning: PolygonWsMCP not found. Streaming market data from Polygon will be unavailable.")

try:
    from mcp_tools.data_mcp.polygon_news_mcp import PolygonNewsMCP
    HAVE_POLYGON_NEWS = True
except ImportError:
    HAVE_POLYGON_NEWS = False
    print("Warning: PolygonNewsMCP not found. News data from Polygon will be unavailable.")

try:
    from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP
    HAVE_YAHOO_FINANCE = True
except ImportError:
    HAVE_YAHOO_FINANCE = False
    print("Warning: YahooFinanceMCP not found. Financial data from Yahoo Finance will be unavailable.")

try:
    from mcp_tools.data_mcp.yahoo_news_mcp import YahooNewsMCP
    HAVE_YAHOO_NEWS = True
except ImportError:
    HAVE_YAHOO_NEWS = False
    print("Warning: YahooNewsMCP not found. News data from Yahoo News will be unavailable.")

try:
    from mcp_tools.data_mcp.reddit_mcp import RedditMCP
    HAVE_REDDIT = True
except ImportError:
    HAVE_REDDIT = False
    print("Warning: RedditMCP not found. Social sentiment data from Reddit will be unavailable.")

try:
    from mcp_tools.data_mcp.unusual_whales_mcp import UnusualWhalesMCP
    HAVE_UNUSUAL_WHALES = True
except ImportError:
    HAVE_UNUSUAL_WHALES = False
    print("Warning: UnusualWhalesMCP not found. Unusual options activity data will be unavailable.")


class FinancialDataMCP(BaseMCPServer):
    """
    Consolidated MCP server for retrieving financial data from multiple sources.
    Provides a unified API for market data, news, social sentiment, and options flow.
    
    Features:
    - Comprehensive monitoring and metrics collection
    - Detailed health reporting with source-specific diagnostics
    - Performance tracking for all data operations
    - Automatic visualization of market and performance data
    - Cache management with detailed metrics
    - Health checking with proactive monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Financial Data MCP server.

        Args:
            config: Optional configuration dictionary. If None, loads from
                  config/mcp_tools/financial_data_mcp_config.json
        """
        init_start_time = time.time()
        
        if config is None:
            config_path = os.path.join("config", "mcp_tools", "financial_data_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Error loading config from {config_path}: {e}")
                    config = {}
            else:
                print(f"Warning: Config file not found at {config_path}. Using default settings.")
                config = {}

        super().__init__(name="financial_data_mcp", config=config)

        # Initialize monitoring locks
        self.monitoring_lock = threading.RLock()

        # Initialize detailed performance metrics for financial data operations
        self.market_data_count = 0
        self.news_request_count = 0
        self.social_sentiment_count = 0
        self.options_data_count = 0
        self.fundamentals_count = 0
        self.market_status_count = 0
        self.ticker_list_count = 0
        self.market_movers_count = 0
        
        # Initialize source usage tracking
        self.source_usage_counts = {}
        self.source_error_counts = {}
        self.source_latencies = {}
        
        # Initialize API rate limit tracking
        self.api_rate_limits = {}
        self.api_rate_limit_resets = {}
        
        # Initialize cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_expirations = 0
        
        # Initialize metrics by symbol to track most queried financial instruments
        self.symbol_query_counts = {}
        
        # Configure performance thresholds
        self.slow_market_data_threshold_ms = self.config.get("slow_market_data_threshold_ms", 1000)
        self.slow_news_threshold_ms = self.config.get("slow_news_threshold_ms", 1500)
        self.slow_fundamental_data_threshold_ms = self.config.get("slow_fundamental_data_threshold_ms", 2000)
        
        # Configure cache settings
        self.enable_cache = self.config.get("enable_cache", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)  # Default 5 minutes cache
        
        # Configure from config and initialize clients first
        self._configure_from_config()
        self._initialize_data_clients()
        
        # Start health check thread specifically for financial data sources
        # (after clients are initialized)
        if self.config.get("enable_financial_health_check", True):
            self._start_financial_health_check_thread()

        # Register tools
        self._register_tools()
        
        # Register monitoring tools
        if self.config.get("enable_monitoring", True):
            self._register_financial_monitoring_tools()
        
        # Record initialization time
        init_duration = (time.time() - init_start_time) * 1000  # milliseconds
        self.logger.timing("financial_data_mcp.initialization_time_ms", init_duration)
        
        # Log detailed initialization information
        self.logger.info(f"Financial Data MCP initialized successfully - init_time_ms: {init_duration}, sources_count: {len(self.clients)}, cache_enabled: {self.enable_cache}, cache_ttl: {self.cache_ttl}")

    def _start_financial_health_check_thread(self):
        """
        Start a background thread for periodic health checks specific to financial data sources.
        This thread will monitor the health of each individual data source and API rate limits.
        """
        def financial_health_check_loop():
            self.logger.info("Starting financial data health check thread")
            
            # Get health check interval from config (with default of 60 seconds)
            health_check_interval = self.config.get("financial_health_check_interval", 60)
            
            # Initialize health check counter for periodic detailed reports
            health_check_counter = 0
            
            while not hasattr(self, 'shutdown_requested') or not self.shutdown_requested:
                try:
                    # Increment the counter
                    health_check_counter += 1
                    
                    # Check the health of each data source
                    source_health = {}
                    
                    with self.monitoring_lock:
                        # For each data source, check its health
                        for source_name, client in self.clients.items():
                            try:
                                # Check if the client has a health_check method
                                if hasattr(client, "health_check"):
                                    health_result = client.health_check()
                                    source_health[source_name] = health_result
                                else:
                                    # Default health check based on error rates
                                    usage_count = self.source_usage_counts.get(source_name, 0)
                                    error_count = self.source_error_counts.get(source_name, 0)
                                    error_rate = error_count / max(1, usage_count) if usage_count > 0 else 0
                                    
                                    # Set health status based on error rate
                                    if error_rate > 0.2:  # 20% error rate is high
                                        status = "unhealthy"
                                    elif error_rate > 0.05:  # 5% error rate is concerning
                                        status = "degraded"
                                    else:
                                        status = "healthy"
                                    
                                    # Record the health information
                                    source_health[source_name] = {
                                        "status": status,
                                        "error_rate": f"{error_rate:.2%}",
                                        "request_count": usage_count,
                                        "error_count": error_count
                                    }
                                    
                                    # Check API rate limits if applicable
                                    if source_name in self.api_rate_limits:
                                        # Add rate limit information
                                        source_health[source_name]["rate_limit"] = {
                                            "remaining": self.api_rate_limits.get(source_name, 0),
                                            "reset_time": self.api_rate_limit_resets.get(source_name, 0)
                                        }
                                        
                                        # Add warning if we're close to rate limit
                                        if self.api_rate_limits.get(source_name, 0) < 50:
                                            source_health[source_name]["warnings"] = [
                                                f"API rate limit running low: {self.api_rate_limits.get(source_name, 0)} remaining"
                                            ]
                            except Exception as e:
                                self.logger.error(f"Error checking health for {source_name}: {e}")
                                source_health[source_name] = {
                                    "status": "unknown",
                                    "error": str(e)
                                }
                    
                    # Determine overall financial data health
                    unhealthy_sources = [name for name, health in source_health.items() 
                                        if health.get("status") == "unhealthy"]
                    degraded_sources = [name for name, health in source_health.items() 
                                       if health.get("status") == "degraded"]
                    
                    if len(unhealthy_sources) > len(self.clients) / 2:
                        # More than half of sources are unhealthy
                        overall_status = "critical"
                        self.logger.critical("Financial data sources in critical state", 
                                           unhealthy_count=len(unhealthy_sources),
                                           total_sources=len(self.clients))
                    elif unhealthy_sources:
                        # Some sources are unhealthy
                        overall_status = "degraded"
                        self.logger.warning("Some financial data sources are unhealthy", 
                                          unhealthy_sources=unhealthy_sources)
                    elif degraded_sources:
                        # Some sources are degraded
                        overall_status = "warning"
                        self.logger.warning("Some financial data sources are degraded", 
                                          degraded_sources=degraded_sources)
                    else:
                        # All sources are healthy
                        overall_status = "healthy"
                    
                    # Update metrics based on health check
                    if overall_status == "healthy":
                        self.logger.gauge("financial_data_health_status", 3)  # 3 = healthy
                    elif overall_status == "warning":
                        self.logger.gauge("financial_data_health_status", 2)  # 2 = warning
                    elif overall_status == "degraded":
                        self.logger.gauge("financial_data_health_status", 1)  # 1 = degraded
                    else:
                        self.logger.gauge("financial_data_health_status", 0)  # 0 = critical
                        
                    # Every 5th check, log detailed health information
                    if health_check_counter % 5 == 0:
                        self.logger.info(f"Detailed financial data health check completed - overall_status: {overall_status}, healthy_sources: {len(self.clients) - len(unhealthy_sources) - len(degraded_sources)}, degraded_sources: {len(degraded_sources)}, unhealthy_sources: {len(unhealthy_sources)}, total_sources: {len(self.clients)}")
                    else:
                        # More concise log for regular checks
                        self.logger.info(f"Financial data health check completed - overall_status: {overall_status}")
                    
                    # Check if we need to generate charts (every 10th check)
                    if health_check_counter % 10 == 0:
                        self._generate_financial_performance_charts()
                    
                except Exception as e:
                    self.logger.error(f"Error in financial health check thread: {e}", exc_info=True)
                    self.logger.counter("financial_health_check_errors")
                
                # Sleep until next check interval
                time.sleep(health_check_interval)
        
        # Start health check thread
        health_thread = threading.Thread(target=financial_health_check_loop, daemon=True)
        health_thread.start()
        self.financial_health_thread = health_thread
        self.logger.info("Financial health check thread started")
    
    def _generate_financial_performance_charts(self):
        """
        Generate performance charts for financial data visualization.
        """
        try:
            # Generate charts only if chart generator is available
            if not hasattr(self, 'chart_generator'):
                return
            
            with self.monitoring_lock:
                # Generate API usage chart
                api_usage_data = {
                    'Market Data': self.market_data_count,
                    'News': self.news_request_count,
                    'Social Sentiment': self.social_sentiment_count,
                    'Options Data': self.options_data_count,
                    'Fundamentals': self.fundamentals_count,
                    'Market Status': self.market_status_count,
                    'Ticker List': self.ticker_list_count,
                    'Market Movers': self.market_movers_count
                }
                
                try:
                    api_chart = self.chart_generator.create_performance_chart(
                        api_usage_data,
                        title="Financial Data API Usage",
                        include_timestamps=True
                    )
                    self.logger.info("Generated API usage chart", chart_file=api_chart)
                except Exception as e:
                    self.logger.warning(f"Failed to create API usage chart: {e}")
                
                # Generate source usage chart
                if self.source_usage_counts:
                    try:
                        source_chart = self.chart_generator.create_performance_chart(
                            self.source_usage_counts,
                            title="Data Source Usage",
                            include_timestamps=True
                        )
                        self.logger.info("Generated source usage chart", chart_file=source_chart)
                    except Exception as e:
                        self.logger.warning(f"Failed to create source usage chart: {e}")
                
                # Generate cache performance chart
                cache_data = {
                    'Hits': self.cache_hits,
                    'Misses': self.cache_misses,
                    'Expirations': self.cache_expirations
                }
                
                try:
                    cache_chart = self.chart_generator.create_performance_chart(
                        cache_data,
                        title="Cache Performance",
                        include_timestamps=True
                    )
                    self.logger.info("Generated cache performance chart", chart_file=cache_chart)
                except Exception as e:
                    self.logger.warning(f"Failed to create cache performance chart: {e}")
                
                # Generate top symbols chart - get top 10 most queried symbols
                if self.symbol_query_counts:
                    try:
                        # Sort by query count and take top 10
                        top_symbols = dict(sorted(
                            self.symbol_query_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:10])
                        
                        if top_symbols:
                            symbol_chart = self.chart_generator.create_performance_chart(
                                top_symbols,
                                title="Top Queried Symbols",
                                include_timestamps=True
                            )
                            self.logger.info("Generated top symbols chart", chart_file=symbol_chart)
                    except Exception as e:
                        self.logger.warning(f"Failed to create top symbols chart: {e}")
        
        except Exception as e:
            self.logger.error(f"Error generating financial performance charts: {e}", exc_info=True)

    def _configure_from_config(self):
        """Extract configuration values."""
        # Use the 'sources' key from the config file
        self.sources_config = self.config.get("sources", {})
        self.default_source_priority = self.config.get("default_source_priority",
                                                      ["polygon_rest", "yahoo_finance", "reddit", "unusual_whales", "polygon_news", "yahoo_news", "polygon_ws"]) # Updated priority list
        self.cache_ttl = self.config.get("cache_ttl", 300) # Default 5 minutes cache

        # Log loaded sources
        enabled_sources = [src for src, cfg in self.sources_config.items() if cfg.get("enabled", False)]
        self.logger.info("Data sources configured", enabled_sources=enabled_sources)


    def _initialize_data_clients(self):
        """Initialize clients for each configured data source."""
        init_start_time = time.time()
        self.clients = {}
        successful_inits = 0
        failed_inits = 0

        # Initialize clients based on availability and config with detailed metrics
        if HAVE_POLYGON_REST and self.sources_config.get("polygon_rest", {}).get("enabled", True):
            try:
                client_init_start = time.time()
                config = self.sources_config.get("polygon_rest", {})
                # Add monitoring context to config
                if "logger" not in config:
                    config["logger"] = self.logger
                if "metrics_collector" not in config:
                    config["metrics_collector"] = self.metrics_collector
                
                # Load API key from environment variable
                config["api_key"] = os.getenv("POLYGON_API_KEY")

                self.clients["polygon_rest"] = PolygonRestMCP(config)
                client_init_time = (time.time() - client_init_start) * 1000
                self.logger.timing("client_init.polygon_rest_ms", client_init_time, source="polygon_rest")
                self.logger.info("Polygon REST client initialized", init_time_ms=client_init_time)
                successful_inits += 1
                
                # Set initial source usage counts for monitoring
                self.source_usage_counts["polygon_rest"] = 0
                self.source_error_counts["polygon_rest"] = 0
                self.source_latencies["polygon_rest"] = 0
            except Exception as e:
                self.logger.error(f"Failed to initialize Polygon REST client: {e}", exc_info=True)
                failed_inits += 1

        if HAVE_POLYGON_WS and self.sources_config.get("polygon_ws", {}).get("enabled", False): # WS usually needs explicit enable
            try:
                client_init_start = time.time()
                config = self.sources_config.get("polygon_ws", {})
                # Add monitoring context to config
                if "logger" not in config:
                    config["logger"] = self.logger
                if "metrics_collector" not in config:
                    config["metrics_collector"] = self.metrics_collector
                
                # Load API key from environment variable
                config["api_key"] = os.getenv("POLYGON_API_KEY")
                    
                self.clients["polygon_ws"] = PolygonWsMCP(config)
                client_init_time = (time.time() - client_init_start) * 1000
                self.logger.timing("client_init.polygon_ws_ms", client_init_time, source="polygon_ws")
                self.logger.info("Polygon WebSocket client initialized", init_time_ms=client_init_time)
                successful_inits += 1
                
                # Set initial source usage counts for monitoring
                self.source_usage_counts["polygon_ws"] = 0
                self.source_error_counts["polygon_ws"] = 0
                self.source_latencies["polygon_ws"] = 0
            except Exception as e:
                self.logger.error(f"Failed to initialize Polygon WebSocket client: {e}", exc_info=True)
                failed_inits += 1

        if HAVE_POLYGON_NEWS and self.sources_config.get("polygon_news", {}).get("enabled", True):
            try:
                client_init_start = time.time()
                config = self.sources_config.get("polygon_news", {})
                # Add monitoring context to config
                if "logger" not in config:
                    config["logger"] = self.logger
                if "metrics_collector" not in config:
                    config["metrics_collector"] = self.metrics_collector
                
                # Load API key from environment variable
                config["api_key"] = os.getenv("POLYGON_API_KEY")
                    
                self.clients["polygon_news"] = PolygonNewsMCP(config)
                client_init_time = (time.time() - client_init_start) * 1000
                self.logger.timing("client_init.polygon_news_ms", client_init_time, source="polygon_news")
                self.logger.info("Polygon News client initialized", init_time_ms=client_init_time)
                successful_inits += 1
                
                # Set initial source usage counts for monitoring
                self.source_usage_counts["polygon_news"] = 0
                self.source_error_counts["polygon_news"] = 0
                self.source_latencies["polygon_news"] = 0
            except Exception as e:
                self.logger.error(f"Failed to initialize Polygon News client: {e}", exc_info=True)
                failed_inits += 1

        if HAVE_YAHOO_FINANCE and self.sources_config.get("yahoo_finance", {}).get("enabled", True):
            try:
                client_init_start = time.time()
                config = self.sources_config.get("yahoo_finance", {})
                # Add monitoring context to config
                if "logger" not in config:
                    config["logger"] = self.logger
                if "metrics_collector" not in config:
                    config["metrics_collector"] = self.metrics_collector
                
                # Load API key from environment variable if needed
                config["api_key"] = os.getenv("YAHOO_FINANCE_API_KEY")
                    
                self.clients["yahoo_finance"] = YahooFinanceMCP(config)
                client_init_time = (time.time() - client_init_start) * 1000
                self.logger.timing("client_init.yahoo_finance_ms", client_init_time, source="yahoo_finance")
                self.logger.info("Yahoo Finance client initialized", init_time_ms=client_init_time)
                successful_inits += 1
                
                # Set initial source usage counts for monitoring
                self.source_usage_counts["yahoo_finance"] = 0
                self.source_error_counts["yahoo_finance"] = 0
                self.source_latencies["yahoo_finance"] = 0
            except Exception as e:
                self.logger.error(f"Failed to initialize Yahoo Finance client: {e}", exc_info=True)
                failed_inits += 1

        if HAVE_REDDIT and self.sources_config.get("reddit", {}).get("enabled", True):
            try:
                client_init_start = time.time()
                config = self.sources_config.get("reddit", {})
                # Add monitoring context to config
                if "logger" not in config:
                    config["logger"] = self.logger
                if "metrics_collector" not in config:
                    config["metrics_collector"] = self.metrics_collector
                
                # Load API keys from environment variables
                config["client_id"] = os.getenv("REDDIT_CLIENT_ID")
                config["client_secret"] = os.getenv("REDDIT_CLIENT_SECRET")
                config["username"] = os.getenv("REDDIT_USERNAME")
                config["password"] = os.getenv("REDDIT_PASSWORD")
                config["user_agent"] = os.getenv("REDDIT_USER_AGENT")
                    
                self.clients["reddit"] = RedditMCP(config)
                client_init_time = (time.time() - client_init_start) * 1000
                self.logger.timing("client_init.reddit_ms", client_init_time, source="reddit")
                self.logger.info("Reddit client initialized", init_time_ms=client_init_time)
                successful_inits += 1
                
                # Set initial source usage counts for monitoring
                self.source_usage_counts["reddit"] = 0
                self.source_error_counts["reddit"] = 0
                self.source_latencies["reddit"] = 0
            except Exception as e:
                self.logger.error(f"Failed to initialize Reddit client: {e}", exc_info=True)
                failed_inits += 1

        if HAVE_UNUSUAL_WHALES and self.sources_config.get("unusual_whales", {}).get("enabled", True):
            try:
                client_init_start = time.time()
                config = self.sources_config.get("unusual_whales", {})
                # Add monitoring context to config
                if "logger" not in config:
                    config["logger"] = self.logger
                if "metrics_collector" not in config:
                    config["metrics_collector"] = self.metrics_collector
                
                # Load API key from environment variable
                config["api_key"] = os.getenv("UNUSUAL_WHALES_API_KEY")
                    
                self.clients["unusual_whales"] = UnusualWhalesMCP(config)
                client_init_time = (time.time() - client_init_start) * 1000
                self.logger.timing("client_init.unusual_whales_ms", client_init_time, source="unusual_whales")
                self.logger.info("Unusual Whales client initialized", init_time_ms=client_init_time)
                successful_inits += 1
                
                # Set initial source usage counts for monitoring
                self.source_usage_counts["unusual_whales"] = 0
                self.source_error_counts["unusual_whales"] = 0
                self.source_latencies["unusual_whales"] = 0
            except Exception as e:
                self.logger.error(f"Failed to initialize Unusual Whales client: {e}", exc_info=True)
                failed_inits += 1

        # Record metrics about client initialization
        total_init_time = (time.time() - init_start_time) * 1000
        self.logger.timing("client_init.total_time_ms", total_init_time, source="all_clients")
        self.logger.gauge("client_init.success_count", successful_inits)
        self.logger.gauge("client_init.failed_count", failed_inits)
        
        if not self.clients:
            self.logger.warning("No data source clients were initialized. FinancialDataMCP may not function.")
            self.logger.gauge("client_init.success_rate", 0)
        else:
            success_rate = successful_inits / (successful_inits + failed_inits) * 100
            self.logger.gauge("client_init.success_rate", success_rate)
            self.logger.info(f"Data source client initialization complete - success_count: {successful_inits}, failed_count: {failed_inits}, success_rate: {success_rate:.1f}%, total_time_ms: {total_init_time}")


    def _register_tools(self):
        """Register unified data retrieval tools."""
        # Record the start time for timing metrics
        start_time = time.time()
        
        # Register core financial data tools
        self.register_tool(
            self.get_market_data,
            "get_market_data",
            "Get historical or real-time market data (OHLCV, trades, quotes) for symbols."
        )
        self.register_tool(
            self.get_news,
            "get_news",
            "Get financial news for specific symbols or general market."
        )
        self.register_tool(
            self.get_social_sentiment,
            "get_social_sentiment",
            "Get social media sentiment or mentions for symbols (e.g., from Reddit)."
        )
        self.register_tool(
            self.get_options_data,
            "get_options_data",
            "Get options chain data or unusual options activity."
        )
        self.register_tool(
            self.get_fundamentals,
            "get_fundamentals",
            "Get fundamental company data (e.g., earnings, financials, metrics)."
        )
        self.register_tool(
            self.get_market_status,
            "get_market_status",
            "Get current market status (open/closed) and related info (e.g., VIX, SPY)."
        )
        self.register_tool(
            self.get_ticker_list,
            "get_ticker_list",
            "Get a list of tickers with optional filtering."
        )
        self.register_tool(
            "get_social_sentiment",
            "Get social media sentiment or mentions for symbols (e.g., from Reddit)."
        )
        self.register_tool(
            self.get_options_data,
            "get_options_data",
            "Get options chain data or unusual options activity."
        )
        self.register_tool(
            self.get_fundamentals,
            "get_fundamentals",
            "Get fundamental company data (e.g., earnings, financials, metrics)."
        )
        self.register_tool(
            self.get_market_status,
            "get_market_status",
            "Get current market status (open/closed) and related info (e.g., VIX, SPY)."
        )
        self.register_tool(
            self.get_ticker_list,
            "get_ticker_list",
            "Get a list of tickers with optional filtering."
        )
        self.register_tool(
            self.get_market_movers,
            "get_market_movers",
            "Get top market movers (gainers or losers)."
        )
        
        # Record registration time
        registration_time = (time.time() - start_time) * 1000
        self.logger.timing("tool_registration_time_ms", registration_time, source="financial_data_mcp")
        self.logger.info("Registered financial data tools", 
                       tool_count=8, 
                       registration_time_ms=registration_time)
    
    def _register_financial_monitoring_tools(self):
        """
        Register monitoring tools specific to financial data operations.
        These tools provide detailed performance and health metrics for financial data sources.
        """
        # Record the start time for timing metrics
        start_time = time.time()
        
        # Register financial health report tool
        self.register_tool(
            self.generate_financial_health_report,
            "get_financial_health_report",
            "Get detailed health information for all financial data sources."
        )
        
        # Register financial performance report tool
        self.register_tool(
            self.generate_financial_performance_report,
            "get_financial_performance_report", 
            "Generate a comprehensive performance report for financial data operations with metrics and visualizations."
        )
        
        # Register source usage stats tool
        self.register_tool(
            self.get_source_usage_stats,
            "get_source_usage_stats",
            "Get detailed usage statistics for all financial data sources."
        )
        
        # Register symbol stats tool
        self.register_tool(
            self.get_symbol_stats,
            "get_symbol_stats",
            "Get statistics about most frequently queried financial symbols."
        )
        
        # Register cache status tool
        self.register_tool(
            self.get_cache_stats,
            "get_cache_stats",
            "Get detailed statistics about cache performance for financial data."
        )
        
        # Record registration time
        registration_time = (time.time() - start_time) * 1000
        self.logger.timing("monitoring_tool_registration_time_ms", registration_time, source="financial_data_mcp")
        self.logger.info(f"Registered financial monitoring tools - tool_count: 5, registration_time_ms: {registration_time}")


    def _get_client(self, capability: str) -> Optional[Any]:
        """Find the best available client for a given capability based on priority."""
        start_time = time.time()
        
        # Define which clients provide which capabilities (this needs refinement based on actual client methods)
        capability_map = {
            "market_data_hist": ["polygon_rest", "yahoo_finance"],
            "market_data_stream": ["polygon_ws"],
            "news": ["polygon_news", "yahoo_news"],
            "social": ["reddit"],
            "options_chain": ["polygon_rest", "yahoo_finance"],
            "options_flow": ["unusual_whales"],
            "fundamentals": ["polygon_rest", "yahoo_finance"],
            "market_info": ["polygon_rest"], # For market status, holidays etc.
            "reference": ["polygon_rest"] # For ticker lists, exchanges etc.
        }

        possible_sources = capability_map.get(capability, [])
        for source_name in self.default_source_priority:
            if source_name in possible_sources and source_name in self.clients:
                # Track client selection time for performance measurement
                selection_time = (time.time() - start_time) * 1000
                self.logger.timing("client_selection_time_ms", selection_time, source="client_selector")
                
                # Log client selection with timing
                self.logger.debug(f"Selected client '{source_name}' for capability '{capability}'", 
                               selection_time_ms=selection_time)
                
                # Update source selection metrics
                with self.monitoring_lock:
                    if source_name not in self.source_usage_counts:
                        self.source_usage_counts[source_name] = 0
                
                return self.clients[source_name]

        # No suitable client found - track as a failure
        selection_time = (time.time() - start_time) * 1000
        self.logger.timing("client_selection_failed_time_ms", selection_time, source="client_selector")
        self.logger.warning(f"No suitable client found for capability: {capability}",
                         capability=capability,
                         possible_sources=possible_sources,
                         selection_time_ms=selection_time)
        self.logger.counter("client_selection_failures")
        
        return None
        
    def generate_financial_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report for all financial data sources.
        
        Returns:
            Dictionary containing health metrics for all financial data sources
        """
        # Record start time for performance measurement
        start_time = time.time()
        
        # Check the health of each data source
        source_health = {}
        
        with self.monitoring_lock:
            # For each data source, check its health
            for source_name, client in self.clients.items():
                try:
                    # Check if the client has a health_check method
                    if hasattr(client, "health_check"):
                        health_result = client.health_check()
                        source_health[source_name] = health_result
                    else:
                        # Default health check based on error rates
                        usage_count = self.source_usage_counts.get(source_name, 0)
                        error_count = self.source_error_counts.get(source_name, 0)
                        error_rate = error_count / max(1, usage_count) if usage_count > 0 else 0
                        
                        # Calculate average response time if available
                        avg_response_time = 0
                        if source_name in self.source_latencies and usage_count > 0:
                            avg_response_time = self.source_latencies[source_name] / usage_count
                        
                        # Set health status based on error rate
                        if error_rate > 0.2:  # 20% error rate is high
                            status = "unhealthy"
                        elif error_rate > 0.05:  # 5% error rate is concerning
                            status = "degraded"
                        else:
                            status = "healthy"
                        
                        # Record the health information
                        source_health[source_name] = {
                            "status": status,
                            "error_rate": f"{error_rate:.2%}",
                            "request_count": usage_count,
                            "error_count": error_count,
                            "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time > 0 else "unknown"
                        }
                        
                        # Check API rate limits if applicable
                        if source_name in self.api_rate_limits:
                            # Add rate limit information
                            source_health[source_name]["rate_limit"] = {
                                "remaining": self.api_rate_limits.get(source_name, 0),
                                "reset_time": self.api_rate_limit_resets.get(source_name, 0)
                            }
                except Exception as e:
                    self.logger.error(f"Error checking health for {source_name}: {e}")
                    source_health[source_name] = {
                        "status": "unknown",
                        "error": str(e)
                    }
        
        # Determine overall financial data health
        unhealthy_sources = [name for name, health in source_health.items() 
                            if health.get("status") == "unhealthy"]
        degraded_sources = [name for name, health in source_health.items() 
                           if health.get("status") == "degraded"]
        
        if len(unhealthy_sources) > len(self.clients) / 2:
            # More than half of sources are unhealthy
            overall_status = "critical"
        elif unhealthy_sources:
            # Some sources are unhealthy
            overall_status = "degraded"
        elif degraded_sources:
            # Some sources are degraded
            overall_status = "warning"
        else:
            # All sources are healthy
            overall_status = "healthy"
            
        # Calculate health score (100 = perfect, 0 = complete failure)
        if not self.clients:
            health_score = 0  # No clients = no health
        else:
            # Healthy sources contribute 100%, degraded 50%, unhealthy 0%
            healthy_sources = len(self.clients) - len(unhealthy_sources) - len(degraded_sources)
            health_score = (healthy_sources * 100 + len(degraded_sources) * 50) / len(self.clients)
        
        # Compile the health report
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "health_score": round(health_score, 1),
            "healthy_sources": len(self.clients) - len(unhealthy_sources) - len(degraded_sources),
            "degraded_sources": len(degraded_sources),
            "unhealthy_sources": len(unhealthy_sources),
            "total_sources": len(self.clients),
            "source_health": source_health
        }
        
        # Add overall request metrics
        report["requests"] = {
            "market_data": self.market_data_count,
            "news": self.news_request_count,
            "social_sentiment": self.social_sentiment_count,
            "options_data": self.options_data_count,
            "fundamentals": self.fundamentals_count,
            "market_status": self.market_status_count,
            "ticker_list": self.ticker_list_count,
            "market_movers": self.market_movers_count
        }
        
        # Add total counts
        report["totals"] = {
            "total_requests": sum(report["requests"].values()),
            "total_errors": sum(self.source_error_counts.values() if self.source_error_counts else [0])
        }
        
        # Calculate report generation time
        report_time = (time.time() - start_time) * 1000
        report["generation_time_ms"] = round(report_time, 2)
        
        # Log report generation
        self.logger.info("Generated financial health report", 
                       overall_status=overall_status,
                       health_score=f"{health_score:.1f}",
                       generation_time_ms=report_time)
        
        return report
        
    def generate_financial_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report for financial data operations.
        Includes detailed metrics and charts for each data source and operation type.
        
        Returns:
            Dictionary with performance metrics and paths to generated charts
        """
        # Record start time for performance measurement
        start_time = time.time()
        
        # Get latest health report for status information
        health_report = self.generate_financial_health_report()
        
        # Generate performance charts
        charts = []
        
        with self.monitoring_lock:
            try:
                # Create financial API usage chart
                api_usage_data = {
                    'Market Data': self.market_data_count,
                    'News': self.news_request_count,
                    'Social Sentiment': self.social_sentiment_count,
                    'Options Data': self.options_data_count,
                    'Fundamentals': self.fundamentals_count,
                    'Market Status': self.market_status_count,
                    'Ticker List': self.ticker_list_count,
                    'Market Movers': self.market_movers_count
                }
                
                # Generate chart if we have a chart generator
                if hasattr(self, 'chart_generator'):
                    try:
                        api_chart = self.chart_generator.create_performance_chart(
                            api_usage_data,
                            title="Financial Data API Usage",
                            include_timestamps=True
                        )
                        charts.append({
                            "name": "api_usage",
                            "path": api_chart,
                            "description": "Financial data API usage by method"
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to create API usage chart: {e}")
                
                # Generate source usage chart
                if self.source_usage_counts and hasattr(self, 'chart_generator'):
                    try:
                        source_chart = self.chart_generator.create_performance_chart(
                            self.source_usage_counts,
                            title="Data Source Usage",
                            include_timestamps=True
                        )
                        charts.append({
                            "name": "source_usage",
                            "path": source_chart,
                            "description": "Usage frequency by data source"
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to create source usage chart: {e}")
                
                # Generate error rate chart
                if self.source_error_counts and hasattr(self, 'chart_generator'):
                    try:
                        # Calculate error rates for each source
                        error_rates = {}
                        for source_name in self.source_usage_counts:
                            usage = self.source_usage_counts.get(source_name, 0)
                            errors = self.source_error_counts.get(source_name, 0)
                            if usage > 0:
                                error_rates[source_name] = (errors / usage) * 100  # As percentage
                        
                        if error_rates:
                            error_chart = self.chart_generator.create_performance_chart(
                                error_rates,
                                title="Data Source Error Rates (%)",
                                include_timestamps=True
                            )
                            charts.append({
                                "name": "error_rates",
                                "path": error_chart,
                                "description": "Error rates by data source"
                            })
                    except Exception as e:
                        self.logger.warning(f"Failed to create error rate chart: {e}")
                
                # Generate cache performance chart
                if hasattr(self, 'chart_generator'):
                    try:
                        cache_data = {
                            'Hits': self.cache_hits,
                            'Misses': self.cache_misses,
                            'Expirations': self.cache_expirations
                        }
                        
                        cache_chart = self.chart_generator.create_performance_chart(
                            cache_data,
                            title="Cache Performance",
                            include_timestamps=True
                        )
                        charts.append({
                            "name": "cache_performance",
                            "path": cache_chart,
                            "description": "Cache hit/miss statistics"
                        })
                        
                        # Calculate hit rate percentage
                        total_cache_attempts = self.cache_hits + self.cache_misses
                        if total_cache_attempts > 0:
                            hit_rate = (self.cache_hits / total_cache_attempts) * 100
                            hit_rate_chart = self.chart_generator.create_performance_chart(
                                {"Hit Rate": hit_rate},
                                title="Cache Hit Rate (%)",
                                include_timestamps=True
                            )
                            charts.append({
                                "name": "cache_hit_rate",
                                "path": hit_rate_chart,
                                "description": "Cache hit rate percentage"
                            })
                    except Exception as e:
                        self.logger.warning(f"Failed to create cache performance chart: {e}")
            
            except Exception as e:
                self.logger.error(f"Error generating performance charts: {e}", exc_info=True)
        
        # Create the performance report
        total_requests = sum(api_usage_data.values())
        total_errors = sum(self.source_error_counts.values()) if self.source_error_counts else 0
        error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate response time metrics across all sources
        avg_response_times = {}
        for source_name in self.source_usage_counts:
            usage = self.source_usage_counts.get(source_name, 0)
            total_time = self.source_latencies.get(source_name, 0)
            if usage > 0:
                avg_response_times[source_name] = round(total_time / usage, 2)
        
        # Build the report
        report = {
            "timestamp": datetime.now().isoformat(),
            "operations": {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": f"{error_rate:.2f}%",
                "requests_by_type": api_usage_data
            },
            "sources": {
                "total_count": len(self.clients),
                "usage_counts": self.source_usage_counts,
                "error_counts": self.source_error_counts,
                "avg_response_times_ms": avg_response_times
            },
            "cache": {
                "enabled": self.enable_cache,
                "ttl_seconds": self.cache_ttl,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "expirations": self.cache_expirations,
                "hit_rate": f"{(self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100):.2f}%"
            },
            "charts": charts,
            "health_status": {
                "overall_status": health_report["overall_status"],
                "health_score": health_report["health_score"]
            }
        }
        
        # Add top queried symbols if available
        if self.symbol_query_counts:
            # Get top 10 most queried symbols
            top_symbols = dict(sorted(
                self.symbol_query_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])
            
            report["symbols"] = {
                "top_queried": top_symbols,
                "total_unique": len(self.symbol_query_counts)
            }
        
        # Calculate report generation time
        generation_time = (time.time() - start_time) * 1000
        report["generation_time_ms"] = round(generation_time, 2)
        
        # Log report generation
        self.logger.info("Generated financial performance report", 
                       total_requests=total_requests,
                       error_rate=f"{error_rate:.2f}%",
                       charts_generated=len(charts),
                       generation_time_ms=generation_time)
        
        return report
        
    def get_source_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed usage statistics for all financial data sources.
        
        Returns:
            Dictionary with usage statistics by source
        """
        start_time = time.time()
        
        with self.monitoring_lock:
            # Calculate total usage across all sources
            total_usage = sum(self.source_usage_counts.values())
            
            # Calculate percentage usage for each source
            source_percentages = {}
            for source_name, count in self.source_usage_counts.items():
                if total_usage > 0:
                    percentage = (count / total_usage) * 100
                    source_percentages[source_name] = f"{percentage:.2f}%"
            
            # Calculate error rates for each source
            error_rates = {}
            for source_name in self.source_usage_counts:
                usage = self.source_usage_counts.get(source_name, 0)
                errors = self.source_error_counts.get(source_name, 0)
                if usage > 0:
                    error_rate = (errors / usage) * 100
                    error_rates[source_name] = f"{error_rate:.2f}%"
            
            # Calculate average response times
            avg_response_times = {}
            for source_name in self.source_usage_counts:
                usage = self.source_usage_counts.get(source_name, 0)
                total_time = self.source_latencies.get(source_name, 0)
                if usage > 0:
                    avg_response_times[source_name] = round(total_time / usage, 2)
            
            # Get rate limit information where available
            rate_limits = {}
            for source_name in self.api_rate_limits:
                remaining = self.api_rate_limits.get(source_name, 0)
                reset_time = self.api_rate_limit_resets.get(source_name, 0)
                
                # Calculate time until reset if we have a reset timestamp
                time_until_reset = None
                if reset_time > 0:
                    current_time = time.time()
                    if reset_time > current_time:
                        time_until_reset = f"{int(reset_time - current_time)} seconds"
                
                rate_limits[source_name] = {
                    "remaining": remaining,
                    "reset_time": datetime.fromtimestamp(reset_time).isoformat() if reset_time > 0 else "unknown",
                    "time_until_reset": time_until_reset
                }
            
            # Compile the report
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_usage": total_usage,
                "source_counts": self.source_usage_counts,
                "source_percentages": source_percentages,
                "error_counts": self.source_error_counts,
                "error_rates": error_rates,
                "avg_response_times_ms": avg_response_times
            }
            
            # Add rate limit information if available
            if rate_limits:
                report["rate_limits"] = rate_limits
            
            # Calculate report generation time
            generation_time = (time.time() - start_time) * 1000
            report["generation_time_ms"] = round(generation_time, 2)
            
            # Log report generation
            self.logger.info("Generated source usage statistics", 
                           total_usage=total_usage,
                           source_count=len(self.source_usage_counts),
                           generation_time_ms=generation_time)
            
            return report
    
    def get_symbol_stats(self) -> Dict[str, Any]:
        """
        Get statistics about most frequently queried financial symbols.
        
        Returns:
            Dictionary with statistics by symbol
        """
        start_time = time.time()
        
        with self.monitoring_lock:
            # Check if we have any symbol data
            if not self.symbol_query_counts:
                return {
                    "status": "no_data",
                    "message": "No symbol query data available yet"
                }
            
            # Get total query count
            total_queries = sum(self.symbol_query_counts.values())
            
            # Get top 20 most queried symbols
            top_symbols = dict(sorted(
                self.symbol_query_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20])
            
            # Calculate percentage for each symbol
            symbol_percentages = {}
            for symbol, count in top_symbols.items():
                percentage = (count / total_queries) * 100
                symbol_percentages[symbol] = f"{percentage:.2f}%"
            
            # Create the report
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_symbol_queries": total_queries,
                "unique_symbols": len(self.symbol_query_counts),
                "top_symbols": top_symbols,
                "symbol_percentages": symbol_percentages
            }
            
            # Calculate report generation time
            generation_time = (time.time() - start_time) * 1000
            report["generation_time_ms"] = round(generation_time, 2)
            
            # Log report generation
            self.logger.info("Generated symbol statistics", 
                           total_queries=total_queries,
                           unique_symbols=len(self.symbol_query_counts),
                           generation_time_ms=generation_time)
            
            return report
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about cache performance for financial data.
        
        Returns:
            Dictionary with cache performance statistics
        """
        start_time = time.time()
        
        with self.monitoring_lock:
            # Calculate hit rate
            total_cache_attempts = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_cache_attempts * 100) if total_cache_attempts > 0 else 0
            
            # Create the report
            report = {
                "timestamp": datetime.now().isoformat(),
                "enabled": self.enable_cache,
                "ttl_seconds": self.cache_ttl,
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "expirations": self.cache_expirations,
                "total_attempts": total_cache_attempts,
                "hit_rate": f"{hit_rate:.2f}%",
                "efficiency": {
                    "score": round(hit_rate, 1),  # Score is just the hit rate for now
                    "status": "good" if hit_rate > 80 else "fair" if hit_rate > 50 else "poor"
                }
            }
            
            # Calculate report generation time
            generation_time = (time.time() - start_time) * 1000
            report["generation_time_ms"] = round(generation_time, 2)
            
            # Log report generation
            self.logger.info("Generated cache statistics", 
                           hit_rate=f"{hit_rate:.2f}%",
                           total_attempts=total_cache_attempts,
                           generation_time_ms=generation_time)
            
            return report

    # --- Unified Tool Implementations ---

    def get_market_data(self, symbols: List[str], timeframe: str = '1d', start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None, stream: bool = False, data_type: str = 'bars') -> Dict[str, Any]:
        """Unified method to get market data (bars, trades, quotes)."""
        start_time = time.time()
        
        # Record metrics about the request
        with self.monitoring_lock:
            self.market_data_count += 1
            
            # Track symbols queried for analytics
            for symbol in symbols:
                if symbol not in self.symbol_query_counts:
                    self.symbol_query_counts[symbol] = 0
                self.symbol_query_counts[symbol] += 1
        
        # Check cache if enabled for historical data
        cache_key = None
        cache_hit = False
        
        if self.enable_cache and not stream:
            # Create a unique cache key based on the request parameters
            cache_params = f"{','.join(symbols)}|{timeframe}|{start_date}|{end_date}|{limit}|{data_type}"
            cache_key = f"market_data:{cache_params}"
            
            # Check if we have this data in cache
            if hasattr(self, 'cache') and hasattr(self, 'cache_timestamps'):
                if not hasattr(self, 'cache'):
                    self.cache = {}
                    self.cache_timestamps = {}
                    
                if cache_key in self.cache:
                    # Check if cache entry is still valid
                    cache_time = self.cache_timestamps.get(cache_key, 0)
                    if (time.time() - cache_time) < self.cache_ttl:
                        # Cache hit - return cached data
                        with self.monitoring_lock:
                            self.cache_hits += 1
                        self.logger.info("Cache hit for market data", 
                                       symbols=symbols, 
                                       data_type=data_type,
                                       cache_age=round(time.time() - cache_time, 2))
                        
                        # Record response time for cache hit
                        response_time = (time.time() - start_time) * 1000
                        # Include data_type in the metric name instead of as a parameter
                        metric_name = f"get_market_data_{data_type}_cache_hit_time_ms"
                        self.logger.timing(metric_name, response_time,
                                          source="cache")
                        
                        # Return cached data with cache metadata
                        cached_response = self.cache[cache_key]
                        cached_response["cache"] = {
                            "hit": True,
                            "age_seconds": round(time.time() - cache_time, 2),
                            "source": "memory"
                        }
                        return cached_response
                    else:
                        # Cache expired
                        with self.monitoring_lock:
                            self.cache_expirations += 1
                        self.logger.debug("Cache expired for market data",
                                        symbols=symbols,
                                        data_type=data_type,
                                        age=round(time.time() - cache_time, 2))
                else:
                    # Cache miss
                    with self.monitoring_lock:
                        self.cache_misses += 1
        
        # Log the request
        self.logger.info("Getting market data", 
                       symbols=symbols, 
                       timeframe=timeframe, 
                       stream=stream, 
                       data_type=data_type,
                       cached=cache_hit)
        
        # Create metrics label for the specific type of request
        request_metric_name = f"market_data_{data_type}"
        self.logger.counter(request_metric_name)

        # Streaming data request
        if stream:
            client = self._get_client("market_data_stream")
            client_name = client.name if client else "none"
            
            # Record source selection for metrics
            with self.monitoring_lock:
                if client_name in self.source_usage_counts:
                    self.source_usage_counts[client_name] += 1
                
            # Track streaming subscription metrics
            self.logger.counter("market_data_stream_subscriptions")
            
            if client:
                try:
                    # Implement basic streaming functionality
                    if data_type == 'bars':
                        if hasattr(client, "subscribe_bars"):
                            # Start a subscription and return a subscription ID
                            subscription_id = client.subscribe_bars(
                                symbols=symbols,
                                timeframe=timeframe,
                                callback=None  # We'll use the client's default callback handler
                            )
                            
                            # Record successful subscription
                            self.logger.counter("market_data_stream_success")
                            
                            # Calculate and record response time
                            response_time = (time.time() - start_time) * 1000
                            self.logger.timing("market_data_stream_setup_time_ms", response_time, 
                                             source=client_name, data_type="bars")
                            
                            return {
                                "status": "subscribed",
                                "subscription_id": subscription_id,
                                "message": "Streaming bars subscription started. Data will be processed by the configured callback.",
                                "source": client_name,
                                "response_time_ms": round(response_time, 2)
                            }
                        else:
                            # Record client capability error
                            with self.monitoring_lock:
                                if client_name in self.source_error_counts:
                                    self.source_error_counts[client_name] += 1
                            self.logger.counter("market_data_stream_capability_error")
                            
                            return {"error": f"Client {client_name} does not support streaming bars.", "source": client_name}
                    
                    elif data_type == 'trades':
                        if hasattr(client, "subscribe_trades"):
                            subscription_id = client.subscribe_trades(
                                symbols=symbols,
                                callback=None  # We'll use the client's default callback handler
                            )
                            
                            # Record successful subscription
                            self.logger.counter("market_data_stream_success")
                            
                            # Calculate and record response time
                            response_time = (time.time() - start_time) * 1000
                            self.logger.timing("market_data_stream_setup_time_ms", response_time, 
                                             source=client_name, data_type="trades")
                            
                            return {
                                "status": "subscribed",
                                "subscription_id": subscription_id,
                                "message": "Streaming trades subscription started. Data will be processed by the configured callback.",
                                "source": client_name,
                                "response_time_ms": round(response_time, 2)
                            }
                        else:
                            # Record client capability error
                            with self.monitoring_lock:
                                if client_name in self.source_error_counts:
                                    self.source_error_counts[client_name] += 1
                            self.logger.counter("market_data_stream_capability_error")
                            
                            return {"error": f"Client {client_name} does not support streaming trades.", "source": client_name}
                    
                    elif data_type == 'quotes':
                        if hasattr(client, "subscribe_quotes"):
                            subscription_id = client.subscribe_quotes(
                                symbols=symbols,
                                callback=None  # We'll use the client's default callback handler
                            )
                            
                            # Record successful subscription
                            self.logger.counter("market_data_stream_success")
                            
                            # Calculate and record response time
                            response_time = (time.time() - start_time) * 1000
                            self.logger.timing("market_data_stream_setup_time_ms", response_time, 
                                             source=client_name, data_type="quotes")
                            
                            return {
                                "status": "subscribed",
                                "subscription_id": subscription_id,
                                "message": "Streaming quotes subscription started. Data will be processed by the configured callback.",
                                "source": client_name,
                                "response_time_ms": round(response_time, 2)
                            }
                        else:
                            # Record client capability error
                            with self.monitoring_lock:
                                if client_name in self.source_error_counts:
                                    self.source_error_counts[client_name] += 1
                            self.logger.counter("market_data_stream_capability_error")
                            
                            return {"error": f"Client {client_name} does not support streaming quotes.", "source": client_name}
                    
                    else:
                        # Invalid data type error
                        self.logger.counter("market_data_invalid_type_error")
                        return {"error": f"Invalid streaming data type: {data_type}. Use 'bars', 'trades', or 'quotes'.", "source": client_name}
                
                except Exception as e:
                    # Record streaming setup error
                    with self.monitoring_lock:
                        if client_name in self.source_error_counts:
                            self.source_error_counts[client_name] += 1
                    self.logger.counter("market_data_stream_setup_error")
                    self.logger.error(f"Error setting up streaming for {data_type} from {client_name}: {e}", exc_info=True)
                    
                    # Calculate and record error response time
                    error_time = (time.time() - start_time) * 1000
                    self.logger.timing("market_data_stream_error_time_ms", error_time, 
                                     source=client_name, data_type=data_type)
                    
                    return {
                        "error": f"Streaming setup failed: {str(e)}", 
                        "source": client_name,
                        "response_time_ms": round(error_time, 2)
                    }
            else:
                # No streaming client available error
                self.logger.counter("market_data_stream_no_client_error")
                
                # Calculate and record error response time
                error_time = (time.time() - start_time) * 1000
                # Include data_type in the metric name instead of as a parameter
                metric_name = f"market_data_no_client_{data_type}_error_time_ms"
                self.logger.timing(metric_name, error_time)
                
                return {
                    "error": "No streaming client available.",
                    "response_time_ms": round(error_time, 2)
                }
        
        # Historical data request
        else:
            # Handle different types of historical data
            if data_type == 'bars':
                client = self._get_client("market_data_hist")
                client_name = client.name if client else "none"
                
                # Record source selection for metrics
                with self.monitoring_lock:
                    if client_name in self.source_usage_counts:
                        self.source_usage_counts[client_name] += 1
                
                if client:
                    try:
                        all_data = {}
                        success_count = 0
                        error_count = 0
                        
                        # Process each symbol
                        for symbol in symbols:
                            symbol_start_time = time.time()
                            
                            # Assuming client has a method like get_aggregates or get_historical_prices
                            if hasattr(client, "get_aggregates"): # Polygon style
                                data = client.get_aggregates(
                                    ticker=symbol,
                                    multiplier=1, # Assuming multiplier based on timeframe
                                    timespan=timeframe, # Assuming direct mapping
                                    from_=start_date,
                                    to=end_date,
                                    limit=limit
                                )
                                
                                # Record per-symbol timing
                                symbol_time = (time.time() - symbol_start_time) * 1000
                                # Include symbol in metric name instead of as parameter
                                symbol_safe = symbol.replace('.', '_').replace('-', '_')
                                self.logger.timing(f"market_data_symbol_{symbol_safe}_time_ms", symbol_time,
                                                  source=client_name)
                                
                                if data and not data.get("error"):
                                    all_data[symbol] = data.get("results", [])
                                    success_count += 1
                                else:
                                    all_data[symbol] = {"error": f"Failed to fetch data for {symbol} from {client_name}"}
                                    error_count += 1
                                    
                                    # Track error
                                    with self.monitoring_lock:
                                        if client_name in self.source_error_counts:
                                            self.source_error_counts[client_name] += 1
                            
                            elif hasattr(client, "get_historical_data"): # Yahoo style
                                data = client.get_historical_data(
                                    symbol=symbol,
                                    start_date=start_date,
                                    end_date=end_date,
                                    interval=timeframe # Assuming direct mapping
                                )
                                
                                # Record per-symbol timing
                                symbol_time = (time.time() - symbol_start_time) * 1000
                                # Include symbol in metric name instead of as parameter
                                symbol_safe = symbol.replace('.', '_').replace('-', '_')
                                self.logger.timing(f"market_data_symbol_{symbol_safe}_time_ms", symbol_time,
                                                  source=client_name)
                                
                                if data and not data.get("error"):
                                    all_data[symbol] = data.get("prices", []) # Assuming key name
                                    success_count += 1
                                else:
                                    all_data[symbol] = {"error": f"Failed to fetch data for {symbol} from {client_name}"}
                                    error_count += 1
                                    
                                    # Track error
                                    with self.monitoring_lock:
                                        if client_name in self.source_error_counts:
                                            self.source_error_counts[client_name] += 1
                            else:
                                all_data[symbol] = {"error": f"Client {client_name} does not support getting historical bars."}
                                error_count += 1
                                
                                # Track error
                                with self.monitoring_lock:
                                    if client_name in self.source_error_counts:
                                        self.source_error_counts[client_name] += 1

                        # Calculate and record overall processing time
                        processing_time = time.time() - start_time
                        
                        # Check if the response time is slower than threshold
                        if processing_time * 1000 > self.slow_market_data_threshold_ms:
                            self.logger.warning("Slow market data request detected", 
                                             symbols=symbols, 
                                             data_type=data_type,
                                             response_time_ms=round(processing_time * 1000, 2),
                                             threshold_ms=self.slow_market_data_threshold_ms)
                        
                        # Record metrics
                        # Only pass name, value, and source to timing method
                        self.logger.timing("get_market_data_bars_time_ms", processing_time * 1000,
                                          source=client_name)
                        
                        # Log additional details using info method which accepts kwargs
                        self.logger.info("Market data bars metrics", 
                                       symbol_count=len(symbols),
                                       success_count=success_count,
                                       error_count=error_count,
                                       response_time_ms=round(processing_time * 1000, 2))
                        
                        # Update source latency metrics
                        with self.monitoring_lock:
                            if client_name in self.source_latencies:
                                self.source_latencies[client_name] += processing_time * 1000
                        
                        # Create response
                        response = {
                            "data": all_data, 
                            "source": client_name, 
                            "processing_time_ms": round(processing_time * 1000, 2),
                            "success_rate": f"{(success_count / max(1, len(symbols)) * 100):.1f}%",
                            "symbol_count": len(symbols),
                            "success_count": success_count,
                            "error_count": error_count
                        }
                        
                        # Cache the response if caching is enabled
                        if self.enable_cache and cache_key and error_count == 0:
                            if not hasattr(self, 'cache'):
                                self.cache = {}
                            if not hasattr(self, 'cache_timestamps'):
                                self.cache_timestamps = {}
                            
                            self.cache[cache_key] = response
                            self.cache_timestamps[cache_key] = time.time()
                            self.logger.debug("Cached market data", 
                                            symbols=symbols, 
                                            data_type=data_type,
                                            cache_key=cache_key)
                        
                        return response

                    except Exception as e:
                        # Record error metrics
                        with self.monitoring_lock:
                            if client_name in self.source_error_counts:
                                self.source_error_counts[client_name] += 1
                        
                        # Calculate error response time
                        error_time = (time.time() - start_time) * 1000
                        # Include data_type in the metric name instead of as a parameter
                        metric_name = f"market_data_{data_type}_error_time_ms"
                        self.logger.timing(metric_name, error_time, 
                                         source=client_name)
                        
                        self.logger.error(f"Error getting historical market data from {client_name}: {e}", exc_info=True)
                        return {
                            "error": str(e), 
                            "source": client_name,
                            "response_time_ms": round(error_time, 2)
                        }
                else:
                    # No suitable client available error
                    self.logger.counter("market_data_no_client_error")
                    
                    # Calculate error response time
                    error_time = (time.time() - start_time) * 1000
                    # Include data_type in the metric name instead of as a parameter
                    metric_name = f"market_data_no_client_{data_type}_error_time_ms"
                    self.logger.timing(metric_name, error_time,
                                       source="no_client")
                    
                    return {
                        "error": "No suitable historical market data client available.",
                        "response_time_ms": round(error_time, 2)
                    }

            elif data_type == 'trades':
                client = self._get_client("market_data_hist") # Trades might be available via REST
                client_name = client.name if client else "none"
                
                # Record source selection for metrics
                with self.monitoring_lock:
                    if client_name in self.source_usage_counts:
                        self.source_usage_counts[client_name] += 1
                
                if client and hasattr(client, "get_trades"): # Assuming method name
                    try:
                        all_trades = {}
                        success_count = 0
                        error_count = 0
                        
                        for symbol in symbols:
                            symbol_start_time = time.time()
                            
                            # Parameters might need adaptation
                            trades = client.get_trades(ticker=symbol, limit=limit, timestamp_gte=start_date, timestamp_lte=end_date) # Example params
                            
                            # Record per-symbol timing
                            symbol_time = (time.time() - symbol_start_time) * 1000
                            # Include symbol in metric name instead of as parameter
                            symbol_safe = symbol.replace('.', '_').replace('-', '_')
                            self.logger.timing(f"market_data_trades_symbol_{symbol_safe}_time_ms", symbol_time,
                                              source=client_name)
                            
                            if trades and not trades.get("error"):
                                all_trades[symbol] = trades.get("results", [])
                                success_count += 1
                            else:
                                all_trades[symbol] = {"error": f"Failed to fetch trades for {symbol} from {client_name}"}
                                error_count += 1
                                
                                # Track error
                                with self.monitoring_lock:
                                    if client_name in self.source_error_counts:
                                        self.source_error_counts[client_name] += 1

                        # Calculate and record overall processing time
                        processing_time = time.time() - start_time
                        
                        # Check if the response time is slower than threshold
                        if processing_time * 1000 > self.slow_market_data_threshold_ms:
                            self.logger.warning("Slow trades request detected", 
                                             symbols=symbols, 
                                             response_time_ms=round(processing_time * 1000, 2),
                                             threshold_ms=self.slow_market_data_threshold_ms)
                        
                        # Record metrics
                        # Only pass name, value, and source to timing method
                        self.logger.timing("get_market_data_trades_time_ms", processing_time * 1000,
                                          source=client_name)
                        
                        # Log additional details using info method which accepts kwargs
                        self.logger.info("Market data trades metrics", 
                                       symbol_count=len(symbols),
                                       success_count=success_count,
                                       error_count=error_count,
                                       response_time_ms=round(processing_time * 1000, 2))
                        
                        # Update source latency metrics
                        with self.monitoring_lock:
                            if client_name in self.source_latencies:
                                self.source_latencies[client_name] += processing_time * 1000
                        
                        # Create response
                        response = {
                            "data": all_trades, 
                            "source": client_name, 
                            "processing_time_ms": round(processing_time * 1000, 2),
                            "success_rate": f"{(success_count / max(1, len(symbols)) * 100):.1f}%",
                            "symbol_count": len(symbols),
                            "success_count": success_count,
                            "error_count": error_count
                        }
                        
                        # Cache the response if caching is enabled
                        if self.enable_cache and cache_key and error_count == 0:
                            if not hasattr(self, 'cache'):
                                self.cache = {}
                            if not hasattr(self, 'cache_timestamps'):
                                self.cache_timestamps = {}
                            
                            self.cache[cache_key] = response
                            self.cache_timestamps[cache_key] = time.time()
                            self.logger.debug("Cached trades data", 
                                            symbols=symbols, 
                                            cache_key=cache_key)
                        
                        return response
                    
                    except Exception as e:
                        # Record error metrics
                        with self.monitoring_lock:
                            if client_name in self.source_error_counts:
                                self.source_error_counts[client_name] += 1
                        
                        # Calculate error response time
                        error_time = (time.time() - start_time) * 1000
                        self.logger.timing("market_data_trades_error_time_ms", error_time, 
                                         source=client_name)
                        
                        self.logger.error(f"Error getting trades from {client_name}: {e}", exc_info=True)
                        return {
                            "error": str(e), 
                            "source": client_name,
                            "response_time_ms": round(error_time, 2)
                        }
                else:
                    # No suitable client available error
                    self.logger.counter("market_data_trades_no_client_error")
                    
                    # Calculate error response time
                    error_time = (time.time() - start_time) * 1000
                    self.logger.timing("market_data_no_client_trades_error_time_ms", error_time,
                                       source="no_client")
                    
                    return {
                        "error": "No suitable trades client available.",
                        "response_time_ms": round(error_time, 2)
                    }

            elif data_type == 'quotes':
                client = self._get_client("market_data_hist") # Quotes might be available via REST
                client_name = client.name if client else "none"
                
                # Record source selection for metrics
                with self.monitoring_lock:
                    if client_name in self.source_usage_counts:
                        self.source_usage_counts[client_name] += 1
                
                if client and hasattr(client, "get_quotes"): # Assuming method name
                    try:
                        all_quotes = {}
                        success_count = 0
                        error_count = 0
                        
                        for symbol in symbols:
                            symbol_start_time = time.time()
                            
                            # Parameters might need adaptation
                            quotes = client.get_quotes(ticker=symbol, limit=limit, timestamp_gte=start_date, timestamp_lte=end_date) # Example params
                            
                            # Record per-symbol timing
                            symbol_time = (time.time() - symbol_start_time) * 1000
                            # Include symbol in metric name instead of as parameter
                            symbol_safe = symbol.replace('.', '_').replace('-', '_')
                            self.logger.timing(f"market_data_quotes_symbol_{symbol_safe}_time_ms", symbol_time,
                                              source=client_name)
                            
                            if quotes and not quotes.get("error"):
                                all_quotes[symbol] = quotes.get("results", [])
                                success_count += 1
                            else:
                                all_quotes[symbol] = {"error": f"Failed to fetch quotes for {symbol} from {client_name}"}
                                error_count += 1
                                
                                # Track error
                                with self.monitoring_lock:
                                    if client_name in self.source_error_counts:
                                        self.source_error_counts[client_name] += 1

                        # Calculate and record overall processing time
                        processing_time = time.time() - start_time
                        
                        # Check if the response time is slower than threshold
                        if processing_time * 1000 > self.slow_market_data_threshold_ms:
                            self.logger.warning("Slow quotes request detected", 
                                             symbols=symbols, 
                                             response_time_ms=round(processing_time * 1000, 2),
                                             threshold_ms=self.slow_market_data_threshold_ms)
                        
                        # Record metrics
                        # Only pass name, value, and source to timing method
                        self.logger.timing("get_market_data_quotes_time_ms", processing_time * 1000,
                                          source=client_name)
                        
                        # Log additional details using info method which accepts kwargs
                        self.logger.info("Market data quotes metrics", 
                                       symbol_count=len(symbols),
                                       success_count=success_count,
                                       error_count=error_count,
                                       response_time_ms=round(processing_time * 1000, 2))
                        
                        # Update source latency metrics
                        with self.monitoring_lock:
                            if client_name in self.source_latencies:
                                self.source_latencies[client_name] += processing_time * 1000
                        
                        # Create response
                        response = {
                            "data": all_quotes, 
                            "source": client_name, 
                            "processing_time_ms": round(processing_time * 1000, 2),
                            "success_rate": f"{(success_count / max(1, len(symbols)) * 100):.1f}%",
                            "symbol_count": len(symbols),
                            "success_count": success_count,
                            "error_count": error_count
                        }
                        
                        # Cache the response if caching is enabled
                        if self.enable_cache and cache_key and error_count == 0:
                            if not hasattr(self, 'cache'):
                                self.cache = {}
                            if not hasattr(self, 'cache_timestamps'):
                                self.cache_timestamps = {}
                            
                            self.cache[cache_key] = response
                            self.cache_timestamps[cache_key] = time.time()
                            self.logger.debug("Cached quotes data", 
                                            symbols=symbols, 
                                            cache_key=cache_key)
                        
                        return response
                    
                    except Exception as e:
                        # Record error metrics
                        with self.monitoring_lock:
                            if client_name in self.source_error_counts:
                                self.source_error_counts[client_name] += 1
                        
                        # Calculate error response time
                        error_time = (time.time() - start_time) * 1000
                        self.logger.timing("market_data_quotes_error_time_ms", error_time, 
                                         source=client_name)
                        
                        self.logger.error(f"Error getting quotes from {client_name}: {e}", exc_info=True)
                        return {
                            "error": str(e), 
                            "source": client_name,
                            "response_time_ms": round(error_time, 2)
                        }
                else:
                    # No suitable client available error
                    self.logger.counter("market_data_quotes_no_client_error")
                    
                    # Calculate error response time
                    error_time = (time.time() - start_time) * 1000
                    self.logger.timing("market_data_no_client_quotes_error_time_ms", error_time,
                                       source="no_client")
                    
                    return {
                        "error": "No suitable quotes client available.",
                        "response_time_ms": round(error_time, 2)
                    }

            else:
                # Invalid data type error
                self.logger.counter("market_data_invalid_type_error")
                
                # Calculate error response time
                error_time = (time.time() - start_time) * 1000
                self.logger.timing("market_data_invalid_type_time_ms", error_time,
                                   source="invalid_type")
                
                return {
                    "error": f"Invalid market data type requested: {data_type}. Use 'bars', 'trades', or 'quotes'.",
                    "response_time_ms": round(error_time, 2)
                }


    def get_news(self, symbols: Optional[List[str]] = None, query: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Unified method to get financial news."""
        start_time = time.time()
        self.logger.info("Getting news", symbols=symbols, query=query, limit=limit)
        client = self._get_client("news")

        if client:
            try:
                news_items = []
                # Prioritize symbol-specific news if symbols are provided
                if symbols:
                    if hasattr(client, "get_ticker_news"): # Polygon style
                        for symbol in symbols:
                             result = client.get_ticker_news(ticker=symbol, limit=limit // len(symbols) if symbols else limit)
                             if result and not result.get("error"):
                                 news_items.extend(result.get("results", []))
                    elif hasattr(client, "get_company_news"): # Yahoo style
                         for symbol in symbols:
                              result = client.get_company_news(symbol=symbol, limit=limit // len(symbols) if symbols else limit)
                              if result and not result.get("error"):
                                  news_items.extend(result.get("items", []))
                elif query:
                     if hasattr(client, "search_news"): # Yahoo News style
                          result = client.search_news(query=query, limit=limit)
                          if result and not result.get("error"):
                               news_items.extend(result.get("results", []) or result.get("items", []))
                     # Add other news search methods if available in other clients
                else:
                     # Attempt a general fetch if possible (e.g., Polygon market news)
                     if hasattr(client, "get_market_news"): # Polygon style
                          result = client.get_market_news(limit=limit)
                          if result and not result.get("error"):
                               news_items.extend(result.get("results", []))


                # Simple deduplication based on title or URL if available
                seen = set()
                unique_news = []
                for item in news_items:
                    identifier = item.get('article_url') or item.get('link') or item.get('title')
                    if identifier and identifier not in seen:
                        unique_news.append(item)
                        seen.add(identifier)

                processing_time = time.time() - start_time
                self.logger.timing("get_news_time_ms", processing_time * 1000,
                                   source=client.name, news_count=len(unique_news))
                return {"news": unique_news[:limit], "source": client.name, "processing_time": processing_time}

            except Exception as e:
                 self.logger.error(f"Error getting news from {client.name}: {e}", exc_info=True)
                 return {"error": str(e), "source": client.name}
        else:
            return {"error": "No suitable news client available."}


    def get_social_sentiment(self, symbols: Optional[List[str]] = None, query: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        """Unified method to get social media sentiment."""
        start_time = time.time()
        self.logger.info("Getting social sentiment", symbols=symbols, query=query, limit=limit)
        client = self._get_client("social")

        if client and hasattr(client, "search_posts"): # Assuming RedditMCP style
            try:
                search_term = query
                if symbols:
                    # Combine symbols into a search query if needed, or search one by one
                    search_term = " OR ".join(symbols) if not query else f"({query}) AND ({' OR '.join(symbols)})"

                if not search_term:
                    return {"error": "Please provide symbols or a query for social sentiment."}

                result = client.search_posts(query=search_term, limit=limit, sort='relevance') # Example params
                posts = result.get("posts", []) if result and not result.get("error") else []

                processing_time = time.time() - start_time
                self.logger.timing("get_social_sentiment_time_ms", processing_time * 1000,
                                   source=client.name, post_count=len(posts))
                return {"posts": posts, "source": client.name, "processing_time": processing_time}

            except Exception as e:
                 self.logger.error(f"Error getting social sentiment from {client.name}: {e}", exc_info=True)
                 return {"error": str(e), "source": client.name}
        else:
            return {"error": "No suitable social sentiment client available."}


    def get_options_data(self, symbol: str, date: Optional[str] = None, type: str = 'chain') -> Dict[str, Any]:
        """Unified method to get options data (chain or flow)."""
        start_time = time.time()
        self.logger.info("Getting options data", symbol=symbol, date=date, type=type)

        if type == 'chain':
            client = self._get_client("options_chain")
            if client:
                try:
                    data = {}
                    if hasattr(client, "get_options_chain"): # Polygon style
                        # Need expiration date logic - assuming 'date' is expiration date
                        result = client.get_options_chain(underlying_ticker=symbol, expiration_date=date) # Example params
                        data = result.get("results", []) if result and not result.get("error") else []
                    elif hasattr(client, "get_options"): # Yahoo style
                         result = client.get_options(symbol=symbol, date=date) # Assuming 'date' is expiration date
                         data = result if result and not result.get("error") else {}
                    else:
                         return {"error": f"Client {client.name} does not support getting options chain."}

                    processing_time = time.time() - start_time
                    self.logger.timing("get_options_chain_time_ms", processing_time * 1000,
                                       source=client.name, symbol=symbol)
                    return {"options_chain": data, "source": client.name, "processing_time": processing_time}

                except Exception as e:
                     self.logger.error(f"Error getting options chain from {client.name}: {e}", exc_info=True)
                     return {"error": str(e), "source": client.name}
            else:
                 return {"error": "No suitable options chain client available."}

        elif type == 'flow':
            client = self._get_client("options_flow")
            if client and hasattr(client, "get_option_activity"): # Assuming UnusualWhales style
                try:
                    # Need date/timeframe logic - assuming 'date' is start date
                    result = client.get_option_activity(ticker=symbol, start_date=date) # Example params
                    flow_data = result.get("data", []) if result and not result.get("error") else []
                    processing_time = time.time() - start_time
                    self.logger.timing("get_options_flow_time_ms", processing_time * 1000,
                                       source=client.name, symbol=symbol)
                    return {"options_flow": flow_data, "source": client.name, "processing_time": processing_time}
                except Exception as e:
                     self.logger.error(f"Error getting options flow from {client.name}: {e}", exc_info=True)
                     return {"error": str(e), "source": client.name}
            else:
                 return {"error": "No suitable options flow client available."}
        else:
            return {"error": f"Invalid options data type requested: {type}. Use 'chain' or 'flow'."}


    def get_fundamentals(self, symbol: str, type: str = 'financials', limit: int = 5) -> Dict[str, Any]:
        """Unified method to get fundamental company data (financials, earnings, metrics)."""
        start_time = time.time()
        self.logger.info("Getting fundamentals", symbol=symbol, type=type, limit=limit)
        client = self._get_client("fundamentals")

        if client:
            try:
                data = {}
                if type == 'financials':
                    if hasattr(client, "get_financials_vx"): # Polygon style
                         result = client.get_financials_vx(ticker=symbol, limit=limit) # Example: get last 'limit' reports
                         data = result.get("results", []) if result and not result.get("error") else []
                    elif hasattr(client, "get_financials"): # Yahoo style (assuming a unified method)
                         # Yahoo finance client might have separate methods for income, balance, cashflow
                         # Assuming get_financials method exists and takes type and limit
                         result = client.get_financials(symbol=symbol, type='quarterly', limit=limit) # Example params
                         data = result if result and not result.get("error") else {}
                    else:
                         return {"error": f"Client {client.name} does not support getting financials."}

                elif type == 'earnings':
                     if hasattr(client, "get_earnings_calendar"): # Yahoo style
                          result = client.get_earnings_calendar(symbol=symbol)
                          data = result if result and not result.get("error") else {}
                     # Add other earnings methods if available
                     else:
                         return {"error": f"Client {client.name} does not support getting earnings data."}

                elif type == 'metrics':
                     if hasattr(client, "get_company_metrics"): # Polygon style (example method name)
                          result = client.get_company_metrics(ticker=symbol)
                          data = result.get("results", {}) if result and not result.get("error") else {}
                     elif hasattr(client, "get_key_statistics"): # Yahoo style (example method name)
                          result = client.get_key_statistics(symbol=symbol)
                          data = result if result and not result.get("error") else {}
                     else:
                         return {"error": f"Client {client.name} does not support getting company metrics."}

                # Add other fundamental types as needed (e.g., analyst ratings)

                processing_time = time.time() - start_time
                self.logger.timing("get_fundamentals_time_ms", processing_time * 1000,
                                   source=client.name, symbol=symbol, type=type)
                return {"fundamentals": data, "source": client.name, "processing_time": processing_time}

            except Exception as e:
                 self.logger.error(f"Error getting fundamentals from {client.name}: {e}", exc_info=True)
                 return {"error": str(e), "source": client.name}
        else:
            return {"error": "No suitable fundamentals client available."}

    def get_market_status(self) -> Dict[str, Any]:
        """Unified method to get current market status and related info."""
        start_time = time.time()
        self.logger.info("Getting market status")
        client = self._get_client("market_info")

        if client:
            try:
                status_data = {}
                if hasattr(client, "get_market_status"): # Polygon style
                    result = client.get_market_status()
                    if result and not result.get("error"):
                        status_data.update(result)
                
                # Get VIX, SPY and other market indicators
                market_indicators = ["VIX", "SPY", "DIA", "QQQ", "IWM"]
                indicators_data = {}
                
                # Try to get last trade for each indicator
                if hasattr(client, "get_last_trade"):
                    for ticker in market_indicators:
                        try:
                            trade = client.get_last_trade(ticker=ticker)
                            if trade and not trade.get("error"):
                                price = trade.get("price") or trade.get("p")  # Different APIs might use different keys
                                if price:
                                    indicators_data[f"{ticker.lower()}_price"] = price
                                    
                                    # Get additional data if available
                                    if trade.get("size") or trade.get("s"):
                                        indicators_data[f"{ticker.lower()}_size"] = trade.get("size") or trade.get("s")
                                    if trade.get("timestamp") or trade.get("t"):
                                        indicators_data[f"{ticker.lower()}_timestamp"] = trade.get("timestamp") or trade.get("t")
                        except Exception as e:
                            self.logger.warning(f"Failed to get {ticker} data: {e}")
                
                # Alternative: try to get snapshot if last_trade is not available
                elif hasattr(client, "get_snapshot"):
                    for ticker in market_indicators:
                        try:
                            snapshot = client.get_snapshot(ticker=ticker)
                            if snapshot and not snapshot.get("error"):
                                last_quote = snapshot.get("last_quote") or snapshot.get("quote")
                                last_trade = snapshot.get("last_trade") or snapshot.get("trade")
                                
                                if last_trade and (last_trade.get("price") or last_trade.get("p")):
                                    indicators_data[f"{ticker.lower()}_price"] = last_trade.get("price") or last_trade.get("p")
                                elif last_quote and (last_quote.get("midpoint") or last_quote.get("mp")):
                                    # Use midpoint if trade price not available
                                    indicators_data[f"{ticker.lower()}_price"] = last_quote.get("midpoint") or last_quote.get("mp")
                        except Exception as e:
                            self.logger.warning(f"Failed to get {ticker} snapshot: {e}")
                
                # Add the indicators data to the status data
                if indicators_data:
                    status_data["market_indicators"] = indicators_data


                processing_time = time.time() - start_time
                self.logger.timing("get_market_status_time_ms", processing_time * 1000,
                                   source=client.name)
                return {"data": status_data, "source": client.name, "processing_time": processing_time}

            except Exception as e:
                 self.logger.error(f"Error getting market status from {client.name}: {e}", exc_info=True)
                 return {"error": str(e), "source": client.name}
        else:
            return {"error": "No suitable market info client available."}

    def get_ticker_list(self, market: str = 'stocks', active: bool = True, limit: int = 100) -> Dict[str, Any]:
        """Unified method to get a list of tickers."""
        start_time = time.time()
        self.logger.info("Getting ticker list", market=market, active=active, limit=limit)
        client = self._get_client("reference")

        if client and hasattr(client, "get_ticker_list"): # Assuming method name
            try:
                # Parameters might need adaptation
                result = client.get_ticker_list(market=market, active=active, limit=limit)
                tickers = result.get("results", []) if result and not result.get("error") else []

                processing_time = time.time() - start_time
                self.logger.timing("get_ticker_list_time_ms", processing_time * 1000,
                                   source=client.name, ticker_count=len(tickers))
                return {"tickers": tickers, "source": client.name, "processing_time": processing_time}
            except Exception as e:
                 self.logger.error(f"Error getting ticker list from {client.name}: {e}", exc_info=True)
                 return {"error": str(e), "source": client.name}
        else:
            return {"error": "No suitable reference data client available for ticker lists."}

    def get_market_movers(self, category: str = 'gainers', limit: int = 10) -> Dict[str, Any]:
        """Unified method to get top market movers (gainers or losers)."""
        start_time = time.time()
        self.logger.info("Getting market movers", category=category, limit=limit)
        client = self._get_client("market_info") # Market info client might provide movers

        if client and hasattr(client, "get_market_movers"): # Assuming method name
            try:
                # Parameters might need adaptation
                result = client.get_market_movers(category=category, limit=limit)
                movers = result.get("results", []) if result and not result.get("error") else []

                processing_time = time.time() - start_time
                self.logger.timing("get_market_movers_time_ms", processing_time * 1000,
                                   source=client.name, mover_count=len(movers))
                return {"movers": movers, "source": client.name, "processing_time": processing_time}
            except Exception as e:
                 self.logger.error(f"Error getting market movers from {client.name}: {e}", exc_info=True)
                 return {"error": str(e), "source": client.name}
        else:
            return {"error": "No suitable market info client available for market movers."}


    def shutdown(self):
        """
        Shutdown the MCP server and its clients.
        Performs proper cleanup of resources and generates final metrics.
        """
        start_time = time.time()
        self.logger.info(f"Shutting down {self.name} MCP server")
        
        # First, generate a final performance report for record-keeping
        try:
            performance_report = self.generate_financial_performance_report()
            self.logger.info("Generated final performance report", 
                           total_requests=performance_report["operations"]["total_requests"],
                           error_rate=performance_report["operations"]["error_rate"])
        except Exception as e:
            self.logger.error(f"Error generating final performance report: {e}", exc_info=True)
        
        # Get all active client names for metrics
        active_clients = list(self.clients.keys())
        
        # Gracefully shut down all clients
        successful_shutdowns = 0
        failed_shutdowns = 0
        
        for client_name, client in self.clients.items():
            try:
                # Record client shutdown start time for metrics
                client_shutdown_start = time.time()
                
                # Check if client has active subscriptions that need to be closed
                if hasattr(client, "close_all_subscriptions"):
                    client.close_all_subscriptions()
                    self.logger.info(f"Closed all subscriptions for client '{client_name}'.")
                
                # Shut down the client
                if hasattr(client, "shutdown"):
                    client.shutdown()
                    
                    # Calculate and record shutdown time
                    client_shutdown_time = (time.time() - client_shutdown_start) * 1000
                    self.logger.timing("client_shutdown_time_ms", client_shutdown_time,
                                       source=client_name)
                    
                    self.logger.info(f"Client '{client_name}' shut down successfully.", 
                                   shutdown_time_ms=client_shutdown_time)
                    
                    successful_shutdowns += 1
                else:
                    self.logger.warning(f"Client '{client_name}' has no shutdown method.")
            except Exception as e:
                self.logger.error(f"Error shutting down client '{client_name}': {e}", exc_info=True)
                failed_shutdowns += 1
        
        # Stop the financial health check thread if it exists
        if hasattr(self, 'financial_health_thread') and self.financial_health_thread.is_alive():
            try:
                # Set shutdown flag (should already be set by BaseMCPServer)
                self.shutdown_requested = True
                
                # Give thread a chance to terminate
                self.financial_health_thread.join(timeout=1.0)
                if self.financial_health_thread.is_alive():
                    self.logger.warning("Financial health check thread did not terminate gracefully")
                else:
                    self.logger.info("Financial health check thread terminated successfully")
            except Exception as e:
                self.logger.error(f"Error terminating financial health check thread: {e}", exc_info=True)
        
        # Clear cache to free memory
        if hasattr(self, 'cache'):
            try:
                cache_size = len(self.cache)
                self.cache.clear()
                if hasattr(self, 'cache_timestamps'):
                    self.cache_timestamps.clear()
                self.logger.info(f"Cleared data cache with {cache_size} entries")
            except Exception as e:
                self.logger.error(f"Error clearing cache: {e}", exc_info=True)
        
        # Log final statistics before super().shutdown()
        shutdown_duration = (time.time() - start_time) * 1000
        
        self.logger.info("Financial data MCP shutdown statistics", 
                       total_clients=len(active_clients),
                       successful_shutdowns=successful_shutdowns,
                       failed_shutdowns=failed_shutdowns,
                       total_market_data_requests=self.market_data_count,
                       total_news_requests=self.news_request_count,
                       total_social_sentiment_requests=self.social_sentiment_count,
                       total_options_data_requests=self.options_data_count,
                       shutdown_time_ms=shutdown_duration)
        
        # Call parent class shutdown to complete the shutdown process
        super().shutdown()
        
    def unsubscribe(self, subscription_id: str) -> Dict[str, Any]:
        """
        Unsubscribe from a streaming data subscription.
        
        Args:
            subscription_id: The ID of the subscription to cancel
            
        Returns:
            Dictionary with status of the unsubscribe operation
        """
        self.logger.info(f"Unsubscribing from subscription {subscription_id}")
        
        # Find which client has this subscription
        for client_name, client in self.clients.items():
            if hasattr(client, "unsubscribe"):
                try:
                    result = client.unsubscribe(subscription_id)
                    if result and result.get("status") == "unsubscribed":
                        self.logger.info(f"Successfully unsubscribed from {subscription_id} on client {client_name}")
                        return {
                            "status": "unsubscribed",
                            "subscription_id": subscription_id,
                            "source": client_name
                        }
                except Exception as e:
                    self.logger.error(f"Error unsubscribing from {subscription_id} on client {client_name}: {e}")
        
        # If we get here, no client successfully unsubscribed
        return {
            "error": f"Failed to unsubscribe from {subscription_id}. Subscription not found or error occurred."
        }

# Example usage (for testing purposes) - Removed to keep file clean
# if __name__ == "__main__":
#     pass
