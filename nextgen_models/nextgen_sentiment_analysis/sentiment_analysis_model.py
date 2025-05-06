"""
Sentiment Analysis Model

This module defines the SentimentAnalysisModel, responsible for processing text data
to extract entities and determine sentiment scores using dedicated MCP tools.
It also handles storage and retrieval of sentiment data for the trading system.
"""

import asyncio
import json
import time
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set
import uuid # Import uuid for generating unique IDs
import threading

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# MCP tools
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function


class SentimentAnalysisModel:
    """
    Analyzes text data (e.g., news, social media) to extract relevant financial entities
    and calculate sentiment scores associated with them.
    This model interacts with FinancialDataMCP.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, user_proxy_agent: Optional[UserProxyAgent] = None):
        """
        Initialize the SentimentAnalysisModel.

        Args:
            config: Configuration dictionary, expected to contain:
                # - financial_text_config: Config for FinancialTextMCP. (Removed, merged into financial_data_config)
                - financial_data_config: Config for FinancialDataMCP.
                - symbol_entity_mapping: Optional mapping of entities to symbols.
                - batch_size: Size of batches for processing (default: 10).
                - sentiment_ttl: Time-to-live for sentiment data in seconds (default: 86400 - 1 day).
                - llm_config: Configuration for AutoGen LLM.
            user_proxy_agent: Optional UserProxyAgent instance to register functions with.

        Args:
            config: Configuration dictionary, expected to contain:
                # - financial_text_config: Config for FinancialTextMCP. (Removed, merged into financial_data_config)
                - financial_data_config: Config for FinancialDataMCP.
                - symbol_entity_mapping: Optional mapping of entities to symbols.
                - batch_size: Size of batches for processing (default: 10).
                - sentiment_ttl: Time-to-live for sentiment data in seconds (default: 86400 - 1 day).
                - llm_config: Configuration for AutoGen LLM.
        """
        init_start_time = time.time()
        
        # Initialize monitoring lock for thread safety
        self.monitoring_lock = threading.RLock()
        
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-sentiment-analysis-model")
        self.logger.info("Starting SentimentAnalysisModel initialization")
        
        # Initialize SystemMetricsCollector for resource monitoring
        from monitoring.system_metrics import SystemMetricsCollector
        self.system_metrics = SystemMetricsCollector(self.logger)
        self.logger.info("SystemMetricsCollector initialized")

        # Initialize StockChartGenerator
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")

        # Initialize counters for sentiment analysis metrics
        self.texts_analyzed_count = 0
        self.entities_extracted_count = 0
        self.sentiment_scores_generated_count = 0
        self.positive_sentiment_count = 0
        self.negative_sentiment_count = 0
        self.neutral_sentiment_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0 # Errors during analysis process
        
        # Initialize performance metrics with timestamps
        self.performance_metrics = {
            "start_time": datetime.now(),
            "last_activity_time": datetime.now(),
            "response_times": [],  # List of recent response times
            "error_timestamps": [],  # Timestamps of recent errors
            "processing_times_by_function": {},  # Dict of function processing times
            "memory_usage_history": [],  # Track memory usage over time
            "cpu_usage_history": [],  # Track CPU usage over time
            "symbols_processed": set(),  # Set of unique symbols processed
            "sources_processed": set(),  # Set of unique news sources processed
            "entities_processed": set(),  # Set of unique entities processed
            "batch_sizes": [],  # Track batch sizes processed
            "slow_operations": [],  # List of slow operations (>1s)
            "peak_memory_usage": 0,  # Peak memory usage
            "cache_hits": 0,  # Count of cache hits
            "cache_misses": 0,  # Count of cache misses
            "sentiment_distribution": {  # Distribution of sentiment scores
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "hourly_distribution": {  # Distribution of processing by hour
                str(i): 0 for i in range(24)
            }
        }
        
        # Initialize monitoring data for LLM API calls
        self.llm_api_metrics = {
            "total_calls": 0,
            "total_tokens": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_response_time": 0,
            "calls_by_model": {},
            "tokens_by_model": {},
            "costs_by_model": {},  # If cost tracking is enabled
        }
        
        # Initialize health data structure
        self.health_data = {
            "last_health_check": datetime.now(),
            "health_check_count": 0,
            "health_status": "initializing",  # initializing, healthy, degraded, error
            "health_score": 100,  # 0-100 score
            "component_health": {  # Health of individual components
                "financial_text_mcp": "unknown",
                "financial_data_mcp": "unknown",
                "redis_mcp": "unknown",
                "llm": "unknown",
            },
            "recent_issues": [],  # List of recent health issues
        }
        
        # Initialize shutdown flag
        self.shutdown_requested = False

        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_sentiment_analysis", "sentiment_analysis_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}")
                    self.execution_errors += 1
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}")
                    self.execution_errors += 1
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}")
                self.config = {}
        else:
            self.config = config

        # Initialize FinancialDataMCP client (handles data sources, entity extraction, and sentiment scoring)
        self.financial_data_mcp = FinancialDataMCP(
            self.config.get("financial_data_config") # Use financial_data_config key
        )

        # For backward compatibility or clarity, point aliases to the consolidated client
        self.entity_client = self.financial_data_mcp
        self.sentiment_client = self.financial_data_mcp
        self.news_client = self.financial_data_mcp
        self.social_client = self.financial_data_mcp

        # Initialize Redis MCP client
        self.redis_mcp = RedisMCP(self.config.get("redis_mcp"))
        self.redis_client = self.redis_mcp # Alias for backward compatibility if needed


        # Configuration parameters
        self.batch_size = self.config.get("batch_size", 10)
        self.sentiment_ttl = self.config.get("sentiment_ttl", 86400)  # Default: 1 day

        # Configure health and performance thresholds
        self.monitoring_config = self.config.get("monitoring_config", {})
        self.health_check_interval = self.monitoring_config.get("health_check_interval", 60)  # seconds
        self.performance_check_interval = self.monitoring_config.get("performance_check_interval", 300)  # seconds
        self.slow_operation_threshold = self.monitoring_config.get("slow_operation_threshold", 1000)  # ms
        self.high_memory_threshold = self.monitoring_config.get("high_memory_threshold", 75)  # percent
        self.high_cpu_threshold = self.monitoring_config.get("high_cpu_threshold", 80)  # percent
        self.enable_detailed_metrics = self.monitoring_config.get("enable_detailed_metrics", True)
        
        # Entity to symbol mapping (can be updated dynamically)
        self.symbol_entity_mapping = self.config.get("symbol_entity_mapping", {})
        # Attempt to load mapping from Redis on startup
        asyncio.run(self._load_symbol_entity_mapping())


        # Redis keys for data storage and inter-model communication
        self.redis_keys = {
            "sentiment_data": "sentiment:data", # Overall sentiment data (if aggregated)
            "entity_sentiment": "sentiment:entity:", # Prefix for entity sentiment data
            "symbol_sentiment": "sentiment:symbol:", # Prefix for symbol sentiment data
            "sentiment_history": "sentiment:history:", # Prefix for sentiment history per symbol (e.g., daily aggregates)
            "latest_analysis_timestamp": "sentiment:latest_analysis_timestamp", # Latest analysis timestamp (single key)
            "symbol_entity_mapping": "sentiment:symbol_entity_mapping", # Key for symbol-entity mapping
            "selection_data": "selection:data", # Selection model data (for feedback)
            "selection_feedback_stream": "sentiment:selection_feedback", # Feedback to selection model (stream)
            "sentiment_analysis_reports_stream": "model:sentiment:analysis", # Stream for publishing sentiment analysis reports
        }

        # Ensure Redis streams exist (optional, but good practice)
        try:
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_feedback_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['selection_feedback_stream']}' exists.")
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["sentiment_analysis_reports_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['sentiment_analysis_reports_stream']}' exists.")
        except Exception as e:
            self.logger.warning(f"Could not ensure Redis streams exist: {e}")


        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Define the score_sentiment_func before registering it
        def score_sentiment_func(text: str) -> Dict[str, Any]:
            """
            Score the sentiment of a given text.
            
            Args:
                text: Text to analyze
                
            Returns:
                Sentiment analysis result
            """
            try:
                # Use the financial data MCP to score sentiment
                result = self.financial_data_mcp.call_tool(
                    "score_sentiment", {"text": text}
                )
                
                if result and not result.get("error"):
                    return result
                else:
                    return {
                        "error": result.get("error", "Failed to score sentiment")
                    }
            except Exception as e:
                self.logger.error(f"Error scoring sentiment: {e}", exc_info=True)
                return {"error": str(e)}

        self.score_sentiment_func = score_sentiment_func

        # Register functions with the agents
        self._register_functions()
        
        # Start system metrics collection
        try:
            self.system_metrics.start()
            self.logger.info("System metrics collection started")
        except Exception as e:
            self.logger.error(f"Error starting system metrics collection: {e}", exc_info=True)
            
        # Start health check thread if enabled
        if self.monitoring_config.get("enable_health_check", True):
            self._start_health_check_thread()
        
        # Update overall health status
        self._update_overall_health_status()
        
        # Save initial performance metrics to Redis
        asyncio.run(self._save_performance_metrics())

        # Record initialization completion
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("sentiment_analysis_model.initialization_time_ms", init_duration)
        
        # Log comprehensive initialization statistics
        self.logger.info("SentimentAnalysisModel initialization complete", 
                       init_time_ms=init_duration,
                       health_status=self.health_data["health_status"],
                       health_score=self.health_data["health_score"],
                       components_initialized=[
                           k for k, v in self.health_data["component_health"].items() 
                           if v in ["healthy", "warning"]
                       ])


    def _start_health_check_thread(self):
        """
        Start background thread for periodic health checks and performance monitoring.
        """
        def health_check_loop():
            self.logger.info(f"Starting health check thread with interval of {self.health_check_interval} seconds")
            
            # Initialize health check counter for periodic detailed reports
            health_check_counter = 0
            
            # Use the existing shutdown_requested flag
            while not self.shutdown_requested:
                try:
                    # Increment counter
                    health_check_counter += 1
                    
                    # Update health check timestamp
                    with self.monitoring_lock:
                        self.health_data["last_health_check"] = datetime.now()
                        self.health_data["health_check_count"] += 1
                    
                    # Perform health check
                    self._check_health()
                    
                    # Update performance metrics
                    if health_check_counter % 5 == 0:  # Every 5th check
                        asyncio.run(self._update_performance_metrics())
                    
                    # Generate performance visualization
                    if health_check_counter % 10 == 0:  # Every 10th check
                        asyncio.run(self._generate_performance_visualizations())
                    
                    # Log health status
                    if health_check_counter % 5 == 0:  # Detailed log every 5th check
                        self.logger.info("Detailed health check completed", 
                                       health_status=self.health_data["health_status"],
                                       health_score=self.health_data["health_score"],
                                       component_health=self.health_data["component_health"],
                                       recent_issues_count=len(self.health_data["recent_issues"]),
                                       texts_analyzed_count=self.texts_analyzed_count,
                                       entities_extracted_count=self.entities_extracted_count,
                                       execution_errors=self.execution_errors)
                    else:
                        self.logger.info("Health check completed", 
                                       health_status=self.health_data["health_status"],
                                       health_score=self.health_data["health_score"])
                    
                except Exception as e:
                    self.logger.error(f"Error in health check thread: {e}", exc_info=True)
                    self.execution_errors += 1
                
                # Sleep until next check
                time.sleep(self.health_check_interval)
        
        # Start health check thread as daemon
        self.health_thread = threading.Thread(target=health_check_loop, daemon=True)
        self.health_thread.start()
        self.logger.info("Health check thread started")
        
    def _check_health(self):
        """
        Perform a comprehensive health check of all system components.
        Updates the health_data dictionary with current status.
        """
        try:
            with self.monitoring_lock:
                # Check Financial Data MCP health
                if self.financial_data_mcp is not None:
                    try:
                        # Try a simple operation to check if it's responsive
                        if hasattr(self.financial_data_mcp, "health_check"):
                            data_health = self.financial_data_mcp.health_check()
                            self.health_data["component_health"]["financial_data_mcp"] = data_health.get("status", "unknown")
                        else:
                            # Fallback to checking if it responds to a method call
                            method_exists = hasattr(self.financial_data_mcp, "call_tool")
                            self.health_data["component_health"]["financial_data_mcp"] = "healthy" if method_exists else "degraded"
                    except Exception as e:
                        self.health_data["component_health"]["financial_data_mcp"] = "error"
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "financial_data_mcp",
                            "error": str(e)
                        })
                        self.logger.error(f"Error checking Financial Data MCP health: {e}")
                
                # Check Redis MCP health
                if self.redis_mcp is not None:
                    try:
                        # Try a simple ping operation to check if Redis is responsive
                        if hasattr(self.redis_mcp, "health_check"):
                            redis_health = self.redis_mcp.health_check()
                            self.health_data["component_health"]["redis_mcp"] = redis_health.get("status", "unknown")
                        else:
                            # Fallback to checking if it responds to a method call
                            ping_result = self.redis_mcp.call_tool("ping", {})
                            self.health_data["component_health"]["redis_mcp"] = "healthy" if ping_result and not ping_result.get("error") else "degraded"
                    except Exception as e:
                        self.health_data["component_health"]["redis_mcp"] = "error"
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "redis_mcp",
                            "error": str(e)
                        })
                        self.logger.error(f"Error checking Redis MCP health: {e}")
                
                # Check LLM API health
                try:
                    # Assuming LLM health can be checked by examining recent API calls
                    if self.llm_api_metrics["total_calls"] > 0:
                        # Calculate success rate
                        success_rate = self.llm_api_metrics["successful_calls"] / max(1, self.llm_api_metrics["total_calls"])
                        
                        if success_rate >= 0.95:  # More than 95% successful
                            self.health_data["component_health"]["llm"] = "healthy"
                        elif success_rate >= 0.8:  # 80-95% successful
                            self.health_data["component_health"]["llm"] = "warning"
                        else:  # Less than 80% successful
                            self.health_data["component_health"]["llm"] = "degraded"
                except Exception as e:
                    self.health_data["component_health"]["llm"] = "unknown"
                    self.logger.error(f"Error checking LLM API health: {e}")
                
                # Check system resources
                try:
                    # Get current CPU and memory usage
                    cpu_percent = self.system_metrics.get_cpu_usage()
                    memory_percent = self.system_metrics.get_memory_usage()
                    
                    # Update maximum memory usage
                    self.performance_metrics["peak_memory_usage"] = max(
                        self.performance_metrics["peak_memory_usage"], 
                        memory_percent
                    )
                    
                    # Add to history (keeping last 10 data points)
                    self.performance_metrics["memory_usage_history"].append(memory_percent)
                    self.performance_metrics["cpu_usage_history"].append(cpu_percent)
                    if len(self.performance_metrics["memory_usage_history"]) > 10:
                        self.performance_metrics["memory_usage_history"].pop(0)
                    if len(self.performance_metrics["cpu_usage_history"]) > 10:
                        self.performance_metrics["cpu_usage_history"].pop(0)
                    
                    # Log resource usage
                    self.logger.gauge("sentiment_analysis_model.cpu_usage", cpu_percent)
                    self.logger.gauge("sentiment_analysis_model.memory_usage", memory_percent)
                    
                    # Check for resource issues
                    if cpu_percent > self.high_cpu_threshold:
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "system_resources",
                            "issue": f"High CPU usage: {cpu_percent}%"
                        })
                        self.logger.warning(f"High CPU usage detected: {cpu_percent}%")
                    
                    if memory_percent > self.high_memory_threshold:
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "system_resources",
                            "issue": f"High memory usage: {memory_percent}%"
                        })
                        self.logger.warning(f"High memory usage detected: {memory_percent}%")
                        
                except Exception as e:
                    self.logger.error(f"Error checking system resources: {e}")
                
                # Limit recent issues list to most recent 20
                if len(self.health_data["recent_issues"]) > 20:
                    self.health_data["recent_issues"] = self.health_data["recent_issues"][-20:]
                
                # Update overall health status based on component health
                self._update_overall_health_status()
                
                # Store current health status in Redis
                if self.redis_mcp is not None:
                    self.redis_mcp.call_tool("set_json", {
                        "key": "sentiment:health:status",
                        "value": {
                            "status": self.health_data["health_status"],
                            "score": self.health_data["health_score"],
                            "components": self.health_data["component_health"],
                            "last_check": datetime.now().isoformat(),
                            "issues_count": len(self.health_data["recent_issues"])
                        }
                    })
        
        except Exception as e:
            self.logger.error(f"Error in health check: {e}", exc_info=True)
            self.execution_errors += 1
    
    def _update_overall_health_status(self):
        """
        Update the overall health status based on component health.
        """
        # Count components by status
        component_counts = {"healthy": 0, "warning": 0, "degraded": 0, "error": 0, "unknown": 0}
        for status in self.health_data["component_health"].values():
            if status in component_counts:
                component_counts[status] += 1
        
        # Determine overall status
        if component_counts["error"] > 0:
            self.health_data["health_status"] = "error"
            self.health_data["health_score"] = max(0, 40 - component_counts["error"] * 20)
        elif component_counts["degraded"] > 0:
            self.health_data["health_status"] = "degraded"
            self.health_data["health_score"] = max(40, 70 - component_counts["degraded"] * 10)
        elif component_counts["warning"] > 0:
            self.health_data["health_status"] = "warning"
            self.health_data["health_score"] = max(70, 90 - component_counts["warning"] * 5)
        elif component_counts["unknown"] == len(self.health_data["component_health"]):
            self.health_data["health_status"] = "unknown"
            self.health_data["health_score"] = 50
        else:
            self.health_data["health_status"] = "healthy"
            self.health_data["health_score"] = 100
        
        # Update health gauge
        self.logger.gauge("sentiment_analysis_model.health_score", self.health_data["health_score"])

    async def _update_performance_metrics(self):
        """
        Update and collect comprehensive performance metrics.
        This enhanced version includes trend analysis and anomaly detection.
        """
        try:
            with self.monitoring_lock:
                # Update timestamp
                current_time = datetime.now()
                self.performance_metrics["last_activity_time"] = current_time
                
                # Get current system resource usage
                cpu_percent = self.system_metrics.get_cpu_usage()
                memory_percent = self.system_metrics.get_memory_usage()
                
                # Update resource usage history
                self.performance_metrics["cpu_usage_history"].append(cpu_percent)
                self.performance_metrics["memory_usage_history"].append(memory_percent)
                
                # Keep history at reasonable size (last 30 data points)
                if len(self.performance_metrics["cpu_usage_history"]) > 30:
                    self.performance_metrics["cpu_usage_history"] = self.performance_metrics["cpu_usage_history"][-30:]
                if len(self.performance_metrics["memory_usage_history"]) > 30:
                    self.performance_metrics["memory_usage_history"] = self.performance_metrics["memory_usage_history"][-30:]
                
                # Update peak memory usage
                self.performance_metrics["peak_memory_usage"] = max(
                    self.performance_metrics["peak_memory_usage"], 
                    memory_percent
                )
                
                # Update response time metrics (if we have any)
                if len(self.performance_metrics["response_times"]) > 30:
                    self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-30:]
                
                # Initialize trend tracking if needed
                if not hasattr(self, "performance_trends"):
                    self.performance_trends = {
                        "response_time": {
                            "history": [],
                            "last_update": time.time(),
                            "trend": "stable",  # stable, improving, degrading
                            "anomalies": []
                        },
                        "memory_usage": {
                            "history": [],
                            "last_update": time.time(),
                            "trend": "stable",
                            "anomalies": []
                        },
                        "sentiment_distribution": {
                            "history": [],
                            "last_update": time.time(),
                            "trend": "stable",
                            "anomalies": []
                        },
                        "error_rate": {
                            "history": [],
                            "last_update": time.time(),
                            "trend": "stable",
                            "anomalies": []
                        }
                    }
                
                # Update trend data
                await self._update_trend_data()
                
                # Perform anomaly detection
                await self._detect_anomalies()
                
                # Update API usage metrics
                self._update_api_usage_metrics()
                
                # Clean up outdated metrics
                self._cleanup_old_metrics()
                
                # Save performance metrics to Redis
                await self._save_performance_metrics()
                
                # Log metrics update
                self.logger.info("Performance metrics updated",
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    peak_memory_usage=self.performance_metrics["peak_memory_usage"],
                    slow_operations_count=len(self.performance_metrics["slow_operations"])
                )
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}", exc_info=True)
            self.execution_errors += 1
    
    async def _update_trend_data(self):
        """Update performance trend data using linear regression."""
        try:
            # Only update trends every 5 minutes
            for metric_name, trend_data in self.performance_trends.items():
                if time.time() - trend_data["last_update"] < 300:  # 300 seconds = 5 minutes
                    continue
                
                # Get appropriate metric history
                if metric_name == "response_time" and self.performance_metrics["response_times"]:
                    values = self.performance_metrics["response_times"]
                elif metric_name == "memory_usage":
                    values = self.performance_metrics["memory_usage_history"]
                elif metric_name == "sentiment_distribution":
                    # Create a ratio of positive to negative sentiment
                    positive = self.performance_metrics["sentiment_distribution"]["positive"]
                    negative = self.performance_metrics["sentiment_distribution"]["negative"]
                    values = [positive / max(1, negative + positive)] * 5  # Create synthetic history if needed
                elif metric_name == "error_rate":
                    # Calculate error rate from error timestamps
                    error_counts = []
                    now = time.time()
                    for minute in range(10):  # Last 10 minutes
                        minute_start = now - (minute + 1) * 60
                        minute_end = now - minute * 60
                        count = sum(1 for ts in self.performance_metrics["error_timestamps"] 
                                    if minute_start <= ts <= minute_end)
                        error_counts.append(count)
                    values = error_counts
                else:
                    continue
                
                # Need at least 5 data points for meaningful trend
                if len(values) < 5:
                    continue
                
                # Update trend history
                trend_data["history"].append((time.time(), values[-1]))
                # Keep history at reasonable size
                if len(trend_data["history"]) > 100:
                    trend_data["history"] = trend_data["history"][-100:]
                
                # Simple linear regression for trend
                if len(trend_data["history"]) >= 5:
                    x = [i for i in range(len(trend_data["history"]))]
                    y = [value for _, value in trend_data["history"]]
                    n = len(x)
                    
                    # Calculate slope
                    x_mean = sum(x) / n
                    y_mean = sum(y) / n
                    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        # Determine trend direction based on slope
                        threshold = 0.05 * y_mean  # 5% of mean as significance threshold
                        if abs(slope) < threshold:
                            trend_data["trend"] = "stable"
                        elif (metric_name in ["response_time", "memory_usage", "error_rate"] 
                              and slope > 0):
                            trend_data["trend"] = "degrading"  # Higher is worse for these metrics
                        elif metric_name == "sentiment_distribution" and slope < 0:
                            trend_data["trend"] = "degrading"  # Lower positive/negative ratio is worse
                        else:
                            trend_data["trend"] = "improving"
                
                # Update last update time
                trend_data["last_update"] = time.time()
                
                # Log trend update
                self.logger.info(f"Performance trend updated for {metric_name}", 
                    trend=trend_data["trend"],
                    current_value=values[-1] if values else None
                )
        except Exception as e:
            self.logger.error(f"Error updating performance trends: {e}")
    
    async def _detect_anomalies(self):
        """Detect anomalies in performance metrics using statistical methods."""
        try:
            # Anomaly detection for response times
            if len(self.performance_metrics["response_times"]) >= 10:
                mean = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
                variance = sum((x - mean) ** 2 for x in self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
                std_dev = variance ** 0.5
                
                # Check latest response time
                if len(self.performance_metrics["response_times"]) > 0:
                    latest = self.performance_metrics["response_times"][-1]
                    # Anomaly if more than 3 standard deviations away
                    if abs(latest - mean) > 3 * std_dev and std_dev > 0:
                        anomaly = {
                            "timestamp": time.time(),
                            "metric": "response_time",
                            "value": latest,
                            "mean": mean,
                            "std_dev": std_dev,
                            "deviation": (latest - mean) / std_dev
                        }
                        self.performance_trends["response_time"]["anomalies"].append(anomaly)
                        # Keep only recent anomalies
                        if len(self.performance_trends["response_time"]["anomalies"]) > 10:
                            self.performance_trends["response_time"]["anomalies"] = self.performance_trends["response_time"]["anomalies"][-10:]
                        self.logger.warning(f"Response time anomaly detected: {latest:.2f}ms vs mean {mean:.2f}ms")
            
            # Similar approach for memory
            if len(self.performance_metrics["memory_usage_history"]) >= 10:
                mean = sum(self.performance_metrics["memory_usage_history"]) / len(self.performance_metrics["memory_usage_history"])
                variance = sum((x - mean) ** 2 for x in self.performance_metrics["memory_usage_history"]) / len(self.performance_metrics["memory_usage_history"])
                std_dev = variance ** 0.5
                
                if len(self.performance_metrics["memory_usage_history"]) > 0:
                    latest = self.performance_metrics["memory_usage_history"][-1]
                    # Anomaly if more than 3 standard deviations away
                    if abs(latest - mean) > 3 * std_dev and std_dev > 0:
                        anomaly = {
                            "timestamp": time.time(),
                            "metric": "memory_usage",
                            "value": latest,
                            "mean": mean,
                            "std_dev": std_dev,
                            "deviation": (latest - mean) / std_dev
                        }
                        self.performance_trends["memory_usage"]["anomalies"].append(anomaly)
                        # Keep only recent anomalies
                        if len(self.performance_trends["memory_usage"]["anomalies"]) > 10:
                            self.performance_trends["memory_usage"]["anomalies"] = self.performance_trends["memory_usage"]["anomalies"][-10:]
                        self.logger.warning(f"Memory usage anomaly detected: {latest:.2f}% vs mean {mean:.2f}%")
            
            # Check for sentiment distribution anomalies
            total_sentiments = (
                self.performance_metrics["sentiment_distribution"]["positive"] + 
                self.performance_metrics["sentiment_distribution"]["negative"] + 
                self.performance_metrics["sentiment_distribution"]["neutral"]
            )
            
            if total_sentiments > 20:  # Only check if we have enough data
                # Calculate expected distribution based on historical data
                pos_ratio = self.performance_metrics["sentiment_distribution"]["positive"] / max(1, total_sentiments)
                neg_ratio = self.performance_metrics["sentiment_distribution"]["negative"] / max(1, total_sentiments)
                
                # If there's a sudden spike in sentiment directionality
                if pos_ratio > 0.8 or neg_ratio > 0.8:  # Extremely skewed distribution
                    anomaly = {
                        "timestamp": time.time(),
                        "metric": "sentiment_distribution",
                        "value": {"positive": pos_ratio, "negative": neg_ratio},
                        "mean": {"positive": 0.5, "negative": 0.25},  # Expected approximate distribution
                        "std_dev": 0.15,  # Approximate standard deviation
                        "deviation": max(abs(pos_ratio - 0.5), abs(neg_ratio - 0.25)) / 0.15
                    }
                    self.performance_trends["sentiment_distribution"]["anomalies"].append(anomaly)
                    # Keep only recent anomalies
                    if len(self.performance_trends["sentiment_distribution"]["anomalies"]) > 10:
                        self.performance_trends["sentiment_distribution"]["anomalies"] = self.performance_trends["sentiment_distribution"]["anomalies"][-10:]
                    self.logger.warning(f"Sentiment distribution anomaly detected: positive={pos_ratio:.2f}, negative={neg_ratio:.2f}")
        
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
    
    def _update_api_usage_metrics(self):
        """Update API usage metrics for monitoring rate limits and costs."""
        # This is already being tracked in llm_api_metrics in the initialization
        # Just log current status
        if self.llm_api_metrics["total_calls"] > 0:
            success_rate = (self.llm_api_metrics["successful_calls"] / 
                           max(1, self.llm_api_metrics["total_calls"])) * 100
            self.logger.gauge("sentiment_analysis_model.llm_api_success_rate", success_rate)
            
            if "total_tokens" in self.llm_api_metrics and self.llm_api_metrics["total_tokens"] > 0:
                self.logger.gauge("sentiment_analysis_model.llm_api_total_tokens", self.llm_api_metrics["total_tokens"])
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory growth."""
        # Clean up error timestamps older than 24 hours
        current_time = time.time()
        day_ago = current_time - 86400  # 24 hours in seconds
        
        self.performance_metrics["error_timestamps"] = [
            ts for ts in self.performance_metrics["error_timestamps"] if ts > day_ago
        ]
        
        # Clean up slow operations older than 12 hours
        half_day_ago = current_time - 43200  # 12 hours in seconds
        
        self.performance_metrics["slow_operations"] = [
            op for op in self.performance_metrics["slow_operations"] 
            if op.get("timestamp", 0) > half_day_ago
        ]
        
    async def _save_performance_metrics(self):
        """Save performance metrics to Redis for external monitoring."""
        try:
            if self.redis_mcp is None:
                return
            
            # Create summary of performance metrics for Redis
            metrics_summary = {
                "updated_at": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                "cpu_usage": self.performance_metrics["cpu_usage_history"][-1] if self.performance_metrics["cpu_usage_history"] else None,
                "memory_usage": self.performance_metrics["memory_usage_history"][-1] if self.performance_metrics["memory_usage_history"] else None,
                "peak_memory_usage": self.performance_metrics["peak_memory_usage"],
                "avg_response_time": sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0,
                "recent_errors_count": len(self.performance_metrics["error_timestamps"]),
                "slow_operations_count": len(self.performance_metrics["slow_operations"]),
                "texts_analyzed_count": self.texts_analyzed_count,
                "entities_extracted_count": self.entities_extracted_count,
                "sentiment_scores": {
                    "positive": self.positive_sentiment_count,
                    "neutral": self.neutral_sentiment_count,
                    "negative": self.negative_sentiment_count,
                    "total": self.sentiment_scores_generated_count
                },
                "symbols_processed_count": len(self.performance_metrics["symbols_processed"]),
                "sources_processed_count": len(self.performance_metrics["sources_processed"]),
                "trends": {
                    name: {"trend": data["trend"], "anomalies_count": len(data["anomalies"])}
                    for name, data in getattr(self, "performance_trends", {}).items()
                }
            }
            
            # Store in Redis
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("set_json", {
                "key": "sentiment:health:performance",
                "value": metrics_summary,
                "expiry": 3600  # 1 hour expiry
            })
            
            # Log successful save
            self.logger.info("Performance metrics saved to Redis")
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics to Redis: {e}")
            self.mcp_tool_error_count += 1
    
    async def _generate_performance_visualizations(self):
        """Generate visualizations of performance metrics for monitoring dashboard."""
        try:
            if not self.chart_generator:
                self.logger.warning("Chart generator not available for generating visualizations")
                return
            
            output_dir = "logs/sentiment_performance_charts"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate CPU and memory usage chart
            if len(self.performance_metrics["cpu_usage_history"]) > 5 and len(self.performance_metrics["memory_usage_history"]) > 5:
                # Prepare data
                times = [(datetime.now() - timedelta(minutes=i)).strftime('%H:%M') 
                         for i in range(len(self.performance_metrics["cpu_usage_history"]) - 1, -1, -1)]
                cpu_values = self.performance_metrics["cpu_usage_history"]
                memory_values = self.performance_metrics["memory_usage_history"]
                
                # Generate chart
                chart_path = os.path.join(output_dir, "resource_usage.png")
                self.chart_generator.create_dual_line_chart(
                    times, cpu_values, memory_values,
                    "System Resource Usage", "Time", "Percentage",
                    "CPU Usage", "Memory Usage", chart_path
                )
                self.logger.info(f"Generated resource usage chart at {chart_path}")
            
            # Generate sentiment distribution chart
            if self.sentiment_scores_generated_count > 0:
                # Prepare data
                labels = ["Positive", "Neutral", "Negative"]
                values = [
                    self.positive_sentiment_count,
                    self.neutral_sentiment_count,
                    self.negative_sentiment_count
                ]
                
                # Generate chart
                chart_path = os.path.join(output_dir, "sentiment_distribution.png")
                self.chart_generator.create_pie_chart(
                    labels, values, "Sentiment Distribution", chart_path
                )
                self.logger.info(f"Generated sentiment distribution chart at {chart_path}")
            
            # Generate hourly distribution chart
            hour_values = list(self.performance_metrics["hourly_distribution"].values())
            if sum(hour_values) > 0:
                # Prepare data
                labels = list(self.performance_metrics["hourly_distribution"].keys())
                
                # Generate chart
                chart_path = os.path.join(output_dir, "hourly_distribution.png")
                self.chart_generator.create_bar_chart(
                    labels, hour_values, 
                    "Hourly Distribution of Sentiment Analysis", 
                    "Hour of Day", "Analysis Count",
                    chart_path
                )
                self.logger.info(f"Generated hourly distribution chart at {chart_path}")
            
            self.logger.info("Performance visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Error generating performance visualizations: {e}", exc_info=True)
            self.execution_errors += 1
            
    async def _load_symbol_entity_mapping(self):
        """Attempt to load the symbol-entity mapping from Redis."""
        try:
            if not hasattr(self, 'redis_keys'):
                self.logger.warning("redis_keys attribute not found. Skipping symbol-entity mapping load.")
                return

            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool("get_json", {"key": self.redis_keys["symbol_entity_mapping"]})
            if result and not result.get("error") and result.get("value"):
                self.symbol_entity_mapping = result.get("value")
                self.logger.info("Loaded symbol-entity mapping from Redis.")
            elif result and result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.warning(f"Failed to load symbol-entity mapping from Redis: {result.get('error')}")
            else:
                self.logger.info("No symbol-entity mapping found in Redis. Using default.")
        except AttributeError:
            self.logger.warning("redis_mcp attribute not found. Skipping symbol-entity mapping load.")
        except Exception as e:
            self.logger.error(f"Error loading symbol-entity mapping from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")


    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen from the main configuration.
        """
        # Get LLM configuration from model settings in the main config
        model_settings = self.config.get("model_settings", {})
        llm_config = self.config.get("llm_config", {})

        # Default configuration if not provided in model_settings
        if not llm_config:
            config_list = [
                {
                    "model": model_settings.get("default_model", "anthropic/claude-3-opus-20240229"),
                    "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_type": "openai",
                    "api_version": None,
                }
            ]
        else:
            config_list = llm_config.get("config_list", [])

        # Use model settings parameters with fallbacks
        temperature = model_settings.get("temperature", 0.1)
        max_tokens = model_settings.get("max_tokens", 2000)
        top_p = model_settings.get("top_p", 0.9)
        timeout = llm_config.get("timeout", 600)

        return {
            "config_list": config_list,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "timeout": timeout,
            "seed": 42,  # Adding seed for reproducibility
        }

    def _setup_agents(self) -> Dict[str, Agent]:
        """
        Initialize AutoGen agents for sentiment analysis.
        """
        agents = {}

        # Create the sentiment analysis assistant agent
        agents["sentiment_assistant"] = AssistantAgent(
            name="SentimentAssistantAgent",
            system_message="""You are a financial sentiment analysis specialist. Your role is to:
            1. Analyze news, social media, and other text sources to extract sentiment
            2. Identify relevant entities (companies, products, people, etc.)
            3. Determine sentiment scores for overall text and specific entities
            4. Provide insights on how sentiment might impact trading decisions
 5. Publish sentiment analysis reports to the sentiment analysis reports stream for other models to consume.

            You have tools for extracting entities, scoring sentiment, and retrieving historical sentiment data.
            Always provide clear reasoning for your sentiment assessments.""",
            llm_config=self.llm_config,
            description="A specialist in financial sentiment analysis",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="SentimentToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        agents["user_proxy"] = user_proxy

        return agents

    # --- Tool Methods (to be registered) ---

    async def _tool_extract_entities(self, text: str) -> Dict[str, Any]:
        """Internal method for entity extraction tool."""
        result = await self._call_entity_extraction(text)
        return result or {"entities": [], "error": "Entity extraction failed"}

    async def _tool_score_sentiment(
        self, text: str, entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Internal method for sentiment scoring tool."""
        result = await self._call_sentiment_scoring(text, entities)
        return result or {
            "overall_sentiment": None,
            "entity_sentiments": {},
            "error": "Sentiment scoring failed",
        }

    async def _tool_analyze_text(self, text: str) -> Dict[str, Any]:
        """Internal method for full text analysis tool."""
        result = await self._analyze_single_text(text)
        return result

    async def _tool_process_news_batch(
        self, news_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Internal method for batch processing tool."""
        result = await self.process_news_batch(news_items)
        return result

    async def _tool_get_symbol_sentiment(
        self, symbol: str, lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """Internal method for symbol sentiment retrieval tool."""
        result = await self.get_symbol_sentiment(symbol, lookback_hours)
        return result

    async def _tool_get_all_symbols_sentiment(
        self, lookback_hours: int = 24,
    ) -> Dict[str, Dict[str, Any]]:
        """Internal method for retrieving all symbols' sentiment."""
        result = await self.get_all_symbols_sentiment(lookback_hours)
        return result

    async def _tool_get_sentiment_history(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """Internal method for sentiment history retrieval tool."""
        result = await self.get_sentiment_history(symbol, days)
        return result

    async def _tool_update_symbol_entity_mapping(
        self, mapping: Dict[str, str],
    ) -> Dict[str, Any]:
        """Internal method for updating symbol-entity mapping."""
        result = await self.update_symbol_entity_mapping(mapping)
        return result

    async def _tool_fetch_latest_news(
        self, symbol: Optional[str] = None, topic: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        """Internal method for fetching latest news."""
        result = await self.fetch_latest_news(symbol, topic, limit)
        return result

    def _tool_get_selection_data(self) -> Dict[str, Any]:
        """Internal method for getting selection data."""
        return self.get_selection_data()

    def _tool_send_feedback_to_selection(self, sentiment_data: Dict[str, Any]) -> bool:
        """Internal method for sending feedback to selection model."""
        return self.send_feedback_to_selection(sentiment_data)

    def _tool_use_financial_data_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Internal method for using FinancialDataMCP tools."""
        self.mcp_tool_call_count += 1
        result = self.financial_data_mcp.call_tool(tool_name, arguments)
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    # --- Registration Methods ---

    def _register_functions(self, user_proxy_agent: Optional[UserProxyAgent] = None):
        """
        Register functions with the user proxy agent.

        Args:
            user_proxy_agent: Optional UserProxyAgent instance to register functions with.
                              If None, uses the default user_proxy agent created internally.
        """
        # Use the provided user_proxy_agent if available, otherwise use the internal one
        user_proxy = user_proxy_agent if user_proxy_agent is not None else self.agents["user_proxy"]
        sentiment_assistant = self.agents["sentiment_assistant"]

        # Register entity extraction functions with type annotations
        def extract_entities_func(text: str) -> Dict[str, Any]:
            return self._tool_extract_entities(text)
            
        # Register sentiment scoring functions with type annotations
        register_function(
            self.score_sentiment_func,
            name="score_sentiment",
            description="Score sentiment for text and entities",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register full analysis function with type annotations
        def analyze_text_func(text: str) -> Dict[str, Any]:
            return self._tool_analyze_text(text)
            
        register_function(
            analyze_text_func,
            name="analyze_text",
            description="Perform full sentiment analysis on text",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register batch processing function
        def process_news_batch_func(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
            return self._tool_process_news_batch(news_items)
            
        register_function(
            process_news_batch_func,
            name="process_news_batch",
            description="Process a batch of news items",
            caller=sentiment_assistant,
            executor=user_proxy,
        )
        # Register symbol sentiment retrieval
        async def get_symbol_sentiment_func(symbol: str, lookback_hours: int = 24) -> Dict[str, Any]:
            return await self._tool_get_symbol_sentiment(symbol, lookback_hours)
            
        register_function(
            get_symbol_sentiment_func,
            name="get_symbol_sentiment",
            description="Get aggregated sentiment for a symbol from Redis",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register all symbols sentiment retrieval
        async def get_all_symbols_sentiment_func(lookback_hours: int = 24) -> Dict[str, Dict[str, Any]]:
            return await self._tool_get_all_symbols_sentiment(lookback_hours)
            
        register_function(
            get_all_symbols_sentiment_func,
            name="get_all_symbols_sentiment",
            description="Get sentiment for all symbols from Redis",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register sentiment history retrieval
        async def get_sentiment_history_func(symbol: str, days: int = 7) -> Dict[str, Any]:
            return await self._tool_get_sentiment_history(symbol, days)
            
        register_function(
            get_sentiment_history_func,
            name="get_sentiment_history",
            description="Get historical sentiment for a symbol from Redis",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register symbol-entity mapping update
        async def update_symbol_entity_mapping_func(mapping: Dict[str, str]) -> Dict[str, Any]:
            return await self._tool_update_symbol_entity_mapping(mapping)
            
        register_function(
            update_symbol_entity_mapping_func,
            name="update_symbol_entity_mapping",
            description="Update the mapping between entities and symbols and store in Redis",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register direct news fetching functions
        async def fetch_latest_news_func(symbol: Optional[str] = None, topic: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
            return await self._tool_fetch_latest_news(symbol, topic, limit)
            
        register_function(
            fetch_latest_news_func,
            name="fetch_latest_news",
            description="Fetch latest news for a symbol or topic using FinancialDataMCP",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register selection model integration functions
        def get_selection_data_func() -> Dict[str, Any]:
            return self._tool_get_selection_data()
            
        register_function(
            get_selection_data_func,
            name="get_selection_data",
            description="Get data from the Selection Model from Redis",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        def send_feedback_to_selection_func(sentiment_data: Dict[str, Any]) -> bool:
            return self._tool_send_feedback_to_selection(sentiment_data)
            
        register_function(
            send_feedback_to_selection_func,
            name="send_feedback_to_selection",
            description="Send sentiment feedback to the Selection Model via Redis Stream",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Register MCP tool access functions
        self._register_mcp_tool_access(user_proxy_agent)

    def _register_mcp_tool_access(self, user_proxy_agent: Optional[UserProxyAgent] = None):
        """
        Register MCP tool access functions with the user proxy agent.

        Args:
            user_proxy_agent: Optional UserProxyAgent instance to register functions with.

                               If None, uses the default user_proxy agent created internally.
        """
        # Use the provided user_proxy_agent if available, otherwise use the internal one
        user_proxy = user_proxy_agent if user_proxy_agent is not None else self.agents["user_proxy"]
        sentiment_assistant = self.agents["sentiment_assistant"]

        # Define MCP tool access functions for consolidated MCPs
        def use_financial_data_tool_func(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self._tool_use_financial_data_tool(tool_name, arguments)
            
        register_function(
            use_financial_data_tool_func,
            name="use_financial_data_tool",
            description="Use a tool provided by the Financial Data MCP server (for news, social, entity extraction, sentiment scoring, etc.)",
            caller=sentiment_assistant,
            executor=user_proxy,
        )

        # Old MCP tool access functions removed

    async def _call_entity_extraction(self, text: str) -> Optional[Dict[str, Any]]:
        """Helper to call the entity extraction service via FinancialTextMCP."""
        try:
            # Use the financial_text_mcp to extract entities
            # Assuming the tool name for entity extraction is 'extract_entities'
            self.mcp_tool_call_count += 1
            result = self.financial_data_mcp.call_tool("extract_entities", {"text": text})

            if result and not result.get("error"):
                return result
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(
                    f"Entity extraction MCP call failed: {result.get('error')}"
                )
                return None
        except Exception as e:
            self.mcp_tool_error_count += 1
            self.logger.error(
                f"Error calling entity extraction service: {e}", exc_info=True
            )
            return None

    async def _call_sentiment_scoring(
        self, text: str, entities: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Helper to call the sentiment scoring service via FinancialTextMCP."""
        try:
            # Use the financial_text_mcp to analyze sentiment
            # Assuming the tool name for sentiment analysis is 'analyze_sentiment'
            self.mcp_tool_call_count += 1
            result = self.financial_data_mcp.call_tool(
                "analyze_sentiment", {"text": text, "entities": entities}
            )

            if result and not result.get("error"):
                return result
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(
                    f"Sentiment scoring MCP call failed: {result.get('error')}"
                )
                return None
        except Exception as e:
            self.mcp_tool_error_count += 1
            self.logger.error(
                f"Error calling sentiment scoring service: {e}", exc_info=True
            )
            return None

    async def analyze_sentiment(
        self, text_data: Union[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes sentiment for given text data using configured MCP clients.

        Args:
            text_data: A single string or a list of strings containing the text to analyze.

        Returns:
            A list of dictionaries, each containing the analysis results for one input text:
            {
                "input_text": str,
                "overall_sentiment": {"label": str, "score": float} | None,
                "entities": [
                    {"text": str, "type": str, "sentiment": {"label": str, "score": float}}
                ],
                "error": Optional[str]
            }
            Returns None for sentiment fields if the respective MCP call fails.
        """
        if isinstance(text_data, str):
            texts = [text_data]
        elif isinstance(text_data, list):
            texts = text_data
        else:
            self.logger.error(f"Invalid input type for text_data: {type(text_data)}")
            raise TypeError("text_data must be a string or a list of strings.")

        #
        # Use asyncio.gather to run analyses concurrently if multiple texts are
        # provided
        tasks = [self._analyze_single_text(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling potential exceptions from gather
        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.error(
                    f"Exception during sentiment analysis for text index {i}: {res}",
                    exc_info=res,
                )
                final_results.append(
                    {
                        "input_text": texts[i],
                        "overall_sentiment": None,
                        "entities": [],
                        "error": f"An unexpected error occurred during analysis: {str(res)}",
                    }
                )
            else:
                final_results.append(res)

        return final_results

    async def _analyze_single_text(self, text: str) -> Dict[str, Any]:
        """Analyzes a single piece of text."""
        start_time = time.time()
        
        analysis_result = {
            "input_text": text,
            "overall_sentiment": None,
            "entities": [],
            "error": None,
            "timestamp": datetime.now().isoformat() # Add timestamp
        }
        entity_error = None
        sentiment_error = None
        
        # Track this operation in hourly distribution
        current_hour = datetime.now().hour
        with self.monitoring_lock:
            self.performance_metrics["hourly_distribution"][str(current_hour)] += 1
            self.logger.gauge(f"sentiment_analysis_model.hourly_analysis_count_{current_hour}", 
                             self.performance_metrics["hourly_distribution"][str(current_hour)])

        try:
            # 1. Extract Entities
            entity_start_time = time.time()
            entity_result = await self._call_entity_extraction(text)
            entity_duration = (time.time() - entity_start_time) * 1000
            
            # Track entity extraction performance
            with self.monitoring_lock:
                if "processing_times_by_function" not in self.performance_metrics:
                    self.performance_metrics["processing_times_by_function"] = {}
                if "entity_extraction" not in self.performance_metrics["processing_times_by_function"]:
                    self.performance_metrics["processing_times_by_function"]["entity_extraction"] = []
                
                self.performance_metrics["processing_times_by_function"]["entity_extraction"].append({
                    "timestamp": time.time(),
                    "duration_ms": entity_duration,
                    "text_length": len(text)
                })
                
                # Keep history manageable
                if len(self.performance_metrics["processing_times_by_function"]["entity_extraction"]) > 100:
                    self.performance_metrics["processing_times_by_function"]["entity_extraction"] = self.performance_metrics["processing_times_by_function"]["entity_extraction"][-100:]
                
                # Track as slow operation if it exceeds threshold
                if entity_duration > self.slow_operation_threshold:
                    self.performance_metrics["slow_operations"].append({
                        "timestamp": time.time(),
                        "operation": "entity_extraction",
                        "duration_ms": entity_duration,
                        "details": {"text_length": len(text)}
                    })
            
            extracted_entities = []
            entity_texts = []
            entity_map = {}
            if entity_result and entity_result.get("entities"):
                extracted_entities = entity_result["entities"]
                entity_texts = [
                    entity["text"]
                    for entity in extracted_entities
                    if entity.get("text")
                ]
                entity_map = {
                    entity["text"]: entity
                    for entity in extracted_entities
                    if entity.get("text")
                }
                analysis_result["entities"] = extracted_entities # Store extracted entities
                
                # Track processed entities
                with self.monitoring_lock:
                    for entity in extracted_entities:
                        if entity.get("text"):
                            self.performance_metrics["entities_processed"].add(entity["text"])
                
            elif entity_result is None or entity_result.get("error"):
                entity_error = f"Entity extraction failed: {entity_result.get('error', 'Unknown error') if entity_result else 'Service call failed'}"
                self.logger.warning(f"{entity_error} for text: '{text[:50]}...'")
                analysis_result["error"] = entity_error # Add entity error to result
                
                # Track error
                with self.monitoring_lock:
                    self.performance_metrics["error_timestamps"].append(time.time())
                    self.logger.counter("sentiment_analysis_model.entity_extraction_errors")

            # 2. Score Sentiment (Overall and for Entities if any)
            sentiment_start_time = time.time()
            sentiment_result = await self._call_sentiment_scoring(
                text, entities=entity_texts if entity_texts else None
            )
            sentiment_duration = (time.time() - sentiment_start_time) * 1000
            
            # Track sentiment scoring performance
            with self.monitoring_lock:
                if "sentiment_scoring" not in self.performance_metrics["processing_times_by_function"]:
                    self.performance_metrics["processing_times_by_function"]["sentiment_scoring"] = []
                
                self.performance_metrics["processing_times_by_function"]["sentiment_scoring"].append({
                    "timestamp": time.time(),
                    "duration_ms": sentiment_duration,
                    "text_length": len(text),
                    "entity_count": len(entity_texts) if entity_texts else 0
                })
                
                # Keep history manageable
                if len(self.performance_metrics["processing_times_by_function"]["sentiment_scoring"]) > 100:
                    self.performance_metrics["processing_times_by_function"]["sentiment_scoring"] = self.performance_metrics["processing_times_by_function"]["sentiment_scoring"][-100:]
                
                # Track as slow operation if it exceeds threshold
                if sentiment_duration > self.slow_operation_threshold:
                    self.performance_metrics["slow_operations"].append({
                        "timestamp": time.time(),
                        "operation": "sentiment_scoring",
                        "duration_ms": sentiment_duration,
                        "details": {"text_length": len(text), "entity_count": len(entity_texts) if entity_texts else 0}
                    })

            if sentiment_result:
                analysis_result["overall_sentiment"] = sentiment_result.get(
                    "overall_sentiment"
                )

                # Merge entity sentiment with extracted entity details
                entity_sentiments = sentiment_result.get("entity_sentiments", {})
                processed_entities = []
                # Use the entity_map created earlier
                for entity_text, sentiment_score in entity_sentiments.items():
                    entity_detail = entity_map.get(
                        entity_text, {"text": entity_text, "type": "UNKNOWN"}
                    )  # Fallback
                    processed_entities.append(
                        {
                            "text": entity_text,
                            "type": entity_detail.get("type"),
                            "sentiment": sentiment_score,
                        }
                    )
                analysis_result["entities"] = processed_entities # Update entities with sentiment
            else:
                sentiment_error = f"Sentiment scoring failed: {sentiment_result.get('error', 'Unknown error') if sentiment_result else 'Service call failed'}"
                self.logger.warning(f"{sentiment_error} for text: '{text[:50]}...'")
                if analysis_result["error"]: # Append to existing error if entity extraction also failed
                     analysis_result["error"] += f"; {sentiment_error}"
                else:
                     analysis_result["error"] = sentiment_error
                     
                # Track error
                with self.monitoring_lock:
                    self.performance_metrics["error_timestamps"].append(time.time())
                    self.logger.counter("sentiment_analysis_model.sentiment_scoring_errors")

            # 3. Store and Publish Result
            if not analysis_result["error"]: # Only store/publish if no errors in analysis steps
                 store_start_time = time.time()
                 await self._store_sentiment_result(analysis_result)
                 await self._publish_sentiment_report(analysis_result)
                 store_duration = (time.time() - store_start_time) * 1000
                 
                 # Track storage performance
                 with self.monitoring_lock:
                     if "result_storage" not in self.performance_metrics["processing_times_by_function"]:
                         self.performance_metrics["processing_times_by_function"]["result_storage"] = []
                     
                     self.performance_metrics["processing_times_by_function"]["result_storage"].append({
                         "timestamp": time.time(),
                         "duration_ms": store_duration,
                         "entity_count": len(analysis_result.get("entities", []))
                     })
                     
                     # Track counters and distributions
                     self.texts_analyzed_count += 1
                     self.logger.counter("sentiment_analysis_model.texts_analyzed")
                     
                     self.entities_extracted_count += len(analysis_result.get("entities", []))
                     self.logger.counter("sentiment_analysis_model.entities_extracted", 
                                       count=len(analysis_result.get("entities", [])))
                     
                     # Extract symbols from entities and track
                     symbols = self._extract_symbols_from_entities(analysis_result.get("entities", []))
                     for symbol in symbols:
                         self.performance_metrics["symbols_processed"].add(symbol)
                     
                     # Update sentiment distribution
                     if analysis_result.get("overall_sentiment"):
                          self.sentiment_scores_generated_count += 1
                          self.logger.counter("sentiment_analysis_model.sentiment_scores_generated")
                          
                          score = analysis_result["overall_sentiment"].get("score", 0)
                          if score > 0.2: 
                              self.positive_sentiment_count += 1
                              self.performance_metrics["sentiment_distribution"]["positive"] += 1
                              self.logger.counter("sentiment_analysis_model.positive_sentiment")
                          elif score < -0.2: 
                              self.negative_sentiment_count += 1
                              self.performance_metrics["sentiment_distribution"]["negative"] += 1
                              self.logger.counter("sentiment_analysis_model.negative_sentiment")
                          else: 
                              self.neutral_sentiment_count += 1
                              self.performance_metrics["sentiment_distribution"]["neutral"] += 1
                              self.logger.counter("sentiment_analysis_model.neutral_sentiment")

        except Exception as e:
            self.logger.error(
                f"Unexpected error during single text analysis: '{text[:100]}...': {e}",
                exc_info=True,
            )
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            analysis_result["error"] = (
                f"An unexpected internal error occurred: {str(e)}"
            )
            
            # Track error
            with self.monitoring_lock:
                self.performance_metrics["error_timestamps"].append(time.time())

        # Calculate and log total duration
        total_duration = (time.time() - start_time) * 1000
        self.logger.timing("sentiment_analysis_model.analyze_text_duration_ms", total_duration)
        
        # Add to response time metrics
        with self.monitoring_lock:
            self.performance_metrics["response_times"].append(total_duration)
            
            # Log as slow operation if very slow
            if total_duration > self.slow_operation_threshold * 2:  # Extra slow threshold
                self.logger.warning(f"Very slow text analysis: {total_duration:.2f}ms for text length {len(text)}")

        return analysis_result

    async def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report with trend analysis and recommendations.
        
        Returns:
            Dictionary containing detailed health information including trends and anomalies
        """
        try:
            # Create a detailed health report
            report = {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                "health_status": self.health_data["health_status"],
                "health_score": self.health_data["health_score"],
                "component_health": self.health_data["component_health"].copy(),
                "recent_issues_count": len(self.health_data["recent_issues"]),
                "recent_issues": self.health_data["recent_issues"][-5:],  # Last 5 issues
                "resources": {
                    "current_cpu": self.performance_metrics["cpu_usage_history"][-1] if self.performance_metrics["cpu_usage_history"] else None,
                    "current_memory": self.performance_metrics["memory_usage_history"][-1] if self.performance_metrics["memory_usage_history"] else None,
                    "peak_memory_usage": self.performance_metrics["peak_memory_usage"],
                    "slow_operations_count": len(self.performance_metrics["slow_operations"]),
                    "slow_operations": self.performance_metrics["slow_operations"][-5:] if self.performance_metrics["slow_operations"] else []
                },
                "performance": {
                    "avg_response_time_ms": sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0,
                    "texts_analyzed_count": self.texts_analyzed_count,
                    "entities_extracted_count": self.entities_extracted_count,
                    "sentiment_scores_generated_count": self.sentiment_scores_generated_count
                },
                "sentiment_distribution": {
                    "positive": self.positive_sentiment_count,
                    "neutral": self.neutral_sentiment_count,
                    "negative": self.negative_sentiment_count,
                    "positive_ratio": self.positive_sentiment_count / max(1, self.sentiment_scores_generated_count),
                    "negative_ratio": self.negative_sentiment_count / max(1, self.sentiment_scores_generated_count),
                    "neutral_ratio": self.neutral_sentiment_count / max(1, self.sentiment_scores_generated_count)
                },
                "entity_stats": {
                    "unique_entities_count": len(self.performance_metrics["entities_processed"]),
                    "unique_symbols_count": len(self.performance_metrics["symbols_processed"]),
                    "unique_sources_count": len(self.performance_metrics["sources_processed"])
                }
            }
            
            # Add trend analysis data if available
            if hasattr(self, "performance_trends"):
                report["trends"] = {}
                for metric_name, trend_data in self.performance_trends.items():
                    report["trends"][metric_name] = {
                        "trend": trend_data["trend"],
                        "anomalies_count": len(trend_data["anomalies"]),
                        "latest_anomalies": trend_data["anomalies"][-3:] if trend_data["anomalies"] else []
                    }
            
            # Add recommendations based on health data
            report["recommendations"] = await self._generate_health_recommendations()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}", exc_info=True)
            return {
                "error": f"Error generating health report: {str(e)}",
                "health_status": self.health_data["health_status"],
                "health_score": self.health_data["health_score"]
            }
    
    async def _generate_health_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on health and performance data."""
        recommendations = []
        
        # Check if any components are in error or degraded state
        for component, status in self.health_data["component_health"].items():
            if status == "error":
                recommendations.append({
                    "priority": "high",
                    "component": component,
                    "issue": f"{component} is in error state",
                    "recommendation": f"Restart or check {component} immediately"
                })
            elif status == "degraded":
                recommendations.append({
                    "priority": "medium",
                    "component": component,
                    "issue": f"{component} is in degraded state",
                    "recommendation": f"Investigate {component} performance issues"
                })
        
        # Check for high memory usage
        if (self.performance_metrics["memory_usage_history"] and 
            self.performance_metrics["memory_usage_history"][-1] > self.high_memory_threshold):
            recommendations.append({
                "priority": "high",
                "component": "system_resources",
                "issue": f"High memory usage ({self.performance_metrics['memory_usage_history'][-1]}%)",
                "recommendation": "Investigate memory usage and consider restarting the service"
            })
        
        # Check for high CPU usage
        if (self.performance_metrics["cpu_usage_history"] and 
            self.performance_metrics["cpu_usage_history"][-1] > self.high_cpu_threshold):
            recommendations.append({
                "priority": "medium",
                "component": "system_resources",
                "issue": f"High CPU usage ({self.performance_metrics['cpu_usage_history'][-1]}%)",
                "recommendation": "Check for CPU-intensive operations and optimize if possible"
            })
        
        # Check for slow operations
        if len(self.performance_metrics["slow_operations"]) > 10:
            from collections import Counter
            operation_counter = Counter([op["operation"] for op in self.performance_metrics["slow_operations"]])
            most_common = operation_counter.most_common(1)
            if most_common:
                operation, count = most_common[0]
                recommendations.append({
                    "priority": "medium",
                    "component": "performance",
                    "issue": f"Frequent slow operations detected ({count} occurrences of '{operation}')",
                    "recommendation": f"Optimize the '{operation}' operation"
                })
        
        # Check for skewed sentiment distribution
        if self.sentiment_scores_generated_count > 50:  # Only check if we have enough data
            positive_ratio = self.positive_sentiment_count / max(1, self.sentiment_scores_generated_count)
            negative_ratio = self.negative_sentiment_count / max(1, self.sentiment_scores_generated_count)
            
            if positive_ratio > 0.8 or negative_ratio > 0.8:
                recommendations.append({
                    "priority": "medium",
                    "component": "sentiment_analysis",
                    "issue": f"Highly skewed sentiment distribution (positive: {positive_ratio:.2f}, negative: {negative_ratio:.2f})",
                    "recommendation": "Check sentiment analysis calibration and entity detection accuracy"
                })
        
        # Check for high error rate
        error_count = len(self.performance_metrics["error_timestamps"])
        total_operations = max(1, self.texts_analyzed_count + self.entities_extracted_count)
        error_rate = error_count / total_operations
        if error_rate > 0.1:  # More than 10% error rate
            recommendations.append({
                "priority": "high",
                "component": "error_handling",
                "issue": f"High error rate ({error_rate:.2%})",
                "recommendation": "Review error logs and fix common error causes"
            })
        
        # Check MCP health implications
        component_issues = []
        if self.health_data["component_health"]["financial_text_mcp"] in ["error", "degraded"]:
            component_issues.append("entity extraction and sentiment scoring")
        if self.health_data["component_health"]["financial_data_mcp"] in ["error", "degraded"]:
            component_issues.append("news and data retrieval")
        if component_issues:
            recommendations.append({
                "priority": "high",
                "component": "mcp_integration",
                "issue": f"MCP issues affecting {', '.join(component_issues)}",
                "recommendation": "Check MCP servers and connections"
            })
        
        return recommendations
        
    async def shutdown(self) -> Dict[str, Any]:
        """
        Perform a graceful shutdown of the sentiment analysis model.
        
        This includes:
        - Setting the shutdown_requested flag to stop background threads
        - Shutting down MCP clients
        - Stopping system metrics collection
        - Generating a final health report
        - Saving final performance metrics to Redis
        
        Returns:
            Dictionary with shutdown status and results
        """
        self.logger.info("Starting graceful shutdown of Sentiment Analysis Model...")
        start_time = time.time()
        
        # Set shutdown flag to stop background threads
        self.shutdown_requested = True
        self.logger.info("Shutdown flag set, background threads will terminate")
        
        # Generate final health report and save performance metrics
        try:
            final_health_report = await self.get_health_report()
            self.logger.info("Generated final health report", 
                           health_status=final_health_report.get("health_status"),
                           health_score=final_health_report.get("health_score"))
            
            # Save final performance metrics
            await self._save_performance_metrics()
            self.logger.info("Saved final performance metrics to Redis")
        except Exception as e:
            self.logger.error(f"Error generating final reports during shutdown: {e}", exc_info=True)
        
        # Stop system metrics collection
        try:
            if self.system_metrics:
                self.system_metrics.stop()
                self.logger.info("Stopped system metrics collection")
        except Exception as e:
            self.logger.error(f"Error stopping system metrics: {e}")
        
        # Wait for health check thread to terminate (if it exists)
        if hasattr(self, "health_thread") and self.health_thread:
            try:
                self.health_thread.join(timeout=5)  # Wait up to 5 seconds
                if self.health_thread.is_alive():
                    self.logger.warning("Health check thread did not terminate within timeout")
                else:
                    self.logger.info("Health check thread terminated successfully")
            except Exception as e:
                self.logger.error(f"Error waiting for health thread: {e}")
        
        # Shutdown MCP clients if they support shutdown
        for mcp_name, mcp in [
            ("financial_data_mcp", self.financial_data_mcp),
            ("redis_mcp", self.redis_mcp)
        ]:
            if mcp is not None:
                try:
                    if hasattr(mcp, "shutdown"):
                        mcp.shutdown()
                    self.logger.info(f"Shutdown {mcp_name} successfully")
                except Exception as e:
                    self.logger.error(f"Error shutting down {mcp_name}: {e}")
        
        # Publish shutdown event to Redis
        try:
            if self.redis_mcp:
                self.redis_mcp.call_tool("xadd", {
                    "stream": "model:sentiment:events",
                    "data": {
                        "event_type": "model_shutdown",
                        "component": "sentiment_analysis_model",
                        "timestamp": datetime.now().isoformat(),
                        "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                        "texts_analyzed_count": self.texts_analyzed_count,
                        "entities_extracted_count": self.entities_extracted_count,
                        "sentiment_scores_generated_count": self.sentiment_scores_generated_count
                    }
                })
                self.logger.info("Published shutdown event to Redis stream")
        except Exception as e:
            self.logger.error(f"Error publishing shutdown event: {e}")
        
        # Calculate shutdown duration
        duration = (time.time() - start_time) * 1000
        self.logger.info(f"Sentiment Analysis Model shutdown completed in {duration:.2f}ms")
        
        return {
            "status": "success",
            "shutdown_duration_ms": duration,
            "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
            "texts_analyzed_count": self.texts_analyzed_count,
            "entities_extracted_count": self.entities_extracted_count,
            "sentiment_scores_generated_count": self.sentiment_scores_generated_count,
            "message": "Sentiment Analysis Model shutdown completed successfully"
        }
    
    async def process_news_batch(
        self, news_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a batch of news items and analyze sentiment.

        Args:
            news_items: List of news items, each containing at least:
                - text: The news text
                - source: Source of the news
                - published_at: Publication timestamp
                - url: Optional URL to the news

        Returns:
            Dictionary with processing results and statistics
        """
        # Initialize monitoring
        batch_start_time = time.time()
        batch_size = len(news_items)
        
        # Track batch size in performance metrics
        with self.monitoring_lock:
            self.performance_metrics["batch_sizes"].append(batch_size)
            # Keep reasonable history
            if len(self.performance_metrics["batch_sizes"]) > 100:
                self.performance_metrics["batch_sizes"] = self.performance_metrics["batch_sizes"][-100:]
            self.logger.gauge("sentiment_analysis_model.batch_size", batch_size)
        
        self.logger.info(f"Processing batch of {batch_size} news items")

        # Initialize tracking variables
        processed_items = []

        self.logger.info(
            f"Processing batch of {batch_size} news items",
            extra={
                "component": "sentiment_analysis",
                "action": "process_news_batch",
                "batch_size": batch_size,
            },
        )

        # Process each news item
        analysis_tasks = []
        source_set = set()  # Track unique sources for monitoring
        
        # Track batch statistics
        batch_text_total_length = 0
        items_with_text_count = 0
        
        for news_item in news_items:
            text = news_item.get("text", "")
            source = news_item.get("source", "unknown")
            
            # Track sources for monitoring
            if source:
                with self.monitoring_lock:
                    self.performance_metrics["sources_processed"].add(source)
                    source_set.add(source)
            
            if text:
                batch_text_total_length += len(text)
                items_with_text_count += 1
                analysis_tasks.append(self._analyze_single_text(text))
            else:
                self.logger.warning(f"Skipping news item without text: {news_item.get('title', 'Unknown')}")
                # Add a placeholder for skipped items to keep results aligned with input
                processed_items.append({"news_item": news_item, "sentiment_analysis": {"error": "No text provided"}})
        
        # Log batch statistics
        if items_with_text_count > 0:
            avg_text_length = batch_text_total_length / items_with_text_count
            self.logger.info(f"Batch statistics: {items_with_text_count} items with text, avg length: {avg_text_length:.1f} chars, sources: {len(source_set)}")
            self.logger.gauge("sentiment_analysis_model.avg_text_length", avg_text_length)
        else:
            self.logger.warning("No items with text in batch")

        # Run analysis tasks concurrently
        analysis_start_time = time.time()
        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        analysis_duration = (time.time() - analysis_start_time) * 1000
        
        # Log analysis timing
        if items_with_text_count > 0:
            avg_analysis_time = analysis_duration / items_with_text_count
            self.logger.timing("sentiment_analysis_model.batch_avg_analysis_time_ms", avg_analysis_time)
            self.logger.info(f"Completed batch analysis in {analysis_duration:.2f}ms, avg: {avg_analysis_time:.2f}ms per item")

        # Combine news items with analysis results
        news_items_with_analysis = []
        analysis_index = 0
        
        # Tracking variables for batch summary
        successful_analyses = 0
        error_analyses = 0
        entities_found = 0
        symbols_found = set()
        sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        
        for news_item in news_items:
            if news_item.get("text"): # Only process items that had text
                analysis_result = analysis_results[analysis_index]
                analysis_index += 1

                if isinstance(analysis_result, Exception):
                    self.logger.error(f"Exception during analysis of news item: {news_item.get('title', 'Unknown')}: {analysis_result}", exc_info=analysis_result)
                    news_item_with_analysis = {"news_item": news_item, "sentiment_analysis": {"error": f"An unexpected error occurred: {str(analysis_result)}"}}
                    error_analyses += 1
                    
                    # Track error in monitoring
                    with self.monitoring_lock:
                        self.performance_metrics["error_timestamps"].append(time.time())
                else:
                    news_item_with_analysis = {"news_item": news_item, "sentiment_analysis": analysis_result}
                    successful_analyses += 1
                    
                    # Track entities
                    if "entities" in analysis_result:
                        entities_found += len(analysis_result["entities"])
                    
                    # Extract symbols from entities in the analysis result
                    symbols = self._extract_symbols_from_entities(
                        analysis_result.get("entities", [])
                    )
                    if symbols:
                        news_item_with_analysis["related_symbols"] = list(symbols)
                        symbols_found.update(symbols)
                    
                    # Track sentiment distribution for batch summary
                    if analysis_result.get("overall_sentiment") and "score" in analysis_result["overall_sentiment"]:
                        score = analysis_result["overall_sentiment"]["score"]
                        if score > 0.2:
                            sentiment_distribution["positive"] += 1
                        elif score < -0.2:
                            sentiment_distribution["negative"] += 1
                        else:
                            sentiment_distribution["neutral"] += 1

                processed_items.append(news_item_with_analysis)
            # Skipped items (without text) were already added as placeholders

        # Calculate processing time
        processing_time = time.time() - batch_start_time
        
        # Log batch completion metrics
        self.logger.info(f"Batch processing summary: {successful_analyses} successful, {error_analyses} errors, {entities_found} entities, {len(symbols_found)} unique symbols")
        self.logger.gauge("sentiment_analysis_model.batch_success_rate", successful_analyses / max(1, items_with_text_count) * 100)
        self.logger.gauge("sentiment_analysis_model.batch_entities_per_item", entities_found / max(1, successful_analyses))
        
        # Update processing time tracking
        with self.monitoring_lock:
            if "batch_processing" not in self.performance_metrics["processing_times_by_function"]:
                self.performance_metrics["processing_times_by_function"]["batch_processing"] = []
            
            self.performance_metrics["processing_times_by_function"]["batch_processing"].append({
                "timestamp": time.time(),
                "duration_ms": processing_time * 1000,
                "batch_size": batch_size,
                "success_rate": successful_analyses / max(1, items_with_text_count),
                "entities_found": entities_found
            })
            
            # Keep history manageable
            if len(self.performance_metrics["processing_times_by_function"]["batch_processing"]) > 50:
                self.performance_metrics["processing_times_by_function"]["batch_processing"] = self.performance_metrics["processing_times_by_function"]["batch_processing"][-50:]

        # Update latest analysis timestamp in Redis
        self.mcp_tool_call_count += 1
        redis_update_start = time.time()
        self.redis_mcp.call_tool(
            "set_json",
            {
                "key": self.redis_keys["latest_analysis_timestamp"],
                "value": {
                    "timestamp": datetime.now().isoformat(),
                    "count": len(processed_items),
                    "successful": successful_analyses,
                    "errors": error_analyses,
                    "sources": list(source_set),
                    "entities_found": entities_found,
                    "symbols_found": list(symbols_found),
                    "sentiment_distribution": sentiment_distribution,
                    "processing_time_ms": int(processing_time * 1000)
                },
            }
        )
        redis_update_duration = (time.time() - redis_update_start) * 1000
        self.logger.info(f"Updated latest sentiment analysis timestamp in Redis in {redis_update_duration:.2f}ms")
        
        # Track Redis performance
        with self.monitoring_lock:
            if "redis_operations" not in self.performance_metrics["processing_times_by_function"]:
                self.performance_metrics["processing_times_by_function"]["redis_operations"] = []
            
            self.performance_metrics["processing_times_by_function"]["redis_operations"].append({
                "timestamp": time.time(),
                "duration_ms": redis_update_duration,
                "operation": "set_json",
                "key": self.redis_keys["latest_analysis_timestamp"]
            })
            
            # Keep history manageable
            if len(self.performance_metrics["processing_times_by_function"]["redis_operations"]) > 100:
                self.performance_metrics["processing_times_by_function"]["redis_operations"] = self.performance_metrics["processing_times_by_function"]["redis_operations"][-100:]

        # Send feedback to selection model (e.g., overall sentiment for symbols found)
        # Aggregate sentiment by symbol from the batch
        feedback_start_time = time.time()
        aggregated_sentiment = {}
        for item in processed_items:
             sentiment_analysis = item.get("sentiment_analysis", {})
             related_symbols = item.get("related_symbols", [])
             overall_sentiment = sentiment_analysis.get("overall_sentiment", {})
             
             if overall_sentiment and "score" in overall_sentiment:
                  score = overall_sentiment["score"]
                  for symbol in related_symbols:
                       if symbol not in aggregated_sentiment:
                            aggregated_sentiment[symbol] = {"scores": [], "count": 0}
                       aggregated_sentiment[symbol]["scores"].append(score)
                       aggregated_sentiment[symbol]["count"] += 1

        feedback_data = {
            "symbols": {},
            "timestamp": datetime.now().isoformat(),
            "batch_id": str(uuid.uuid4()),
            "batch_size": batch_size,
            "sources": list(source_set)
        }
        
        for symbol, data in aggregated_sentiment.items():
             if data["scores"]:
                  avg_score = sum(data["scores"]) / len(data["scores"])
                  label = "NEUTRAL"
                  if avg_score > 0.2: label = "POSITIVE"
                  elif avg_score < -0.2: label = "NEGATIVE"
                  feedback_data["symbols"][symbol] = {
                       "score": avg_score,
                       "label": label,
                       "item_count": data["count"]
                  }

        if feedback_data["symbols"]:
             self.send_feedback_to_selection(feedback_data)
             feedback_duration = (time.time() - feedback_start_time) * 1000
             self.logger.info(f"Sent sentiment feedback to selection model for {len(feedback_data['symbols'])} symbols in {feedback_duration:.2f}ms")
        
        # Save batch performance data for trend analysis
        with self.monitoring_lock:
            # Update sentiment distributions
            self.performance_metrics["sentiment_distribution"]["positive"] += sentiment_distribution["positive"]
            self.performance_metrics["sentiment_distribution"]["neutral"] += sentiment_distribution["neutral"]
            self.performance_metrics["sentiment_distribution"]["negative"] += sentiment_distribution["negative"]
            
            # Log current sentiment distribution
            total_sentiments = sentiment_distribution["positive"] + sentiment_distribution["neutral"] + sentiment_distribution["negative"]
            if total_sentiments > 0:
                pos_pct = sentiment_distribution["positive"] / total_sentiments * 100
                neg_pct = sentiment_distribution["negative"] / total_sentiments * 100
                neu_pct = sentiment_distribution["neutral"] / total_sentiments * 100
                self.logger.gauge("sentiment_analysis_model.batch_positive_pct", pos_pct)
                self.logger.gauge("sentiment_analysis_model.batch_negative_pct", neg_pct)
                self.logger.gauge("sentiment_analysis_model.batch_neutral_pct", neu_pct)

        # Create detailed result
        result = {
            "processed": len(processed_items),
            "successful": successful_analyses,
            "errors": error_analyses,
            "processing_time_ms": int(processing_time * 1000),
            "entities_found": entities_found,
            "sentiment_distribution": sentiment_distribution,
            "symbols_found": {symbol: 1 for symbol in symbols_found},  # Format compatible with _count_unique_symbols
            "items": processed_items
        }
        
        # Log completion
        self.logger.info(f"Completed batch processing: {successful_analyses}/{batch_size} items in {processing_time*1000:.2f}ms")
        
        return result

    def _extract_symbols_from_entities(
        self, entities: List[Dict[str, Any]]
    ) -> Set[str]:
        """Extract trading symbols from entity list using mapping."""
        symbols = set()

        for entity in entities:
            entity_text = entity.get("text", "").upper()
            entity_type = entity.get("type", "")

            # Direct match if entity is a stock ticker
            if entity_type == "TICKER" or entity_type == "STOCK_TICKER":
                symbols.add(entity_text)

            # Look up in mapping
            elif entity_text in self.symbol_entity_mapping:
                symbols.add(self.symbol_entity_mapping[entity_text])

        return symbols

    def _count_unique_symbols(
        self, processed_items: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Count occurrences of each symbol in processed items."""
        symbol_counts = {}

        for item in processed_items:
            for symbol in item.get("related_symbols", []):
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return symbol_counts

    async def _store_sentiment_result(
        self, sentiment_analysis_result: Dict[str, Any]
    ) -> None:
        """Store sentiment result in Redis."""
        try:
            # Store the full analysis result
            analysis_id = str(uuid.uuid4())
            sentiment_analysis_result["analysis_id"] = analysis_id
            
            # Store overall sentiment data (if available)
            overall_sentiment = sentiment_analysis_result.get("overall_sentiment")
            if overall_sentiment and "score" in overall_sentiment:
                 overall_key = f"{self.redis_keys['sentiment_data']}:{analysis_id}"
                 self.mcp_tool_call_count += 1
                 self.redis_mcp.call_tool(
                      "set_json",
                      {"key": overall_key, "value": overall_sentiment, "expiry": self.sentiment_ttl}
                 )

            # Store entity sentiment data (if available)
            entities = sentiment_analysis_result.get("entities", [])
            for entity in entities:
                 if entity.get("sentiment") and "score" in entity["sentiment"]:
                      entity_key = f"{self.redis_keys['entity_sentiment']}{entity.get('text', 'unknown').replace(' ', '_')}:{analysis_id}"
                      self.mcp_tool_call_count += 1
                      self.redis_mcp.call_tool(
                           "set_json",
                           {"key": entity_key, "value": entity, "expiry": self.sentiment_ttl}
                      )

            # Store symbol sentiment data (if related symbols found)
            related_symbols = self._extract_symbols_from_entities(entities)
            if related_symbols:
                 for symbol in related_symbols:
                      symbol_key = f"{self.redis_keys['symbol_sentiment']}{symbol}:{analysis_id}"
                      # Store a simplified version or reference to the full analysis
                      symbol_sentiment_data = {
                           "analysis_id": analysis_id,
                           "overall_sentiment": overall_sentiment,
                           "entity_sentiment": entity.get("sentiment") if entity.get("text", "").upper() == symbol else None, # Store specific entity sentiment if it matches symbol
                           "timestamp": sentiment_analysis_result.get("timestamp")
                      }
                      self.mcp_tool_call_count += 1
                      self.redis_mcp.call_tool(
                           "set_json",
                           {"key": symbol_key, "value": symbol_sentiment_data, "expiry": self.sentiment_ttl}
                      )

            self.logger.info(f"Stored sentiment analysis result with ID: {analysis_id}")

        except Exception as e:
            self.logger.error(f"Error storing sentiment result in Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")


    async def _publish_sentiment_report(self, sentiment_analysis_result: Dict[str, Any]) -> None:
        """Publish sentiment analysis report to a Redis stream."""
        try:
            # Publish the analysis result to the sentiment reports stream
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                 "xadd",
                 {
                      "stream": self.redis_keys["sentiment_analysis_reports_stream"],
                      "data": sentiment_analysis_result
                 }
            )
            if result and not result.get("error"):
                 self.logger.info("Published sentiment analysis report to stream.")
            else:
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to publish sentiment analysis report to stream: {result.get('error') if result else 'Unknown error'}")

        except Exception as e:
            self.logger.error(f"Error publishing sentiment report: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")


    async def get_symbol_sentiment(
        self, symbol: str, lookback_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get aggregated sentiment for a specific symbol from Redis.

        Args:
            symbol: Trading symbol to get sentiment for
            lookback_hours: Hours to look back for sentiment data

        Returns:
            Aggregated sentiment data for the symbol
        """
        try:
            # Calculate the start time for lookback
            start_time = datetime.now() - timedelta(hours=lookback_hours)
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(datetime.now().timestamp())

            # Get sentiment analyses from Redis for the symbol within the time range
            # Using ZRANGEBYSCORE on a sorted set if history is stored that way,
            # or iterating through keys if stored individually with timestamps.
            # Assuming individual keys with timestamps for now, as per _store_sentiment_result.
            key_pattern = f"{self.redis_keys['symbol_sentiment']}{symbol}:*"
            self.mcp_tool_call_count += 1
            keys_result = self.redis_mcp.call_tool("keys", {"pattern": key_pattern})

            if not keys_result or keys_result.get("error") or not keys_result.get("keys"):
                self.logger.warning(f"No sentiment data found for symbol {symbol}")
                return {
                    "symbol": symbol,
                    "sentiment": None,
                    "entity_count": 0,
                    "item_count": 0, # Renamed from news_count for generality
                    "lookback_hours": lookback_hours,
                    "timestamp": datetime.now().isoformat()
                }

            # Get data for each key and filter by timestamp
            analyses = []
            keys_to_fetch = keys_result.get("keys", [])
            if keys_to_fetch:
                 # Fetch data for multiple keys in one go if RedisMCP supports MGET or similar
                 # For now, fetching individually
                 for key in keys_to_fetch:
                      self.mcp_tool_call_count += 1
                      data_result = self.redis_mcp.call_tool("get_json", {"key": key})
                      data = data_result.get("value") if data_result and not data_result.get("error") else None

                      if data and "timestamp" in data:
                          # Check if the data is within the lookback period
                          try:
                              data_timestamp = datetime.fromisoformat(data["timestamp"]).timestamp()
                              if data_timestamp >= start_timestamp and data_timestamp <= end_timestamp:
                                  analyses.append(data)
                          except (ValueError, TypeError):
                              self.logger.warning(f"Invalid timestamp format in data for key {key}")
                      elif data_result and data_result.get("error"):
                           self.mcp_tool_error_count += 1
                           self.logger.error(f"Failed to get data for key {key}: {data_result.get('error')}")


            # Aggregate the sentiment data
            result = self._aggregate_symbol_sentiment(symbol, analyses, lookback_hours)
            result["item_count"] = len(analyses) # Update item count based on filtered data
            return result

        except Exception as e:
            self.logger.error(f"Error getting sentiment for symbol {symbol}: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {
                "symbol": symbol,
                "sentiment": None,
                "entity_count": 0,
                "item_count": 0,
                "lookback_hours": lookback_hours,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _aggregate_symbol_sentiment(
        self, symbol: str, analyses: List[Dict[str, Any]], lookback_hours: int
    ) -> Dict[str, Any]:
        """Aggregate sentiment from multiple analyses for a symbol."""
        if not analyses:
            return {
                "symbol": symbol,
                "sentiment": None,
                "entity_count": 0,
                "item_count": 0,
                "lookback_hours": lookback_hours,
            }

        # Extract sentiment scores
        overall_scores = []
        entity_scores = []
        sources = set()
        entity_count = 0

        for analysis in analyses:
            # Assuming analysis structure from _store_sentiment_result
            overall = analysis.get("overall_sentiment", {})
            if overall and "score" in overall:
                overall_scores.append(float(overall["score"]))

            # Entity sentiment - need to retrieve full analysis if only simplified stored
            # Or, if _store_sentiment_result stores entity sentiment directly, use that
            # Assuming _store_sentiment_result stores simplified data with entity_sentiment key
            entity_sentiment_data = analysis.get("entity_sentiment")
            if entity_sentiment_data and "sentiment" in entity_sentiment_data and "score" in entity_sentiment_data["sentiment"]:
                 entity_scores.append(float(entity_sentiment_data["sentiment"]["score"]))
                 entity_count += 1 # Count entities with sentiment

            # Track sources (if source info is available in the stored data)
            # Assuming source info is not directly in the stored symbol sentiment data,
            # would need to retrieve full analysis or store source info with symbol sentiment.
            # Skipping source tracking for now based on current stored data assumption.


        # Calculate aggregated sentiment
        avg_overall = (
            sum(overall_scores) / len(overall_scores) if overall_scores else None
        )
        avg_entity = sum(entity_scores) / len(entity_scores) if entity_scores else None

        # Determine sentiment label
        sentiment_label = "NEUTRAL"
        # Prioritize entity sentiment if available, otherwise use overall
        sentiment_score = avg_entity if avg_entity is not None else avg_overall

        if sentiment_score is not None:
            if sentiment_score > 0.2:
                sentiment_label = "POSITIVE"
            elif sentiment_score < -0.2:
                sentiment_label = "NEGATIVE"

        return {
            "symbol": symbol,
            "sentiment": {
                "label": sentiment_label,
                "score": sentiment_score,
                "overall_score": avg_overall,
                "entity_score": avg_entity,
            },
            "entity_count": entity_count,
            "item_count": len(analyses), # Number of analysis items contributing
            "lookback_hours": lookback_hours,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_all_symbols_sentiment(
        self, lookback_hours: int = 24
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get sentiment for all symbols that have recent data from Redis.

        Args:
            lookback_hours: Hours to look back for sentiment data

        Returns:
            Dictionary mapping symbols to their sentiment data
        """
        try:
            # Get all symbol sentiment keys
            key_pattern = f"{self.redis_keys['symbol_sentiment']}*"
            self.mcp_tool_call_count += 1
            keys_result = self.redis_mcp.call_tool("keys", {"pattern": key_pattern})

            if not keys_result or keys_result.get("error") or not keys_result.get("keys"):
                self.logger.warning("No sentiment data found for any symbols")
                return {
                    "symbols": {},
                    "count": 0,
                    "lookback_hours": lookback_hours,
                    "timestamp": datetime.now().isoformat()
                }

            # Extract unique symbols from keys
            symbols = set()
            for key in keys_result.get("keys", []):
                # Extract symbol from key pattern "sentiment:symbol:SYMBOL:*"
                parts = key.split(":")
                if len(parts) >= 3:
                    symbols.add(parts[2])

            # Get sentiment for each symbol concurrently
            result = {"symbols": {}, "count": 0, "lookback_hours": lookback_hours, "timestamp": datetime.now().isoformat()}

            tasks = [self.get_symbol_sentiment(symbol, lookback_hours) for symbol in symbols]
            sentiment_results = await asyncio.gather(*tasks, return_exceptions=True)

            for symbol, sentiment_result in zip(symbols, sentiment_results):
                 if isinstance(sentiment_result, Exception):
                      self.logger.error(f"Error getting sentiment for symbol {symbol} in batch: {sentiment_result}", exc_info=sentiment_result)
                      # Optionally add an error entry for this symbol
                      # result["symbols"][symbol] = {"error": str(sentiment_result)}
                 elif sentiment_result and sentiment_result.get("sentiment"):
                      result["symbols"][symbol] = sentiment_result
                      result["count"] += 1

            return result

        except Exception as e:
            self.logger.error(f"Error getting all symbols sentiment: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {
                "symbols": {},
                "count": 0,
                "lookback_hours": lookback_hours,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    async def update_symbol_entity_mapping(
        self, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Update the mapping between entities and symbols and store in Redis.

        Args:
            mapping: Dictionary mapping entity names to symbols

        Returns:
            Status of the update operation
        """
        try:
            # Update the internal mapping
            self.symbol_entity_mapping.update(mapping)

            # Store in Redis
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                "set_json",
                {"key": self.redis_keys["symbol_entity_mapping"], "value": self.symbol_entity_mapping}
            )

            if result and not result.get("error"):
                self.logger.info("Updated symbol-entity mapping in Redis.")
                return {
                    "status": "success",
                    "updated_count": len(mapping),
                    "total_mappings": len(self.symbol_entity_mapping),
                }
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to update symbol-entity mapping in Redis: {result.get('error') if result else 'Unknown error'}")
                return {"status": "error", "error": result.get('error', 'Failed to update mapping in Redis')}

        except Exception as e:
            self.logger.error(
                f"Error updating symbol-entity mapping: {e}", exc_info=True
            )
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {"status": "error", "error": str(e)}

    async def get_sentiment_history(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get historical sentiment data for a symbol from Redis.

        Args:
            symbol: Trading symbol to get history for
            days: Number of days to look back

        Returns:
            Historical sentiment data by day
        """
        try:
            # Calculate the start time for lookback
            start_time = datetime.now() - timedelta(days=days)
            start_timestamp = int(start_time.timestamp())
            end_timestamp = int(datetime.now().timestamp())

            # Get sentiment history from Redis
            # Assuming history is stored in a sorted set keyed by timestamp
            history_key = f"{self.redis_keys['sentiment_history']}{symbol}"
            self.mcp_tool_call_count += 1
            results = self.redis_mcp.call_tool(
                 "zrangebyscore", # Assuming zrangebyscore tool exists
                 {
                      "key": history_key,
                      "min": start_timestamp,
                      "max": end_timestamp
                 }
            )

            if not results or results.get("error") or not results.get("members"):
                self.logger.warning(f"No sentiment history found for symbol {symbol} in the last {days} days.")
                return {
                    "symbol": symbol,
                    "history": [],
                    "days": days,
                    "timestamp": datetime.now().isoformat()
                }

            # Process and aggregate data by day
            history_by_day = {}
            for item_json in results.get("members", []):
                 try:
                      item = json.loads(item_json)
                      if "timestamp" in item and "sentiment" in item and "score" in item["sentiment"]:
                           item_timestamp = datetime.fromisoformat(item["timestamp"])
                           day_key = item_timestamp.strftime("%Y-%m-%d")
                           if day_key not in history_by_day:
                                history_by_day[day_key] = []
                           history_by_day[day_key].append(item)
                 except (json.JSONDecodeError, ValueError, TypeError) as e:
                      self.logger.warning(f"Skipping invalid history item: {item_json[:100]}... Error: {e}")


            # Aggregate sentiment for each day
            daily_sentiment = []
            for day, items in sorted(history_by_day.items()):
                # Aggregate sentiment scores for the day
                scores = [item.get("sentiment", {}).get("score", 0) for item in items if item.get("sentiment", {}).get("score") is not None]
                avg_score = sum(scores) / len(scores) if scores else None

                # Determine sentiment label
                sentiment_label = "NEUTRAL"
                if avg_score is not None:
                    if avg_score > 0.2:
                        sentiment_label = "POSITIVE"
                    elif avg_score < -0.2:
                        sentiment_label = "NEGATIVE"

                daily_sentiment.append({
                    "date": day,
                    "sentiment": {
                        "label": sentiment_label,
                        "score": avg_score
                    },
                    "count": len(items)
                })

            return {
                "symbol": symbol,
                "history": daily_sentiment,
                "days": days,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting sentiment history for symbol {symbol}: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {
                "symbol": symbol,
                "history": [],
                "days": days,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _get_sentiment_for_timerange(
        self, symbol: str, start_time: int, end_time: int
    ) -> Optional[Dict[str, Any]]:
        """Get sentiment for a specific time range from Redis."""
        # This method is similar to get_sentiment_history but might return raw data or a different aggregation
        # Re-implementing based on the assumption of individual analysis results stored with timestamps
        try:
            key_pattern = f"{self.redis_keys['symbol_sentiment']}{symbol}:*"
            self.mcp_tool_call_count += 1
            keys_result = self.redis_mcp.call_tool("keys", {"pattern": key_pattern})

            if not keys_result or keys_result.get("error") or not keys_result.get("keys"):
                self.logger.warning(f"No sentiment data found for symbol {symbol} in the specified time range.")
                return None

            analyses = []
            keys_to_fetch = keys_result.get("keys", [])
            if keys_to_fetch:
                 for key in keys_to_fetch:
                      self.mcp_tool_call_count += 1
                      data_result = self.redis_mcp.call_tool("get_json", {"key": key})
                      data = data_result.get("value") if data_result and not data_result.get("error") else None

                      if data and "timestamp" in data:
                          try:
                              data_timestamp = datetime.fromisoformat(data["timestamp"]).timestamp()
                              if data_timestamp >= start_time and data_timestamp <= end_time:
                                  analyses.append(data)
                          except (ValueError, TypeError):
                              self.logger.warning(f"Invalid timestamp format in data for key {key}")
                      elif data_result and data_result.get("error"):
                           self.mcp_tool_error_count += 1
                           self.logger.error(f"Failed to get data for key {key}: {data_result.get('error')}")

            if analyses:
                 # Aggregate sentiment for the time range
                 aggregated_sentiment = self._aggregate_symbol_sentiment(symbol, analyses, (end_time - start_time) / 3600) # Calculate hours
                 return aggregated_sentiment
            else:
                 self.logger.warning(f"No sentiment data found for symbol {symbol} within the specified time range after filtering.")
                 return None

        except Exception as e:
            self.logger.error(f"Error getting sentiment for time range for symbol {symbol}: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {"error": str(e)}


    async def fetch_latest_news(
        self, symbol: Optional[str] = None, topic: Optional[str] = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Fetch latest news for a symbol or topic from various sources using FinancialDataMCP.

        Args:
            symbol: Optional stock symbol to fetch news for
            topic: Optional topic to fetch news for
            limit: Maximum number of news items to fetch

        Returns:
            Dictionary with news items from FinancialDataMCP
        """
        results = {"news_items": [], "sources": [], "count": 0}

        try:
            # Use the financial_data_mcp to get news
            # Assuming the tool name for getting news is 'get_news'
            self.mcp_tool_call_count += 1
            news_result = self.financial_data_mcp.call_tool(
                "get_news", {"symbols": [symbol] if symbol else None, "query": topic, "limit": limit}
            )

            if news_result and not news_result.get("error"):
                results["news_items"] = news_result.get("news", []) # Assuming 'news' key in result
                results["sources"] = news_result.get("sources", []) # Assuming 'sources' key in result
                results["count"] = len(results["news_items"])

                # Process the fetched news for sentiment analysis
                if results["news_items"]:
                     # The process_news_batch method expects a list of news items
                     # Need to ensure the format from FinancialDataMCP matches
                     # process_news_batch expects items with 'text', 'source', 'published_at'
                     # If format differs, transformation is needed here.
                     # Assuming for now it's compatible or can be adapted within process_news_batch.
                     await self.process_news_batch(results["news_items"])

            else:
                self.mcp_tool_error_count += 1
                self.logger.error(
                    f"FinancialDataMCP 'get_news' call failed: {news_result.get('error') if news_result else 'Unknown error'}"
                )
                results["error"] = news_result.get('error', 'FinancialDataMCP news fetch failed') if news_result else 'FinancialDataMCP news fetch failed'


            return results

        except Exception as e:
            self.mcp_tool_error_count += 1
            self.logger.error(f"Error fetching news via FinancialDataMCP: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {"error": str(e), "news_items": [], "count": 0}

    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get data from the Selection Model using Redis.

        Returns:
            Selection Model data (or empty dict/error if not available)
        """
        self.logger.info("Getting selection data from Redis...")
        try:
            # Assuming Selection Model stores candidates in a list or set at a known key
            self.mcp_tool_call_count += 1
            # Example: Get members of a set
            result = self.redis_mcp.call_tool("smembers", {"key": self.redis_keys["selection_candidates"]})

            if result and not result.get("error"):
                candidates = result.get("members", [])
                self.logger.info(f"Retrieved {len(candidates)} selection candidates.")
                # Assuming candidates are just symbols, format as expected by analyze_company caller
                return {"selected_companies": [{"symbol": c} for c in candidates]}
            elif result and result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to get selection data from Redis: {result.get('error')}")
                 return {"error": result.get('error', 'Failed to get selection data from Redis')}
            else:
                 self.logger.warning("No selection data found in Redis.")
                 return {"selected_companies": []} # Return empty list if no data

        except Exception as e:
            self.logger.error(f"Error getting selection data from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {"error": str(e)}

    def send_feedback_to_selection(self, sentiment_data: Dict[str, Any]) -> bool:
        """
        Send sentiment feedback to the Selection Model using Redis Stream.

        Args:
            sentiment_data: Sentiment data to send

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Sending feedback to selection model for symbols: {sentiment_data.get('symbols', [])}")
        try:
            # Add timestamp to the data if not already present
            if "timestamp" not in sentiment_data:
                sentiment_data["timestamp"] = datetime.now().isoformat()

            # Publish feedback to the selection feedback stream
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                 "xadd", # Using stream for feedback
                 {
                      "stream": self.redis_keys["selection_feedback_stream"],
                      "data": sentiment_data
                 }
            )

            if result and not result.get("error"):
                self.logger.info("Sent feedback to selection model via stream.")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to send feedback to selection model: {result.get('error') if result else 'Unknown error'}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending feedback to selection model: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return False

    def run_sentiment_analysis(self, query: str) -> Dict[str, Any]:
        """
        Run sentiment analysis using AutoGen agents.

        Args:
            query: Query or instruction for sentiment analysis

        Returns:
            Results of the sentiment analysis
        """
        self.logger.info(f"Running sentiment analysis with query: {query}")

        sentiment_assistant = self.agents.get("sentiment_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not sentiment_assistant or not user_proxy:
            return {"error": "AutoGen agents not initialized"}

        try:
            # Initiate chat with the sentiment assistant
            user_proxy.initiate_chat(sentiment_assistant, message=query)

            # Get the last message from the assistant
            last_message = user_proxy.last_message(sentiment_assistant)
            content = last_message.get("content", "")

            # Extract structured data if possible
            try:
                # Find JSON blocks in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    result_str = content[json_start:json_end]
                    result = json.loads(result_str)
                    return result
            except json.JSONDecodeError:
                # Return the raw content if JSON parsing fails
                pass

            return {"analysis": content}

        except Exception as e:
            self.logger.error(f"Error during AutoGen chat: {e}")
            self.execution_errors += 1
            self.logger.counter("sentiment_analysis_model.execution_errors")
            return {"error": str(e)}


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.

    def analyze_sentiment_sync(self, text_data: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for analyze_sentiment for use in non-async contexts.
        
        Args:
            text_data: A single string or a list of strings containing the text to analyze.
            
        Returns:
            A list of dictionaries, each containing the analysis results for one input text.
        """
        return asyncio.run(self.analyze_sentiment(text_data))
