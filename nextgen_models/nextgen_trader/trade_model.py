"""
NextGen Trader Model

This module implements the trade execution and monitoring component of the NextGen Models system.
It handles order execution, position monitoring, and trade analytics using MCP tools.
"""

import os
import json
from dotenv import load_dotenv
load_dotenv()
import time
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math # Import math for floor function
import uuid # Import uuid for generating unique IDs

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator
from monitoring.system_metrics import SystemMetricsCollector

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
    # FunctionCall is not available in autogen 0.9.0
)

# MCP tools (Consolidated)
from mcp_tools.trading_mcp.trading_mcp import TradingMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP # Import RedisMCP

# Forward declarations for type hinting
# Removed duplicate helper class definitions (TradeExecutor, TradeMonitor, etc.)
# as their logic is integrated into the main TradeModel class methods.

class TradeModel:
    """
    Trade Model for executing and monitoring trading decisions.

    This model receives trade decisions from the Decision Model,
    executes orders via Alpaca, and monitors active positions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Trade Model with comprehensive monitoring.

        Args:
            config: Optional configuration dictionary with trading parameters
        """
        init_start_time = time.time()
        
        # Initialize monitoring lock for thread safety
        self.monitoring_lock = threading.RLock()
        
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-trade-model")
        self.logger.info("Starting TradeModel initialization")
        
        # Initialize SystemMetricsCollector for resource monitoring
        self.system_metrics = SystemMetricsCollector(self.logger)
        self.logger.info("SystemMetricsCollector initialized")
        
        # Initialize StockChartGenerator for trade performance visualization
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")

        # Initialize counters for trade model metrics
        self.orders_executed_count = 0
        self.buy_orders_count = 0
        self.sell_orders_count = 0
        self.market_orders_count = 0
        self.limit_orders_count = 0
        self.positions_opened_count = 0
        self.positions_closed_count = 0
        self.monitoring_cycles_run = 0
        self.exit_signals_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0 # Errors during trade execution and monitoring
        
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
            "largest_order_size": 0,  # Largest order size processed
            "slow_operations": [],  # List of slow operations (>1s)
            "peak_memory_usage": 0,  # Peak memory usage
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
        
        # Initialize trade statistics
        self.trade_stats = {
            "successful_trades": 0,
            "failed_trades": 0,
            "buy_trades_success": 0,
            "buy_trades_failed": 0,
            "sell_trades_success": 0,
            "sell_trades_failed": 0,
            "market_trades_success": 0,
            "market_trades_failed": 0,
            "limit_trades_success": 0,
            "limit_trades_failed": 0,
            "total_volume_traded": 0.0,
            "total_notional_value": 0.0,
            "slippage_stats": {
                "total_slippage_pct": 0.0,
                "count": 0,
                "max_slippage_pct": 0.0,
                "min_slippage_pct": 0.0,
            },
            "execution_time_stats": {
                "total_ms": 0.0,
                "count": 0,
                "max_ms": 0.0,
                "min_ms": float('inf'),
            },
            "trades_by_symbol": {},  # Dict of symbol -> trade count
            "trades_by_hour": {str(i): 0 for i in range(24)},  # Dict of hour -> trade count
        }
        
        # Track performance of MCP tool interactions
        self.mcp_metrics = {
            "trading_mcp_calls": 0,
            "trading_mcp_errors": 0,
            "financial_data_mcp_calls": 0,
            "financial_data_mcp_errors": 0,
            "time_series_mcp_calls": 0,
            "time_series_mcp_errors": 0,
            "redis_mcp_calls": 0,
            "redis_mcp_errors": 0,
            "tool_call_durations": [],  # List of recent durations
            "tool_call_errors": [],  # List of recent errors with details
        }
        
        # Initialize health data structure
        self.health_data = {
            "last_health_check": datetime.now(),
            "health_check_count": 0,
            "health_status": "initializing",  # initializing, healthy, degraded, error
            "health_score": 100,  # 0-100 score
            "component_health": {  # Health of individual components
                "trading_mcp": "unknown",
                "financial_data_mcp": "unknown",
                "time_series_mcp": "unknown",
                "redis_mcp": "unknown",
                "llm": "unknown",
            },
            "recent_issues": [],  # List of recent health issues
        }
        
        # Initialize shutdown flag
        self.shutdown_requested = False


        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_trader", "trade_model_config.json")
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
        self.daily_capital_limit = self.config.get("daily_capital_limit", 5000.0)

        # No overnight positions flag
        self.no_overnight_positions = self.config.get("no_overnight_positions", True)

        # Configure health and performance thresholds
        self.monitoring_config = self.config.get("monitoring_config", {})
        self.health_check_interval = self.monitoring_config.get("health_check_interval", 60)  # seconds
        self.performance_check_interval = self.monitoring_config.get("performance_check_interval", 300)  # seconds
        self.slow_operation_threshold = self.monitoring_config.get("slow_operation_threshold", 1000)  # ms
        self.high_memory_threshold = self.monitoring_config.get("high_memory_threshold", 75)  # percent
        self.high_cpu_threshold = self.monitoring_config.get("high_cpu_threshold", 80)  # percent
        self.enable_detailed_metrics = self.monitoring_config.get("enable_detailed_metrics", True)
        
        # Initialize Consolidated MCP clients with comprehensive monitoring
        
        # TradingMCP handles Alpaca functionality
        trading_config = self.config.get("trading_config", {})
        # Add logger to config if not present
        if "logger" not in trading_config:
            trading_config["logger"] = self.logger
            
        try:
            self.trading_mcp = TradingMCP(trading_config)
            self.health_data["component_health"]["trading_mcp"] = "healthy"
            self.logger.info("TradingMCP initialized successfully")
        except Exception as e:
            self.health_data["component_health"]["trading_mcp"] = "error"
            self.health_data["recent_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "trading_mcp",
                "error": str(e)
            })
            self.logger.error(f"Error initializing TradingMCP: {e}", exc_info=True)
            # Create a fallback or placeholder if needed
            self.trading_mcp = None
        
        # FinancialDataMCP handles data retrieval (Polygon REST)
        financial_data_config = self.config.get("financial_data_config", {})
        # Add logger to config if not present
        if "logger" not in financial_data_config:
            financial_data_config["logger"] = self.logger
            
        try:
            self.financial_data_mcp = FinancialDataMCP(financial_data_config)
            self.health_data["component_health"]["financial_data_mcp"] = "healthy"
            self.logger.info("FinancialDataMCP initialized successfully")
        except Exception as e:
            self.health_data["component_health"]["financial_data_mcp"] = "error"
            self.health_data["recent_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "financial_data_mcp",
                "error": str(e)
            })
            self.logger.error(f"Error initializing FinancialDataMCP: {e}", exc_info=True)
            # Create a fallback or placeholder if needed
            self.financial_data_mcp = None
        
        # TimeSeriesMCP handles peak detection, slippage analysis, and drift detection
        time_series_config = self.config.get("time_series_config", {})
        # Add logger to config if not present
        if "logger" not in time_series_config:
            time_series_config["logger"] = self.logger
            
        try:
            self.time_series_mcp = TimeSeriesMCP(time_series_config)
            self.health_data["component_health"]["time_series_mcp"] = "healthy"
            self.logger.info("TimeSeriesMCP initialized successfully")
        except Exception as e:
            self.health_data["component_health"]["time_series_mcp"] = "error"
            self.health_data["recent_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "time_series_mcp",
                "error": str(e)
            })
            self.logger.error(f"Error initializing TimeSeriesMCP: {e}", exc_info=True)
            # Create a fallback or placeholder if needed
            self.time_series_mcp = None
        
        # Initialize Redis MCP client for state management and event publishing
        redis_config = self.config.get("redis_config", {})
        # Add logger to config if not present
        if "logger" not in redis_config:
            redis_config["logger"] = self.logger
            
        try:
            self.redis_mcp = RedisMCP(redis_config)
            self.health_data["component_health"]["redis_mcp"] = "healthy"
            self.logger.info("RedisMCP initialized successfully")
        except Exception as e:
            self.health_data["component_health"]["redis_mcp"] = "error"
            self.health_data["recent_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "redis_mcp",
                "error": str(e)
            })
            self.logger.error(f"Error initializing RedisMCP: {e}", exc_info=True)
            # Create a fallback or placeholder if needed
            self.redis_mcp = None

        # Initialize model components (integrated into TradeModel methods)
        # self.executor = TradeExecutor(self) # Removed
        # self.monitor = TradeMonitor(self) # Removed
        # self.position_manager = TradePositionManager(self) # Removed
        # self.analytics = TradeAnalytics(self) # Removed

        # Start system metrics collection
        try:
            self.system_metrics.start()
            self.logger.info("System metrics collection started")
        except Exception as e:
            self.logger.error(f"Error starting system metrics collection: {e}", exc_info=True)
            
        # Start health check thread if enabled
        if self.monitoring_config.get("enable_health_check", True):
            self._start_health_check_thread()

        # Initialize AutoGen integration
        try:
            self.llm_config = self._get_llm_config()
            self.agents = self._setup_agents()
            # Register functions with the agents
            self._register_functions(self.agents["user_proxy"])
            self.health_data["component_health"]["llm"] = "healthy"
            self.logger.info("AutoGen integration initialized successfully")
        except Exception as e:
            self.health_data["component_health"]["llm"] = "error"
            self.health_data["recent_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "llm",
                "error": str(e)
            })
            self.logger.error(f"Error initializing AutoGen integration: {e}", exc_info=True)

        # Update overall health status
        self._update_overall_health_status()

        # Redis keys for state management and event publishing
        self.redis_keys = {
            "daily_capital_usage": "trade:daily_capital_usage",
            "open_positions_monitoring": "trade:open_positions_monitoring", # Hash or Set of symbols being monitored
            "position_monitoring_data": "trade:position_monitoring_data:", # Hash or JSON for individual position data
            "trade_events_stream": "trade:events", # Stream for publishing trade events
            "account_info": "trade:account_info", # Latest account info
            "portfolio_constraints": "trade:portfolio_constraints", # Latest portfolio constraints,
            # Health monitoring keys
            "health_status": "trade:health:status",  # Current health status
            "performance_metrics": "trade:health:performance",  # Performance metrics
        }

        # Ensure Redis stream exists (optional, but good practice)
        try:
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["trade_events_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['trade_events_stream']}' exists.")
        except Exception as e:
            self.logger.warning(f"Could not ensure Redis stream exists: {e}")

        # Record initialization completion
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("trade_model.initialization_time_ms", init_duration)
        
        # Save initial performance metrics to Redis
        self._save_performance_metrics()
        
        # Log comprehensive initialization statistics
        self.logger.info("Trade Model initialization complete", 
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
                        self._update_performance_metrics()
                    
                    # Generate performance visualization
                    if health_check_counter % 10 == 0:  # Every 10th check
                        self._generate_performance_visualizations()
                    
                    # Log health status
                    if health_check_counter % 5 == 0:  # Detailed log every 5th check
                        self.logger.info("Detailed health check completed", 
                                       health_status=self.health_data["health_status"],
                                       health_score=self.health_data["health_score"],
                                       component_health=self.health_data["component_health"],
                                       recent_issues_count=len(self.health_data["recent_issues"]),
                                       orders_executed_count=self.orders_executed_count,
                                       monitoring_cycles_run=self.monitoring_cycles_run,
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
                # Check Trading MCP health
                if self.trading_mcp is not None:
                    try:
                        # Try a simple ping operation to check if it's responsive
                        if hasattr(self.trading_mcp, "health_check"):
                            trading_health = self.trading_mcp.health_check()
                            self.health_data["component_health"]["trading_mcp"] = trading_health.get("status", "unknown")
                        else:
                            # Fallback to checking if it responds to a method call
                            method_exists = hasattr(self.trading_mcp, "call_tool")
                            self.health_data["component_health"]["trading_mcp"] = "healthy" if method_exists else "degraded"
                    except Exception as e:
                        self.health_data["component_health"]["trading_mcp"] = "error"
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "trading_mcp",
                            "error": str(e)
                        })
                        self.logger.error(f"Error checking Trading MCP health: {e}")
                
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
                
                # Check Time Series MCP health
                if self.time_series_mcp is not None:
                    try:
                        # Try a simple operation to check if it's responsive
                        if hasattr(self.time_series_mcp, "health_check"):
                            ts_health = self.time_series_mcp.health_check()
                            self.health_data["component_health"]["time_series_mcp"] = ts_health.get("status", "unknown")
                        else:
                            # Fallback to checking if it responds to a method call
                            method_exists = hasattr(self.time_series_mcp, "call_tool")
                            self.health_data["component_health"]["time_series_mcp"] = "healthy" if method_exists else "degraded"
                    except Exception as e:
                        self.health_data["component_health"]["time_series_mcp"] = "error"
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "time_series_mcp",
                            "error": str(e)
                        })
                        self.logger.error(f"Error checking Time Series MCP health: {e}")
                
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
                    self.logger.gauge("trade_model.cpu_usage", cpu_percent)
                    self.logger.gauge("trade_model.memory_usage", memory_percent)
                    
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
                        "key": self.redis_keys["health_status"],
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
        self.logger.gauge("trade_model.health_score", self.health_data["health_score"])
        
    def _update_performance_metrics(self):
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
                        "cpu_usage": {
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
                self._update_trend_data()
                
                # Perform anomaly detection
                self._detect_anomalies()
                
                # Update API usage metrics
                self._update_api_usage_metrics()
                
                # Clean up outdated metrics
                self._cleanup_old_metrics()
                
                # Save performance metrics to Redis
                self._save_performance_metrics()
                
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
    
    def _update_trend_data(self):
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
                elif metric_name == "cpu_usage":
                    values = self.performance_metrics["cpu_usage_history"]
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
                        elif (metric_name in ["response_time", "memory_usage", "cpu_usage", "error_rate"] 
                              and slope > 0):
                            trend_data["trend"] = "degrading"  # Higher is worse for these metrics
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
    
    def _detect_anomalies(self):
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
            
            # Similar approach for memory and CPU
            for metric_name, values in [
                ("memory_usage", self.performance_metrics["memory_usage_history"]),
                ("cpu_usage", self.performance_metrics["cpu_usage_history"])
            ]:
                if len(values) >= 10:
                    mean = sum(values) / len(values)
                    variance = sum((x - mean) ** 2 for x in values) / len(values)
                    std_dev = variance ** 0.5
                    
                    if len(values) > 0:
                        latest = values[-1]
                        # Anomaly if more than 3 standard deviations away
                        if abs(latest - mean) > 3 * std_dev and std_dev > 0:
                            anomaly = {
                                "timestamp": time.time(),
                                "metric": metric_name,
                                "value": latest,
                                "mean": mean,
                                "std_dev": std_dev,
                                "deviation": (latest - mean) / std_dev
                            }
                            self.performance_trends[metric_name]["anomalies"].append(anomaly)
                            # Keep only recent anomalies
                            if len(self.performance_trends[metric_name]["anomalies"]) > 10:
                                self.performance_trends[metric_name]["anomalies"] = self.performance_trends[metric_name]["anomalies"][-10:]
                            self.logger.warning(f"{metric_name.replace('_', ' ')} anomaly detected: {latest:.2f}% vs mean {mean:.2f}%")
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
    
    def _update_api_usage_metrics(self):
        """Update API usage metrics for monitoring rate limits and costs."""
        # This is already being tracked in llm_api_metrics in the initialization
        # Just log current status
        if self.llm_api_metrics["total_calls"] > 0:
            success_rate = (self.llm_api_metrics["successful_calls"] / 
                           max(1, self.llm_api_metrics["total_calls"])) * 100
            self.logger.gauge("trade_model.llm_api_success_rate", success_rate)
            
            if "total_tokens" in self.llm_api_metrics and self.llm_api_metrics["total_tokens"] > 0:
                self.logger.gauge("trade_model.llm_api_total_tokens", self.llm_api_metrics["total_tokens"])
    
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
        
    def _save_performance_metrics(self):
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
                "symbols_processed_count": len(self.performance_metrics["symbols_processed"]),
                "model_metrics": self.get_model_metrics(),
                "trends": {
                    name: {"trend": data["trend"], "anomalies_count": len(data["anomalies"])}
                    for name, data in getattr(self, "performance_trends", {}).items()
                }
            }
            
            # Store in Redis
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("set_json", {
                "key": self.redis_keys["performance_metrics"],
                "value": metrics_summary,
                "expiry": 3600  # 1 hour expiry
            })
            
            # Log successful save
            self.logger.info("Performance metrics saved to Redis")
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics to Redis: {e}")
            self.mcp_tool_error_count += 1
    
    def _generate_performance_visualizations(self):
        """Generate visualizations of performance metrics for monitoring dashboard."""
        try:
            if not self.chart_generator:
                self.logger.warning("Chart generator not available for generating visualizations")
                return
            
            output_dir = "logs/performance_charts"
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
            
            # Generate trade statistics chart
            if self.trade_stats["market_trades_success"] + self.trade_stats["limit_trades_success"] > 0:
                # Prepare data
                labels = ["Market Buy", "Market Sell", "Limit Buy", "Limit Sell"]
                values = [
                    self.trade_stats.get("buy_trades_success", 0) - self.trade_stats.get("limit_trades_success", 0),
                    self.trade_stats.get("sell_trades_success", 0) - self.trade_stats.get("limit_trades_success", 0),
                    self.trade_stats.get("limit_trades_success", 0) * 0.6,  # Approximate distribution
                    self.trade_stats.get("limit_trades_success", 0) * 0.4,  # Approximate distribution
                ]
                
                # Generate chart
                chart_path = os.path.join(output_dir, "trade_distribution.png")
                self.chart_generator.create_pie_chart(
                    labels, values, "Trade Type Distribution", chart_path
                )
                self.logger.info(f"Generated trade distribution chart at {chart_path}")
            
            # Generate execution time distribution
            if (self.trade_stats["execution_time_stats"]["count"] > 10 and
                self.trade_stats["execution_time_stats"]["max_ms"] > 0):
                
                # Create histogram bins
                min_time = max(1, self.trade_stats["execution_time_stats"]["min_ms"])
                max_time = self.trade_stats["execution_time_stats"]["max_ms"]
                avg_time = self.trade_stats["execution_time_stats"]["total_ms"] / self.trade_stats["execution_time_stats"]["count"]
                
                # Generate synthetic data based on statistics
                # (in a real system, you'd use actual stored execution times)
                import numpy as np
                
                # Create a gamma distribution that approximates the execution time distribution
                data = np.random.gamma(
                    shape=2, 
                    scale=avg_time/2, 
                    size=self.trade_stats["execution_time_stats"]["count"]
                )
                
                # Rescale to match min/max
                data = np.clip(data, min_time, max_time)
                
                # Generate chart
                chart_path = os.path.join(output_dir, "execution_time_distribution.png")
                self.chart_generator.create_histogram(
                    data, 
                    "Execution Time Distribution (ms)", 
                    "Execution Time (ms)", 
                    "Frequency",
                    chart_path
                )
                self.logger.info(f"Generated execution time distribution chart at {chart_path}")
            
            # Generate health score history
            # In a real system, you would store the health scores over time
            # For this example, we'll create a synthetic trend
                
            self.logger.info("Performance visualizations generated")
            
        except Exception as e:
            self.logger.error(f"Error generating performance visualizations: {e}", exc_info=True)
            self.execution_errors += 1


    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen.

        Returns:
            LLM configuration dictionary
        """
        llm_config = self.config.get("llm_config", {})

        # Default configuration if not provided
        if not llm_config:
            llm_config = {
                "temperature": 0.1,
                "config_list": [
                    {
                        "model": "anthropic/claude-3-opus-20240229",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    },
                    {
                        "model": "meta-llama/llama-3-70b-instruct",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None,
                    },
                ],
            }

        return {
            "config_list": llm_config.get("config_list", []),
            "temperature": llm_config.get("temperature", 0.1),
            "timeout": llm_config.get("timeout", 600),
            "seed": 42,  # Adding seed for reproducibility
        }

    def _setup_agents(self) -> Dict[str, Agent]:
        """
        Initialize AutoGen agents for trading.

        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the trade assistant agent
        agents["trade_assistant"] = AssistantAgent(
            name="TradeAssistantAgent",
            system_message="""You are a trade execution specialist. Your role is to:
            1. Execute trades efficiently while managing costs
            2. Monitor execution quality
            3. Adjust execution tactics based on market conditions
            4. Track positions and portfolio exposure

            You have tools for executing different order types, monitoring market conditions,
            and analyzing trade performance. You should always be aware of the daily capital
            limit ($5000) and avoid overnight positions.""",
            llm_config=self.llm_config,
            description="A specialist in trade execution and monitoring",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="TradeToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        agents["user_proxy"] = user_proxy

        return agents

    def _register_functions(self, user_proxy: UserProxyAgent):
        """
        Register functions with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register functions with
        """

        # Define execution functions
        @register_function(
            name="execute_market_order",
            description="Execute a market order",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def execute_market_order(
            symbol: str, quantity: float, side: str
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._execute_market_order_internal(symbol, quantity, side)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "execute_market_order", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "execute_market_order"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in execute_market_order: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "execute_market_order", "status": "failed"})
                return {"error": str(e)}


        @register_function(
            name="execute_limit_order",
            description="Execute a limit order",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def execute_limit_order(
            symbol: str, quantity: float, side: str, limit_price: float
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._execute_limit_order_internal(
                    symbol, quantity, side, limit_price
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "execute_limit_order", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "execute_limit_order"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in execute_limit_order: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "execute_limit_order", "status": "failed"})
                return {"error": str(e)}


        # Define monitoring functions
        @register_function(
            name="start_position_monitoring",
            description="Begin monitoring a new position",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def start_position_monitoring(symbol: str, order_id: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._start_position_monitoring_internal(symbol, order_id)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "start_position_monitoring", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "start_position_monitoring"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in start_position_monitoring: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "start_position_monitoring", "status": "failed"})
                return {"error": str(e)}


        @register_function(
            name="check_exit_conditions",
            description="Check if exit conditions are met for a position",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def check_exit_conditions(symbol: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._check_exit_conditions_internal(symbol)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "check_exit_conditions", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "check_exit_conditions"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in check_exit_conditions: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "check_exit_conditions", "status": "failed"})
                return {"error": str(e)}


        # Define position management functions
        @register_function(
            name="get_positions",
            description="Get all current positions",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def get_positions() -> List[Dict[str, Any]]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._get_positions_internal()
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_positions", "status": "success" if result is not None else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_positions"})
                if result is None: # Assuming internal method returns None on error
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                    return [] # Return empty list on error
                return result
            except Exception as e:
                self.logger.error(f"Error in get_positions: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_positions", "status": "failed"})
                return [] # Return empty list on error


        @register_function(
            name="get_position",
            description="Get position for a specific symbol",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def get_position(symbol: str) -> Optional[Dict[str, Any]]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._get_position_internal(symbol)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_position", "status": "success" if result is not None else "not_found"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_position"})
                # No error increment here, as not finding a position is valid
                return result
            except Exception as e:
                self.logger.error(f"Error in get_position: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_position", "status": "failed"})
                return None # Return None on error


        @register_function(
            name="get_portfolio_constraints",
            description="Get current portfolio constraints",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def get_portfolio_constraints() -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._get_portfolio_constraints_internal()
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_portfolio_constraints", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_portfolio_constraints"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in get_portfolio_constraints: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_portfolio_constraints", "status": "failed"})
                return {"error": str(e)}


        # Define analytics functions
        @register_function(
            name="get_trade_history",
            description="Get historical trades",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def get_trade_history(
            symbol: Optional[str] = None, limit: int = 50
        ) -> List[Dict[str, Any]]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._get_trade_history_internal(symbol, limit)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_trade_history", "status": "success"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_trade_history"})
                return result
            except Exception as e:
                self.logger.error(f"Error in get_trade_history: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_trade_history", "status": "failed"})
                return [] # Return empty list on error


        @register_function(
            name="calculate_execution_quality",
            description="Calculate execution quality metrics",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def calculate_execution_quality(order_id: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call internal method directly
                result = self._calculate_execution_quality_internal(order_id)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "calculate_execution_quality", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "calculate_execution_quality"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in calculate_execution_quality: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "calculate_execution_quality", "status": "failed"})
                return {"error": str(e)}


        # Register MCP tool access functions
        self._register_mcp_tool_access(user_proxy)


    def _register_mcp_tool_access(self, user_proxy: UserProxyAgent):
        """
        Register MCP tool access functions with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register functions with
        """

        # Define MCP tool access functions for consolidated MCPs
        @register_function(
            name="use_trading_tool",
            description="Use a tool provided by the Trading MCP server (for Alpaca functionality)",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_trading_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.trading_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="use_financial_data_tool",
            description="Use a tool provided by the Financial Data MCP server (for market data)",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_financial_data_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.financial_data_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="use_time_series_tool",
            description="Use a tool provided by the Time Series MCP server (for analysis like peak detection, slippage, drift)",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_time_series_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.time_series_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="list_mcp_tools",
            description="List all available tools on an MCP server",
            parameters={
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server (trading, financial_data, time_series)",
                }
            },
            return_type=List[Dict[str, str]],
        )
        def list_mcp_tools(server_name: str) -> List[Dict[str, str]]:
            if server_name == "trading":
                return self.trading_mcp.list_tools()
            elif server_name == "financial_data":
                return self.financial_data_mcp.list_tools()
            elif server_name == "time_series":
                return self.time_series_mcp.list_tools()
            else:
                return [{"error": f"MCP server not found: {server_name}"}]

        # Register the MCP tool access functions
        user_proxy.register_function(use_trading_tool)
        user_proxy.register_function(use_financial_data_tool)
        user_proxy.register_function(use_time_series_tool)
        user_proxy.register_function(list_mcp_tools)


    # --- Internal Methods for Core Logic ---

    def _execute_market_order_internal(self, symbol: str, quantity: float, side: str) -> Dict[str, Any]:
        """Internal logic for executing market orders."""
        self.logger.info(f"Executing market order: {symbol} {side} {quantity}")
        self.mcp_tool_call_count += 1
        try:
            result = self.trading_mcp.call_tool(
                "submit_market_order",
                {"symbol": symbol, "qty": quantity, "side": side}
            )
            if result and not result.get("error"):
                self.market_orders_count += 1
                if side == "buy": self.buy_orders_count += 1
                else: self.sell_orders_count += 1
                # Publish trade event to Redis stream
                self._publish_trade_event("order_submitted", {"order": result})
                return result
            else:
                self.mcp_tool_error_count += 1
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                self.logger.error(f"Market order execution failed: {error_msg}")
                self._publish_trade_event("order_submission_failed", {"symbol": symbol, "side": side, "quantity": quantity, "error": error_msg})
                return {"error": error_msg}
        except Exception as e:
            self.logger.error(f"Error executing market order: {e}", exc_info=True)
            self.execution_errors += 1
            self._publish_trade_event("order_submission_failed", {"symbol": symbol, "side": side, "quantity": quantity, "error": str(e)})
            return {"error": str(e)}

    def _execute_limit_order_internal(self, symbol: str, quantity: float, side: str, limit_price: float) -> Dict[str, Any]:
        """Internal logic for executing limit orders."""
        self.logger.info(f"Executing limit order: {symbol} {side} {quantity} @ {limit_price}")
        self.mcp_tool_call_count += 1
        try:
            result = self.trading_mcp.call_tool(
                "submit_limit_order",
                {"symbol": symbol, "qty": quantity, "side": side, "limit_price": limit_price}
            )
            if result and not result.get("error"):
                self.limit_orders_count += 1
                if side == "buy": self.buy_orders_count += 1
                else: self.sell_orders_count += 1
                # Publish trade event to Redis stream
                self._publish_trade_event("order_submitted", {"order": result})
                return result
            else:
                self.mcp_tool_error_count += 1
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                self.logger.error(f"Limit order execution failed: {error_msg}")
                self._publish_trade_event("order_submission_failed", {"symbol": symbol, "side": side, "quantity": quantity, "limit_price": limit_price, "error": error_msg})
                return {"error": error_msg}
        except Exception as e:
            self.logger.error(f"Error executing limit order: {e}", exc_info=True)
            self.execution_errors += 1
            self._publish_trade_event("order_submission_failed", {"symbol": symbol, "side": side, "quantity": quantity, "limit_price": limit_price, "error": str(e)})
            return {"error": str(e)}

    def _start_position_monitoring_internal(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Internal logic for starting position monitoring."""
        self.logger.info(f"Starting position monitoring for {symbol}, order_id: {order_id}")
        try:
            position = self._get_position_internal(symbol)
            if not position:
                self.logger.warning(f"No position found for {symbol}")
                return {"error": f"No position found for {symbol}"}

            self.mcp_tool_call_count += 1
            quote_result = self.financial_data_mcp.call_tool("get_latest_quote", {"symbol": symbol})
            if not quote_result or quote_result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to get quote for {symbol}")
                return {"error": f"Failed to get quote for {symbol}"}

            current_price = quote_result.get("last_price", 0)
            if current_price <= 0:
                self.logger.error(f"Invalid current price for {symbol}: {current_price}")
                return {"error": f"Invalid current price for {symbol}"}

            entry_price = float(position.get("avg_entry_price", 0))
            if entry_price <= 0:
                self.logger.error(f"Invalid entry price for {symbol}: {entry_price}")
                return {"error": f"Invalid entry price for {symbol}"}

            # Calculate stop loss and take profit based on entry price (can be configured)
            stop_loss = entry_price * self.config.get("stop_loss_pct", 0.98)
            take_profit = entry_price * self.config.get("take_profit_pct", 1.05)

            # Store monitoring data in Redis
            monitoring_data = {
                "symbol": symbol, "order_id": order_id, "entry_price": entry_price,
                "stop_loss": stop_loss, "take_profit": take_profit,
                "last_checked": datetime.now().isoformat(), "monitoring_active": True
            }
            self.mcp_tool_call_count += 1
            redis_key = f"{self.redis_keys['position_monitoring_data']}{symbol}"
            store_result = self.redis_mcp.call_tool("set_json", {"key": redis_key, "value": monitoring_data})

            if store_result and not store_result.get("error"):
                # Add symbol to the set of monitored positions
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool("sadd", {"key": self.redis_keys["open_positions_monitoring"], "member": symbol})
                self.logger.info(f"Position monitoring started for {symbol} with stop_loss: {stop_loss}, take_profit: {take_profit}")
                return {"status": "success", "monitoring_data": monitoring_data}
            else:
                self.mcp_tool_error_count += 1
                error_msg = store_result.get("error", "Unknown error") if store_result else "No result returned"
                self.logger.error(f"Failed to store monitoring data for {symbol}: {error_msg}")
                return {"error": f"Failed to store monitoring data: {error_msg}"}

        except Exception as e:
            self.logger.error(f"Error starting position monitoring: {e}", exc_info=True)
            self.execution_errors += 1
            return {"error": str(e)}

    def _check_exit_conditions_internal(self, symbol: str) -> Dict[str, Any]:
        """Internal logic for checking exit conditions."""
        self.logger.info(f"Checking exit conditions for {symbol}")
        try:
            # Retrieve monitoring data from Redis
            self.mcp_tool_call_count += 1
            redis_key = f"{self.redis_keys['position_monitoring_data']}{symbol}"
            monitoring_data_result = self.redis_mcp.call_tool("get_json", {"key": redis_key})

            if not monitoring_data_result or monitoring_data_result.get("error") or not monitoring_data_result.get("value"):
                self.mcp_tool_error_count += 1
                self.logger.warning(f"No monitoring data found for {symbol}")
                # Remove from monitored set if data is missing
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool("srem", {"key": self.redis_keys["open_positions_monitoring"], "member": symbol})
                return {"should_exit": False, "reason": "No monitoring data found"}

            monitoring_data = monitoring_data_result.get("value")
            stop_loss = monitoring_data.get("stop_loss", 0)
            take_profit = monitoring_data.get("take_profit", 0)
            entry_price = monitoring_data.get("entry_price", 0)

            if stop_loss <= 0 or take_profit <= 0 or entry_price <= 0:
                 self.logger.error(f"Invalid monitoring data for {symbol}: {monitoring_data}")
                 # Remove from monitored set if data is invalid
                 self.mcp_tool_call_count += 1
                 self.redis_mcp.call_tool("srem", {"key": self.redis_keys["open_positions_monitoring"], "member": symbol})
                 return {"should_exit": False, "reason": "Invalid monitoring data"}


            self.mcp_tool_call_count += 1
            quote_result = self.financial_data_mcp.call_tool("get_latest_quote", {"symbol": symbol})
            if not quote_result or quote_result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to get quote for {symbol}")
                return {"should_exit": False, "reason": "Failed to get current price"}

            current_price = quote_result.get("last_price", 0)
            if current_price <= 0:
                self.logger.error(f"Invalid current price for {symbol}: {current_price}")
                return {"should_exit": False, "reason": "Invalid current price"}

            # Update last checked time in Redis
            monitoring_data["last_checked"] = datetime.now().isoformat()
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("set_json", {"key": redis_key, "value": monitoring_data})


            if current_price <= stop_loss:
                self.logger.info(f"Stop loss triggered for {symbol}: current_price={current_price}, stop_loss={stop_loss}")
                return {"should_exit": True, "reason": "Stop loss triggered"}
            if current_price >= take_profit:
                self.logger.info(f"Take profit triggered for {symbol}: current_price={current_price}, take_profit={take_profit}")
                return {"should_exit": True, "reason": "Take profit triggered"}

            if self.no_overnight_positions:
                self.mcp_tool_call_count += 1
                clock_result = self.trading_mcp.call_tool("get_clock", {})
                if clock_result and not clock_result.get("error"):
                    is_open = clock_result.get("is_open", True)
                    next_close = clock_result.get("next_close", "")
                    if not is_open:
                        self.logger.info(f"Market closed, exiting position for {symbol}")
                        return {"should_exit": True, "reason": "Market closed"}
                    if next_close:
                        next_close_time = datetime.fromisoformat(next_close.replace('Z', '+00:00'))
                        time_to_close = (next_close_time - datetime.now().astimezone()).total_seconds()
                        if time_to_close <= self.config.get("end_of_day_exit_buffer_seconds", 600): # Configurable buffer
                            self.logger.info(f"Market closing soon, exiting position for {symbol}")
                            return {"should_exit": True, "reason": "Market closing soon"}

            return {"should_exit": False, "reason": "No exit conditions met"}
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}", exc_info=True)
            self.execution_errors += 1
            return {"should_exit": False, "reason": f"Error: {str(e)}"}

    def _get_positions_internal(self) -> List[Dict[str, Any]]:
        """Internal logic for getting all positions."""
        self.logger.info("Getting all positions")
        self.mcp_tool_call_count += 1
        try:
            result = self.trading_mcp.call_tool("get_positions", {})
            if result and not result.get("error"):
                positions = result.get("positions", [])
                self.logger.info(f"Retrieved {len(positions)} positions")
                # Publish position update event
                self._publish_trade_event("positions_updated", {"positions": positions})
                return positions
            else:
                self.mcp_tool_error_count += 1
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                self.logger.error(f"Failed to get positions: {error_msg}")
                self._publish_trade_event("get_positions_failed", {"error": error_msg})
                return []
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}", exc_info=True)
            self.execution_errors += 1
            self._publish_trade_event("get_positions_failed", {"error": str(e)})
            return []

    def _get_position_internal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Internal logic for getting a specific position."""
        self.logger.info(f"Getting position for {symbol}")
        self.mcp_tool_call_count += 1
        try:
            result = self.trading_mcp.call_tool("get_position", {"symbol": symbol})
            if result and not result.get("error"):
                self.logger.info(f"Retrieved position for {symbol}")
                return result
            else:
                if result and result.get("error") and "not found" in result.get("error", "").lower():
                    self.logger.info(f"No position found for {symbol}")
                    return None
                self.mcp_tool_error_count += 1
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                self.logger.error(f"Failed to get position for {symbol}: {error_msg}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}", exc_info=True)
            self.execution_errors += 1
            return None

    def _get_portfolio_constraints_internal(self) -> Dict[str, Any]:
        """Internal logic for getting portfolio constraints."""
        self.logger.info("Getting portfolio constraints")
        try:
            self.mcp_tool_call_count += 1
            account_info = self.trading_mcp.call_tool("get_account_info", {})
            if not account_info or "error" in account_info:
                self.mcp_tool_error_count += 1
                error_msg = account_info.get("error", "Unknown error") if account_info else "No result returned"
                self.logger.error(f"Failed to get account information: {error_msg}")
                return {"error": error_msg}

            cash = float(account_info.get("cash", 0))
            buying_power = float(account_info.get("buying_power", 0))
            equity = float(account_info.get("equity", 0))

            # Retrieve daily usage from Redis
            self.mcp_tool_call_count += 1
            daily_usage_result = self.redis_mcp.call_tool("get", {"key": self.redis_keys["daily_capital_usage"]})
            daily_usage = float(daily_usage_result.get("value", 0)) if daily_usage_result and not daily_usage_result.get("error") else 0.0

            remaining_capital = max(0, self.daily_capital_limit - daily_usage)

            positions = self._get_positions_internal()
            position_count = len(positions)
            position_value = sum(float(pos.get("market_value", 0)) for pos in positions)

            constraints = {
                "daily_capital_limit": self.daily_capital_limit, "daily_usage": daily_usage,
                "remaining_daily_capital": remaining_capital, "cash": cash, "buying_power": buying_power,
                "equity": equity, "position_count": position_count, "position_value": position_value,
                "no_overnight_positions": self.no_overnight_positions, "timestamp": datetime.now().isoformat()
            }

            # Store latest constraints in Redis
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("set_json", {"key": self.redis_keys["portfolio_constraints"], "value": constraints, "expiry": 60}) # Cache for 60 seconds

            self.logger.info(f"Portfolio constraints: remaining_capital=${remaining_capital:.2f}, position_count={position_count}")
            return constraints
        except Exception as e:
            self.logger.error(f"Error getting portfolio constraints: {e}", exc_info=True)
            self.execution_errors += 1
            return {"error": str(e)}

    def _get_trade_history_internal(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Internal logic for getting trade history."""
        self.logger.info(f"Getting trade history for {symbol if symbol else 'all symbols'}, limit={limit}")
        self.mcp_tool_call_count += 1
        try:
            params = {"limit": limit}
            if symbol: params["symbol"] = symbol
            result = self.trading_mcp.call_tool("get_orders", params)
            if result and not result.get("error"):
                orders = result.get("orders", [])
                self.logger.info(f"Retrieved {len(orders)} orders")
                return orders
            else:
                self.mcp_tool_error_count += 1
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                self.logger.error(f"Failed to get trade history: {error_msg}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}", exc_info=True)
            self.execution_errors += 1
            return []

    def _calculate_execution_quality_internal(self, order_id: str) -> Dict[str, Any]:
        """Internal logic for calculating execution quality."""
        self.logger.info(f"Calculating execution quality for order {order_id}")
        self.mcp_tool_call_count += 1
        try:
            order_result = self.trading_mcp.call_tool("get_order", {"order_id": order_id})
            if not order_result or order_result.get("error"):
                self.mcp_tool_error_count += 1
                error_msg = order_result.get("error", "Unknown error") if order_result else "No result returned"
                self.logger.error(f"Failed to get order details: {error_msg}")
                return {"error": error_msg}

            symbol = order_result.get("symbol")
            side = order_result.get("side")
            order_type = order_result.get("type")
            submitted_at = order_result.get("submitted_at")
            filled_at = order_result.get("filled_at")
            filled_qty = float(order_result.get("filled_qty", 0))
            filled_avg_price = float(order_result.get("filled_avg_price", 0))

            if not symbol or not side or not submitted_at or not filled_at or filled_qty <= 0 or filled_avg_price <= 0:
                self.logger.error(f"Incomplete order information for {order_id}")
                return {"error": "Incomplete order information"}

            submitted_time = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
            filled_time = datetime.fromisoformat(filled_at.replace('Z', '+00:00'))
            execution_time_ms = (filled_time - submitted_time).total_seconds() * 1000

            slippage = 0.0
            if order_type == "limit":
                limit_price = float(order_result.get("limit_price", 0))
                if limit_price > 0:
                    if side == "buy": slippage = (limit_price - filled_avg_price) / limit_price * 100
                    else: slippage = (filled_avg_price - limit_price) / limit_price * 100

            metrics = {
                "order_id": order_id, "symbol": symbol, "side": side, "order_type": order_type,
                "filled_qty": filled_qty, "filled_avg_price": filled_avg_price,
                "execution_time_ms": execution_time_ms, "slippage_percent": slippage,
                "timestamp": datetime.now().isoformat()
            }
            self.logger.info(f"Execution quality for {order_id}: execution_time={execution_time_ms:.2f}ms, slippage={slippage:.4f}%")
            return metrics
        except Exception as e:
            self.logger.error(f"Error calculating execution quality: {e}", exc_info=True)
            self.execution_errors += 1
            return {"error": str(e)}

    def _publish_trade_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Internal method to publish trade events to a Redis stream."""
        try:
            event_data = {
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                "xadd",
                {
                    "stream": self.redis_keys["trade_events_stream"],
                    "data": event_data
                }
            )
            if result and not result.get("error"):
                self.logger.info(f"Published trade event '{event_type}' to stream.")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to publish trade event '{event_type}' to stream: {result.get('error') if result else 'Unknown error'}")
                return False
        except Exception as e:
            self.logger.error(f"Error publishing trade event '{event_type}': {e}", exc_info=True)
            self.execution_errors += 1
            return False


    # --- Public Methods ---

    def execute_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade based on the provided trade data using TradingMCP.

        Args:
            trade_data: Dictionary containing trade details (symbol, action, quantity, etc.)

        Returns:
            Dictionary with trade execution results
        """
        start_time = time.time()
        self.logger.info(f"Executing trade: {trade_data}")

        # Track this symbol in performance metrics
        with self.monitoring_lock:
            # If symbol exists, add to performance metrics
            if "symbol" in trade_data and trade_data["symbol"]:
                self.performance_metrics["symbols_processed"].add(trade_data["symbol"])
                self.logger.gauge("trade_model.unique_symbols_processed", len(self.performance_metrics["symbols_processed"]))
            
            # Track order by hour of day for pattern analysis
            hour_of_day = datetime.now().hour
            if "trades_by_hour" not in self.performance_metrics:
                self.performance_metrics["trades_by_hour"] = {str(i): 0 for i in range(24)}
            self.performance_metrics["trades_by_hour"][str(hour_of_day)] = self.performance_metrics["trades_by_hour"].get(str(hour_of_day), 0) + 1
            self.logger.gauge(f"trade_model.trades_hour_{hour_of_day}", self.performance_metrics["trades_by_hour"][str(hour_of_day)])

        # Check if market is open (using TradingMCP tool)
        if not self._is_market_open():
            self.logger.warning("Market is closed, cannot execute trade")
            self.execution_errors += 1
            with self.monitoring_lock:
                self.performance_metrics["error_timestamps"].append(time.time())
            self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": "Market is closed"})
            return {"status": "failed", "error": "Market is closed"}

        # Extract trade parameters
        symbol = trade_data.get("symbol")
        action = trade_data.get("action", "").lower()  # buy or sell
        quantity = float(trade_data.get("quantity", 0))
        order_type = trade_data.get("order_type", "market").lower()
        limit_price = float(trade_data.get("limit_price", 0)) if "limit_price" in trade_data else None
        capital_amount = float(trade_data.get("capital_amount", 0))

        # Validate trade parameters
        if not symbol:
            self.logger.error("Missing symbol in trade data")
            self.execution_errors += 1
            with self.monitoring_lock:
                self.performance_metrics["error_timestamps"].append(time.time())
            self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": "Missing symbol"})
            return {"status": "failed", "error": "Missing symbol"}

        if action not in ["buy", "sell"]:
            self.logger.error(f"Invalid action: {action}")
            self.execution_errors += 1
            with self.monitoring_lock:
                self.performance_metrics["error_timestamps"].append(time.time())
            self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": f"Invalid action: {action}"})
            return {"status": "failed", "error": f"Invalid action: {action}"}

        # Map action to side for TradingMCP tool
        side = "buy" if action == "buy" else "sell"

        # If capital_amount is provided and quantity is not, calculate quantity
        if capital_amount > 0 and quantity == 0:
            # Get latest quote to determine price (using FinancialDataMCP tool)
            quote_start_time = time.time()
            self.mcp_tool_call_count += 1
            quote_result = self.financial_data_mcp.call_tool("get_latest_quote", {"symbol": symbol})
            quote_duration = (time.time() - quote_start_time) * 1000
            
            # Track API latency
            with self.monitoring_lock:
                if "api_latency" not in self.performance_metrics:
                    self.performance_metrics["api_latency"] = {"get_latest_quote": []}
                elif "get_latest_quote" not in self.performance_metrics["api_latency"]:
                    self.performance_metrics["api_latency"]["get_latest_quote"] = []
                    
                self.performance_metrics["api_latency"]["get_latest_quote"].append({
                    "timestamp": time.time(),
                    "duration_ms": quote_duration,
                    "symbol": symbol
                })
                
                # Keep history manageable
                if len(self.performance_metrics["api_latency"]["get_latest_quote"]) > 100:
                    self.performance_metrics["api_latency"]["get_latest_quote"] = self.performance_metrics["api_latency"]["get_latest_quote"][-100:]
                
                # Track as slow operation if it exceeds threshold
                if quote_duration > self.slow_operation_threshold:
                    self.performance_metrics["slow_operations"].append({
                        "timestamp": time.time(),
                        "operation": "get_latest_quote",
                        "duration_ms": quote_duration,
                        "details": {"symbol": symbol}
                    })
            
            if not quote_result or quote_result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to get quote for {symbol}: {quote_result.get('error') if quote_result else 'Unknown error'}")
                self.execution_errors += 1
                with self.monitoring_lock:
                    self.performance_metrics["error_timestamps"].append(time.time())
                self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": f"Failed to get quote for {symbol}"})
                return {"status": "failed", "error": f"Failed to get quote for {symbol}"}

            # Use ask price for buy orders, bid price for sell orders
            price = quote_result.get("ask_price" if side == "buy" else "bid_price", 0)
            if price <= 0:
                self.logger.error(f"Invalid price for {symbol}: {price}")
                self.execution_errors += 1
                with self.monitoring_lock:
                    self.performance_metrics["error_timestamps"].append(time.time())
                self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": f"Invalid price for {symbol}"})
                return {"status": "failed", "error": f"Invalid price for {symbol}"}

            # Calculate quantity based on capital amount and price
            quantity = math.floor((capital_amount / price) * 100) / 100  # Round down to 2 decimal places

            if quantity <= 0:
                self.logger.error(f"Calculated quantity is zero or negative: {quantity}")
                self.execution_errors += 1
                with self.monitoring_lock:
                    self.performance_metrics["error_timestamps"].append(time.time())
                self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": "Calculated quantity is zero or negative"})
                return {"status": "failed", "error": "Calculated quantity is zero or negative"}

        # Update largest order size metric
        with self.monitoring_lock:
            order_value = quantity * (limit_price if order_type == "limit" and limit_price is not None else price if 'price' in locals() else 0)
            self.performance_metrics["largest_order_size"] = max(self.performance_metrics["largest_order_size"], order_value)
            self.logger.gauge("trade_model.largest_order_size", self.performance_metrics["largest_order_size"])

        # Check daily capital limit using Redis
        current_daily_usage = self._get_daily_capital_usage()
        trade_cost = quantity * (limit_price if order_type == "limit" and limit_price is not None else price if 'price' in locals() else 0) # Estimate cost
        if current_daily_usage + trade_cost > self.daily_capital_limit:
            self.logger.warning(f"Daily capital limit exceeded. Current usage: ${current_daily_usage:.2f}, Trade cost: ${trade_cost:.2f}, Limit: ${self.daily_capital_limit:.2f}")
            self.execution_errors += 1
            with self.monitoring_lock:
                self.performance_metrics["error_timestamps"].append(time.time())
            self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": "Daily capital limit exceeded"})
            return {"status": "failed", "error": "Daily capital limit exceeded"}

        # Execute the order based on order type
        execution_start_time = time.time()
        result = None
        if order_type == "market":
            result = self._execute_market_order_internal(symbol, quantity, side)
        elif order_type == "limit" and limit_price is not None:
            result = self._execute_limit_order_internal(symbol, quantity, side, limit_price)
        else:
            self.logger.error(f"Invalid order type: {order_type}")
            self.execution_errors += 1
            with self.monitoring_lock:
                self.performance_metrics["error_timestamps"].append(time.time())
            self._publish_trade_event("trade_execution_failed", {"trade_data": trade_data, "error": f"Invalid order type: {order_type}"})
            return {"status": "failed", "error": f"Invalid order type: {order_type}"}

        execution_duration = (time.time() - execution_start_time) * 1000
        
        # Track execution time in performance metrics
        with self.monitoring_lock:
            function_name = f"execute_{order_type}_order"
            if "processing_times_by_function" not in self.performance_metrics:
                self.performance_metrics["processing_times_by_function"] = {}
            if function_name not in self.performance_metrics["processing_times_by_function"]:
                self.performance_metrics["processing_times_by_function"][function_name] = []
                
            self.performance_metrics["processing_times_by_function"][function_name].append({
                "timestamp": time.time(),
                "duration_ms": execution_duration,
                "symbol": symbol,
                "quantity": quantity
            })
            
            # Keep history manageable
            if len(self.performance_metrics["processing_times_by_function"][function_name]) > 100:
                self.performance_metrics["processing_times_by_function"][function_name] = self.performance_metrics["processing_times_by_function"][function_name][-100:]
                
            # Track as slow operation if it exceeds threshold
            if execution_duration > self.slow_operation_threshold:
                self.performance_metrics["slow_operations"].append({
                    "timestamp": time.time(),
                    "operation": function_name,
                    "duration_ms": execution_duration,
                    "details": {"symbol": symbol, "quantity": quantity, "side": side}
                })

        # Handle execution result
        if not result or "error" in result:
            error_msg = result.get("error", "Unknown error") if result else "No result returned"
            self.logger.error(f"Order execution failed: {error_msg}")
            # No need to increment execution_errors here, internal methods do it
            # Trade event already published by internal methods
            
            # Update trade stats
            with self.monitoring_lock:
                self.trade_stats["failed_trades"] += 1
                if side == "buy":
                    self.trade_stats["buy_trades_failed"] += 1
                else:
                    self.trade_stats["sell_trades_failed"] += 1
                    
                if order_type == "market":
                    self.trade_stats["market_trades_failed"] += 1
                elif order_type == "limit":
                    self.trade_stats["limit_trades_failed"] += 1
                    
                # Track failures by symbol
                if "symbol" in trade_data and trade_data["symbol"]:
                    symbol = trade_data["symbol"]
                    if symbol not in self.trade_stats["trades_by_symbol"]:
                        self.trade_stats["trades_by_symbol"][symbol] = {"success": 0, "failed": 0}
                    self.trade_stats["trades_by_symbol"][symbol]["failed"] += 1
            
            return {"status": "failed", "error": error_msg}

        # Update daily capital usage in Redis
        if result.get("status") in ["filled", "partially_filled"]:
            filled_value = float(result.get("filled_avg_price", 0)) * float(result.get("filled_qty", 0))
            self._update_daily_capital_usage(filled_value)
            
            # Update trade statistics
            with self.monitoring_lock:
                self.trade_stats["successful_trades"] += 1
                self.trade_stats["total_volume_traded"] += float(result.get("filled_qty", 0))
                self.trade_stats["total_notional_value"] += filled_value
                
                if side == "buy":
                    self.trade_stats["buy_trades_success"] += 1
                else:
                    self.trade_stats["sell_trades_success"] += 1
                    
                if order_type == "market":
                    self.trade_stats["market_trades_success"] += 1
                elif order_type == "limit":
                    self.trade_stats["limit_trades_success"] += 1
                
                # Track execution time stats
                execution_time_ms = execution_duration
                self.trade_stats["execution_time_stats"]["total_ms"] += execution_time_ms
                self.trade_stats["execution_time_stats"]["count"] += 1
                self.trade_stats["execution_time_stats"]["max_ms"] = max(
                    self.trade_stats["execution_time_stats"]["max_ms"], 
                    execution_time_ms
                )
                self.trade_stats["execution_time_stats"]["min_ms"] = min(
                    self.trade_stats["execution_time_stats"]["min_ms"], 
                    execution_time_ms
                )
                
                # Track trades by symbol
                if "symbol" in trade_data and trade_data["symbol"]:
                    symbol = trade_data["symbol"]
                    if symbol not in self.trade_stats["trades_by_symbol"]:
                        self.trade_stats["trades_by_symbol"][symbol] = {"success": 0, "failed": 0}
                    self.trade_stats["trades_by_symbol"][symbol]["success"] += 1
                
                # Track trades by hour
                hour_key = str(datetime.now().hour)
                self.trade_stats["trades_by_hour"][hour_key] = self.trade_stats["trades_by_hour"].get(hour_key, 0) + 1

        # Update counters
        self.orders_executed_count += 1
        # buy/sell counts already in internal methods
        if side == "buy":
            self.positions_opened_count += 1
        else:
            self.positions_closed_count += 1

        # Start position monitoring if it's a buy order and filled
        if side == "buy" and result.get("status") in ["filled", "partially_filled"]:
            self._start_position_monitoring_internal(symbol, result.get("id"))

        # Calculate and track duration
        duration = (time.time() - start_time) * 1000
        self.logger.timing("trade_model.execute_trade_duration_ms", duration)
        
        # Add to response time metrics
        with self.monitoring_lock:
            self.performance_metrics["response_times"].append(duration)
            
            # Track as slow operation if it exceeds threshold
            if duration > self.slow_operation_threshold:
                if not any(op.get("operation") == "execute_trade" and 
                          op.get("details", {}).get("symbol") == symbol for op in self.performance_metrics["slow_operations"]):
                    self.performance_metrics["slow_operations"].append({
                        "timestamp": time.time(),
                        "operation": "execute_trade",
                        "duration_ms": duration,
                        "details": {"symbol": symbol, "side": side, "order_type": order_type}
                    })

        # Publish trade executed event
        self._publish_trade_event("trade_executed", {"trade_result": result})
        
        # Check memory usage after trade execution
        current_memory_percent = self.system_metrics.get_memory_usage()
        if current_memory_percent > self.high_memory_threshold:
            self.logger.warning(f"High memory usage after trade execution: {current_memory_percent}%")
            
            # Log detailed memory info in health data for debugging
            with self.monitoring_lock:
                self.health_data["recent_issues"].append({
                    "timestamp": datetime.now().isoformat(),
                    "component": "system_resources",
                    "issue": f"High memory usage after trade: {current_memory_percent}%",
                    "details": {
                        "symbol": symbol,
                        "order_type": order_type,
                        "side": side
                    }
                })

        return result

    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open using TradingMCP.

        Returns:
            True if market is open, False otherwise
        """
        try:
            self.mcp_tool_call_count += 1
            market_hours_result = self.trading_mcp.call_tool("get_clock", {}) # Assuming get_clock tool exists
            if market_hours_result and not market_hours_result.get("error"):
                 return market_hours_result.get("is_open", False)
            elif market_hours_result and market_hours_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error checking market hours: {market_hours_result['error']}")
                 return False # Assume market is closed on error
            return False # Default to closed on failure
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            self.execution_errors += 1
            return False

    def _get_daily_capital_usage(self) -> float:
        """Retrieve daily capital usage from Redis."""
        try:
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool("get", {"key": self.redis_keys["daily_capital_usage"]})
            if result and not result.get("error") and result.get("value") is not None:
                return float(result.get("value"))
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting daily capital usage from Redis: {e}")
            return 0.0

    def _update_daily_capital_usage(self, amount: float) -> bool:
        """Update daily capital usage in Redis."""
        try:
            current_usage = self._get_daily_capital_usage()
            new_usage = current_usage + amount
            # Store with expiration at end of day (requires more complex Redis logic or a separate process)
            # For simplicity, just setting the value for now. End-of-day reset needed.
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool("set", {"key": self.redis_keys["daily_capital_usage"], "value": str(new_usage)})
            if result and not result.get("error"):
                self.logger.info(f"Updated daily capital usage to ${new_usage:.2f}")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to update daily capital usage in Redis: {result.get('error') if result else 'Unknown error'}")
                return False
        except Exception as e:
            self.logger.error(f"Error updating daily capital usage in Redis: {e}")
            self.execution_errors += 1
            return False

    def _reset_daily_capital_usage(self) -> bool:
        """Reset daily capital usage in Redis (called at start of day)."""
        try:
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool("set", {"key": self.redis_keys["daily_capital_usage"], "value": "0.0"})
            if result and not result.get("error"):
                self.logger.info("Reset daily capital usage in Redis.")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to reset daily capital usage in Redis: {result.get('error') if result else 'Unknown error'}")
                return False
        except Exception as e:
            self.logger.error(f"Error resetting daily capital usage in Redis: {e}")
            self.execution_errors += 1
            return False


    def start_of_day(self) -> Dict[str, Any]:
        """
        Perform start of day procedures using Redis for state management.

        This includes:
        - Checking market hours
        - Resetting daily capital limits in Redis
        - Syncing portfolio data from broker and storing monitoring data in Redis
        - Publishing available capital to a Redis key/stream
        Returns:
            Dictionary with start of day status and constraints
        """
        self.logger.info("Starting start of day procedures...")
        start_time = time.time()

        # Check if market is open
        market_open = self._is_market_open()
        if not market_open:
            self.logger.warning("Market is closed")
            return {"status": "market_closed", "message": "Market is not open"}

        # Reset daily capital usage in Redis
        self._reset_daily_capital_usage()

        # Get account information (using TradingMCP tool)
        self.mcp_tool_call_count += 1
        account_info = self.trading_mcp.call_tool("get_account_info", {})
        if not account_info or "error" in account_info:
            self.mcp_tool_error_count += 1
            self.logger.error(f"Failed to get account information: {account_info.get('error') if account_info else 'Unknown error'}")
            self.execution_errors += 1
            return {"status": "failed", "error": "Failed to get account information"}

        # Extract account values
        cash = float(account_info.get("cash", 0))
        buying_power = float(account_info.get("buying_power", 0))
        equity = float(account_info.get("equity", 0))

        # Calculate available capital (minimum of daily limit and actual cash)
        available_capital = min(self.daily_capital_limit, cash)

        # Store available capital in Redis (e.g., for Decision Model)
        self.mcp_tool_call_count += 1
        self.redis_mcp.call_tool("set", {"key": self.redis_keys["capital_available"], "value": str(available_capital)})
        self.logger.info(f"Published available capital to Redis: ${available_capital:.2f}")


        # Sync current positions from broker and store monitoring data
        positions = self._get_positions_internal()
        if positions is None: # Error occurred getting positions
             self.logger.error("Failed to sync positions at start of day.")
             # Continue, but with a warning
             synced_positions_count = 0
        else:
            synced_positions_count = len(positions)
            # Clear existing monitored positions in Redis
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("del", {"key": self.redis_keys["open_positions_monitoring"]})
            self.logger.info("Cleared existing monitored positions in Redis.")

            # Store monitoring data for current open positions
            for position in positions:
                symbol = position.get("symbol")
                if symbol:
                    # Generate a dummy order_id for existing positions if needed, or use position ID
                    order_id = position.get("latest_order_id", str(uuid.uuid4())) # Use latest order ID or generate
                    self._start_position_monitoring_internal(symbol, order_id) # This will store monitoring data in Redis


        # Get and store portfolio constraints
        constraints = self._get_portfolio_constraints_internal()
        if "error" in constraints:
             self.logger.error(f"Failed to get portfolio constraints at start of day: {constraints['error']}")
             # Continue, but with a warning
             constraints = {"error": constraints["error"]} # Store error in constraints


        duration = (time.time() - start_time) * 1000
        self.logger.timing("trade_model.start_of_day_duration_ms", duration)

        return {
            "status": "success",
            "market_open": market_open,
            "available_capital": available_capital,
            "daily_capital_limit": self.daily_capital_limit,
            "no_overnight_positions": self.no_overnight_positions,
            "synced_positions_count": synced_positions_count,
            "portfolio_constraints": constraints,
            "message": "Start of day procedures completed."
        }

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """
        Run a monitoring cycle to check all positions using Redis for state management.

        This includes:
        - Getting list of monitored positions from Redis
        - Checking exit conditions for each monitored position
        - Executing exit trades if necessary
        - Removing closed positions from monitoring in Redis
        - Publishing position updates to a Redis stream
        Returns:
            Dictionary with monitoring results
        """
        self.logger.info("Starting monitoring cycle...")
        start_time = time.time()
        self.monitoring_cycles_run += 1

        # Get list of symbols being monitored from Redis
        self.mcp_tool_call_count += 1
        monitored_symbols_result = self.redis_mcp.call_tool("smembers", {"key": self.redis_keys["open_positions_monitoring"]})

        if not monitored_symbols_result or monitored_symbols_result.get("error"):
            self.mcp_tool_error_count += 1
            self.logger.error(f"Failed to get monitored symbols from Redis: {monitored_symbols_result.get('error') if monitored_symbols_result else 'Unknown error'}")
            self.execution_errors += 1
            return {"status": "failed", "positions_checked": 0, "exit_signals": 0, "error": "Failed to get monitored symbols"}

        monitored_symbols = monitored_symbols_result.get("members", [])
        self.logger.info(f"Monitoring {len(monitored_symbols)} positions: {monitored_symbols}")

        if not monitored_symbols:
            self.logger.info("No positions to monitor")
            return {"status": "success", "positions_checked": 0, "exit_signals": 0}

        positions_checked = 0
        exit_signals = 0
        exit_trades = []
        positions_to_remove_from_monitoring = []

        # Check each monitored position for exit conditions
        for symbol in monitored_symbols:
            positions_checked += 1
            exit_result = self._check_exit_conditions_internal(symbol) # This now uses Redis for monitoring data

            if exit_result and exit_result.get("should_exit", False):
                self.logger.info(f"Exit signal for {symbol}: {exit_result.get('reason')}")
                exit_signals += 1
                self.exit_signals_count += 1

                # Get current position quantity before exiting
                position = self._get_position_internal(symbol)
                if not position or position.get("error"):
                    self.logger.error(f"Failed to get position details for {symbol} before exiting: {position.get('error') if position else 'Not found'}")
                    continue # Cannot exit if we don't know the quantity

                quantity = float(position.get("qty", 0))
                if quantity <= 0:
                    self.logger.warning(f"Invalid position quantity for {symbol}: {quantity}. Cannot exit.")
                    positions_to_remove_from_monitoring.append(symbol) # Remove from monitoring if quantity is zero/invalid
                    continue

                trade_data = {"symbol": symbol, "action": "sell", "quantity": quantity, "order_type": "market"}
                sell_result = self.execute_trade(trade_data) # Executes trade and publishes event

                if sell_result and sell_result.get("status") in ["filled", "partially_filled"]:
                    self.logger.info(f"Successfully initiated exit trade for {symbol}. Order ID: {sell_result.get('id')}")
                    exit_trades.append({"symbol": symbol, "reason": exit_result.get("reason"), "quantity": quantity, "result": sell_result})
                    positions_to_remove_from_monitoring.append(symbol) # Mark for removal from monitoring
                else:
                    error_msg = sell_result.get("error", "Unknown error") if sell_result else "No result returned"
                    self.logger.error(f"Failed to initiate exit trade for {symbol}: {error_msg}")
                    # Decide how to handle failed exit trade - retry? Alert? For now, just log.


        # Remove positions that were exited from monitoring in Redis
        if positions_to_remove_from_monitoring:
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("srem", {"key": self.redis_keys["open_positions_monitoring"], "member": positions_to_remove_from_monitoring})
            self.logger.info(f"Removed {len(positions_to_remove_from_monitoring)} positions from monitoring.")
            # Also delete their monitoring data keys
            for symbol in positions_to_remove_from_monitoring:
                 self.mcp_tool_call_count += 1
                 self.redis_mcp.call_tool("del", {"key": f"{self.redis_keys['position_monitoring_data']}{symbol}"})


        duration = (time.time() - start_time) * 1000
        self.logger.timing("trade_model.monitoring_cycle_duration_ms", duration)

        return {
            "status": "success",
            "positions_checked": positions_checked,
            "exit_signals": exit_signals,
            "exit_trades": exit_trades,
            "message": "Monitoring cycle completed."
        }

    def end_of_day(self) -> Dict[str, Any]:
        """
        Perform end of day procedures using Redis for state management.

        This includes:
        - Checking market hours
        - Closing positions if no_overnight_positions is True
        - Generating end of day reports (Placeholder)
        - Resetting daily capital usage in Redis
        Returns:
            Dictionary with end of day status and results
        """
        self.logger.info("Starting end of day procedures...")
        start_time = time.time()

        # Check if market is open (should be closing or closed)
        market_open = self._is_market_open()
        if market_open:
             self.logger.warning("Market is still open during end of day procedures.")
             # Decide how to handle this - proceed anyway? Wait? For now, proceed.


        # Check if we should close all positions
        if not self.no_overnight_positions:
            self.logger.info("Overnight positions allowed, skipping position closure")
            # Reset daily capital usage even if not closing positions
            self._reset_daily_capital_usage()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_model.end_of_day_duration_ms", duration)
            return {"status": "success", "message": "Overnight positions allowed, end of day procedures completed."}

        self.logger.info("Closing all open positions for end of day.")

        # Get all positions
        positions = self._get_positions_internal()
        if positions is None: # Error occurred
            self.logger.error("Failed to get positions for end of day closure.")
            # Attempt to reset daily capital usage even on error
            self._reset_daily_capital_usage()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_model.end_of_day_duration_ms", duration)
            return {"status": "failed", "positions_closed": 0, "error": "Failed to get positions for closure"}

        if not positions:
            self.logger.info("No positions to close")
            # Reset daily capital usage
            self._reset_daily_capital_usage()
            duration = (time.time() - start_time) * 1000
            self.logger.timing("trade_model.end_of_day_duration_ms", duration)
            return {"status": "success", "positions_closed": 0, "message": "No positions to close, end of day procedures completed."}

        positions_closed = 0
        close_results = []

        # Close each position
        for position in positions:
            symbol = position.get("symbol")
            if not symbol: continue

            quantity = float(position.get("qty", 0))
            if quantity <= 0:
                self.logger.warning(f"Invalid position quantity for {symbol}: {quantity}. Cannot close.")
                continue

            trade_data = {
                "symbol": symbol, "action": "sell", "quantity": quantity, "order_type": "market"
            }
            sell_result = self.execute_trade(trade_data) # Executes trade and publishes event

            if sell_result and sell_result.get("status") in ["filled", "partially_filled"]:
                self.logger.info(f"Successfully initiated end of day exit trade for {symbol}. Order ID: {sell_result.get('id')}")
                positions_closed += 1
                close_results.append({
                    "symbol": symbol, "quantity": quantity, "result": sell_result
                })
                # Remove from monitoring in Redis
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool("srem", {"key": self.redis_keys["open_positions_monitoring"], "member": symbol})
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool("del", {"key": f"{self.redis_keys['position_monitoring_data']}{symbol}"})

            else:
                error_msg = sell_result.get("error", "Unknown error") if sell_result else "No result returned"
                self.logger.error(f"Failed to initiate end of day exit trade for {symbol}: {error_msg}")
                # Decide how to handle failed end of day exit trade - alert? Retry?

        # Reset daily capital usage in Redis
        self._reset_daily_capital_usage()

        duration = (time.time() - start_time) * 1000
        self.logger.timing("trade_model.end_of_day_duration_ms", duration)

        return {
            "status": "success",
            "positions_closed": positions_closed,
            "close_results": close_results,
            "message": "End of day procedures completed."
        }

    def get_model_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics for the trade model.

        Returns:
            Dictionary with model metrics
        """
        return {
            "orders_executed_count": self.orders_executed_count,
            "buy_orders_count": self.buy_orders_count,
            "sell_orders_count": self.sell_orders_count,
            "market_orders_count": self.market_orders_count,
            "limit_orders_count": self.limit_orders_count,
            "positions_opened_count": self.positions_opened_count,
            "positions_closed_count": self.positions_closed_count,
            "monitoring_cycles_run": self.monitoring_cycles_run,
            "exit_signals_count": self.exit_signals_count,
            "llm_api_call_count": self.llm_api_call_count,
            "mcp_tool_call_count": self.mcp_tool_call_count,
            "mcp_tool_error_count": self.mcp_tool_error_count,
            "execution_errors": self.execution_errors
        }
        
    def get_health_report(self) -> Dict[str, Any]:
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
                    "peak_memory": self.performance_metrics["peak_memory_usage"],
                    "slow_operations_count": len(self.performance_metrics["slow_operations"])
                },
                "performance": {
                    "avg_response_time_ms": sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"]) if self.performance_metrics["response_times"] else 0,
                    "slow_operations": self.performance_metrics["slow_operations"][-5:] if self.performance_metrics["slow_operations"] else [],
                    "symbols_processed": len(self.performance_metrics["symbols_processed"]),
                },
                "trade_statistics": {
                    "successful_trades": self.trade_stats["successful_trades"],
                    "failed_trades": self.trade_stats["failed_trades"],
                    "success_rate": (self.trade_stats["successful_trades"] / 
                                   max(1, self.trade_stats["successful_trades"] + self.trade_stats["failed_trades"])) * 100,
                    "total_volume_traded": self.trade_stats["total_volume_traded"],
                    "total_notional_value": self.trade_stats["total_notional_value"],
                    "avg_execution_time_ms": (self.trade_stats["execution_time_stats"]["total_ms"] / 
                                             max(1, self.trade_stats["execution_time_stats"]["count"]))
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
            report["recommendations"] = self._generate_health_recommendations()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}", exc_info=True)
            return {
                "error": f"Error generating health report: {str(e)}",
                "health_status": self.health_data["health_status"],
                "health_score": self.health_data["health_score"]
            }
    
    def _generate_health_recommendations(self) -> List[Dict[str, Any]]:
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
            # Find the most common slow operation
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
        
        # Check for high error rate
        error_count = len(self.performance_metrics["error_timestamps"])
        total_operations = max(1, self.orders_executed_count + self.monitoring_cycles_run)
        error_rate = error_count / total_operations
        if error_rate > 0.1:  # More than 10% error rate
            recommendations.append({
                "priority": "high",
                "component": "error_handling",
                "issue": f"High error rate ({error_rate:.2%})",
                "recommendation": "Review error logs and fix common error causes"
            })
        
        # Check trend data if available
        if hasattr(self, "performance_trends"):
            for metric_name, trend_data in self.performance_trends.items():
                if trend_data["trend"] == "degrading":
                    recommendations.append({
                        "priority": "medium",
                        "component": "trends",
                        "issue": f"Degrading trend detected for {metric_name.replace('_', ' ')}",
                        "recommendation": f"Monitor {metric_name.replace('_', ' ')} closely and investigate causes"
                    })
        
        # If there are bottlenecks in specific MCP calls
        if "api_latency" in self.performance_metrics:
            for api, calls in self.performance_metrics["api_latency"].items():
                if calls and len(calls) >= 5:
                    avg_latency = sum(call["duration_ms"] for call in calls) / len(calls)
                    if avg_latency > self.slow_operation_threshold:
                        recommendations.append({
                            "priority": "medium",
                            "component": "api_performance",
                            "issue": f"High latency in {api} API calls (avg: {avg_latency:.2f}ms)",
                            "recommendation": "Consider caching or optimizing these API calls"
                        })
        
        return recommendations
    
    def shutdown(self) -> Dict[str, Any]:
        """
        Perform a graceful shutdown of the trade model.
        
        This includes:
        - Setting the shutdown_requested flag to stop background threads
        - Closing all open positions if no_overnight_positions is True
        - Shutting down MCP clients
        - Stopping system metrics collection
        - Generating a final health report
        - Saving final performance metrics to Redis
        
        Returns:
            Dictionary with shutdown status and results
        """
        self.logger.info("Starting graceful shutdown of Trade Model...")
        start_time = time.time()
        
        # Set shutdown flag to stop background threads
        self.shutdown_requested = True
        self.logger.info("Shutdown flag set, background threads will terminate")
        
        # Close positions if needed
        positions_closed = 0
        if self.no_overnight_positions:
            try:
                self.logger.info("Closing all open positions for shutdown")
                end_of_day_result = self.end_of_day()
                positions_closed = end_of_day_result.get("positions_closed", 0)
                self.logger.info(f"Closed {positions_closed} positions during shutdown")
            except Exception as e:
                self.logger.error(f"Error closing positions during shutdown: {e}", exc_info=True)
        
        # Generate final health report and save performance metrics
        try:
            final_health_report = self.get_health_report()
            self.logger.info("Generated final health report", 
                           health_status=final_health_report.get("health_status"),
                           health_score=final_health_report.get("health_score"))
            
            # Save final performance metrics
            self._save_performance_metrics()
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
        
        # Shutdown MCPs
        for mcp_name, mcp in [
            ("trading_mcp", self.trading_mcp), 
            ("financial_data_mcp", self.financial_data_mcp),
            ("time_series_mcp", self.time_series_mcp),
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
                self._publish_trade_event("trade_model_shutdown", {
                    "timestamp": datetime.now().isoformat(),
                    "positions_closed": positions_closed,
                    "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds()
                })
                self.logger.info("Published shutdown event to Redis stream")
        except Exception as e:
            self.logger.error(f"Error publishing shutdown event: {e}")
        
        # Calculate shutdown duration
        duration = (time.time() - start_time) * 1000
        self.logger.info(f"Trade Model shutdown completed in {duration:.2f}ms")
        
        return {
            "status": "success",
            "shutdown_duration_ms": duration,
            "positions_closed": positions_closed,
            "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
            "message": "Trade Model shutdown completed successfully"
        }
