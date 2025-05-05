"""
Risk Assessment Model

This module defines the RiskAssessmentModel, responsible for evaluating portfolio risk
using scenario generation, risk attribution, and stress testing capabilities.
It integrates with various MCP tools to provide comprehensive risk analysis.

Features:
- Advanced risk metrics calculation (VaR, Expected Shortfall)
- Scenario generation and stress testing
- Risk attribution and factor analysis
- Portfolio optimization recommendations
- Risk limit monitoring and alerting
- Comprehensive monitoring and performance metrics
- Health checking and diagnostics
- Visualization of risk metrics and performance
"""

import json
import time
import os
import asyncio
import hashlib
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
load_dotenv()

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator
from monitoring.system_metrics import SystemMetricsCollector

# MCP tools (Consolidated)
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
)


class PortfolioAnalyzer:
    pass


class RiskManager:
    pass


class MarketStateAnalyzer:
    pass


class DecisionModel:
    pass


class RiskAssessmentModel:
    """
    Evaluates portfolio risk using scenario generation, risk attribution, and stress testing.

    Acts as a central processing hub that:
    1. Collects and analyzes reports from all processing models
    2. Integrates data into comprehensive risk-assessed packages
    3. Provides consolidated reports to the Decision Model
    4. Coordinates with MCP tools to evaluate portfolio and position risk
    
    Features:
    - Real-time risk metrics calculation and monitoring
    - Comprehensive performance tracking and analytics
    - Detailed health checking and diagnostics
    - Visualization of risk metrics and market conditions
    - Advanced risk attribution and scenario analysis
    - Inter-model communication and data integration
    - Auto-scaling computation based on portfolio complexity
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RiskAssessmentModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - risk_analysis_config: Config for RiskAnalysisMCP
                - redis_config: Config for RedisMCP
                - llm_config: Configuration for AutoGen LLM
                - default_confidence_level: Default confidence level for risk metrics (default: 0.95)
                - default_time_horizon: Default time horizon in days (default: 20)
                - risk_data_ttl: Time-to-live for risk data in seconds (default: 86400 - 1 day)
                - monitoring_config: Configuration for system monitoring and metrics collection
        """
        init_start_time = time.time()
        
        # Initialize monitoring lock for thread safety
        self.monitoring_lock = threading.RLock()
        
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-risk-assessment-model")
        self.logger.info("Starting RiskAssessmentModel initialization")
        
        # Initialize system metrics collector for resource usage monitoring
        self.system_metrics = SystemMetricsCollector(self.logger)
        self.logger.info("SystemMetricsCollector initialized")
        
        # Initialize StockChartGenerator for risk visualization
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")
        
        # Initialize counters for risk assessment metrics
        self.portfolio_analysis_count = 0
        self.risk_limit_checks_count = 0
        self.risk_limit_exceeded_count = 0
        self.risk_alerts_generated_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0  # Errors during risk assessment process
        self.total_risk_analysis_cycles = 0  # Total times analyze_portfolio_risk is called
        
        # Initialize performance metrics with timestamps
        self.performance_metrics = {
            "start_time": datetime.now(),
            "last_activity_time": datetime.now(),
            "response_times": [],  # List of recent response times
            "error_timestamps": [],  # Timestamps of recent errors
            "processing_times_by_function": {},  # Dict of function processing times
            "memory_usage_history": [],  # Track memory usage over time
            "cpu_usage_history": [],  # Track CPU usage over time
            "risk_metric_calculation_times": {},  # Track time to calculate different risk metrics
            "symbols_processed": set(),  # Set of unique symbols processed
            "largest_portfolio_size": 0,  # Largest portfolio size processed
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
        
        # Initialize statistics for risk metrics
        self.risk_metrics_stats = {
            "var_calculations": 0,
            "es_calculations": 0,
            "volatility_calculations": 0,
            "beta_calculations": 0,
            "correlation_calculations": 0,
            "scenario_generations": 0,
            "optimization_runs": 0,
            "factor_analyses": 0,
            "var_distribution": {},  # Distribution of VaR values by range
            "es_distribution": {},  # Distribution of ES values by range
            "vol_distribution": {},  # Distribution of volatility values by range
        }
        
        # Track performance of inter-model communication
        self.intermodel_metrics = {
            "model_report_collection_count": 0,
            "model_report_collection_times": [],
            "decision_model_notifications": 0,
            "selection_model_interactions": 0,
            "last_model_data_timestamps": {},
            "missing_model_data_counts": {},
        }
        
        # Cache for risk calculations to improve performance
        self.calculation_cache = {
            "var_cache": {},  # Cache for VaR calculations
            "es_cache": {},  # Cache for ES calculations
            "scenario_cache": {},  # Cache for scenario generations
            "hit_count": 0,
            "miss_count": 0,
            "last_cleanup": datetime.now(),
        }
        
        # Initialize health check data
        self.health_data = {
            "last_health_check": datetime.now(),
            "health_check_count": 0,
            "health_status": "initializing",  # initializing, healthy, degraded, error
            "health_score": 100,  # 0-100 score
            "component_health": {  # Health of individual components
                "risk_analysis_mcp": "unknown",
                "redis_mcp": "unknown",
                "llm": "unknown",
                "intermodel_communication": "unknown",
            },
            "recent_issues": [],  # List of recent health issues
        }
        
        # Internal state (Example: could store portfolio details or capital)
        self.current_portfolio = {}  # Example: {'AAPL': {'shares': 100, 'avg_price': 150.0}, ...}
        self.current_available_capital = None  # Example: 100000.0
        
        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_risk_assessment", "risk_assessment_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}", file_path=config_path)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}", error_type="JSONDecodeError")
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}", error_type=type(e).__name__)
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}", file_path=config_path)
                self.config = {}
        else:
            self.config = config
            self.logger.info("Using provided configuration")
        
        # Configure health and performance thresholds
        self.monitoring_config = self.config.get("monitoring_config", {})
        self.health_check_interval = self.monitoring_config.get("health_check_interval", 60)  # seconds
        self.performance_check_interval = self.monitoring_config.get("performance_check_interval", 300)  # seconds
        self.slow_operation_threshold = self.monitoring_config.get("slow_operation_threshold", 1000)  # ms
        self.high_memory_threshold = self.monitoring_config.get("high_memory_threshold", 75)  # percent
        self.high_cpu_threshold = self.monitoring_config.get("high_cpu_threshold", 80)  # percent
        self.enable_detailed_metrics = self.monitoring_config.get("enable_detailed_metrics", True)
        self.cache_ttl = self.monitoring_config.get("cache_ttl", 3600)  # 1 hour default for calculation cache
        
        # Initialize RiskAnalysisMCP with comprehensive monitoring
        risk_analysis_config = self.config.get("risk_analysis_config", {})
        # Add logger to config if not present
        if "logger" not in risk_analysis_config:
            risk_analysis_config["logger"] = self.logger
        try:
            self.risk_analysis_mcp = RiskAnalysisMCP(risk_analysis_config)
            self.health_data["component_health"]["risk_analysis_mcp"] = "healthy"
            self.logger.info("RiskAnalysisMCP initialized successfully")
        except Exception as e:
            self.health_data["component_health"]["risk_analysis_mcp"] = "error"
            self.health_data["recent_issues"].append({
                "timestamp": datetime.now().isoformat(),
                "component": "risk_analysis_mcp",
                "error": str(e)
            })
            self.logger.error(f"Error initializing RiskAnalysisMCP: {e}", exc_info=True)
            # Create a fallback or placeholder if needed
            self.risk_analysis_mcp = None
        
        # Initialize Redis MCP client with monitoring
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
            
        # Configuration parameters
        self.default_confidence_level = self.config.get("default_confidence_level", 0.95)
        self.default_time_horizon = self.config.get("default_time_horizon", 20)  # 20 trading days (approx. 1 month)
        self.risk_data_ttl = self.config.get("risk_data_ttl", 86400)  # Default: 1 day
        
        # Redis keys for data storage and model interactions
        self.redis_keys = {
            # Risk model internal keys
            "portfolio_risk": "risk:portfolio:",  # Prefix for portfolio risk data
            "scenario_results": "risk:scenarios:",  # Prefix for scenario results
            "risk_attribution": "risk:attribution:",  # Prefix for risk attribution data
            "risk_metrics": "risk:metrics:",  # Prefix for risk metrics
            "optimization_results": "risk:optimization:",  # Prefix for optimization results
            "latest_analysis": "risk:latest_analysis",  # Latest analysis timestamp
            "risk_limits": "risk:limits:",  # Prefix for risk limits
            "risk_alerts": "risk:alerts",  # Risk alerts
            "internal_portfolio_state": "risk:internal:portfolio",  # Key for storing internal portfolio state
            "internal_capital_state": "risk:internal:capital",  # Key for storing internal capital state
            
            # Keys for accessing other model reports (using Redis for inter-model communication)
            "sentiment_data": "sentiment:data",  # Sentiment analysis reports
            "fundamental_data": "fundamental:data:",  # Prefix for fundamental data per symbol
            "technical_data": "market_analysis:indicators:",  # Prefix for technical data per symbol
            "market_data": "market_analysis:data",  # Overall market analysis data
            "selection_data": "selection:data",  # Selection Model data
            
            # Keys for storing consolidated packages
            "consolidated_package": "risk:consolidated_package:",  # Prefix for consolidated risk packages
            "package_history": "risk:package_history",  # History of package IDs
            
            # Trade model interaction keys
            "trade_events": "trade:events",  # Trade event stream
            "capital_available": "trade:capital_available",  # Available capital
            
            # Streams for receiving reports from other models
            "sentiment_reports_stream": "model:sentiment:analysis",
            "fundamental_reports_stream": "model:fundamental:insights",
            "market_reports_stream": "model:market:trends",
            "technical_reports_stream": "model:technical:analysis",
            "selection_reports_stream": "model:selection:candidates",
            # Note: Selection Model publishes to model:selection:candidates, not selection:data
            
            # Stream for sending consolidated packages to Decision Model
            "decision_package_stream": "model:risk:assessments",
            
            # Health monitoring keys
            "health_status": "risk:health:status",  # Current health status
            "performance_metrics": "risk:health:performance",  # Performance metrics
        }
        
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
            self._register_functions()
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
        
        # Record initialization completion
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("risk_assessment_model.initialization_time_ms", init_duration)
        
        # Save initial performance metrics to Redis
        self._save_performance_metrics()
        
        # Log comprehensive initialization statistics
        self.logger.info("RiskAssessmentModel initialization complete", 
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
            
            # Initialize shutdown flag
            self.shutdown_requested = False
            
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
                    
                    # Perform cache cleanup if needed
                    if (datetime.now() - self.calculation_cache["last_cleanup"]).total_seconds() > self.cache_ttl:
                        self._cleanup_calculation_cache()
                    
                    # Log health status
                    if health_check_counter % 5 == 0:  # Detailed log every 5th check
                        self.logger.info("Detailed health check completed", 
                                       health_status=self.health_data["health_status"],
                                       health_score=self.health_data["health_score"],
                                       component_health=self.health_data["component_health"],
                                       recent_issues_count=len(self.health_data["recent_issues"]),
                                       portfolio_analysis_count=self.portfolio_analysis_count,
                                       risk_limit_checks_count=self.risk_limit_checks_count,
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
                # Check RiskAnalysisMCP health
                if self.risk_analysis_mcp is not None:
                    try:
                        # Try a simple operation to check if it's responsive
                        if hasattr(self.risk_analysis_mcp, "health_check"):
                            risk_health = self.risk_analysis_mcp.health_check()
                            self.health_data["component_health"]["risk_analysis_mcp"] = risk_health.get("status", "unknown")
                        else:
                            # Fallback to checking if it responds to a method call
                            method_exists = hasattr(self.risk_analysis_mcp, "call_tool")
                            self.health_data["component_health"]["risk_analysis_mcp"] = "healthy" if method_exists else "degraded"
                    except Exception as e:
                        self.health_data["component_health"]["risk_analysis_mcp"] = "error"
                        self.health_data["recent_issues"].append({
                            "timestamp": datetime.now().isoformat(),
                            "component": "risk_analysis_mcp",
                            "error": str(e)
                        })
                        self.logger.error(f"Error checking RiskAnalysisMCP health: {e}")
                
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
                
                # Check inter-model communication health
                try:
                    # Check if we're successfully communicating with other models
                    if self.intermodel_metrics["model_report_collection_count"] > 0:
                        # Calculate success rate based on missing data counts
                        total_interactions = sum(self.intermodel_metrics["missing_model_data_counts"].values())
                        if total_interactions == 0:
                            self.health_data["component_health"]["intermodel_communication"] = "healthy"
                        else:
                            miss_rate = total_interactions / max(1, self.intermodel_metrics["model_report_collection_count"])
                            if miss_rate < 0.1:  # Less than 10% missing data
                                self.health_data["component_health"]["intermodel_communication"] = "healthy"
                            elif miss_rate < 0.3:  # 10-30% missing data
                                self.health_data["component_health"]["intermodel_communication"] = "warning"
                            else:  # More than 30% missing data
                                self.health_data["component_health"]["intermodel_communication"] = "degraded"
                    else:
                        # No interactions yet
                        self.health_data["component_health"]["intermodel_communication"] = "unknown"
                except Exception as e:
                    self.health_data["component_health"]["intermodel_communication"] = "unknown"
                    self.logger.error(f"Error checking inter-model communication health: {e}")
                
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
                    self.logger.gauge("risk_assessment_model.cpu_usage", cpu_percent)
                    self.logger.gauge("risk_assessment_model.memory_usage", memory_percent)
                    
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
        self.logger.gauge("risk_assessment_model.health_score", self.health_data["health_score"])
    
    def _update_performance_metrics(self):
        """
        Update and collect comprehensive performance metrics.
        This enhanced version includes trend analysis and anomaly detection.
        """
        try:
            with self.monitoring_lock:
                # Calculate uptime
                uptime = (datetime.now() - self.performance_metrics["start_time"]).total_seconds()
                
                # Calculate average response time from recent data
                avg_response_time = 0
                if self.performance_metrics["response_times"]:
                    avg_response_time = sum(self.performance_metrics["response_times"]) / len(self.performance_metrics["response_times"])
                
                # Calculate error rate
                error_rate = 0
                if self.total_risk_analysis_cycles > 0:
                    error_rate = self.execution_errors / self.total_risk_analysis_cycles
                
                # Calculate cache efficiency
                cache_hit_rate = 0
                if (self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"]) > 0:
                    cache_hit_rate = self.calculation_cache["hit_count"] / (self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"])
                
                # Calculate average CPU and memory usage
                avg_cpu = 0
                avg_memory = 0
                if self.performance_metrics["cpu_usage_history"]:
                    avg_cpu = sum(self.performance_metrics["cpu_usage_history"]) / len(self.performance_metrics["cpu_usage_history"])
                if self.performance_metrics["memory_usage_history"]:
                    avg_memory = sum(self.performance_metrics["memory_usage_history"]) / len(self.performance_metrics["memory_usage_history"])
                
                # Initialize trend tracking if needed
                if not hasattr(self, "performance_trends"):
                    self.performance_trends = {
                        "response_time": {
                            "history": [],
                            "last_update": time.time(),
                            "trend": "stable",  # stable, improving, degrading
                            "anomalies": []
                        },
                        "error_rate": {
                            "history": [],
                            "last_update": time.time(),
                            "trend": "stable",
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
                        }
                    }
                
                # Update trend data (every 5 minutes)
                now = time.time()
                
                # Add current metrics to the trend history
                if now - self.performance_trends["response_time"]["last_update"] >= 300:  # 5 minutes
                    # Update response time trend
                    self.performance_trends["response_time"]["history"].append({
                        "timestamp": now,
                        "value": avg_response_time
                    })
                    
                    # Update error rate trend
                    self.performance_trends["error_rate"]["history"].append({
                        "timestamp": now,
                        "value": error_rate
                    })
                    
                    # Update memory usage trend
                    self.performance_trends["memory_usage"]["history"].append({
                        "timestamp": now,
                        "value": avg_memory
                    })
                    
                    # Update CPU usage trend
                    self.performance_trends["cpu_usage"]["history"].append({
                        "timestamp": now,
                        "value": avg_cpu
                    })
                    
                    # Keep last 24 hours (288 5-minute samples)
                    for metric in self.performance_trends:
                        if len(self.performance_trends[metric]["history"]) > 288:
                            self.performance_trends[metric]["history"] = self.performance_trends[metric]["history"][-288:]
                    
                    # Calculate trends for each metric
                    for metric_name, metric_data in self.performance_trends.items():
                        history = metric_data["history"]
                        if len(history) >= 3:  # Need at least 3 data points for trend
                            # Get the most recent and oldest values within the last hour
                            recent_values = [entry["value"] for entry in history[-12:]]  # Last hour (12 x 5 minutes)
                            if len(recent_values) >= 3:
                                # Simple linear regression to detect trend
                                x = list(range(len(recent_values)))
                                y = recent_values
                                n = len(x)
                                if n > 0:
                                    # Calculate slope of the trend line
                                    x_mean = sum(x) / n
                                    y_mean = sum(y) / n
                                    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
                                    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                                    
                                    # Avoid division by zero
                                    if denominator != 0:
                                        slope = numerator / denominator
                                        
                                        # Determine trend direction
                                        # For error rate and resource usage, positive slope is bad
                                        # For response time, positive slope is bad
                                        is_improving = slope < 0
                                        
                                        # If response time trend is improving
                                        if is_improving:
                                            metric_data["trend"] = "improving"
                                        elif slope > 0:
                                            metric_data["trend"] = "degrading"
                                        else:
                                            metric_data["trend"] = "stable"
                                        
                                        # Calculate standard deviation to detect anomalies
                                        mean = sum(recent_values) / len(recent_values)
                                        variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)
                                        std_dev = variance ** 0.5
                                        
                                        # Check if current value is an anomaly (> 2 std devs from mean)
                                        current_value = recent_values[-1]
                                        if std_dev > 0 and abs(current_value - mean) > 2 * std_dev:
                                            # Record anomaly
                                            metric_data["anomalies"].append({
                                                "timestamp": now,
                                                "value": current_value,
                                                "mean": mean,
                                                "std_dev": std_dev,
                                                "threshold": 2 * std_dev
                                            })
                                            # Keep only most recent 20 anomalies
                                            if len(metric_data["anomalies"]) > 20:
                                                metric_data["anomalies"] = metric_data["anomalies"][-20:]
                                            
                                            # Log anomaly
                                            self.logger.warning(f"Performance anomaly detected in {metric_name}", 
                                                              current_value=current_value,
                                                              mean=mean,
                                                              deviation=f"{abs(current_value - mean) / std_dev:.1f} σ")
                    
                    # Update last update timestamp
                    for metric in self.performance_trends:
                        self.performance_trends[metric]["last_update"] = now
                        
                    # Generate performance predictions
                    try:
                        self.generate_performance_predictions()
                    except Exception as predict_err:
                        self.logger.error(f"Error generating performance predictions: {predict_err}", exc_info=True)
                
                # Enhanced metrics for gauges
                # Calculate rate of change for key metrics
                if hasattr(self, "last_metrics"):
                    time_diff = now - self.last_metrics["timestamp"]
                    if time_diff > 0:
                        # Calculate rate of change for error count (errors per second)
                        error_diff = self.execution_errors - self.last_metrics["execution_errors"]
                        error_rate_per_second = error_diff / time_diff
                        self.logger.gauge("risk_assessment_model.error_rate_per_second", error_rate_per_second)
                        
                        # Calculate rate of change for portfolio analyses
                        portfolio_analyses_diff = self.portfolio_analysis_count - self.last_metrics["portfolio_analysis_count"]
                        portfolio_analyses_per_second = portfolio_analyses_diff / time_diff
                        self.logger.gauge("risk_assessment_model.portfolio_analyses_per_second", portfolio_analyses_per_second)
                
                # Save current metrics for rate calculations next time
                self.last_metrics = {
                    "timestamp": now,
                    "execution_errors": self.execution_errors,
                    "portfolio_analysis_count": self.portfolio_analysis_count,
                    "total_risk_analysis_cycles": self.total_risk_analysis_cycles
                }
                
                # Log performance metrics with trend information
                trend_info = {
                    "response_time": self.performance_trends["response_time"]["trend"],
                    "error_rate": self.performance_trends["error_rate"]["trend"],
                    "memory_usage": self.performance_trends["memory_usage"]["trend"],
                    "cpu_usage": self.performance_trends["cpu_usage"]["trend"]
                }
                
                self.logger.info("Performance metrics updated", 
                               uptime_seconds=uptime,
                               avg_response_time_ms=avg_response_time,
                               error_rate=f"{error_rate:.2%}",
                               cache_hit_rate=f"{cache_hit_rate:.2%}",
                               risk_analyses=self.total_risk_analysis_cycles,
                               portfolio_analyses=self.portfolio_analysis_count,
                               avg_cpu_usage=f"{avg_cpu:.1f}%",
                               avg_memory_usage=f"{avg_memory:.1f}%",
                               peak_memory_usage=f"{self.performance_metrics['peak_memory_usage']:.1f}%",
                               trends=trend_info)
                
                # Update gauges
                self.logger.gauge("risk_assessment_model.uptime_seconds", uptime)
                self.logger.gauge("risk_assessment_model.avg_response_time_ms", avg_response_time)
                self.logger.gauge("risk_assessment_model.error_rate", error_rate * 100)
                self.logger.gauge("risk_assessment_model.cache_hit_rate", cache_hit_rate * 100)
                
                # Track trend metrics as gauges
                for metric, data in trend_info.items():
                    trend_value = 0  # stable
                    if data == "improving":
                        trend_value = 1
                    elif data == "degrading":
                        trend_value = -1
                    self.logger.gauge(f"risk_assessment_model.{metric}_trend", trend_value)
                
                # Save performance metrics to Redis
                self._save_performance_metrics()
                
                # Prune lists to avoid memory growth
                self._prune_metrics_data()
        
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}", exc_info=True)
            
    def _save_performance_metrics(self):
        """
        Save performance metrics to Redis for external monitoring.
        """
        if self.redis_mcp is not None:
            try:
                # Create summary of performance metrics
                performance_summary = {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                    "risk_analyses": self.total_risk_analysis_cycles,
                    "portfolio_analyses": self.portfolio_analysis_count,
                    "risk_limit_checks": self.risk_limit_checks_count,
                    "risk_limit_violations": self.risk_limit_exceeded_count,
                    "execution_errors": self.execution_errors,
                    "llm_api_calls": self.llm_api_metrics["total_calls"],
                    "mcp_tool_calls": self.mcp_tool_call_count,
                    "mcp_tool_errors": self.mcp_tool_error_count,
                    "unique_symbols_processed": len(self.performance_metrics["symbols_processed"]),
                    "largest_portfolio_size": self.performance_metrics["largest_portfolio_size"],
                    "system_resources": {
                        "cpu_usage": self.system_metrics.get_cpu_usage(),
                        "memory_usage": self.system_metrics.get_memory_usage(),
                        "peak_memory_usage": self.performance_metrics["peak_memory_usage"]
                    },
                    "risk_metrics_stats": {
                        "var_calculations": self.risk_metrics_stats["var_calculations"],
                        "es_calculations": self.risk_metrics_stats["es_calculations"],
                        "scenario_generations": self.risk_metrics_stats["scenario_generations"]
                    },
                    "cache_metrics": {
                        "hit_count": self.calculation_cache["hit_count"],
                        "miss_count": self.calculation_cache["miss_count"],
                        "hit_rate": (self.calculation_cache["hit_count"] / max(1, self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"])) * 100
                    }
                }
                
                # Save to Redis
                self.redis_mcp.call_tool("set_json", {
                    "key": self.redis_keys["performance_metrics"],
                    "value": performance_summary,
                    "expiry": 86400  # 1 day TTL
                })
            except Exception as e:
                self.logger.error(f"Error saving performance metrics to Redis: {e}")
    
    def _generate_performance_visualizations(self):
        """
        Generate visualization charts for performance and risk metrics.
        """
        try:
            # Create directory for charts if it doesn't exist
            charts_dir = os.path.join("monitoring", "charts", "risk_assessment")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Track chart generation success
            charts_generated = []
            
            # 1. Generate risk metrics distribution chart
            if len(self.risk_metrics_stats["var_distribution"]) > 0:
                try:
                    # Prepare data for chart
                    risk_metric_data = {
                        "VaR Distribution": self.risk_metrics_stats["var_distribution"],
                        "ES Distribution": self.risk_metrics_stats["es_distribution"],
                        "Volatility Distribution": self.risk_metrics_stats["vol_distribution"]
                    }
                    
                    # Generate chart with stock chart generator
                    risk_chart = self.chart_generator.create_performance_chart(
                        risk_metric_data,
                        title="Risk Metrics Distribution",
                        include_timestamps=True
                    )
                    charts_generated.append(risk_chart)
                except Exception as e:
                    self.logger.warning(f"Error generating risk metrics chart: {e}")
            
            # 2. Generate performance metrics chart
            try:
                # Prepare data for chart
                perf_data = {
                    "Portfolio Analyses": self.portfolio_analysis_count,
                    "Risk Limit Checks": self.risk_limit_checks_count,
                    "Risk Limit Violations": self.risk_limit_exceeded_count,
                    "Execution Errors": self.execution_errors,
                    "LLM API Calls": self.llm_api_metrics["total_calls"],
                    "MCP Tool Calls": self.mcp_tool_call_count
                }
                
                # Generate chart with stock chart generator
                perf_chart = self.chart_generator.create_performance_chart(
                    perf_data,
                    title="Risk Assessment Performance Metrics",
                    include_timestamps=True
                )
                charts_generated.append(perf_chart)
            except Exception as e:
                self.logger.warning(f"Error generating performance metrics chart: {e}")
            
            # 3. Generate system resource usage chart
            if len(self.performance_metrics["cpu_usage_history"]) > 1:
                try:
                    # Prepare data for chart
                    resource_data = {
                        "CPU Usage (%)": self.performance_metrics["cpu_usage_history"],
                        "Memory Usage (%)": self.performance_metrics["memory_usage_history"],
                    }
                    
                    # Generate chart with stock chart generator
                    resource_chart = self.chart_generator.create_performance_chart(
                        resource_data,
                        title="Resource Usage History",
                        include_timestamps=True
                    )
                    charts_generated.append(resource_chart)
                except Exception as e:
                    self.logger.warning(f"Error generating resource usage chart: {e}")
            
            # 4. Generate trend analysis chart if we have performance trends data
            if hasattr(self, "performance_trends"):
                try:
                    # Check if we have enough history for each metric
                    has_enough_data = all(len(metric_data["history"]) >= 3 
                                      for metric_name, metric_data in self.performance_trends.items())
                    
                    if has_enough_data:
                        # Extract trend data for each metric
                        trend_data = {}
                        
                        for metric_name, metric_data in self.performance_trends.items():
                            if len(metric_data["history"]) >= 3:
                                # Extract the values
                                values = [entry["value"] for entry in metric_data["history"]]
                                # Add to trend data dictionary
                                trend_data[f"{metric_name}"] = values
                        
                        # Generate chart with stock chart generator
                        if trend_data:
                            trend_chart = self.chart_generator.create_performance_chart(
                                trend_data,
                                title="Performance Metric Trends",
                                include_timestamps=True,
                                chart_type="line"
                            )
                            charts_generated.append(trend_chart)
                except Exception as e:
                    self.logger.warning(f"Error generating trend analysis chart: {e}")
            
            # 5. Generate forecast chart if we have predictions
            if hasattr(self, "performance_predictions"):
                try:
                    # Extract prediction data
                    prediction_data = {}
                    
                    # Format: {metric_name: [current_value, predicted_value]}
                    for metric_name, prediction in self.performance_predictions.items():
                        # Get current value and prediction
                        current = prediction.get("current_value", 0)
                        predicted = prediction.get("predicted_value", 0)
                        
                        # Add to chart data
                        prediction_data[f"{metric_name} (Current vs Predicted)"] = [current, predicted]
                    
                    # Generate chart
                    if prediction_data:
                        forecast_chart = self.chart_generator.create_performance_chart(
                            prediction_data,
                            title="Performance Metric Forecasts (1 Hour)",
                            include_timestamps=True,
                            chart_type="bar"
                        )
                        charts_generated.append(forecast_chart)
                except Exception as e:
                    self.logger.warning(f"Error generating forecast chart: {e}")
            
            # 6. Generate cache effectiveness chart
            try:
                # Collect cache metrics
                cache_data = {
                    "Calculation Cache Hit Rate (%)": (self.calculation_cache["hit_count"] / 
                                                max(1, self.calculation_cache["hit_count"] + 
                                                   self.calculation_cache["miss_count"])) * 100
                }
                
                # Add market data cache metrics if available
                if hasattr(self, "market_data_cache_hits") and hasattr(self, "market_data_cache_misses"):
                    market_data_hit_rate = (getattr(self, "market_data_cache_hits", 0) / 
                                        max(1, getattr(self, "market_data_cache_hits", 0) + 
                                            getattr(self, "market_data_cache_misses", 0))) * 100
                    cache_data["Market Data Cache Hit Rate (%)"] = market_data_hit_rate
                
                # Generate chart
                cache_chart = self.chart_generator.create_performance_chart(
                    cache_data,
                    title="Cache Effectiveness",
                    include_timestamps=True,
                    chart_type="bar"
                )
                charts_generated.append(cache_chart)
            except Exception as e:
                self.logger.warning(f"Error generating cache effectiveness chart: {e}")
            
            # Log success
            if charts_generated:
                self.logger.info(f"Generated {len(charts_generated)} performance visualization charts", 
                               chart_files=charts_generated)
            
        except Exception as e:
            self.logger.error(f"Error generating performance visualizations: {e}", exc_info=True)
    
    def generate_performance_predictions(self) -> Dict[str, Any]:
        """
        Generate predictions for key performance metrics based on historical data.
        Uses linear regression and other simple forecasting methods.
        
        Returns:
            Dictionary of predictions for key metrics
        """
        try:
            predictions = {}
            
            # Only generate predictions if we have trends data
            if not hasattr(self, "performance_trends"):
                return predictions
            
            # Initialize performance_predictions if not already present
            if not hasattr(self, "performance_predictions"):
                self.performance_predictions = {}
            
            # For each metric in performance trends
            for metric_name, metric_data in self.performance_trends.items():
                if len(metric_data["history"]) < 6:  # Need at least 6 data points
                    continue
                    
                # Extract the historical values
                history = metric_data["history"]
                values = [entry["value"] for entry in history]
                times = [entry["timestamp"] for entry in history]
                
                # Get current (latest) value
                current_value = values[-1]
                current_time = times[-1]
                
                # Calculate simple linear regression
                x = list(range(len(values)))
                y = values
                n = len(x)
                
                x_mean = sum(x) / n if n > 0 else 0
                y_mean = sum(y) / n if n > 0 else 0
                
                numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) if n > 0 else 0
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n)) if n > 0 else 1
                
                slope = numerator / denominator if denominator != 0 else 0
                intercept = y_mean - (slope * x_mean)
                
                # Only make predictions if the trend is significant
                if abs(slope) > 0.001:
                    # Predict 1 hour ahead (assuming 5-minute intervals, that's 12 steps ahead)
                    future_x = n + 12
                    predicted_value = slope * future_x + intercept
                    
                    # Calculate confidence based on R-squared
                    ss_total = sum((yi - y_mean) ** 2 for yi in y) if n > 1 else 0
                    y_pred = [slope * xi + intercept for xi in x]
                    ss_residual = sum((y[i] - y_pred[i]) ** 2 for i in range(n)) if n > 1 else 0
                    
                    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                    confidence = max(0, min(100, r_squared * 100))  # Convert to percentage, clamped to 0-100
                    
                    # Calculate prediction interval
                    if n > 2:
                        # Calculate standard error of the regression
                        se = (ss_residual / (n - 2)) ** 0.5 if n > 2 else 0
                        
                        # Calculate standard error of the prediction
                        se_pred = se * (1 + 1/n + ((future_x - x_mean) ** 2) / sum((xi - x_mean) ** 2 for xi in x)) ** 0.5
                        
                        # Calculate 95% prediction interval
                        # t-statistic for 95% confidence and n-2 degrees of freedom (approximating with 2)
                        t_value = 2.0
                        prediction_interval = t_value * se_pred
                        
                        lower_bound = predicted_value - prediction_interval
                        upper_bound = predicted_value + prediction_interval
                    else:
                        # If we don't have enough data, use a simplified approach
                        lower_bound = predicted_value * 0.8
                        upper_bound = predicted_value * 1.2
                    
                    # Store prediction
                    predictions[metric_name] = {
                        "current_value": current_value,
                        "current_time": current_time,
                        "predicted_value": predicted_value,
                        "prediction_time": current_time + 3600,  # 1 hour later
                        "confidence": confidence,
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "percent_change": ((predicted_value - current_value) / current_value) * 100 if current_value != 0 else 0
                    }
                    
                    # Determine if this prediction represents a potential issue
                    is_concerning = False
                    concern_reason = None
                    
                    # For certain metrics, increasing values are bad
                    if metric_name in ["response_time", "error_rate", "memory_usage", "cpu_usage"]:
                        if predicted_value > current_value * 1.2:  # 20% increase
                            is_concerning = True
                            concern_reason = f"Predicted {metric_name} is increasing significantly"
                    
                    predictions[metric_name]["is_concerning"] = is_concerning
                    predictions[metric_name]["concern_reason"] = concern_reason
            
            # Store predictions for visualization
            self.performance_predictions = predictions
            
            # Log any concerning predictions
            concerning_predictions = {k: v for k, v in predictions.items() if v.get("is_concerning")}
            if concerning_predictions:
                self.logger.warning("Concerning performance predictions detected", 
                                   prediction_count=len(concerning_predictions),
                                   predictions=concerning_predictions)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error generating performance predictions: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _cleanup_calculation_cache(self):
        """
        Clean up expired entries from the calculation cache.
        """
        try:
            # Record cache size before cleanup
            pre_cleanup_size = sum(len(cache) for cache in [
                self.calculation_cache["var_cache"],
                self.calculation_cache["es_cache"],
                self.calculation_cache["scenario_cache"]
            ])
            
            # Clean up VaR cache
            var_cache_size = len(self.calculation_cache["var_cache"])
            self.calculation_cache["var_cache"] = {}
            
            # Clean up ES cache
            es_cache_size = len(self.calculation_cache["es_cache"])
            self.calculation_cache["es_cache"] = {}
            
            # Clean up scenario cache
            scenario_cache_size = len(self.calculation_cache["scenario_cache"])
            self.calculation_cache["scenario_cache"] = {}
            
            # Update last cleanup timestamp
            self.calculation_cache["last_cleanup"] = datetime.now()
            
            # Log cleanup
            self.logger.info("Calculation cache cleanup completed", 
                           pre_cleanup_entries=pre_cleanup_size,
                           var_cache_entries=var_cache_size,
                           es_cache_entries=es_cache_size,
                           scenario_cache_entries=scenario_cache_size)
        
        except Exception as e:
            self.logger.error(f"Error cleaning up calculation cache: {e}", exc_info=True)
    
    def _prune_metrics_data(self):
        """
        Prune metrics data to prevent memory growth.
        """
        try:
            # Limit response times history to most recent 100
            if len(self.performance_metrics["response_times"]) > 100:
                self.performance_metrics["response_times"] = self.performance_metrics["response_times"][-100:]
            
            # Limit error timestamps to most recent 100
            if len(self.performance_metrics["error_timestamps"]) > 100:
                self.performance_metrics["error_timestamps"] = self.performance_metrics["error_timestamps"][-100:]
            
            # Limit model report collection times to most recent 100
            if len(self.intermodel_metrics["model_report_collection_times"]) > 100:
                self.intermodel_metrics["model_report_collection_times"] = self.intermodel_metrics["model_report_collection_times"][-100:]
            
            # Limit slow operations to most recent 50
            if len(self.performance_metrics["slow_operations"]) > 50:
                self.performance_metrics["slow_operations"] = self.performance_metrics["slow_operations"][-50:]
                
            # Limit recent issues to most recent 20
            if len(self.health_data["recent_issues"]) > 20:
                self.health_data["recent_issues"] = self.health_data["recent_issues"][-20:]
                
        except Exception as e:
            self.logger.error(f"Error pruning metrics data: {e}")
    
    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive health report with trend analysis.
        
        Returns:
            Dictionary containing detailed health information including trends and anomalies
        """
        try:
            # Perform fresh health check
            self._check_health()
            
            # Prepare trend information if available
            trends_info = {}
            anomalies_info = {}
            
            if hasattr(self, "performance_trends"):
                for metric_name, metric_data in self.performance_trends.items():
                    # Add trend direction
                    trends_info[metric_name] = {
                        "trend": metric_data["trend"],
                        "data_points": len(metric_data["history"]),
                    }
                    
                    # Add recent historical values (last 6 data points = 30 minutes)
                    if len(metric_data["history"]) > 0:
                        recent_history = metric_data["history"][-6:]
                        trends_info[metric_name]["recent_values"] = [
                            {"timestamp": entry["timestamp"], "value": entry["value"]}
                            for entry in recent_history
                        ]
                        
                        # Calculate percentage change over the monitoring period
                        if len(metric_data["history"]) >= 2:
                            first_value = metric_data["history"][0]["value"]
                            last_value = metric_data["history"][-1]["value"]
                            if first_value != 0:
                                pct_change = ((last_value - first_value) / abs(first_value)) * 100
                                trends_info[metric_name]["percent_change"] = pct_change
                                
                                # Add trend severity based on percentage change
                                if metric_name in ["response_time", "error_rate", "memory_usage", "cpu_usage"]:
                                    # For these metrics, increasing is generally bad
                                    if pct_change > 50:
                                        trends_info[metric_name]["severity"] = "critical"
                                    elif pct_change > 20:
                                        trends_info[metric_name]["severity"] = "warning"
                                    elif pct_change < -20:
                                        trends_info[metric_name]["severity"] = "improving"
                                    else:
                                        trends_info[metric_name]["severity"] = "stable"
                    
                    # Add recent anomalies
                    if len(metric_data["anomalies"]) > 0:
                        anomalies_info[metric_name] = {
                            "count": len(metric_data["anomalies"]),
                            "most_recent": metric_data["anomalies"][-1]
                        }
            
            # Add market data cache metrics if available
            market_data_cache_info = {}
            if hasattr(self, "market_data_cache"):
                market_data_cache_info = {
                    "cache_size": len(getattr(self, "market_data_cache", {})),
                    "hit_count": getattr(self, "market_data_cache_hits", 0),
                    "miss_count": getattr(self, "market_data_cache_misses", 0),
                    "hit_rate": getattr(self, "market_data_cache_hits", 0) / max(1, 
                                getattr(self, "market_data_cache_hits", 0) + 
                                getattr(self, "market_data_cache_misses", 0)) * 100
                }
            
            # Add API metrics if available
            api_metrics = {}
            if hasattr(self, "market_data_api_latencies"):
                api_metrics = {
                    "avg_latency_ms": sum(self.market_data_api_latencies) / max(1, len(self.market_data_api_latencies)),
                    "max_latency_ms": max(self.market_data_api_latencies) if self.market_data_api_latencies else 0,
                    "min_latency_ms": min(self.market_data_api_latencies) if self.market_data_api_latencies else 0,
                    "success_rate": getattr(self, "market_data_api_successes", 0) / max(1, 
                                  getattr(self, "market_data_api_successes", 0) + 
                                  getattr(self, "market_data_api_failures", 0)) * 100
                }
            
            # Add request volume metrics if available
            request_volume = {}
            if hasattr(self, "data_request_volume"):
                request_volume = {
                    "last_minute": self.data_request_volume["last_minute"],
                    "last_hour": self.data_request_volume["last_hour"],
                    "last_day": self.data_request_volume["last_day"]
                }
            
            # Create comprehensive health report
            report = {
                "timestamp": datetime.now().isoformat(),
                "overall_status": self.health_data["health_status"],
                "health_score": self.health_data["health_score"],
                "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                "component_health": self.health_data["component_health"],
                "recent_issues": self.health_data["recent_issues"],
                "system_resources": {
                    "current_cpu_usage": self.system_metrics.get_cpu_usage(),
                    "current_memory_usage": self.system_metrics.get_memory_usage(),
                    "current_memory_usage_mb": self.system_metrics.get_memory_usage_mb(),
                    "peak_memory_usage": self.performance_metrics["peak_memory_usage"],
                    "avg_cpu_usage": sum(self.performance_metrics["cpu_usage_history"]) / max(1, len(self.performance_metrics["cpu_usage_history"])),
                    "avg_memory_usage": sum(self.performance_metrics["memory_usage_history"]) / max(1, len(self.performance_metrics["memory_usage_history"]))
                },
                "operational_metrics": {
                    "total_risk_analyses": self.total_risk_analysis_cycles,
                    "portfolio_analyses": self.portfolio_analysis_count,
                    "risk_limit_checks": self.risk_limit_checks_count,
                    "risk_limit_violations": self.risk_limit_exceeded_count,
                    "execution_errors": self.execution_errors,
                    "error_rate": self.execution_errors / max(1, self.total_risk_analysis_cycles),
                },
                "llm_api_metrics": {
                    "total_calls": self.llm_api_metrics["total_calls"],
                    "successful_calls": self.llm_api_metrics["successful_calls"],
                    "failed_calls": self.llm_api_metrics["failed_calls"],
                    "success_rate": self.llm_api_metrics["successful_calls"] / max(1, self.llm_api_metrics["total_calls"]),
                    "average_response_time": self.llm_api_metrics["average_response_time"]
                },
                "intermodel_communication": {
                    "model_report_collections": self.intermodel_metrics["model_report_collection_count"],
                    "decision_model_notifications": self.intermodel_metrics["decision_model_notifications"],
                    "selection_model_interactions": self.intermodel_metrics["selection_model_interactions"],
                    "last_data_timestamps": self.intermodel_metrics["last_model_data_timestamps"],
                    "missing_data_counts": self.intermodel_metrics["missing_model_data_counts"]
                },
                "calculation_cache_performance": {
                    "hit_count": self.calculation_cache["hit_count"],
                    "miss_count": self.calculation_cache["miss_count"],
                    "hit_rate": (self.calculation_cache["hit_count"] / max(1, self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"])) * 100,
                    "last_cleanup": self.calculation_cache["last_cleanup"].isoformat()
                },
                "market_data_cache": market_data_cache_info,
                "market_data_api": api_metrics,
                "request_volume": request_volume,
                "performance_trends": trends_info,
                "performance_anomalies": anomalies_info,
                "recommendations": self._generate_health_recommendations()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating health report: {e}", exc_info=True)
            self.execution_errors += 1
            
            # Return basic error information
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "health_score": 0,
                "error": str(e)
            }
            
    def _generate_health_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate automatic health recommendations based on current metrics.
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        try:
            # Check for degrading trends
            if hasattr(self, "performance_trends"):
                # Check response time trend
                response_time_trend = self.performance_trends.get("response_time", {}).get("trend")
                if response_time_trend == "degrading":
                    recommendations.append({
                        "priority": "medium",
                        "category": "performance",
                        "recommendation": "Investigate increasing response times",
                        "details": "Response times are showing an upward trend. Consider checking for resource bottlenecks or slow operations."
                    })
                
                # Check error rate trend
                error_rate_trend = self.performance_trends.get("error_rate", {}).get("trend")
                if error_rate_trend == "degrading":
                    recommendations.append({
                        "priority": "high",
                        "category": "reliability",
                        "recommendation": "Address increasing error rates",
                        "details": "Error rates are trending upward. Review recent errors and consider implementing additional error handling."
                    })
                
                # Check resource usage trends
                memory_trend = self.performance_trends.get("memory_usage", {}).get("trend")
                cpu_trend = self.performance_trends.get("cpu_usage", {}).get("trend")
                
                if memory_trend == "degrading" and cpu_trend == "degrading":
                    recommendations.append({
                        "priority": "high",
                        "category": "resources",
                        "recommendation": "Investigate resource usage growth",
                        "details": "Both CPU and memory usage are trending upward. Check for resource leaks or inefficient operations."
                    })
                elif memory_trend == "degrading":
                    recommendations.append({
                        "priority": "medium",
                        "category": "resources",
                        "recommendation": "Monitor increasing memory usage",
                        "details": "Memory usage is trending upward. Check for memory leaks or consider scaling resources."
                    })
                elif cpu_trend == "degrading":
                    recommendations.append({
                        "priority": "medium",
                        "category": "resources",
                        "recommendation": "Monitor increasing CPU usage",
                        "details": "CPU usage is trending upward. Check for inefficient operations or consider scaling resources."
                    })
            
            # Check for cache efficiency
            calculation_cache_hit_rate = (self.calculation_cache["hit_count"] / 
                                       max(1, self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"])) * 100
            
            if calculation_cache_hit_rate < 40:
                recommendations.append({
                    "priority": "low",
                    "category": "performance",
                    "recommendation": "Optimize calculation cache utilization",
                    "details": f"Cache hit rate is low ({calculation_cache_hit_rate:.1f}%). Consider adjusting cache TTL or precomputing common calculations."
                })
            
            # Check market data cache if available
            if hasattr(self, "market_data_cache_hits") and hasattr(self, "market_data_cache_misses"):
                market_data_hit_rate = (getattr(self, "market_data_cache_hits", 0) / 
                                      max(1, getattr(self, "market_data_cache_hits", 0) + 
                                          getattr(self, "market_data_cache_misses", 0))) * 100
                
                if market_data_hit_rate < 30:
                    recommendations.append({
                        "priority": "low",
                        "category": "performance",
                        "recommendation": "Optimize market data cache",
                        "details": f"Market data cache hit rate is low ({market_data_hit_rate:.1f}%). Consider adjusting cache TTL or preemptively caching common symbols."
                    })
            
            # Check LLM API metrics
            llm_success_rate = self.llm_api_metrics["successful_calls"] / max(1, self.llm_api_metrics["total_calls"]) * 100
            if llm_success_rate < 90:
                recommendations.append({
                    "priority": "medium",
                    "category": "reliability",
                    "recommendation": "Improve LLM API reliability",
                    "details": f"LLM API success rate is below target ({llm_success_rate:.1f}%). Consider implementing retry logic or fallback models."
                })
            
            # Check for intermodel communication issues
            missing_model_data = sum(self.intermodel_metrics["missing_model_data_counts"].values())
            if missing_model_data > 10:
                recommendations.append({
                    "priority": "medium",
                    "category": "integration",
                    "recommendation": "Address missing model data",
                    "details": f"Detected {missing_model_data} instances of missing data from other models. Check model connectivity and availability."
                })
            
            # Check slow operations
            if len(self.performance_metrics["slow_operations"]) > 5:
                recommendations.append({
                    "priority": "medium", 
                    "category": "performance",
                    "recommendation": "Optimize slow operations",
                    "details": f"Detected {len(self.performance_metrics['slow_operations'])} slow operations. Review and optimize the most frequent slow operations."
                })
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating health recommendations: {e}", exc_info=True)
            return [{
                "priority": "medium",
                "category": "system",
                "recommendation": "Review health monitoring system",
                "details": f"Could not generate complete health recommendations due to an error: {str(e)}"
            }]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report with visualizations and trend analysis.
        
        Returns:
            Dictionary containing detailed performance metrics, trends, and links to charts
        """
        try:
            start_time = time.time()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Generate fresh visualizations
            self._generate_performance_visualizations()
            
            # Find chart files
            charts_dir = os.path.join("monitoring", "charts", "risk_assessment")
            chart_files = []
            if os.path.exists(charts_dir):
                chart_files = [os.path.join(charts_dir, f) for f in os.listdir(charts_dir) 
                             if f.endswith('.png') or f.endswith('.jpg')]
                chart_files = sorted(chart_files, key=os.path.getmtime, reverse=True)[:5]  # Get 5 most recent
            
            # Prepare trend information 
            trend_analysis = {}
            if hasattr(self, "performance_trends"):
                # Create trend analysis data
                for metric_name, metric_data in self.performance_trends.items():
                    trend_analysis[metric_name] = {
                        "direction": metric_data["trend"],
                        "data_points": len(metric_data["history"]),
                    }
                    
                    # Add linear regression info if we have enough data
                    if len(metric_data["history"]) >= 3:
                        history = metric_data["history"]
                        x = list(range(len(history)))
                        y = [entry["value"] for entry in history]
                        n = len(x)
                        
                        # Calculate simple linear regression
                        x_mean = sum(x) / n if n > 0 else 0
                        y_mean = sum(y) / n if n > 0 else 0
                        
                        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n)) if n > 0 else 0
                        denominator = sum((x[i] - x_mean) ** 2 for i in range(n)) if n > 0 else 1  # Avoid div by zero
                        
                        slope = numerator / denominator if denominator != 0 else 0
                        intercept = y_mean - (slope * x_mean)
                        
                        trend_analysis[metric_name]["regression"] = {
                            "slope": slope,
                            "intercept": intercept,
                            "trend_strength": abs(slope / (y_mean if y_mean != 0 else 1)),  # Normalized slope
                        }
                        
                        # Calculate R-squared (coefficient of determination)
                        if n > 1 and denominator != 0:
                            # Calculate predicted y values
                            y_pred = [slope * xi + intercept for xi in x]
                            
                            # Calculate R-squared
                            ss_total = sum((yi - y_mean) ** 2 for yi in y)
                            ss_residual = sum((yi - y_pred[i]) ** 2 for i, yi in enumerate(y))
                            
                            if ss_total != 0:
                                r_squared = 1 - (ss_residual / ss_total)
                                trend_analysis[metric_name]["regression"]["r_squared"] = r_squared
                        
                        # Add trend prediction
                        if abs(slope) > 0.001:  # Only predict if trend is significant
                            # Predict value 1 hour ahead (12 more 5-minute intervals)
                            future_x = n + 12
                            predicted_value = slope * future_x + intercept
                            trend_analysis[metric_name]["prediction"] = {
                                "value_in_1_hour": predicted_value,
                                "percent_change": ((predicted_value - y[-1]) / y[-1]) * 100 if y[-1] != 0 else 0
                            }
            
            # Add market data performance metrics if available
            market_data_performance = {}
            if hasattr(self, "market_data_api_latencies") and hasattr(self, "market_data_cache_hits"):
                market_data_performance = {
                    "api": {
                        "avg_latency_ms": sum(self.market_data_api_latencies) / max(1, len(self.market_data_api_latencies)),
                        "max_latency_ms": max(self.market_data_api_latencies) if self.market_data_api_latencies else 0,
                        "min_latency_ms": min(self.market_data_api_latencies) if self.market_data_api_latencies else 0,
                        "success_rate": getattr(self, "market_data_api_successes", 0) / max(1, 
                                    getattr(self, "market_data_api_successes", 0) + 
                                    getattr(self, "market_data_api_failures", 0)) * 100,
                        "total_requests": getattr(self, "market_data_api_successes", 0) + getattr(self, "market_data_api_failures", 0)
                    },
                    "cache": {
                        "size": len(getattr(self, "market_data_cache", {})),
                        "hit_count": getattr(self, "market_data_cache_hits", 0),
                        "miss_count": getattr(self, "market_data_cache_misses", 0),
                        "hit_rate": getattr(self, "market_data_cache_hits", 0) / max(1, 
                                    getattr(self, "market_data_cache_hits", 0) + 
                                    getattr(self, "market_data_cache_misses", 0)) * 100
                    }
                }
                
                # Add hourly request distribution if available
                if hasattr(self, "hourly_api_requests"):
                    market_data_performance["request_distribution"] = {
                        "hourly": self.hourly_api_requests
                    }
                
                # Add request volume metrics if available
                if hasattr(self, "data_request_volume"):
                    market_data_performance["request_volume"] = {
                        "last_minute": self.data_request_volume["last_minute"],
                        "last_hour": self.data_request_volume["last_hour"],
                        "last_day": self.data_request_volume["last_day"]
                    }
            
            # Create enhanced performance report
            report = {
                "timestamp": datetime.now().isoformat(),
                "generation_time_ms": round((time.time() - start_time) * 1000, 2),
                "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                "operational_metrics": {
                    "total_risk_analyses": self.total_risk_analysis_cycles,
                    "portfolio_analyses": self.portfolio_analysis_count,
                    "risk_limit_checks": self.risk_limit_checks_count,
                    "risk_limit_violations": self.risk_limit_exceeded_count,
                    "execution_errors": self.execution_errors,
                    "error_rate": self.execution_errors / max(1, self.total_risk_analysis_cycles),
                    "avg_response_time_ms": sum(self.performance_metrics["response_times"]) / max(1, len(self.performance_metrics["response_times"])),
                    "unique_symbols_processed": len(self.performance_metrics["symbols_processed"]),
                    "largest_portfolio_size": self.performance_metrics["largest_portfolio_size"]
                },
                "risk_metrics_stats": {
                    "var_calculations": self.risk_metrics_stats["var_calculations"],
                    "es_calculations": self.risk_metrics_stats["es_calculations"],
                    "volatility_calculations": self.risk_metrics_stats["volatility_calculations"],
                    "beta_calculations": self.risk_metrics_stats["beta_calculations"],
                    "correlation_calculations": self.risk_metrics_stats["correlation_calculations"],
                    "scenario_generations": self.risk_metrics_stats["scenario_generations"],
                    "optimization_runs": self.risk_metrics_stats["optimization_runs"],
                    "factor_analyses": self.risk_metrics_stats["factor_analyses"]
                },
                "system_resources": {
                    "current_cpu_usage": self.system_metrics.get_cpu_usage(),
                    "current_memory_usage": self.system_metrics.get_memory_usage(),
                    "current_memory_usage_mb": self.system_metrics.get_memory_usage_mb(),
                    "peak_memory_usage": self.performance_metrics["peak_memory_usage"],
                    "cpu_usage_history": self.performance_metrics["cpu_usage_history"],
                    "memory_usage_history": self.performance_metrics["memory_usage_history"]
                },
                "llm_api_performance": {
                    "total_calls": self.llm_api_metrics["total_calls"],
                    "successful_calls": self.llm_api_metrics["successful_calls"],
                    "failed_calls": self.llm_api_metrics["failed_calls"],
                    "success_rate": self.llm_api_metrics["successful_calls"] / max(1, self.llm_api_metrics["total_calls"]),
                    "average_response_time": self.llm_api_metrics["average_response_time"],
                    "calls_by_model": self.llm_api_metrics["calls_by_model"],
                    "tokens_by_model": self.llm_api_metrics["tokens_by_model"]
                },
                "calculation_cache_performance": {
                    "hit_count": self.calculation_cache["hit_count"],
                    "miss_count": self.calculation_cache["miss_count"],
                    "hit_rate": (self.calculation_cache["hit_count"] / max(1, self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"])) * 100,
                    "last_cleanup": self.calculation_cache["last_cleanup"].isoformat()
                },
                "market_data_performance": market_data_performance,
                "slow_operations": self.performance_metrics["slow_operations"][:10],  # Show 10 most recent slow operations
                "trend_analysis": trend_analysis,
                "charts": chart_files,
                "health_status": self.health_data["health_status"],
                "health_score": self.health_data["health_score"],
                "performance_summary": self._generate_performance_summary(trend_analysis)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}", exc_info=True)
            self.execution_errors += 1
            
            # Return basic error information
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                "total_risk_analyses": self.total_risk_analysis_cycles
            }
    
    def _generate_performance_summary(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summarized performance analysis based on trends.
        
        Args:
            trend_analysis: Dictionary containing trend analysis data
            
        Returns:
            Dictionary with performance summary
        """
        try:
            # Extract key metrics
            avg_response_time = sum(self.performance_metrics["response_times"]) / max(1, len(self.performance_metrics["response_times"]))
            error_rate = self.execution_errors / max(1, self.total_risk_analysis_cycles)
            cache_hit_rate = self.calculation_cache["hit_count"] / max(1, self.calculation_cache["hit_count"] + self.calculation_cache["miss_count"])
            
            # Determine overall performance status
            if error_rate > 0.05 or avg_response_time > 1000:  # 5% errors or >1000ms response time
                overall_status = "degraded"
            elif error_rate > 0.01 or avg_response_time > 500:  # 1% errors or >500ms response time
                overall_status = "warning"
            else:
                overall_status = "optimal"
            
            # Analyze performance bottlenecks
            bottlenecks = []
            
            # Check for slow database operations
            if avg_response_time > 500:
                bottlenecks.append({
                    "type": "response_time",
                    "severity": "high" if avg_response_time > 1000 else "medium",
                    "metric": f"{avg_response_time:.2f}ms average response time"
                })
            
            # Check for poor cache utilization
            if cache_hit_rate < 0.5:  # Less than 50% hit rate
                bottlenecks.append({
                    "type": "cache_efficiency",
                    "severity": "high" if cache_hit_rate < 0.3 else "medium",
                    "metric": f"{cache_hit_rate:.1%} cache hit rate"
                })
            
            # Check resource utilization
            if hasattr(self, "performance_metrics") and "cpu_usage_history" in self.performance_metrics:
                avg_cpu = sum(self.performance_metrics["cpu_usage_history"]) / max(1, len(self.performance_metrics["cpu_usage_history"]))
                if avg_cpu > 80:
                    bottlenecks.append({
                        "type": "cpu_usage",
                        "severity": "high" if avg_cpu > 90 else "medium",
                        "metric": f"{avg_cpu:.1f}% average CPU usage"
                    })
            
            if hasattr(self, "performance_metrics") and "memory_usage_history" in self.performance_metrics:
                avg_memory = sum(self.performance_metrics["memory_usage_history"]) / max(1, len(self.performance_metrics["memory_usage_history"]))
                if avg_memory > 80:
                    bottlenecks.append({
                        "type": "memory_usage",
                        "severity": "high" if avg_memory > 90 else "medium",
                        "metric": f"{avg_memory:.1f}% average memory usage"
                    })
            
            # Check for higher error rates
            if error_rate > 0.01:
                bottlenecks.append({
                    "type": "error_rate",
                    "severity": "high" if error_rate > 0.05 else "medium",
                    "metric": f"{error_rate:.2%} error rate"
                })
            
            # Check for slow operations
            if len(self.performance_metrics["slow_operations"]) > 5:
                bottlenecks.append({
                    "type": "slow_operations",
                    "severity": "medium",
                    "metric": f"{len(self.performance_metrics['slow_operations'])} slow operations detected"
                })
            
            # Analyze trends (improving or degrading)
            improving_metrics = []
            degrading_metrics = []
            
            for metric_name, metric_data in trend_analysis.items():
                if metric_data.get("direction") == "improving":
                    improving_metrics.append(metric_name)
                elif metric_data.get("direction") == "degrading":
                    degrading_metrics.append(metric_name)
            
            # Sort bottlenecks by severity
            severity_order = {"high": 0, "medium": 1, "low": 2}
            bottlenecks.sort(key=lambda x: severity_order.get(x["severity"], 3))
            
            # Generate summary
            summary = {
                "overall_status": overall_status,
                "bottlenecks": bottlenecks,
                "improving_metrics": improving_metrics,
                "degrading_metrics": degrading_metrics,
                "total_portfolio_analyses": self.portfolio_analysis_count,
                "avg_response_time_ms": avg_response_time,
                "error_rate": error_rate,
                "cache_hit_rate": cache_hit_rate
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}", exc_info=True)
            return {
                "overall_status": "unknown",
                "error": str(e)
            }

    def shutdown(self) -> None:
        """
        Properly shut down the RiskAssessmentModel, clean up resources, and generate final reports.
        """
        shutdown_start_time = time.time()
        self.logger.info("Starting RiskAssessmentModel shutdown process")
        
        # Flag threads to stop
        self.shutdown_requested = True
        
        # Generate final performance report for record keeping
        try:
            final_performance_report = self.get_performance_report()
            self.logger.info("Generated final performance report during shutdown", 
                           total_risk_analyses=final_performance_report["operational_metrics"]["total_risk_analyses"],
                           error_rate=final_performance_report["operational_metrics"]["error_rate"],
                           generation_time_ms=final_performance_report["generation_time_ms"])
        except Exception as e:
            self.logger.error(f"Error generating final performance report during shutdown: {e}", exc_info=True)
        
        # Clean up system metrics collection
        try:
            if hasattr(self, 'system_metrics'):
                self.system_metrics.stop()
                self.logger.info("System metrics collection stopped")
        except Exception as e:
            self.logger.error(f"Error stopping system metrics collection: {e}", exc_info=True)
        
        # Wait for health check thread to terminate (if it exists)
        try:
            if hasattr(self, 'health_thread') and self.health_thread and self.health_thread.is_alive():
                # Give thread a chance to terminate gracefully
                self.health_thread.join(timeout=2.0)
                if self.health_thread.is_alive():
                    self.logger.warning("Health check thread did not terminate within timeout")
                else:
                    self.logger.info("Health check thread terminated successfully")
        except Exception as e:
            self.logger.error(f"Error terminating health check thread: {e}", exc_info=True)
        
        # Clean up RiskAnalysisMCP resources
        try:
            if self.risk_analysis_mcp is not None and hasattr(self.risk_analysis_mcp, "shutdown"):
                self.risk_analysis_mcp.shutdown()
                self.logger.info("RiskAnalysisMCP shutdown completed")
        except Exception as e:
            self.logger.error(f"Error shutting down RiskAnalysisMCP: {e}", exc_info=True)
        
        # Clean up Redis MCP resources
        try:
            if self.redis_mcp is not None and hasattr(self.redis_mcp, "shutdown"):
                self.redis_mcp.shutdown()
                self.logger.info("RedisMCP shutdown completed")
        except Exception as e:
            self.logger.error(f"Error shutting down RedisMCP: {e}", exc_info=True)
        
        # Clean up any calculation caches
        try:
            if hasattr(self, 'calculation_cache'):
                self.calculation_cache["var_cache"].clear()
                self.calculation_cache["es_cache"].clear()
                self.calculation_cache["scenario_cache"].clear()
                self.logger.info("Calculation caches cleared")
        except Exception as e:
            self.logger.error(f"Error clearing calculation caches: {e}", exc_info=True)
        
        # Store final status in Redis if available
        try:
            if self.redis_mcp is not None:
                # Record shutdown in Redis with final status
                self.redis_mcp.call_tool("set_json", {
                    "key": "risk:shutdown:status",
                    "value": {
                        "timestamp": datetime.now().isoformat(),
                        "uptime_seconds": (datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                        "total_risk_analyses": self.total_risk_analysis_cycles,
                        "portfolio_analyses": self.portfolio_analysis_count,
                        "execution_errors": self.execution_errors,
                        "shutdown_reason": "normal_shutdown",
                        "final_health_status": self.health_data["health_status"]
                    },
                    "expiry": 86400 * 7  # Keep for a week
                })
                self.logger.info("Final status stored in Redis")
        except Exception as e:
            self.logger.error(f"Error storing final status in Redis: {e}", exc_info=True)
        
        # Log shutdown completion with metrics
        shutdown_duration = (time.time() - shutdown_start_time) * 1000
        self.logger.info("RiskAssessmentModel shutdown complete", 
                       shutdown_duration_ms=shutdown_duration,
                       total_uptime_seconds=(datetime.now() - self.performance_metrics["start_time"]).total_seconds(),
                       total_risk_analyses=self.total_risk_analysis_cycles,
                       portfolio_analyses=self.portfolio_analysis_count,
                       risk_limit_checks=self.risk_limit_checks_count,
                       risk_limit_violations=self.risk_limit_exceeded_count,
                       execution_errors=self.execution_errors)
        
    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen.
        """
        start_time = time.time()
        self.logger.info("Retrieving LLM configuration")
        
        llm_config = self.config.get("llm_config", {})

        # Default configuration if not provided
        if not llm_config:
            self.logger.info("Using default LLM configuration")
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
        
        # Create configuration with monitoring-friendly defaults
        config = {
            "config_list": llm_config.get("config_list", []),
            "temperature": llm_config.get("temperature", 0.1),
            "timeout": llm_config.get("timeout", 600),
            "seed": llm_config.get("seed", 42),  # Adding seed for reproducibility
            "context_window": llm_config.get("context_window", 100000),  # Claude-3 context window
        }
        
        # Track API access information for monitoring
        for model_config in config["config_list"]:
            model_name = model_config.get("model", "unknown")
            # Initialize model-specific metrics
            if model_name not in self.llm_api_metrics["calls_by_model"]:
                self.llm_api_metrics["calls_by_model"][model_name] = 0
            if model_name not in self.llm_api_metrics["tokens_by_model"]:
                self.llm_api_metrics["tokens_by_model"][model_name] = 0
            
            # Log API provider information
            api_provider = "unknown"
            if "openrouter" in model_config.get("base_url", ""):
                api_provider = "openrouter"
            elif "anthropic" in model_config.get("base_url", ""):
                api_provider = "anthropic_direct"
            elif "openai" in model_config.get("base_url", ""):
                api_provider = "openai"
            self.logger.info("LLM model configuration loaded", 
                           model=model_name, 
                           provider=api_provider,
                           config_retrieval_time_ms=(time.time() - start_time) * 1000)
            
        return config

    def _setup_agents(self) -> Dict[str, Agent]:
        """
        Initialize AutoGen agents for risk assessment.
        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the risk assessment assistant agent
        agents["risk_assistant"] = AssistantAgent(
            name="RiskAssistantAgent",
            system_message="""You are a financial risk assessment specialist. Your role is to:
            1. Analyze portfolio risk using various metrics and methodologies
            2. Generate and evaluate stress test scenarios
            3. Decompose portfolio risk into factor components
            4. Provide risk-based portfolio optimization recommendations
            5. Monitor risk limits and generate alerts

            You have tools for scenario generation, risk attribution, risk metrics calculation,
            and portfolio optimization. Always provide clear reasoning for your risk assessments.""",
            llm_config=self.llm_config,
            description="A specialist in financial risk assessment and portfolio analysis",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="RiskToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        agents["user_proxy"] = user_proxy

        return agents

    def _register_functions(self):
        """
        Register functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        risk_assistant = self.agents["risk_assistant"]

        # Register scenario generation functions (now part of RiskAnalysisMCP)
        @register_function(
            name="generate_historical_scenario",
            description="Generate a scenario based on a historical market event",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def generate_historical_scenario(
            event_name: str,
            asset_returns: Dict[str, List[float]],
            lookback_days: Optional[int] = None,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "generate_historical_scenario",
                    {
                        "event_name": event_name,
                        "asset_returns": asset_returns,
                        "lookback_days": lookback_days,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "generate_historical_scenario", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "generate_historical_scenario"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Historical scenario generation failed"}
            except Exception as e:
                self.logger.error(f"Error in generate_historical_scenario: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "generate_historical_scenario", "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="generate_monte_carlo_scenario",
            description="Generate scenarios using Monte Carlo simulation",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def generate_monte_carlo_scenario(
            asset_returns: Dict[str, List[float]],
            correlation_matrix: Optional[List[List[float]]] = None,
            num_scenarios: Optional[int] = None,
            time_horizon_days: Optional[int] = None,
            confidence_level: Optional[float] = None,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "generate_monte_carlo_scenario",
                    {
                        "asset_returns": asset_returns,
                        "correlation_matrix": correlation_matrix,
                        "num_scenarios": num_scenarios,
                        "time_horizon_days": time_horizon_days or self.default_time_horizon,
                        "confidence_level": confidence_level or self.default_confidence_level,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "generate_monte_carlo_scenario", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "generate_monte_carlo_scenario"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Monte Carlo scenario generation failed"}
            except Exception as e:
                self.logger.error(f"Error in generate_monte_carlo_scenario: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "generate_monte_carlo_scenario", "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="generate_custom_scenario",
            description="Generate a custom scenario based on specified market movements",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def generate_custom_scenario(
            asset_returns: Dict[str, List[float]],
            shock_factors: Dict[str, float],
            correlation_matrix: Optional[List[List[float]]] = None,
            propagate_shocks: bool = True,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "generate_custom_scenario",
                    {
                        "asset_returns": asset_returns,
                        "shock_factors": shock_factors,
                        "correlation_matrix": correlation_matrix,
                        "propagate_shocks": propagate_shocks,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "generate_custom_scenario", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "generate_custom_scenario"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Custom scenario generation failed"}
            except Exception as e:
                self.logger.error(f"Error in generate_custom_scenario: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "generate_custom_scenario", "status": "failed"})
                return {"error": str(e)}

        # Register risk attribution functions (now part of RiskAnalysisMCP)
        @register_function(
            name="calculate_risk_contributions",
            description="Calculate risk contributions of assets in a portfolio",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def calculate_risk_contributions(
            portfolio_weights: Dict[str, float],
            asset_returns: Dict[str, List[float]],
            use_correlation: bool = True,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "calculate_risk_contributions",
                    {
                        "portfolio_weights": portfolio_weights,
                        "asset_returns": asset_returns,
                        "use_correlation": use_correlation,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_risk_contributions", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "calculate_risk_contributions"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Risk contribution calculation failed"}
            except Exception as e:
                self.logger.error(f"Error in calculate_risk_contributions: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_risk_contributions", "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="perform_factor_analysis",
            description="Perform factor analysis on portfolio returns",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def perform_factor_analysis(
            portfolio_returns: List[float],
            factor_returns: Dict[str, List[float]],
            factors: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "perform_factor_analysis",
                    {
                        "portfolio_returns": portfolio_returns,
                        "factor_returns": factor_returns,
                        "factors": factors,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "perform_factor_analysis", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "perform_factor_analysis"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Factor analysis failed"}
            except Exception as e:
                self.logger.error(f"Error in perform_factor_analysis: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "perform_factor_analysis", "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="decompose_asset_risk",
            description="Decompose risk of individual assets into factor components",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def decompose_asset_risk(
            asset_returns: Dict[str, List[float]],
            factor_returns: Dict[str, List[float]],
            factors: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "decompose_asset_risk",
                    {
                        "asset_returns": asset_returns,
                        "factor_returns": factor_returns,
                        "factors": factors,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "decompose_asset_risk", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "decompose_asset_risk"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Asset risk decomposition failed"}
            except Exception as e:
                self.logger.error(f"Error in decompose_asset_risk: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "decompose_asset_risk", "status": "failed"})
                return {"error": str(e)}

        # Register risk metrics functions (now part of RiskAnalysisMCP)
        @register_function(
            name="calculate_var",
            description="Calculate Value at Risk (VaR) for a portfolio",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def calculate_var(
            portfolio_returns: List[float],
            confidence_level: Optional[float] = None,
            method: str = "historical",
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "calculate_var",
                    {
                        "returns": portfolio_returns,
                        "confidence_level": confidence_level or self.default_confidence_level,
                        "method": method,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_var", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "calculate_var"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "VaR calculation failed"}
            except Exception as e:
                self.logger.error(f"Error in calculate_var: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_var", "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="calculate_expected_shortfall",
            description="Calculate Expected Shortfall (ES) for a portfolio",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def calculate_expected_shortfall(
            portfolio_returns: List[float],
            confidence_level: Optional[float] = None,
            method: str = "historical",
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "calculate_expected_shortfall",
                    {
                        "returns": portfolio_returns,
                        "confidence_level": confidence_level or self.default_confidence_level,
                        "method": method,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_expected_shortfall", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "calculate_expected_shortfall"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Expected Shortfall calculation failed"}
            except Exception as e:
                self.logger.error(f"Error in calculate_expected_shortfall: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_expected_shortfall", "status": "failed"})
                return {"error": str(e)}

        # Register portfolio optimization functions (now part of RiskAnalysisMCP)
        @register_function(
            name="optimize_portfolio",
            description="Optimize a portfolio based on risk and return objectives",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def optimize_portfolio(
            asset_returns: Dict[str, List[float]],
            optimization_objective: str = "sharpe_ratio",
            constraints: Optional[Dict[str, Any]] = None,
            risk_aversion: float = 1.0,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "optimize_portfolio",
                    {
                        "asset_returns": asset_returns,
                        "optimization_objective": optimization_objective,
                        "constraints": constraints,
                        "risk_aversion": risk_aversion,
                    },
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "optimize_portfolio", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "optimize_portfolio"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Portfolio optimization failed"}
            except Exception as e:
                self.logger.error(f"Error in optimize_portfolio: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "optimize_portfolio", "status": "failed"})
                return {"error": str(e)}

        # Register scenario impact function (now part of RiskAnalysisMCP)
        @register_function(
            name="calculate_scenario_impact",
            description="Calculate the impact of a scenario on a portfolio",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def calculate_scenario_impact(
            portfolio: Dict[str, float],
            scenario: Dict[str, float],
            initial_value: float = 1000000,
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Call the tool on RiskAnalysisMCP
                self.mcp_tool_call_count += 1
                result = self.risk_analysis_mcp.call_tool(
                    "calculate_scenario_impact",
                    {"portfolio": portfolio, "scenario": scenario, "initial_value": initial_value},
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_scenario_impact", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "calculate_scenario_impact"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                return result or {"error": "Scenario impact calculation failed"}
            except Exception as e:
                self.logger.error(f"Error in calculate_scenario_impact: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "calculate_scenario_impact", "status": "failed"})
                return {"error": str(e)}

        # Register risk limit functions (Implemented using Redis)
        @register_function(
            name="set_risk_limits",
            description="Set risk limits for a portfolio",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def set_risk_limits(
            portfolio_id: str, risk_limits: Dict[str, Any]
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Store risk limits in Redis
                self.mcp_tool_call_count += 1
                result = self.redis_mcp.call_tool(
                    "set_json",
                    {
                        "key": f"{self.redis_keys['risk_limits']}{portfolio_id}",
                        "value": risk_limits,
                        "expiry": None  # Risk limits typically don't expire
                    }
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "set_risk_limits", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "set_risk_limits"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                    return {"status": "error", "portfolio_id": portfolio_id, "error": result.get("error", "Failed to set risk limits in Redis")}
                return {"status": "success", "portfolio_id": portfolio_id}
            except Exception as e:
                self.logger.error(f"Error setting risk limits: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "set_risk_limits", "status": "failed"})
                return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

        @register_function(
            name="check_risk_limits",
            description="Check if portfolio risk metrics exceed defined limits",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def check_risk_limits(
            portfolio_id: str, risk_metrics: Dict[str, Any]
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Retrieve risk limits from Redis
                self.mcp_tool_call_count += 1
                limits_result = self.redis_mcp.call_tool(
                    "get_json",
                    {"key": f"{self.redis_keys['risk_limits']}{portfolio_id}"}
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "check_risk_limits", "status": "success" if limits_result and "error" not in limits_result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "check_risk_limits"})

                if limits_result and limits_result.get("error"):
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                    self.logger.warning(f"Could not retrieve risk limits for {portfolio_id}: {limits_result.get('error')}")
                    return {"status": "warning", "portfolio_id": portfolio_id, "message": "Could not retrieve risk limits"}

                risk_limits = limits_result.get("value", {}) if limits_result else {}
                exceeded_limits = {}

                # Check each metric against its limit
                for metric, limit_value in risk_limits.items():
                    current_value = risk_metrics.get(metric)
                    if current_value is not None and limit_value is not None:
                        if metric in ["var", "expected_shortfall", "portfolio_volatility"] and current_value > limit_value:
                            exceeded_limits[metric] = {"current": current_value, "limit": limit_value}
                        # Add other metric checks as needed (e.g., max drawdown, concentration)

                if exceeded_limits:
                    self.risk_limit_exceeded_count += 1
                    self.logger.counter("risk_assessment_model.risk_limit_exceeded_count")
                    self.logger.warning(f"Risk limits exceeded for portfolio {portfolio_id}: {exceeded_limits}")
                    return {"status": "exceeded", "portfolio_id": portfolio_id, "exceeded_limits": exceeded_limits}
                else:
                    return {"status": "within_limits", "portfolio_id": portfolio_id}

            except Exception as e:
                self.logger.error(f"Error checking risk limits: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "check_risk_limits", "status": "failed"})
                return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

        # Register data storage and retrieval functions (Implemented using Redis)
        @register_function(
            name="store_risk_analysis",
            description="Store risk analysis results",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def store_risk_analysis(
            portfolio_id: str, analysis_results: Dict[str, Any]
        ) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Store analysis results in Redis
                self.mcp_tool_call_count += 1
                result = self.redis_mcp.call_tool(
                    "set_json",
                    {
                        "key": f"{self.redis_keys['portfolio_risk']}{portfolio_id}",
                        "value": analysis_results,
                        "expiry": self.risk_data_ttl  # Use configured TTL
                    }
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "store_risk_analysis", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "store_risk_analysis"})
                if result and "error" in result:
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                    return {"status": "error", "portfolio_id": portfolio_id, "error": result.get("error", "Failed to store risk analysis in Redis")}
                return {"status": "success", "portfolio_id": portfolio_id}
            except Exception as e:
                self.logger.error(f"Error storing risk analysis: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "store_risk_analysis", "status": "failed"})
                return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

        @register_function(
            name="get_risk_analysis",
            description="Retrieve stored risk analysis results",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def get_risk_analysis(portfolio_id: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                # Retrieve analysis results from Redis
                self.mcp_tool_call_count += 1
                result = self.redis_mcp.call_tool(
                    "get_json",
                    {"key": f"{self.redis_keys['portfolio_risk']}{portfolio_id}"}
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "get_risk_analysis", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "get_risk_analysis"})

                if result and result.get("error"):
                    self.execution_errors += 1
                    self.logger.counter("risk_assessment_model.execution_errors")
                    self.logger.warning(f"Could not retrieve risk analysis for {portfolio_id}: {result.get('error')}")
                    return {"status": "error", "portfolio_id": portfolio_id, "error": result.get("error", "Failed to retrieve risk analysis from Redis")}

                return result.get("value", {}) if result else {}  # Return the stored value

            except Exception as e:
                self.logger.error(f"Error retrieving risk analysis: {e}")
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "get_risk_analysis", "status": "failed"})
                return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

        # Register inter-model communication functions (Implemented using Redis Streams/Keys)
        @register_function(
            name="collect_model_reports",
            description="Collect reports from all processing models for the specified symbols.",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def collect_model_reports(symbols: List[str]) -> Dict[str, Any]:
            start_time = time.time()
            self.logger.info(f"Collecting model reports for symbols: {symbols}")
            collected_reports = {
                "timestamp": datetime.now().isoformat(),
                "symbols_requested": symbols,
                "reports": {}
            }

            try:
                # Collect reports from other models by reading from specific Redis keys or streams
                # This implementation assumes other models store their latest reports in predictable Redis keys.
                # If they use streams, the logic would need to be adapted to read from streams.

                # Example: Get latest sentiment data for each symbol from Redis
                sentiment_reports = {}
                for symbol in symbols:
                    self.mcp_tool_call_count += 1
                    # Assuming SentimentModel stores latest sentiment per symbol in a key like "sentiment:symbol:SYMBOL"
                    sentiment_key = f"sentiment:symbol:{symbol}"
                    sentiment_result = self.redis_mcp.call_tool("get_json", {"key": sentiment_key})
                    if sentiment_result and not sentiment_result.get("error"):
                        sentiment_reports[symbol] = sentiment_result.get("value")
                    elif sentiment_result and sentiment_result.get("error"):
                        self.mcp_tool_error_count += 1
                        self.logger.warning(f"Could not get sentiment for {symbol}: {sentiment_result.get('error')}")

                collected_reports["reports"]["sentiment"] = sentiment_reports

                # Example: Get latest fundamental data for each symbol from Redis
                fundamental_reports = {}
                for symbol in symbols:
                    self.mcp_tool_call_count += 1
                    # Assuming FundamentalAnalysisModel stores latest fundamental data per symbol in a key
                    fundamental_key = f"{self.redis_keys['fundamental_data']}{symbol}"
                    fundamental_result = self.redis_mcp.call_tool("get_json", {"key": fundamental_key})
                    if fundamental_result and not fundamental_result.get("error"):
                        fundamental_reports[symbol] = fundamental_result.get("value")
                    elif fundamental_result and fundamental_result.get("error"):
                        self.mcp_tool_error_count += 1
                        self.logger.warning(f"Could not get fundamental data for {symbol}: {fundamental_result.get('error')}")

                collected_reports["reports"]["fundamental"] = fundamental_reports

                # Add similar logic for technical and market data...
                # Example for technical data (assuming stored per symbol):
                technical_reports = {}
                for symbol in symbols:
                    self.mcp_tool_call_count += 1
                    # Assuming MarketAnalysisModel stores latest technical data per symbol
                    technical_key = f"{self.redis_keys['technical_data']}{symbol}"
                    technical_result = self.redis_mcp.call_tool("get_json", {"key": technical_key})
                    if technical_result and not technical_result.get("error"):
                        technical_reports[symbol] = technical_result.get("value")
                    elif technical_result and technical_result.get("error"):
                        self.mcp_tool_error_count += 1
                        self.logger.warning(f"Could not get technical data for {symbol}: {technical_result.get('error')}")

                collected_reports["reports"]["technical"] = technical_reports

                # Example for overall market data (assuming stored in a single key):
                market_reports = {}
                self.mcp_tool_call_count += 1
                # Assuming MarketAnalysisModel stores overall market data in a key
                market_result = self.redis_mcp.call_tool("get_json", {"key": self.redis_keys['market_data']})
                if market_result and not market_result.get("error"):
                    market_reports = market_result.get("value", {})
                elif market_result and market_result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.warning(f"Could not get overall market data: {market_result.get('error')}")

                collected_reports["reports"]["market"] = market_reports

                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "collect_model_reports", "status": "success"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "collect_model_reports"})
                return collected_reports

            except Exception as e:
                self.logger.error(f"Error collecting model reports: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "collect_model_reports", "status": "failed"})
                return {"error": str(e), "timestamp": datetime.now().isoformat(), "symbols_requested": symbols}

        @register_function(
            name="create_consolidated_package",
            description="Create a consolidated package of reports for the Decision Model.",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def create_consolidated_package(
            symbols: List[str],
            request_id: Optional[str] = None,
            available_capital: Optional[float] = None
        ) -> Dict[str, Any]:
            start_time = time.time()
            self.logger.info(f"Creating consolidated package for symbols: {symbols}")

            try:
                # 1. Collect reports from other models
                collected_reports = await collect_model_reports(symbols)
                if collected_reports.get("error"):
                    return {"error": f"Failed to collect model reports: {collected_reports.get('error')}"}

                # 2. Process and consolidate reports and perform symbol-specific risk assessment
                symbol_analysis_results = {}
                for symbol in symbols:
                    # Use the newly defined method
                    symbol_analysis = await self.process_symbol_for_decision(symbol, collected_reports, available_capital)
                    if symbol_analysis.get("error"):
                        self.logger.warning(f"Failed to process symbol {symbol} for decision: {symbol_analysis.get('error')}")
                        # Optionally skip this symbol or include the error in the package
                        symbol_analysis_results[symbol] = {"error": symbol_analysis.get("error")}
                    else:
                        symbol_analysis_results[symbol] = symbol_analysis

                # 3. Create the consolidated package structure
                consolidated_data = {
                    "package_id": request_id or hashlib.md5(json.dumps(symbols, sort_keys=True).encode()).hexdigest(),
                    "timestamp": datetime.now().isoformat(),
                    "symbols": symbols,
                    "available_capital": available_capital,
                    "symbol_analysis": symbol_analysis_results,  # Store symbol-specific analysis
                    "overall_risk_assessment": {},  # Will be calculated below
                    "recommendations": []  # Will be calculated below
                }

                # Calculate overall portfolio risk assessment
                consolidated_data["overall_risk_assessment"] = await self._calculate_overall_risk_assessment(symbols, symbol_analysis_results)
                
                # Generate recommendations based on risk analysis
                consolidated_data["recommendations"] = await self._generate_risk_recommendations(symbols, symbol_analysis_results, consolidated_data["overall_risk_assessment"])

                # 4. Store the consolidated package in Redis
                self.mcp_tool_call_count += 1
                store_result = self.redis_mcp.call_tool(
                    "set_json",
                    {
                        "key": f"{self.redis_keys['consolidated_package']}{consolidated_data['package_id']}",
                        "value": consolidated_data,
                        "expiry": self.risk_data_ttl  # Use configured TTL
                    }
                )
                if store_result and store_result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to store consolidated package in Redis: {store_result.get('error')}")
                    return {"error": f"Failed to store consolidated package: {store_result.get('error')}"}

                # 5. Notify the Decision Model (Publish to a Redis Stream)
                self.mcp_tool_call_count += 1
                notify_result = self.redis_mcp.call_tool(
                    "xadd",  # Assuming xadd tool exists for streams
                    {
                        "stream": self.redis_keys["decision_package_stream"],
                        "data": {"package_id": consolidated_data["package_id"], "timestamp": consolidated_data["timestamp"]}
                    }
                )
                if notify_result and notify_result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to notify Decision Model via Redis stream: {notify_result.get('error')}")
                    # Decide if this is a critical error or just a warning
                    # For now, return success if package was stored, warn about notification failure
                    consolidated_data["notification_error"] = notify_result.get("error")

                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "create_consolidated_package", "status": "success"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "create_consolidated_package"})
                return consolidated_data

            except Exception as e:
                self.logger.error(f"Error creating consolidated package: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "create_consolidated_package", "status": "failed"})
                return {"error": str(e), "timestamp": datetime.now().isoformat(), "symbols": symbols}

        @register_function(
            name="notify_decision_model",
            description="Notify the Decision Model about a new consolidated package.",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def notify_decision_model(package_id: str) -> Dict[str, Any]:
            start_time = time.time()
            self.logger.info(f"Notifying Decision Model about package: {package_id}")
            try:
                # Publish package ID to Decision Model stream
                self.mcp_tool_call_count += 1
                result = self.redis_mcp.call_tool(
                    "xadd",  # Assuming xadd tool exists for streams
                    {
                        "stream": self.redis_keys["decision_package_stream"],
                        "data": {"package_id": package_id, "timestamp": datetime.now().isoformat()}
                    }
                )
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "notify_decision_model", "status": "success" if result and "error" not in result else "failed"})
                self.logger.counter("risk_assessment_model.function_call_count", tags={"function": "notify_decision_model"})
                if result and "error" in result:
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Failed to notify Decision Model via Redis stream: {result.get('error')}")
                    return {"status": "error", "package_id": package_id, "error": result.get("error", "Failed to notify Decision Model")}
                return {"status": "success", "package_id": package_id}
            except Exception as e:
                self.logger.error(f"Error notifying Decision Model: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("risk_assessment_model.function_call_duration_ms", duration, tags={"function": "notify_decision_model", "status": "failed"})
                return {"status": "error", "package_id": package_id, "error": str(e)}

        @register_function(
            name="monitor_selection_responses",
            description="Monitor for new selection responses and process them.",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def monitor_selection_responses() -> Dict[str, Any]:
            """
            Monitor for new selection responses and process them using Redis Streams.

            This method processes a batch of messages read from the stream.

            Returns:
                Status of the monitoring operation
            """
            self.logger.info("Starting to monitor selection responses stream...")
            try:
                # Read from the selection reports stream
                self.mcp_tool_call_count += 1
                read_result = self.redis_mcp.call_tool(
                    "xread",  # Assuming xread tool exists for streams
                    {
                        "streams": [self.redis_keys["selection_reports_stream"]],
                        "count": 10,  # Read up to 10 messages
                        "block": 1000  # Block for 1 second if no messages
                    }
                )

                if read_result and not read_result.get("error"):
                    messages = read_result.get("messages", [])
                    if messages:
                        self.logger.info(f"Received {len(messages)} messages from selection responses stream.")
                        # Process each message (e.g., extract symbols, trigger analysis)
                        for stream_name, stream_messages in messages:
                            for message_id, message_data in stream_messages:
                                self.logger.info(f"Processing message {message_id} from {stream_name}: {message_data}")
                                # Example: Extract symbols from the message data
                                selected_symbols = message_data.get("symbols", [])  # Assuming message contains a 'symbols' key
                                request_id = message_data.get("request_id")  # Assuming message contains a 'request_id'
                                available_capital = message_data.get("available_capital")  # Assuming message contains 'available_capital'

                                if selected_symbols:
                                    self.logger.info(f"Received selected symbols: {selected_symbols}. Triggering consolidated package creation.")
                                    # Trigger the creation of a consolidated package for these symbols
                                    await self.create_consolidated_package(selected_symbols, request_id, available_capital)
                                else:
                                    self.logger.warning(f"Received message from {stream_name} with no symbols: {message_data}")

                        # Acknowledge processed messages (optional but good practice for streams)
                        # self.redis_mcp.call_tool("xack", {"stream": self.redis_keys["selection_reports_stream"], "group": "some_group", "ids": [msg[0] for stream_name, stream_messages in messages for msg in stream_messages]})

                    else:
                        self.logger.info("No new messages in selection responses stream.")

                    return {"status": "success", "messages_read": len(messages)}

                elif read_result and read_result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Error reading from selection responses stream: {read_result.get('error')}")
                    return {"status": "error", "error": read_result.get("error", "Failed to read from stream")}

                else:
                    self.logger.warning("No result from reading selection responses stream.")
                    return {"status": "warning", "message": "No result from reading stream."}

            except Exception as e:
                self.logger.error(f"Error monitoring selection responses: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                return {"status": "error", "error": str(e)}

        @register_function(
            name="monitor_trade_events",
            description="Monitor trade events stream for position updates and capital availability.",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def monitor_trade_events() -> Dict[str, Any]:
            """
            Monitor trade events stream for position updates and capital availability using Redis Streams.

            This method processes a batch of messages read from the stream.

            Returns:
                Status of the monitoring operation
            """
            self.logger.info("Starting to monitor trade events stream...")
            try:
                # Example: Read from the trade events stream
                # This method processes a batch of messages read from the stream.
                self.mcp_tool_call_count += 1
                read_result = self.redis_mcp.call_tool(
                    "xread",  # Assuming xread tool exists for streams
                    {
                        "streams": [self.redis_keys["trade_events"]],
                        "count": 10,  # Read up to 10 messages
                        "block": 1000  # Block for 1 second if no messages
                    }
                )

                if read_result:
                    if read_result.get("error"):
                        # Handle Redis tool error
                        self.mcp_tool_error_count += 1
                        self.logger.error(f"Error reading from trade events stream: {read_result.get('error')}")
                        return {"status": "error", "error": read_result.get("error", "Failed to read from stream")}
                    else:
                        # Process successful read
                        messages = read_result.get("messages", [])
                        if messages:
                            self.logger.info(f"Received {len(messages)} messages from trade events stream.")
                            processed_count = 0
                            for stream_name, stream_messages in messages:
                                for message_id, message_data in stream_messages:
                                    self.logger.info(f"Processing trade event {message_id}: {message_data}")
                                    try:
                                        event_type = message_data.get("event_type")
                                        if event_type == "position_update":
                                            # Update internal portfolio state
                                            symbol = message_data.get("symbol")
                                            shares = message_data.get("shares")
                                            avg_price = message_data.get("avg_price")
                                            if symbol is not None and shares is not None:
                                                if shares > 0:
                                                    self.current_portfolio[symbol] = {"shares": shares, "avg_price": avg_price}
                                                    self.logger.info(f"Updated position for {symbol}: {shares} shares @ {avg_price}")
                                                else:
                                                    if symbol in self.current_portfolio:
                                                        del self.current_portfolio[symbol]
                                                        self.logger.info(f"Closed position for {symbol}")
                                            else:
                                                self.logger.warning(f"Incomplete position update event: {message_data}")

                                            # Optionally, store updated portfolio state in Redis
                                            self.redis_mcp.call_tool(
                                                "set_json", 
                                                {
                                                    "key": self.redis_keys['internal_portfolio_state'],
                                                    "value": self.current_portfolio
                                                }
                                            )

                                        elif event_type == "capital_update":
                                            # Update available capital
                                            new_capital = message_data.get("available_capital")
                                            if new_capital is not None:
                                                self.current_available_capital = new_capital
                                                self.logger.info(f"Updated available capital: {new_capital}")
                                                # Store updated capital state in Redis
                                                self.redis_mcp.call_tool(
                                                    "set", 
                                                    {
                                                        "key": self.redis_keys['internal_capital_state'],
                                                        "value": str(new_capital)
                                                    }
                                                )
                                            else:
                                                self.logger.warning(f"Incomplete capital update event: {message_data}")

                                        else:
                                            self.logger.warning(f"Unknown trade event type: {event_type}")

                                        processed_count += 1
                                    except Exception as proc_err:
                                        self.logger.error(f"Error processing trade event {message_id}: {proc_err}", exc_info=True)
                                        self.execution_errors += 1
                                        self.logger.counter("risk_assessment_model.execution_errors")

                            # Acknowledge processed messages (optional)
                            # self.redis_mcp.call_tool("xack", {"stream": self.redis_keys["trade_events"], "group": "risk_group", "ids": [msg[0] for stream_name, stream_messages in messages for msg in stream_messages]})
                            return {"status": "success", "messages_processed": processed_count}

                        else:
                            self.logger.info("No new messages in trade events stream.")
                            return {"status": "success", "messages_processed": 0}
                else:
                    # Handle case where read_result is None (e.g., timeout without error)
                    self.logger.info("No result or timeout reading from trade events stream.")
                    return {"status": "success", "messages_processed": 0}

            except Exception as e:
                self.logger.error(f"Error monitoring trade events: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("risk_assessment_model.execution_errors")
                return {"status": "error", "error": str(e)}

    async def _calculate_overall_risk_assessment(
        self, symbols: List[str], symbol_analyses: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate the overall portfolio risk assessment based on symbol-specific analyses.
        
        Args:
            symbols: List of symbols in the potential portfolio
            symbol_analyses: Dictionary of symbol analyses from process_symbol_for_decision
            
        Returns:
            Dictionary containing overall risk assessment metrics
        """
        self.logger.info(f"Calculating overall risk assessment for {len(symbols)} symbols")
        
        try:
            # Extract volatility, beta, and other risk metrics from symbol analyses
            volatilities = {}
            betas = {}
            correlations = {}
            market_caps = {}
            sectors = {}
            
            for symbol, analysis in symbol_analyses.items():
                # Extract risk metrics
                risk_metrics = analysis.get("risk_metrics", {})
                technical_data = analysis.get("technical_summary", {}) or {}
                fundamental_data = analysis.get("fundamental_summary", {}) or {}
                
                # Get volatility (prioritize risk metrics, fall back to technical data)
                volatilities[symbol] = risk_metrics.get("volatility", technical_data.get("volatility", 0.02))
                
                # Get beta (prioritize risk metrics, fall back to technical data)
                betas[symbol] = risk_metrics.get("beta", technical_data.get("beta", 1.0))
                
                # Get market cap from fundamental data (for size factor exposure)
                market_caps[symbol] = fundamental_data.get("market_cap", 10e9)  # Default to $10B
                
                # Get sector from fundamental data (for sector concentration)
                sectors[symbol] = fundamental_data.get("sector", "Unknown")
            
            # Calculate portfolio weights (equal weight if not specified)
            portfolio_weight = 1.0 / len(symbols) if symbols else 0
            weights = {symbol: portfolio_weight for symbol in symbols}
            
            # 1. Calculate portfolio volatility using betas and market correlation
            # This is a more sophisticated approach than simple weighted average
            # It accounts for market correlation through betas
            market_vol = 0.015  # Assumed market volatility (1.5% daily)
            
            # Calculate systematic risk component (market-driven)
            systematic_risk = 0
            for symbol in symbols:
                systematic_risk += weights.get(symbol, 0) * betas.get(symbol, 1.0) * market_vol
            
            # Calculate idiosyncratic risk component (stock-specific)
            idiosyncratic_risk = 0
            for symbol in symbols:
                # Idiosyncratic vol = total vol - systematic vol
                symbol_systematic_vol = betas.get(symbol, 1.0) * market_vol
                symbol_idiosyncratic_vol = max(0, volatilities.get(symbol, 0.02) - symbol_systematic_vol)
                idiosyncratic_risk += (weights.get(symbol, 0) * symbol_idiosyncratic_vol) ** 2
            
            idiosyncratic_risk = idiosyncratic_risk ** 0.5  # Square root for volatility
            
            # Total portfolio volatility (combining systematic and idiosyncratic)
            portfolio_volatility = (systematic_risk ** 2 + idiosyncratic_risk ** 2) ** 0.5
            
            # 2. Calculate portfolio VaR using more sophisticated approach
            confidence_level = 0.95  # 95% confidence level
            z_score = 1.645  # Z-score for 95% confidence in normal distribution
            
            # Parametric VaR with fat-tail adjustment
            fat_tail_multiplier = 1.2  # Adjust for non-normal distribution
            portfolio_var = portfolio_volatility * z_score * fat_tail_multiplier
            
            # 3. Calculate Expected Shortfall (ES) more accurately
            # For fat-tailed distributions, ES is further from VaR
            portfolio_es = portfolio_var * 1.4  # More conservative estimate
            
            # 4. Calculate diversification metrics
            
            # a. Sector diversification
            sector_weights = {}
            for symbol in symbols:
                sector = sectors.get(symbol, "Unknown")
                sector_weights[sector] = sector_weights.get(sector, 0) + weights.get(symbol, 0)
            
            # Herfindahl-Hirschman Index (HHI) for sector concentration
            # Lower is better (more diversified)
            sector_hhi = sum(weight ** 2 for weight in sector_weights.values())
            
            # b. Size diversification (large, mid, small cap exposure)
            size_categories = {"large": 0, "mid": 0, "small": 0}
            for symbol in symbols:
                market_cap = market_caps.get(symbol, 10e9)
                if market_cap > 10e9:  # $10B+
                    size_categories["large"] += weights.get(symbol, 0)
                elif market_cap > 2e9:  # $2B-$10B
                    size_categories["mid"] += weights.get(symbol, 0)
                else:  # < $2B
                    size_categories["small"] += weights.get(symbol, 0)
            
            # c. Overall diversification ratio
            # Combination of number of stocks, sector diversification, and correlation benefits
            num_symbols = len(symbols)
            if num_symbols > 1:
                # More sophisticated diversification ratio
                # 1. Base diversification from number of stocks (diminishing returns)
                stock_count_factor = min(0.8, 0.3 + 0.5 * (1 - 1/num_symbols))
                
                # 2. Sector diversification factor (lower HHI is better)
                sector_factor = max(0, min(0.8, 1 - sector_hhi))
                
                # 3. Beta dispersion (variety of market exposures)
                beta_values = list(betas.values())
                beta_range = max(beta_values) - min(beta_values) if beta_values else 0
                beta_factor = min(0.5, beta_range / 2)  # Cap at 0.5
                
                # Combine factors
                diversification_ratio = (stock_count_factor + sector_factor + beta_factor) / 3
            else:
                diversification_ratio = 0.0  # No diversification with only one symbol
            
            # 5. Calculate risk score with more factors
            # Base score from volatility
            vol_score = 50 * (portfolio_volatility / 0.015)  # Using 1.5% as baseline
            
            # Adjust for diversification (better diversification lowers risk)
            div_adjustment = -10 * diversification_ratio  # Up to -10 points
            
            # Adjust for market beta (higher beta = higher risk)
            portfolio_beta = sum(weights.get(symbol, 0) * betas.get(symbol, 1.0) for symbol in symbols)
            beta_adjustment = 10 * (portfolio_beta - 1.0)  # +10 per beta point above 1
            
            # Adjust for sector concentration
            sector_adjustment = 15 * (sector_hhi - 0.2)  # Penalty for concentration above 0.2 HHI
            
            # Calculate final risk score
            risk_score = vol_score + div_adjustment + beta_adjustment + sector_adjustment
            risk_score = max(0, min(100, risk_score))  # Clamp between 0-100
            
            # 6. Define risk categories based on score
            risk_category = "Moderate"
            if risk_score < 25:
                risk_category = "Very Low"
            elif risk_score < 40:
                risk_category = "Low"
            elif risk_score < 60:
                risk_category = "Moderate"
            elif risk_score < 80:
                risk_category = "High"
            else:
                risk_category = "Very High"
            
            # 7. Generate scenario impacts with more detailed scenarios
            # Calculate more realistic scenario impacts
            
            # a. Market crash scenario (e.g., -20% market)
            market_crash_impact = 0
            for symbol in symbols:
                # Higher beta stocks affected more
                symbol_impact = -0.20 * betas.get(symbol, 1.0)  # 20% market drop
                # Add idiosyncratic component (some stocks might drop more)
                symbol_impact -= 0.05 * (volatilities.get(symbol, 0.02) / 0.02)
                market_crash_impact += weights.get(symbol, 0) * symbol_impact
            
            # b. Interest rate hike scenario (affects sectors differently)
            rate_hike_impact = 0
            sector_sensitivities = {
                "Financials": 0.02,  # Benefit
                "Utilities": -0.05,  # Hurt
                "Real Estate": -0.06,  # Hurt
                "Technology": -0.03,  # Moderate negative
                "Consumer Staples": -0.01,  # Slight negative
                "Unknown": -0.03,  # Default
            }
            
            for symbol in symbols:
                sector = sectors.get(symbol, "Unknown")
                sensitivity = sector_sensitivities.get(sector, sector_sensitivities["Unknown"])
                rate_hike_impact += weights.get(symbol, 0) * sensitivity
            
            # c. Inflation surge scenario
            inflation_impact = 0
            inflation_sensitivities = {
                "Energy": 0.04,  # Benefit
                "Materials": 0.03,  # Benefit
                "Consumer Staples": -0.02,  # Hurt
                "Consumer Discretionary": -0.04,  # Hurt
                "Technology": -0.03,  # Moderate negative
                "Unknown": -0.02,  # Default
            }
            
            for symbol in symbols:
                sector = sectors.get(symbol, "Unknown")
                sensitivity = inflation_sensitivities.get(sector, inflation_sensitivities["Unknown"])
                inflation_impact += weights.get(symbol, 0) * sensitivity
            
            # d. Dollar strength scenario
            dollar_strength_impact = 0
            # Companies with high international exposure are more affected
            for symbol in symbols:
                # Simplified: use beta as proxy for international exposure
                intl_exposure = min(1.0, betas.get(symbol, 1.0) * 0.8)
                dollar_strength_impact += weights.get(symbol, 0) * (-0.03 * intl_exposure)
            
            # Compile scenario impacts
            scenario_impacts = {
                "market_crash_20pct": market_crash_impact,
                "interest_rate_hike_100bp": rate_hike_impact,
                "inflation_surge_5pct": inflation_impact,
                "dollar_strength_10pct": dollar_strength_impact,
            }
            
            # 8. Calculate factor exposures
            factor_exposures = {
                "market_beta": portfolio_beta,
                "size": sum(weights.get(symbol, 0) * (1 if market_caps.get(symbol, 10e9) < 5e9 else 0) for symbol in symbols),
                "value": 0.5,  # Placeholder - would calculate from P/E, P/B ratios
                "momentum": 0.5,  # Placeholder - would calculate from price momentum
                "volatility": portfolio_volatility / 0.015,  # Relative to market
            }
            
            # Create the enhanced overall assessment
            overall_assessment = {
                "overall_risk_score": risk_score,
                "risk_category": risk_category,
                "portfolio_var": portfolio_var,
                "portfolio_es": portfolio_es,
                "portfolio_volatility": portfolio_volatility,
                "portfolio_beta": portfolio_beta,
                "diversification": {
                    "overall_ratio": diversification_ratio,
                    "sector_concentration": sector_weights,
                    "sector_hhi": sector_hhi,
                    "size_exposure": size_categories,
                },
                "factor_exposures": factor_exposures,
                "scenario_impacts": scenario_impacts,
                "risk_decomposition": {
                    "systematic_risk": systematic_risk,
                    "idiosyncratic_risk": idiosyncratic_risk,
                    "systematic_pct": (systematic_risk ** 2) / (portfolio_volatility ** 2) if portfolio_volatility > 0 else 0,
                    "idiosyncratic_pct": (idiosyncratic_risk ** 2) / (portfolio_volatility ** 2) if portfolio_volatility > 0 else 0,
                }
            }
            
            return overall_assessment
            
        except Exception as e:
            self.logger.error(f"Error calculating overall risk assessment: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("risk_assessment_model.execution_errors")
            return {
                "overall_risk_score": None,
                "risk_category": "Unknown",
                "portfolio_var": None,
                "portfolio_es": None,
                "portfolio_volatility": None,
                "diversification_ratio": None,
                "scenario_impacts": {},
                "error": str(e)
            }

    async def _generate_risk_recommendations(
        self, symbols: List[str], symbol_analyses: Dict[str, Dict[str, Any]], 
        overall_assessment: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate risk-based recommendations based on symbol analyses and overall assessment.
        
        Args:
            symbols: List of symbols in the potential portfolio
            symbol_analyses: Dictionary of symbol analyses from process_symbol_for_decision
            overall_assessment: Overall risk assessment from _calculate_overall_risk_assessment
            
        Returns:
            List of recommendation dictionaries
        """
        self.logger.info("Generating risk-based recommendations")
        recommendations = []
        
        try:
            # Extract key metrics from overall assessment
            risk_score = overall_assessment.get("overall_risk_score")
            risk_category = overall_assessment.get("risk_category")
            portfolio_vol = overall_assessment.get("portfolio_volatility")
            portfolio_beta = overall_assessment.get("portfolio_beta")
            portfolio_var = overall_assessment.get("portfolio_var")
            portfolio_es = overall_assessment.get("portfolio_es")
            
            # Extract diversification metrics
            diversification = overall_assessment.get("diversification", {})
            diversification_ratio = diversification.get("overall_ratio")
            sector_concentration = diversification.get("sector_concentration", {})
            sector_hhi = diversification.get("sector_hhi")
            size_exposure = diversification.get("size_exposure", {})
            
            # Extract risk decomposition
            risk_decomposition = overall_assessment.get("risk_decomposition", {})
            systematic_pct = risk_decomposition.get("systematic_pct")
            idiosyncratic_pct = risk_decomposition.get("idiosyncratic_pct")
            
            # Extract scenario impacts
            scenario_impacts = overall_assessment.get("scenario_impacts", {})
            
            # Extract factor exposures
            factor_exposures = overall_assessment.get("factor_exposures", {})
            
            # 1. Overall risk profile recommendations
            if risk_score is not None:
                if risk_score > 70:
                    recommendations.append({
                        "type": "overall_risk",
                        "priority": "high",
                        "recommendation": "Reduce overall portfolio risk exposure",
                        "details": f"Portfolio risk score is {risk_score:.1f} ({risk_category}), which is significantly above the recommended threshold of 60. Consider rebalancing toward lower-risk assets."
                    })
                elif risk_score > 60:
                    recommendations.append({
                        "type": "overall_risk",
                        "priority": "medium",
                        "recommendation": "Monitor elevated portfolio risk level",
                        "details": f"Portfolio risk score is {risk_score:.1f} ({risk_category}), slightly above the recommended threshold. Consider adding some defensive positions."
                    })
                elif risk_score < 30:
                    recommendations.append({
                        "type": "overall_risk",
                        "priority": "low",
                        "recommendation": "Consider increasing risk exposure for better returns",
                        "details": f"Portfolio risk score is {risk_score:.1f} ({risk_category}), which may be too conservative. Consider adding growth-oriented positions."
                    })
            
            # 2. Position sizing recommendations based on VaR and ES
            if portfolio_var is not None and portfolio_es is not None:
                # Calculate maximum acceptable loss (e.g., 2% of portfolio)
                max_acceptable_loss_pct = 0.02  # 2% maximum acceptable daily loss
                
                if portfolio_es > max_acceptable_loss_pct:
                    recommendations.append({
                        "type": "position_sizing",
                        "priority": "high",
                        "recommendation": "Reduce position sizes to limit potential losses",
                        "details": f"Expected Shortfall (ES) is {portfolio_es:.2%}, exceeding the maximum acceptable daily loss of {max_acceptable_loss_pct:.2%}. Consider reducing overall exposure by approximately {((portfolio_es/max_acceptable_loss_pct) - 1) * 100:.0f}%."
                    })
                elif portfolio_es < max_acceptable_loss_pct * 0.5:
                    recommendations.append({
                        "type": "position_sizing",
                        "priority": "medium",
                        "recommendation": "Consider increasing position sizes for better returns",
                        "details": f"Expected Shortfall (ES) is {portfolio_es:.2%}, well below the maximum acceptable daily loss of {max_acceptable_loss_pct:.2%}. There is capacity to increase positions by approximately {((max_acceptable_loss_pct/portfolio_es) - 1) * 100:.0f}%."
                    })
            
            # 3. Diversification recommendations
            if diversification_ratio is not None:
                if diversification_ratio < 0.3 and len(symbols) < 3:
                    recommendations.append({
                        "type": "diversification",
                        "priority": "high",
                        "recommendation": "Add more uncorrelated assets to improve diversification",
                        "details": f"Current diversification ratio is {diversification_ratio:.2f}, well below the recommended minimum of 0.5. Portfolio is vulnerable to concentrated risks."
                    })
                elif 0.3 <= diversification_ratio < 0.6:
                    recommendations.append({
                        "type": "diversification",
                        "priority": "medium",
                        "recommendation": "Consider adding 1-2 more uncorrelated assets",
                        "details": f"Current diversification ratio is {diversification_ratio:.2f}, which could be improved. Additional diversification would enhance risk-adjusted returns."
                    })
            
            # 4. Sector concentration recommendations
            if sector_concentration and sector_hhi:
                # Find the most concentrated sectors
                top_sectors = sorted(sector_concentration.items(), key=lambda x: x[1], reverse=True)
                
                if sector_hhi > 0.5:  # Very high concentration
                    top_sector = top_sectors[0][0] if top_sectors else "Unknown"
                    top_weight = top_sectors[0][1] if top_sectors else 0
                    
                    recommendations.append({
                        "type": "sector_concentration",
                        "priority": "high",
                        "recommendation": f"Reduce exposure to {top_sector} sector",
                        "details": f"Portfolio has {top_weight:.1%} allocation to {top_sector} sector, creating significant concentration risk (HHI: {sector_hhi:.2f}). Consider diversifying across more sectors."
                    })
                elif sector_hhi > 0.3:  # Moderate concentration
                    top_two_sectors = [s[0] for s in top_sectors[:2]] if len(top_sectors) >= 2 else ["Unknown"]
                    
                    recommendations.append({
                        "type": "sector_concentration",
                        "priority": "medium",
                        "recommendation": f"Diversify beyond {' and '.join(top_two_sectors)} sectors",
                        "details": f"Portfolio has moderate sector concentration (HHI: {sector_hhi:.2f}). Consider adding exposure to underrepresented sectors."
                    })
            
            # 5. Size exposure recommendations
            if size_exposure:
                large_cap = size_exposure.get("large", 0)
                mid_cap = size_exposure.get("mid", 0)
                small_cap = size_exposure.get("small", 0)
                
                if small_cap > 0.4:  # High small cap exposure
                    recommendations.append({
                        "type": "size_exposure",
                        "priority": "medium",
                        "recommendation": "Consider reducing small-cap exposure",
                        "details": f"Portfolio has {small_cap:.1%} allocation to small-cap stocks, which may increase volatility and liquidity risk. Consider adding some large-cap positions for stability."
                    })
                elif large_cap > 0.8:  # Very high large cap exposure
                    recommendations.append({
                        "type": "size_exposure",
                        "priority": "low",
                        "recommendation": "Consider adding mid or small-cap exposure for growth",
                        "details": f"Portfolio has {large_cap:.1%} allocation to large-cap stocks. Adding some mid or small-cap exposure may improve long-term growth potential."
                    })
            
            # 6. Systematic vs. idiosyncratic risk recommendations
            if systematic_pct is not None and idiosyncratic_pct is not None:
                if systematic_pct > 0.8:  # Very high systematic risk
                    recommendations.append({
                        "type": "risk_decomposition",
                        "priority": "medium",
                        "recommendation": "Reduce market beta exposure",
                        "details": f"Portfolio has {systematic_pct:.1%} systematic (market) risk, making it highly vulnerable to market downturns. Consider adding low-beta or market-neutral positions."
                    })
                elif idiosyncratic_pct > 0.7:  # Very high idiosyncratic risk
                    recommendations.append({
                        "type": "risk_decomposition",
                        "priority": "medium",
                        "recommendation": "Reduce concentrated single-stock exposures",
                        "details": f"Portfolio has {idiosyncratic_pct:.1%} idiosyncratic (stock-specific) risk. Consider diversifying across more names to reduce company-specific risks."
                    })
            
            # 7. Beta-based recommendations
            if portfolio_beta is not None:
                if portfolio_beta > 1.3:  # High beta
                    recommendations.append({
                        "type": "market_sensitivity",
                        "priority": "medium",
                        "recommendation": "Reduce market sensitivity (beta)",
                        "details": f"Portfolio beta of {portfolio_beta:.2f} indicates high sensitivity to market movements. Consider adding lower-beta positions to reduce downside risk in market corrections."
                    })
                elif portfolio_beta < 0.7:  # Low beta
                    recommendations.append({
                        "type": "market_sensitivity",
                        "priority": "low",
                        "recommendation": "Consider increasing market exposure in bullish conditions",
                        "details": f"Portfolio beta of {portfolio_beta:.2f} indicates low sensitivity to market movements. In bullish markets, this may lead to underperformance."
                    })
            
            # 8. Volatility-based recommendations for individual symbols
            high_vol_symbols = []
            for symbol, analysis in symbol_analyses.items():
                risk_metrics = analysis.get("risk_metrics", {})
                technical_data = analysis.get("technical_summary", {}) or {}
                symbol_vol = risk_metrics.get("volatility", technical_data.get("volatility", 0.02))
                
                if symbol_vol > 0.03:  # 3% daily vol is high
                    high_vol_symbols.append(symbol)
            
            if high_vol_symbols and len(high_vol_symbols) <= 3:  # Only recommend if a few specific symbols
                recommendations.append({
                    "type": "volatility_management",
                    "priority": "medium",
                    "recommendation": f"Consider hedging or reducing positions in high-volatility symbols: {', '.join(high_vol_symbols)}",
                    "details": "These symbols show above-average volatility that may increase overall portfolio risk. Consider using options strategies or position sizing to manage this risk."
                })
            elif len(high_vol_symbols) > 3:  # If many symbols are high volatility
                recommendations.append({
                    "type": "volatility_management",
                    "priority": "medium",
                    "recommendation": "Reduce overall exposure to high-volatility names",
                    "details": f"{len(high_vol_symbols)} of {len(symbols)} symbols in the portfolio exhibit high volatility. Consider rebalancing toward more stable names."
                })
            
            # 9. Market condition recommendations
            market_reports = next(iter([a.get("market_context", {}) for a in symbol_analyses.values() if a.get("market_context")]), {})
            market_trend = market_reports.get("trend", "neutral")
            market_volatility = market_reports.get("volatility", "normal")
            
            if market_trend == "bearish" and market_volatility == "high":
                recommendations.append({
                    "type": "market_conditions",
                    "priority": "high",
                    "recommendation": "Implement defensive measures due to bearish, volatile market",
                    "details": "Current market conditions (bearish trend, high volatility) suggest significant caution. Consider raising cash levels, adding hedges, or rotating to defensive sectors."
                })
            elif market_trend == "bearish":
                recommendations.append({
                    "type": "market_conditions",
                    "priority": "medium",
                    "recommendation": "Adopt defensive positioning in bearish market",
                    "details": "Current bearish market trend suggests caution. Consider reducing high-beta exposures and increasing allocation to defensive sectors."
                })
            elif market_volatility == "high":
                recommendations.append({
                    "type": "market_conditions",
                    "priority": "medium",
                    "recommendation": "Manage risk in volatile market conditions",
                    "details": "Current high market volatility suggests implementing tighter risk controls. Consider reducing position sizes and implementing stop-loss strategies."
                })
            
            # 10. Scenario-based recommendations
            worst_scenarios = []
            for scenario, impact in scenario_impacts.items():
                if impact < -0.05:  # If potential impact is worse than -5%
                    worst_scenarios.append((scenario, impact))
            
            # Sort by impact (most negative first)
            worst_scenarios.sort(key=lambda x: x[1])
            
            if worst_scenarios:
                # Take the worst scenario
                worst_scenario, worst_impact = worst_scenarios[0]
                
                # Generate specific hedging recommendations based on scenario type
                hedge_strategy = "implementing portfolio hedges"
                if "market_crash" in worst_scenario:
                    hedge_strategy = "purchasing protective puts or increasing cash allocation"
                elif "interest_rate" in worst_scenario:
                    hedge_strategy = "reducing duration exposure and rate-sensitive positions"
                elif "inflation" in worst_scenario:
                    hedge_strategy = "adding inflation-protected securities or commodities exposure"
                elif "dollar" in worst_scenario:
                    hedge_strategy = "reducing exposure to companies with high international revenue"
                
                recommendations.append({
                    "type": "scenario_protection",
                    "priority": "high" if worst_impact < -0.1 else "medium",
                    "recommendation": f"Hedge against {worst_scenario.replace('_', ' ')} scenario",
                    "details": f"This scenario could have a {worst_impact:.1%} impact on the portfolio. Consider {hedge_strategy} to mitigate this risk."
                })
                
                # If multiple severe scenarios, add a general recommendation
                if len(worst_scenarios) > 1:
                    recommendations.append({
                        "type": "scenario_protection",
                        "priority": "medium",
                        "recommendation": "Implement broad-based risk mitigation strategies",
                        "details": "Portfolio is vulnerable to multiple adverse scenarios. Consider a combination of hedging strategies and increased diversification."
                    })
            
            # 11. Factor exposure recommendations
            if factor_exposures:
                # Check for extreme factor exposures
                market_beta = factor_exposures.get("market_beta")
                size_factor = factor_exposures.get("size")
                value_factor = factor_exposures.get("value")
                momentum_factor = factor_exposures.get("momentum")
                volatility_factor = factor_exposures.get("volatility")
                
                factor_recommendations = []
                
                if market_beta and market_beta > 1.3:
                    factor_recommendations.append("reducing market beta")
                if size_factor and size_factor > 0.7:
                    factor_recommendations.append("diversifying away from small-cap concentration")
                if value_factor and value_factor < 0.3:
                    factor_recommendations.append("adding some value exposure")
                if momentum_factor and momentum_factor > 0.7:
                    factor_recommendations.append("reducing momentum concentration")
                if volatility_factor and volatility_factor > 1.3:
                    factor_recommendations.append("reducing high-volatility exposure")
                
                if len(factor_recommendations) >= 2:
                    recommendations.append({
                        "type": "factor_exposure",
                        "priority": "medium",
                        "recommendation": "Rebalance factor exposures for better risk-adjusted returns",
                        "details": f"Portfolio has concentrated factor exposures. Consider {', '.join(factor_recommendations)}."
                    })
            
            # 12. Portfolio optimization recommendation
            if risk_score is not None and diversification_ratio is not None:
                if risk_score > 60 or diversification_ratio < 0.4:
                    recommendations.append({
                        "type": "portfolio_optimization",
                        "priority": "medium",
                        "recommendation": "Run portfolio optimization to improve risk-adjusted returns",
                        "details": "Current portfolio structure suggests potential for improvement through formal optimization. Consider running a mean-variance or risk-parity optimization to enhance risk-adjusted returns."
                    })
            
            # Sort recommendations by priority (high, medium, low)
            priority_order = {"high": 0, "medium": 1, "low": 2}
            recommendations.sort(key=lambda x: priority_order.get(x.get("priority", "low"), 3))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating risk recommendations: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("risk_assessment_model.execution_errors")
            return [{
                "type": "error",
                "priority": "high",
                "recommendation": "Unable to generate detailed risk recommendations",
                "details": f"Error: {str(e)}"
            }]

    # --- Main Analysis Methods ---

    async def analyze_portfolio_risk(self, portfolio: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform a comprehensive risk analysis of the given portfolio.

        Args:
            portfolio: Dictionary representing the portfolio holdings.
            market_data: Dictionary containing relevant market data.

        Returns:
            Dictionary containing the portfolio risk analysis results.
        """
        self.logger.info(f"Starting portfolio risk analysis for portfolio: {list(portfolio.keys())}")
        start_time = time.time()
        self.total_risk_analysis_cycles += 1
        self.logger.counter("risk_assessment_model.total_risk_analysis_cycles")

        try:
            # 1. Prepare input data (returns, weights)
            # Convert portfolio holdings to weights and collect historical returns
            total_value = 0
            for symbol, position in portfolio.items():
                shares = position.get('shares', 0)
                avg_price = position.get('avg_price', 0)
                total_value += shares * avg_price
            
            # Create portfolio weights dictionary
            portfolio_weights = {}
            for symbol, position in portfolio.items():
                shares = position.get('shares', 0)
                avg_price = position.get('avg_price', 0)
                if total_value > 0:
                    portfolio_weights[symbol] = (shares * avg_price) / total_value
                else:
                    portfolio_weights[symbol] = 0
            
            # Get historical returns from market data
            # In a real implementation, this would come from a data provider or database
            # Here we'll assume market_data contains returns for each symbol
            asset_returns = {}
            for symbol in portfolio.keys():
                asset_returns[symbol] = market_data.get(f'{symbol}_returns', [0.001] * 252)  # Default to 0.1% daily for a year
            
            # Combine returns into a single portfolio return series
            portfolio_returns = [0] * len(next(iter(asset_returns.values()), []))
            for symbol, weights in portfolio_weights.items():
                symbol_returns = asset_returns.get(symbol, [])
                for i in range(min(len(portfolio_returns), len(symbol_returns))):
                    portfolio_returns[i] += weights * symbol_returns[i]
            
            # 2. Calculate VaR and Expected Shortfall
            var_result = await self.agents["user_proxy"]._execute_function(
                "calculate_var",
                {"portfolio_returns": portfolio_returns, "confidence_level": self.default_confidence_level, "method": "historical"}
            )
            
            es_result = await self.agents["user_proxy"]._execute_function(
                "calculate_expected_shortfall",
                {"portfolio_returns": portfolio_returns, "confidence_level": self.default_confidence_level, "method": "historical"}
            )
            
            # 3. Calculate risk contributions
            contributions_result = await self.agents["user_proxy"]._execute_function(
                "calculate_risk_contributions",
                {"portfolio_weights": portfolio_weights, "asset_returns": asset_returns, "use_correlation": True}
            )
            
            # 4. Generate scenarios and calculate impacts
            # Historical scenario - 2008 Financial Crisis
            crisis_scenario = await self.agents["user_proxy"]._execute_function(
                "generate_historical_scenario",
                {"event_name": "2008_financial_crisis", "asset_returns": asset_returns, "lookback_days": 30}
            )
            
            # Custom scenario - Fed rate hike
            rate_hike_scenario = await self.agents["user_proxy"]._execute_function(
                "generate_custom_scenario",
                {
                    "asset_returns": asset_returns,
                    "shock_factors": {"equities": -0.05, "bonds": -0.03, "commodities": 0.02},
                    "propagate_shocks": True
                }
            )
            
            # Calculate scenario impacts
            crisis_impact = await self.agents["user_proxy"]._execute_function(
                "calculate_scenario_impact",
                {"portfolio": portfolio_weights, "scenario": crisis_scenario.get("scenario", {})}
            )
            
            rate_hike_impact = await self.agents["user_proxy"]._execute_function(
                "calculate_scenario_impact",
                {"portfolio": portfolio_weights, "scenario": rate_hike_scenario.get("scenario", {})}
            )
            
            # 5. Perform factor analysis
            # Assume factor returns are available in market_data
            factor_returns = {
                "market": market_data.get("market_factor", [0.001] * 252),
                "size": market_data.get("size_factor", [0.0005] * 252),
                "value": market_data.get("value_factor", [0.0003] * 252),
                "momentum": market_data.get("momentum_factor", [0.0002] * 252)
            }
            
            factor_analysis = await self.agents["user_proxy"]._execute_function(
                "perform_factor_analysis",
                {"portfolio_returns": portfolio_returns, "factor_returns": factor_returns}
            )
            
            # 6. Check risk limits
            risk_metrics = {
                "var": var_result.get("var", 0),
                "expected_shortfall": es_result.get("expected_shortfall", 0),
                "portfolio_volatility": var_result.get("volatility", 0)
            }
            
            limit_check_result = await self.agents["user_proxy"]._execute_function(
                "check_risk_limits",
                {"portfolio_id": "current_portfolio", "risk_metrics": risk_metrics}
            )
            
            # 7. Compile results
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_id": "current_portfolio",
                "status": "completed",
                "metrics": {
                    "var": var_result.get("var"),
                    "expected_shortfall": es_result.get("expected_shortfall"),
                    "volatility": var_result.get("volatility"),
                    "sharpe_ratio": sum(portfolio_returns) / (var_result.get("volatility", 1) * (len(portfolio_returns) ** 0.5)),
                },
                "contributions": contributions_result.get("contributions", {}),
                "factor_exposure": factor_analysis.get("factor_exposures", {}),
                "scenario_impacts": {
                    "financial_crisis": crisis_impact.get("impact"),
                    "rate_hike": rate_hike_impact.get("impact")
                },
                "limit_check_status": limit_check_result.get("status")
            }
            
            # 8. Store the analysis result
            await self.store_risk_analysis("current_portfolio", analysis_result)
            
            duration = (time.time() - start_time) * 1000
            self.logger.timing("risk_assessment_model.portfolio_analysis_duration_ms", duration)
            self.portfolio_analysis_count += 1
            self.logger.counter("risk_assessment_model.portfolio_analysis_count")
            
            return analysis_result
        
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio risk: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("risk_assessment_model.execution_errors")
            
            # Create error result
            error_result = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_id": "current_portfolio",
                "status": "error",
                "message": f"Portfolio risk analysis failed: {str(e)}",
                "metrics": {},
                "contributions": {},
                "factor_exposure": {},
                "scenario_impacts": {},
                "limit_check_status": "not_checked"
            }
            
            # Store the error result
            await self.store_risk_analysis("current_portfolio", error_result)
            
            duration = (time.time() - start_time) * 1000
            self.logger.timing("risk_assessment_model.portfolio_analysis_duration_ms", duration)
            
            return error_result

    async def process_symbol_for_decision(
        self,
        symbol: str,
        collected_reports: Dict[str, Any],
        available_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process data for a single symbol to prepare it for the Decision Model package.

        This involves:
        1. Extracting relevant data for the symbol from collected_reports.
        2. Performing symbol-specific risk assessment (e.g., standalone VaR, contribution if in portfolio).
        3. Synthesizing information into a structured format.

        Args:
            symbol: The stock symbol to process.
            collected_reports: Dictionary containing reports from various models.
            available_capital: Current available capital (optional).

        Returns:
            Dictionary containing the processed analysis for the symbol.
        """
        self.logger.info(f"Processing symbol '{symbol}' for decision package.")
        start_time = time.time()

        try:
            # Extract data for the symbol from collected_reports
            reports = collected_reports.get("reports", {})
            sentiment_data = reports.get("sentiment", {}).get(symbol, {})
            fundamental_data = reports.get("fundamental", {}).get(symbol, {})
            technical_data = reports.get("technical", {}).get(symbol, {})
            market_context = reports.get("market", {})
            
            # Initialize symbol analysis structure
            symbol_analysis = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "status": "processed",
                "data_sources": [],
                "risk_metrics": {},
                "sentiment_summary": sentiment_data,
                "fundamental_summary": fundamental_data,
                "technical_summary": technical_data,
                "market_context": market_context,
                "potential_trade_size": None
            }
            
            # Track which data sources were available
            if sentiment_data: symbol_analysis["data_sources"].append("sentiment")
            if fundamental_data: symbol_analysis["data_sources"].append("fundamental")
            if technical_data: symbol_analysis["data_sources"].append("technical")
            if market_context: symbol_analysis["data_sources"].append("market")
            
            # Check if we have enough data to assess
            if not symbol_analysis["data_sources"]:
                return {"symbol": symbol, "status": "insufficient_data", "error": "No model reports available for this symbol"}
                
            # Perform symbol-specific risk assessment
            
            # 1. Extract volatility from technical data 
            volatility = technical_data.get("volatility", 0.02)  # Default to 2% daily volatility if not available
            
            # 2. Calculate standalone VaR for this symbol (simplified)
            confidence_level = 0.95  # 95% confidence
            z_score = 1.645  # Z-score for 95% confidence in normal distribution
            symbol_var = volatility * z_score
            
            # 3. Calculate expected shortfall (simplified)
            symbol_es = symbol_var * 1.25  # Simplified - in reality would use proper calculation
            
            # 4. Calculate beta from technical data or market context
            beta = technical_data.get("beta", 1.0)  # Default to market beta if not available
            
            # 5. Assess if the symbol is currently in portfolio and contribution to portfolio risk
            in_portfolio = symbol in self.current_portfolio
            contribution_to_portfolio_risk = None
            if in_portfolio:
                # Simple approximation - in reality would calculate proper risk contribution
                position_size = self.current_portfolio[symbol].get("shares", 0) * self.current_portfolio[symbol].get("avg_price", 0)
                portfolio_total = sum(pos.get("shares", 0) * pos.get("avg_price", 0) for pos in self.current_portfolio.values())
                if portfolio_total > 0:
                    weight = position_size / portfolio_total
                    contribution_to_portfolio_risk = weight * beta * volatility
            
            # Store risk metrics in the analysis
            symbol_analysis["risk_metrics"] = {
                "volatility": volatility,
                "var_95": symbol_var,
                "expected_shortfall_95": symbol_es,
                "beta": beta,
                "in_portfolio": in_portfolio,
                "contribution_to_portfolio_risk": contribution_to_portfolio_risk
            }
            
            # 6. Calculate potential trade size based on risk and available capital
            if available_capital:
                try:
                    # Calculate risk-based position size
                    # Rule: Risk no more than 0.5% of capital per position
                    risk_per_dollar = volatility  # Daily risk per dollar invested
                    max_risk_amount = available_capital * 0.005  # 0.5% of capital at risk
                    
                    # Get current price from technical or fundamental data
                    price = technical_data.get("price") or fundamental_data.get("price")
                    if not price and "last_price" in technical_data:
                        price = technical_data["last_price"]
                    if not price:
                        price = 100  # Default if no price data available
                    
                    # Calculate max position size based on risk
                    if risk_per_dollar > 0:
                        max_position_size = max_risk_amount / risk_per_dollar
                        max_shares = int(max_position_size / price)
                        
                        # Apply minimum position size (e.g., at least $1000)
                        min_position = 1000
                        min_shares = max(1, int(min_position / price))
                        
                        # Final share count: either risk-based or minimum, whichever is smaller
                        shares = min(max_shares, max(min_shares, int(available_capital * 0.1 / price)))
                        
                        symbol_analysis["potential_trade_size"] = {
                            "shares": shares,
                            "estimated_value": shares * price,
                            "estimated_risk_usd": shares * price * volatility,
                            "capital_percentage": (shares * price / available_capital) if available_capital > 0 else 0
                        }
                except Exception as calc_err:
                    self.logger.warning(f"Could not calculate trade size for {symbol}: {calc_err}")
                    symbol_analysis["potential_trade_size"] = {"error": str(calc_err)}
            
            # 7. Add risk-based recommendation (simplified)
            risk_rating = "medium"
            if volatility < 0.01:
                risk_rating = "low"
            elif volatility > 0.03:
                risk_rating = "high"
                
            sentiment_score = sentiment_data.get("score", 0) if isinstance(sentiment_data, dict) else 0
            fundamental_score = fundamental_data.get("score", 0) if isinstance(fundamental_data, dict) else 0
            technical_score = technical_data.get("score", 0) if isinstance(technical_data, dict) else 0
                
            symbol_analysis["risk_recommendation"] = {
                "risk_rating": risk_rating,
                "summary": f"Symbol {symbol} has {risk_rating} risk profile with {volatility:.1%} daily volatility and {beta:.2f} beta.",
                "trade_considerations": f"Consider {'smaller' if risk_rating == 'high' else 'standard'} position size given the risk profile."
            }
            
            duration = (time.time() - start_time) * 1000
            self.logger.timing("risk_assessment_model.symbol_processing_duration_ms", duration, tags={"symbol": symbol})
            return symbol_analysis
            
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol} for decision: {e}", exc_info=True)
            self.execution_errors += 1
            duration = (time.time() - start_time) * 1000
            self.logger.timing("risk_assessment_model.symbol_processing_duration_ms", duration, tags={"symbol": symbol, "status": "failed"})
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # --- Main Execution Loop / Trigger ---

    async def run_assessment_cycle(self):
        """
        Run a full risk assessment cycle.
        This could be triggered periodically or by specific events.
        """
        self.logger.info("Starting new risk assessment cycle.")
        start_time = time.time()

        # 1. Monitor for inputs (e.g., selection responses, trade events)
        # These monitors might run continuously in separate tasks/threads in a real system.
        # For a single cycle, we can call them sequentially.
        await self.monitor_selection_responses()  # Checks for new symbols to analyze
        await self.monitor_trade_events()  # Updates internal portfolio/capital state

        # 2. Perform portfolio-level risk analysis (if portfolio exists)
        if self.current_portfolio:
            # Fetch necessary market data (placeholder)
            # In a real implementation, this would come from a data provider or database
            market_data = await self._fetch_market_data(list(self.current_portfolio.keys()))
            
            # Run the actual portfolio analysis
            portfolio_analysis_result = await self.analyze_portfolio_risk(self.current_portfolio, market_data)
            self.logger.info(f"Portfolio analysis completed: {portfolio_analysis_result.get('status')}")

            # 3. Check risk limits based on the analysis
            limit_check_result = await self.check_risk_limits(
                "current_portfolio", 
                portfolio_analysis_result.get("metrics", {})
            )
            self.risk_limit_checks_count += 1
            self.logger.counter("risk_assessment_model.risk_limit_checks_count")
            
            if limit_check_result.get("status") == "exceeded":
                self.risk_alerts_generated_count += 1
                self.logger.counter("risk_assessment_model.risk_alerts_generated_count")
                self.logger.critical(f"RISK LIMITS EXCEEDED: {limit_check_result.get('exceeded_limits')}")
                
                # Generate alert
                alert_data = {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_id": "current_portfolio",
                    "type": "limit_exceeded",
                    "details": limit_check_result.get('exceeded_limits'),
                    "analytics": portfolio_analysis_result
                }
                
                # Store alert in Redis
                self.redis_mcp.call_tool(
                    "lpush", 
                    {
                        "key": self.redis_keys["risk_alerts"], 
                        "values": [json.dumps(alert_data)]
                    }
                )
                
                # Implement other alert notification mechanisms here if needed
                # For example, sending email, triggering system alerts, etc.
                
        else:
            self.logger.info("No current portfolio to analyze.")

        cycle_duration = (time.time() - start_time) * 1000
        self.logger.timing("risk_assessment_model.assessment_cycle_duration_ms", cycle_duration)
        self.logger.info("Risk assessment cycle finished.")
        
    async def _fetch_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch market data needed for risk analysis.
        
        Args:
            symbols: List of symbols to fetch data for
            
        Returns:
            Dictionary containing market data
        """
        fetch_start_time = time.time()
        self.logger.info(f"Fetching market data for {len(symbols)} symbols")
        
        # Track this request in performance metrics
        with self.monitoring_lock:
            self.performance_metrics["symbols_processed"].update(symbols)
            unique_symbols_count = len(self.performance_metrics["symbols_processed"])
            self.logger.gauge("risk_assessment_model.unique_symbols_processed", unique_symbols_count)
            
            # Track API request patterns
            hour_of_day = datetime.now().hour
            if not hasattr(self, "hourly_api_requests"):
                self.hourly_api_requests = {hour: 0 for hour in range(24)}
            self.hourly_api_requests[hour_of_day] = self.hourly_api_requests.get(hour_of_day, 0) + 1
            
            # Track request volume
            if not hasattr(self, "data_request_volume"):
                self.data_request_volume = {"last_minute": 0, "last_hour": 0, "last_day": 0, "timestamps": []}
            self.data_request_volume["last_minute"] += 1
            self.data_request_volume["last_hour"] += 1
            self.data_request_volume["last_day"] += 1
            self.data_request_volume["timestamps"].append(time.time())
            
            # Prune old timestamps
            now = time.time()
            self.data_request_volume["timestamps"] = [t for t in self.data_request_volume["timestamps"] if now - t <= 86400]  # Keep last day
            
            # Recalculate periods based on timestamps
            minute_ago = now - 60
            hour_ago = now - 3600
            self.data_request_volume["last_minute"] = len([t for t in self.data_request_volume["timestamps"] if t >= minute_ago])
            self.data_request_volume["last_hour"] = len([t for t in self.data_request_volume["timestamps"] if t >= hour_ago])
            self.data_request_volume["last_day"] = len(self.data_request_volume["timestamps"])
            
            # Update metrics for monitoring
            self.logger.gauge("risk_assessment_model.market_data_requests_per_minute", self.data_request_volume["last_minute"])
            self.logger.gauge("risk_assessment_model.market_data_requests_per_hour", self.data_request_volume["last_hour"])
            self.logger.gauge("risk_assessment_model.market_data_requests_per_day", self.data_request_volume["last_day"])
        
        # Check for cached data first
        cache_key = f"market_data:{','.join(sorted(symbols))}"
        cache_hit = False
        
        if hasattr(self, 'market_data_cache') and cache_key in self.market_data_cache:
            cache_timestamp = self.market_data_cache_timestamps.get(cache_key, 0)
            cache_age = time.time() - cache_timestamp
            
            # Use cache if it's less than 1 hour old
            if cache_age < 3600:
                self.logger.info(f"Using cached market data for {len(symbols)} symbols, cache age: {cache_age:.1f} seconds")
                with self.monitoring_lock:
                    if not hasattr(self, 'market_data_cache_hits'):
                        self.market_data_cache_hits = 0
                        self.market_data_cache_misses = 0
                    self.market_data_cache_hits += 1
                    hit_rate = self.market_data_cache_hits / (self.market_data_cache_hits + self.market_data_cache_misses) * 100
                    self.logger.gauge("risk_assessment_model.market_data_cache_hit_rate", hit_rate)
                    
                fetch_duration = (time.time() - fetch_start_time) * 1000
                self.logger.timing("risk_assessment_model.market_data_fetch_duration_ms", fetch_duration, tags={"source": "cache", "symbols_count": len(symbols)})
                return self.market_data_cache[cache_key]
                
        # This is a placeholder function. In a real implementation,
        # this would connect to market data providers, databases, or APIs
        # to get historical prices, returns, factors, etc.
        
        # Track potential rate limiting and API health
        with self.monitoring_lock:
            if not hasattr(self, 'market_data_api_failures'):
                self.market_data_api_failures = 0
                self.market_data_api_successes = 0
                self.market_data_api_latencies = []
            
            if not hasattr(self, 'market_data_cache'):
                self.market_data_cache = {}
                self.market_data_cache_timestamps = {}
                self.market_data_cache_misses = 0
            
            # Record cache miss
            self.market_data_cache_misses += 1
            hit_rate = self.market_data_cache_hits / (self.market_data_cache_hits + self.market_data_cache_misses) * 100 if hasattr(self, 'market_data_cache_hits') else 0
            self.logger.gauge("risk_assessment_model.market_data_cache_hit_rate", hit_rate)
        
        # For now, generate dummy data for testing
        market_data = {}
        
        # Create some random returns for the past year (252 trading days)
        import random
        random.seed(42)  # For reproducibility
        
        try:
            for symbol in symbols:
                symbol_start_time = time.time()
                
                # Create slightly different volatility profiles for each symbol
                vol_factor = 0.01 + (ord(symbol[0]) % 10) * 0.002  # Use first letter to vary volatility
                
                # Generate daily returns with that volatility
                symbol_returns = []
                for _ in range(252):
                    daily_return = random.normalvariate(0.0005, vol_factor)  # Mean 0.05% daily
                    symbol_returns.append(daily_return)
                    
                market_data[f"{symbol}_returns"] = symbol_returns
                
                symbol_duration = (time.time() - symbol_start_time) * 1000
                self.logger.timing("risk_assessment_model.symbol_data_fetch_duration_ms", symbol_duration, tags={"symbol": symbol})
            
            # Add some factor returns
            market_data["market_factor"] = [random.normalvariate(0.0006, 0.01) for _ in range(252)]
            market_data["size_factor"] = [random.normalvariate(0.0002, 0.005) for _ in range(252)]
            market_data["value_factor"] = [random.normalvariate(0.0001, 0.006) for _ in range(252)]
            market_data["momentum_factor"] = [random.normalvariate(0.0003, 0.007) for _ in range(252)]
            
            # Store fetched data in cache
            with self.monitoring_lock:
                self.market_data_cache[cache_key] = market_data
                self.market_data_cache_timestamps[cache_key] = time.time()
                
                # Cleanup cache if too large (keep most recent 1000 entries max)
                if len(self.market_data_cache) > 1000:
                    oldest_keys = sorted(self.market_data_cache_timestamps.items(), key=lambda x: x[1])[:len(self.market_data_cache) - 1000]
                    for key, _ in oldest_keys:
                        del self.market_data_cache[key]
                        del self.market_data_cache_timestamps[key]
                
                # Track API call success
                self.market_data_api_successes += 1
                success_rate = self.market_data_api_successes / (self.market_data_api_successes + self.market_data_api_failures) * 100
                self.logger.gauge("risk_assessment_model.market_data_api_success_rate", success_rate)
            
            fetch_duration = (time.time() - fetch_start_time) * 1000
            self.logger.timing("risk_assessment_model.market_data_fetch_duration_ms", fetch_duration, tags={"source": "api", "symbols_count": len(symbols)})
            
            # Record detailed analytics
            with self.monitoring_lock:
                self.market_data_api_latencies.append(fetch_duration)
                # Keep last 100 latencies
                if len(self.market_data_api_latencies) > 100:
                    self.market_data_api_latencies = self.market_data_api_latencies[-100:]
                # Calculate average latency
                avg_latency = sum(self.market_data_api_latencies) / len(self.market_data_api_latencies)
                self.logger.gauge("risk_assessment_model.market_data_api_avg_latency_ms", avg_latency)
                
                # Log if this request was slow (more than 2x avg latency)
                if fetch_duration > avg_latency * 2 and avg_latency > 0:
                    self.logger.warning("Slow market data fetch detected", 
                                       duration_ms=fetch_duration,
                                       avg_latency_ms=avg_latency,
                                       symbols_count=len(symbols))
                    # Add to slow operations tracking
                    self.performance_metrics["slow_operations"].append({
                        "timestamp": datetime.now().isoformat(),
                        "operation": "market_data_fetch",
                        "duration_ms": fetch_duration,
                        "symbols_count": len(symbols),
                        "symbols": symbols[:5] + (["..."] if len(symbols) > 5 else [])
                    })
            
            return market_data
            
        except Exception as e:
            # Track API call failure
            with self.monitoring_lock:
                self.market_data_api_failures += 1
                success_rate = self.market_data_api_successes / (self.market_data_api_successes + self.market_data_api_failures) * 100
                self.logger.gauge("risk_assessment_model.market_data_api_success_rate", success_rate)
                
                # Log the error with details
                self.logger.error(f"Error fetching market data: {e}", 
                                symbols_count=len(symbols),
                                symbols=symbols[:5] + (["..."] if len(symbols) > 5 else []),
                                exc_info=True)
                
            # Return empty data
            return {}


# Example Usage (Optional - for testing)
async def main():
    # Load config if needed, or pass None to use default path
    model = RiskAssessmentModel(config=None)

    # Example: Manually trigger an assessment cycle
    await model.run_assessment_cycle()

    # Example: Simulate receiving selected symbols (normally done via Redis stream)
    await model.create_consolidated_package(["AAPL", "MSFT", "GOOG"], request_id="test-req-123", available_capital=100000)

if __name__ == "__main__":
    # Setup basic logging if running standalone
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Run the main async function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Risk Assessment Model stopped.")
