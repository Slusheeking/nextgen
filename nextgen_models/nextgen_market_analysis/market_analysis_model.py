"""
Market Analysis Model

This module defines the MarketAnalysisModel, responsible for technical analysis
of market data using indicators, pattern recognition, and market scanning capabilities.
It integrates with AutoGen for advanced analysis and decision making.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
import numpy as np
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import enhanced monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector
from monitoring.stock_charts import StockChartGenerator

# MCP tools (Consolidated)
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP # Import RedisMCP

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function


class MarketAnalysisModel:
    """
    Analyzes market data using technical indicators and pattern recognition.

    This model integrates FinancialDataMCP and TimeSeriesMCP to provide
    comprehensive market analysis, including data retrieval, indicator calculations,
    pattern detection, and market scanning capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, user_proxy_agent: Optional[UserProxyAgent] = None):
        """
        Initialize the MarketAnalysisModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - financial_data_config: Config for FinancialDataMCP
                - time_series_config: Config for TimeSeriesMCP
                - redis_config: Config for RedisMCP
                - llm_config: Configuration for AutoGen LLM
                - scan_interval: Interval for market scanning in seconds (default: 3600)
                - max_scan_symbols: Maximum number of symbols to scan (default: 100)
                - default_indicators: List of default indicators to calculate
                - default_patterns: List of default patterns to detect
            user_proxy_agent: Optional UserProxyAgent instance to register functions with.
        """
        try:
            self.config = config or {}
            
            # Initialize enhanced monitoring with NetdataLogger
            self.logger = NetdataLogger(component_name="market-analysis-model")
            self.logger.info("Initializing MarketAnalysisModel")
            
            # Start system metrics collection
            self.metrics_collector = SystemMetricsCollector(self.logger)
            self.metrics_collector.start()
            
            # Initialize chart generator for visualization
            self.chart_generator = StockChartGenerator()
            
            # Initialize thread-safe metrics tracking with locks
            self._metrics_lock = threading.RLock()
            
            # Initialize performance metrics
            self.start_time = datetime.now()
            
            # Initialize counters for market analysis metrics
            self.mcp_tool_call_count = 0
            self.mcp_tool_error_count = 0
            self.analysis_count = 0
            self.scan_count = 0
            self.execution_errors = 0
            
            # Performance tracking metrics
            self.operation_timing = {
                "get_market_data": [],
                "analyze_symbol": [],
                "scan_market": [],
                "assess_risk": [],
                "tool_calls": {}
            }
            
            # Initialize health metrics
            self.health_check_thread = None
            self.health_check_running = False
            self.last_health_check = None
            self.health_status = {
                "overall_health": 100.0,
                "last_check": None,
                "components": {
                    "financial_data_mcp": {"status": "unknown", "score": 0},
                    "time_series_mcp": {"status": "unknown", "score": 0},
                    "redis_mcp": {"status": "unknown", "score": 0},
                    "autogen": {"status": "unknown", "score": 0}
                }
            }

            # Initialize Consolidated MCP clients with timing metrics
            self._initialize_mcp_clients()

            # Configuration parameters
            self._set_configuration_parameters()
            
            # Start periodic health check thread
            self._start_health_check_thread()

            # Redis keys for data storage and inter-model communication
            self._setup_redis_keys()

            # Initialize AutoGen integration
            self._initialize_autogen()

            # Record total initialization time
            total_init_time = (datetime.now() - self.start_time).total_seconds() * 1000
            self.logger.timing("total_init_time_ms", total_init_time)
            
            # Log initialization metrics
            self.logger.info("MarketAnalysisModel initialized",
                           config=str(self.config.keys()),
                           init_time_ms=total_init_time)
            
            # Send startup metrics to Netdata
            self.logger.gauge("model_startup_time_ms", total_init_time)
            self.logger.gauge("model_version", 1.0)  # Increment on major updates
            
            # Initialize model status in Redis
            self._update_model_status("initialized")

        except Exception as e:
            self.logger.error(f"Error during MarketAnalysisModel initialization: {str(e)}", exc_info=True)
            raise

    def _initialize_mcp_clients(self):
        """Initialize MCP clients."""
        try:
            # FinancialDataMCP
            start_time = time.time()
            self.logger.info("Initializing FinancialDataMCP")
            financial_data_config = self.config.get("financial_data_config")
            if not financial_data_config:
                self.logger.warning("Missing financial_data_config in configuration, using default configuration")
                # Provide a default configuration
                financial_data_config = {
                    "api_keys": {},
                    "data_sources": ["polygon", "yahoo_finance"],
                    "cache_ttl": 3600,
                    "max_retries": 3,
                    "timeout": 30
                }
            self.financial_data_mcp = FinancialDataMCP(financial_data_config)
            financial_data_init_time = (time.time() - start_time) * 1000
            self.logger.timing("financial_data_mcp_init_time_ms", financial_data_init_time)
            self.logger.info("FinancialDataMCP initialized successfully")
            
            # TimeSeriesMCP
            start_time = time.time()
            self.logger.info("Initializing TimeSeriesMCP")
            time_series_config = self.config.get("time_series_mcp_config") # Look for the correct key
            if not time_series_config:
                raise ValueError("Missing time_series_mcp_config in configuration") # Update error message
            self.time_series_mcp = TimeSeriesMCP(time_series_config)
            time_series_init_time = (time.time() - start_time) * 1000
            self.logger.timing("time_series_mcp_init_time_ms", time_series_init_time)
            self.logger.info("TimeSeriesMCP initialized successfully")
            
            # RedisMCP
            start_time = time.time()
            self.logger.info("Initializing RedisMCP")
            redis_config = self.config.get("redis_mcp_config") # Look for the correct key
            if not redis_config:
                raise ValueError("Missing redis_mcp_config in configuration") # Update error message
            self.redis_mcp = RedisMCP(redis_config)
            redis_init_time = (time.time() - start_time) * 1000
            self.logger.timing("redis_mcp_init_time_ms", redis_init_time)
            self.logger.info("RedisMCP initialized successfully")

            # Verify MCP clients are properly initialized
            self._verify_mcp_clients()

        except Exception as e:
            self.logger.error(f"Error initializing MCP clients: {str(e)}", exc_info=True)
            raise

    def _verify_mcp_clients(self):
        """Verify that MCP clients are properly initialized and functional."""
        try:
            # Check FinancialDataMCP
            if not hasattr(self.financial_data_mcp, 'call_tool'):
                raise AttributeError("FinancialDataMCP missing 'call_tool' method")
            
            # Check TimeSeriesMCP
            if not hasattr(self.time_series_mcp, 'call_tool'):
                raise AttributeError("TimeSeriesMCP missing 'call_tool' method")
            
            # Check RedisMCP
            if not hasattr(self.redis_mcp, 'call_tool'):
                raise AttributeError("RedisMCP missing 'call_tool' method")
            
            # Perform a simple operation with each MCP to ensure they're functional
            self.financial_data_mcp.call_tool("get_health", {})
            self.time_series_mcp.call_tool("get_health", {})
            self.redis_mcp.call_tool("ping", {})
            
            self.logger.info("All MCP clients verified and functional")
        except Exception as e:
            self.logger.error(f"Error verifying MCP clients: {str(e)}", exc_info=True)
            raise

    def _set_configuration_parameters(self):
        """Set configuration parameters."""
        self.scan_interval = self.config.get("scan_interval", 3600)  # Default: 1 hour
        self.max_scan_symbols = self.config.get("max_scan_symbols", 100)
        self.default_indicators = self.config.get(
            "default_indicators", ["sma", "ema", "rsi", "macd", "bbands", "atr", "adx"]
        )
        self.default_patterns = self.config.get(
            "default_patterns",
            [
                "double_top",
                "double_bottom",
                "head_and_shoulders",
                "ascending_triangle",
                "descending_triangle",
                "symmetrical_triangle",
            ],
        )

    def _setup_redis_keys(self):
        """Setup Redis keys and ensure streams exist."""
        self.redis_keys = {
            "market_analysis_data": "market_analysis:data",
            "technical_indicators": "market_analysis:indicators:",
            "pattern_recognition": "market_analysis:patterns:",
            "market_scan_results": "market_analysis:scan_results",
            "latest_analysis": "market_analysis:latest_analysis:",
            "selection_data": "selection:data",
            "selection_feedback": "market_analysis:selection_feedback",
            "market_analysis_reports_stream": "model:market:trends",
        }

        try:
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["market_analysis_reports_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['market_analysis_reports_stream']}' exists.")
        except Exception as e:
            self.logger.warning(f"Could not ensure Redis stream exists: {e}")

    def _initialize_autogen(self):
        """Initialize AutoGen integration."""
        start_time = time.time()
        self.logger.info("Initializing AutoGen integration")
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()
        self._register_functions()
        autogen_init_time = (time.time() - start_time) * 1000
        self.logger.timing("autogen_init_time_ms", autogen_init_time)

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
                    }
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
        Initialize AutoGen agents for market analysis.
        """
        agents = {}

        # Create the market analysis assistant agent
        agents["market_analysis_assistant"] = AssistantAgent(
            name="MarketAnalysisAssistant",
            system_message="""You are a technical market analysis specialist. Your role is to:
            1. Analyze market data using technical indicators
            2. Identify chart patterns and potential trading signals
            3. Evaluate market conditions and trends
            4. Provide insights on potential trading opportunities

            You have tools for calculating technical indicators, detecting patterns,
            and scanning the market for opportunities. Always provide clear reasoning
            for your analysis and recommendations.""",
            llm_config=self.llm_config,
            description="A specialist in technical market analysis",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="MarketAnalysisToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
        )

        agents["user_proxy"] = user_proxy

        return agents

    # --- Tool Methods (to be registered) ---

    def _tool_calculate_indicators(
        self, price_data: List[Dict[str, Any]], indicators: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Internal method for calculating technical indicators."""
        indicators = indicators or self.default_indicators
        self.mcp_tool_call_count += 1
        result = self.time_series_mcp.call_tool("calculate_indicators", {"data": price_data, "indicators": indicators})
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    def _tool_calculate_moving_averages(
        self, price_data: List[Dict[str, Any]], ma_types: List[str], periods: List[int]
    ) -> Dict[str, Any]:
        """Internal method for calculating moving averages."""
        self.mcp_tool_call_count += 1
        result = self.time_series_mcp.call_tool("calculate_moving_averages", {"data": price_data, "ma_types": ma_types, "periods": periods})
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    def _tool_calculate_momentum_oscillators(
        self, price_data: List[Dict[str, Any]], oscillators: List[str]
    ) -> Dict[str, Any]:
        """Internal method for calculating momentum oscillators."""
        self.mcp_tool_call_count += 1
        result = self.time_series_mcp.call_tool("calculate_momentum_oscillators", {"data": price_data, "oscillators": oscillators})
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    def _tool_detect_patterns(
        self, price_data: List[Dict[str, Any]], patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Internal method for detecting chart patterns."""
        patterns = patterns or self.default_patterns
        self.mcp_tool_call_count += 1
        result = self.time_series_mcp.call_tool("detect_patterns", {"data": price_data, "patterns": patterns})
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    def _tool_detect_candlestick_patterns(
        self, price_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Internal method for detecting candlestick patterns."""
        self.mcp_tool_call_count += 1
        result = self.time_series_mcp.call_tool("detect_candlestick_patterns", {"data": price_data})
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    def _tool_detect_support_resistance(
        self, price_data: List[Dict[str, Any]], method: str = "peaks"
    ) -> Dict[str, Any]:
        """Internal method for detecting support and resistance."""
        self.mcp_tool_call_count += 1
        result = self.time_series_mcp.call_tool("detect_support_resistance", {"data": price_data, "method": method})
        if result and result.get("error"):
             self.mcp_tool_error_count += 1
        return result

    def _tool_get_market_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 100
    ) -> Dict[str, Any]:
        """Internal method for getting market data."""
        return self.get_market_data(symbol, timeframe, limit)

    def _tool_scan_market(
        self, symbols: List[str], scan_type: str = "technical"
    ) -> Dict[str, Any]:
        """Internal method for scanning the market."""
        return self.scan_market(symbols, scan_type)

    def _tool_get_scan_results(self) -> Dict[str, Any]:
        """Internal method for getting scan results."""
        return self.get_scan_results()

    def _tool_get_selection_data(self) -> Dict[str, Any]:
        """Internal method for getting selection data."""
        return self.get_selection_data()

    def _tool_send_feedback_to_selection(self, analysis_data: Dict[str, Any]) -> bool:
        """Internal method for sending feedback to selection model."""
        return self.send_feedback_to_selection(analysis_data)

    def _tool_use_financial_data_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Any:
        """Internal method for using FinancialDataMCP tools."""
        operation_id = f"financial_data.{tool_name}.{int(time.time() * 1000)}"
        start_time = time.time()
        self.logger.info(f"Calling FinancialDataMCP.{tool_name}", operation_id=operation_id, arguments=str(arguments.keys())[:100])
        try:
            result = self.financial_data_mcp.call_tool(tool_name, arguments)
            duration_ms = (time.time() - start_time) * 1000
            success = not (result and result.get("error"))
            if not success:
                self.logger.error(f"Error calling FinancialDataMCP.{tool_name}", operation_id=operation_id, error=result.get("error", "Unknown error"))
            self._track_tool_call("financial_data", tool_name, duration_ms, success)
            if isinstance(result, dict):
                result["_metadata"] = {"operation_id": operation_id, "duration_ms": duration_ms, "timestamp": datetime.now().isoformat()}
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Exception calling FinancialDataMCP.{tool_name}", operation_id=operation_id, error=str(e))
            self._track_tool_call("financial_data", tool_name, duration_ms, False)
            raise

    def _tool_use_time_series_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Internal method for using TimeSeriesMCP tools."""
        operation_id = f"time_series.{tool_name}.{int(time.time() * 1000)}"
        start_time = time.time()
        self.logger.info(f"Calling TimeSeriesMCP.{tool_name}", operation_id=operation_id, arguments=str(arguments.keys())[:100])
        try:
            result = self.time_series_mcp.call_tool(tool_name, arguments)
            duration_ms = (time.time() - start_time) * 1000
            success = not (result and result.get("error"))
            if not success:
                self.logger.error(f"Error calling TimeSeriesMCP.{tool_name}", operation_id=operation_id, error=result.get("error", "Unknown error"))
            self._track_tool_call("time_series", tool_name, duration_ms, success)
            if isinstance(result, dict):
                result["_metadata"] = {"operation_id": operation_id, "duration_ms": duration_ms, "timestamp": datetime.now().isoformat()}
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Exception calling TimeSeriesMCP.{tool_name}", operation_id=operation_id, error=str(e))
            self._track_tool_call("time_series", tool_name, duration_ms, False)
            raise

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
        market_analysis_assistant = self.agents["market_analysis_assistant"]

        # Register technical indicator functions
        register_function(
            lambda price_data: self._tool_calculate_indicators(price_data),
            name="calculate_indicators",
            description="Calculate technical indicators for price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        register_function(
            lambda price_data, ma_types, periods: self._tool_calculate_moving_averages(price_data, ma_types, periods),
            name="calculate_moving_averages",
            description="Calculate various types of moving averages",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        register_function(
            lambda price_data, oscillators: self._tool_calculate_momentum_oscillators(price_data, oscillators),
            name="calculate_momentum_oscillators",
            description="Calculate momentum oscillators like RSI, Stochastic, MACD",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        # Register pattern recognition functions
        register_function(
            lambda price_data: self._tool_detect_patterns(price_data),
            name="detect_patterns",
            description="Detect technical chart patterns in price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        register_function(
            lambda price_data: self._tool_detect_candlestick_patterns(price_data),
            name="detect_candlestick_patterns",
            description="Detect candlestick patterns in price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        register_function(
            lambda price_data, method="peaks": self._tool_detect_support_resistance(price_data, method),
            name="detect_support_resistance",
            description="Detect support and resistance levels in price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        # Register market data functions
        register_function(
            lambda symbol, timeframe="1d", limit=100: self._tool_get_market_data(symbol, timeframe, limit),
            name="get_market_data",
            description="Get market data for a symbol",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        # Register market scanning functions
        register_function(
            lambda symbols, scan_type="technical": self._tool_scan_market(symbols, scan_type),
            name="scan_market",
            description="Scan the market for trading opportunities",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        register_function(
            lambda: self._tool_get_scan_results(),
            name="get_scan_results",
            description="Get the latest market scan results",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        # Register selection model integration functions
        register_function(
            lambda: self._tool_get_selection_data(),
            name="get_selection_data",
            description="Get data from the Selection Model",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        register_function(
            lambda analysis_data: self._tool_send_feedback_to_selection(analysis_data),
            name="send_feedback_to_selection",
            description="Send technical analysis feedback to the Selection Model",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )

        # Register MCP tool access functions
        self._register_mcp_tool_access(user_proxy_agent)

    def _start_health_check_thread(self, interval=60):
        """Start a background thread for periodic health checks."""
        if self.health_check_thread is not None and self.health_check_thread.is_alive():
            self.logger.info("Health check thread is already running")
            return

        self.health_check_running = True
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            args=(interval,),
            daemon=True
        )
        self.health_check_thread.start()
        self.logger.info("Health check thread started", interval=interval)
    
    def _health_check_loop(self, interval):
        """Background loop for periodic health checks."""
        while self.health_check_running:
            try:
                # Perform a health check
                health_data = self.get_health_check()
                
                # Update the health status
                self.health_status = health_data
                
                # Log health status
                self.logger.info("Health check completed", 
                               overall_health=health_data["overall_health"],
                               components=len(health_data["components"]))
                
                # Send health metrics to Netdata
                self.logger.gauge("health_score", health_data["overall_health"])
                for component, data in health_data["components"].items():
                    self.logger.gauge(f"component_health.{component}", data["score"])
                
                # Store health data in Redis
                try:
                    self.redis_mcp.call_tool(
                        "set_json",
                        {
                            "key": "market_analysis:health_status",
                            "value": health_data,
                            "expiry": interval * 2  # Expire after 2 intervals
                        }
                    )
                except Exception as e:
                    self.logger.error("Failed to store health data in Redis", error=str(e))
                
                # Sleep until next check
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error("Error in health check thread", error=str(e))
                time.sleep(interval)  # Sleep even on error
    
    def _update_model_status(self, status, metadata=None):
        """Update the model status in Redis."""
        try:
            status_data = {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            self.redis_mcp.call_tool(
                "set_json",
                {
                    "key": "market_analysis:model_status",
                    "value": status_data
                }
            )
            
            self.logger.info(f"Model status updated: {status}", metadata=metadata)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update model status: {e}")
            return False
    
    def _track_operation_timing(self, operation_name, duration_ms):
        """
        Thread-safe tracking of operation timing for performance analysis.
        
        Args:
            operation_name: Name of the operation
            duration_ms: Duration in milliseconds
        """
        with self._metrics_lock:
            if operation_name in self.operation_timing:
                # Limit list size to prevent memory issues
                if len(self.operation_timing[operation_name]) >= 1000:
                    self.operation_timing[operation_name].pop(0)
                
                self.operation_timing[operation_name].append(duration_ms)
            else:
                self.operation_timing[operation_name] = [duration_ms]
            
            # Send to Netdata
            self.logger.timing(f"operation.{operation_name}", duration_ms)
    
    def _track_tool_call(self, mcp_name, tool_name, duration_ms, success=True):
        """
        Thread-safe tracking of MCP tool calls.
        
        Args:
            mcp_name: Name of the MCP
            tool_name: Name of the tool
            duration_ms: Duration in milliseconds
            success: Whether the call was successful
        """
        with self._metrics_lock:
            # Increment counters
            self.mcp_tool_call_count += 1
            if not success:
                self.mcp_tool_error_count += 1
            
            # Track timing
            tool_key = f"{mcp_name}.{tool_name}"
            if tool_key not in self.operation_timing["tool_calls"]:
                self.operation_timing["tool_calls"][tool_key] = []
            
            # Limit list size
            if len(self.operation_timing["tool_calls"][tool_key]) >= 100:
                self.operation_timing["tool_calls"][tool_key].pop(0)
            
            self.operation_timing["tool_calls"][tool_key].append(duration_ms)
            
            # Send to Netdata
            self.logger.timing(f"tool_call.{tool_key}", duration_ms)
            self.logger.counter(f"tool_call_count.{tool_key}", 1)
            if not success:
                self.logger.counter(f"tool_call_error.{tool_key}", 1)
    
    def _register_mcp_tool_access(self, user_proxy_agent: Optional[UserProxyAgent] = None):
        """
        Register MCP tool access functions with the user proxy agent.
        
        Args:
            user_proxy_agent: Optional UserProxyAgent instance to register functions with.
                              If None, uses the default user_proxy agent created internally.
        """
        user_proxy = user_proxy_agent if user_proxy_agent is not None else self.agents["user_proxy"]
        market_analysis_assistant = self.agents["market_analysis_assistant"]

        # Define MCP tool access functions for consolidated MCPs
        @register_function(
            name="use_financial_data_tool",
            description="Use a tool provided by the Financial Data MCP server (for market data retrieval)",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_financial_data_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            # Generate a unique operation ID for tracking
            operation_id = f"financial_data.{tool_name}.{int(time.time() * 1000)}"
            
            # Track timing
            start_time = time.time()
            
            # Log the call
            self.logger.info(f"Calling FinancialDataMCP.{tool_name}", 
                           operation_id=operation_id,
                           arguments=str(arguments.keys())[:100])
            
            try:
                # Call the tool
                result = self.financial_data_mcp.call_tool(tool_name, arguments)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Track success or failure
                success = not (result and result.get("error"))
                if not success:
                    self.logger.error(f"Error calling FinancialDataMCP.{tool_name}", 
                                    operation_id=operation_id,
                                    error=result.get("error", "Unknown error"))
                
                # Track metrics
                self._track_tool_call("financial_data", tool_name, duration_ms, success)
                
                # Add metadata to result
                if isinstance(result, dict):
                    result["_metadata"] = {
                        "operation_id": operation_id,
                        "duration_ms": duration_ms,
                        "timestamp": datetime.now().isoformat()
                    }
                
                return result
            except Exception as e:
                # Log and track the exception
                duration_ms = (time.time() - start_time) * 1000
                self.logger.error(f"Exception calling FinancialDataMCP.{tool_name}", 
                                operation_id=operation_id,
                                error=str(e))
                self._track_tool_call("financial_data", tool_name, duration_ms, False)
                raise

        @register_function(
            name="use_time_series_tool",
            description="Use a tool provided by the Time Series MCP server (for indicators and patterns)",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_time_series_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            # Generate a unique operation ID for tracking
            operation_id = f"time_series.{tool_name}.{int(time.time() * 1000)}"
            
            # Track timing
            start_time = time.time()
            
            # Log the call
            self.logger.info(f"Calling TimeSeriesMCP.{tool_name}", 
                           operation_id=operation_id,
                           arguments=str(arguments.keys())[:100])
            
            try:
                # Call the tool
                result = self.time_series_mcp.call_tool(tool_name, arguments)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Track success or failure
                success = not (result and result.get("error"))
                if not success:
                    self.logger.error(f"Error calling TimeSeriesMCP.{tool_name}", 
                                    operation_id=operation_id,
                                    error=result.get("error", "Unknown error"))
                
                # Track metrics
                self._track_tool_call("time_series", tool_name, duration_ms, success)
                
                # Add metadata to result
                if isinstance(result, dict):
                    result["_metadata"] = {
                        "operation_id": operation_id,
                        "duration_ms": duration_ms,
                        "timestamp": datetime.now().isoformat()
                    }
                
                return result
            except Exception as e:
                # Log and track the exception
                duration_ms = (time.time() - start_time) * 1000
                self.logger.error(f"Exception calling TimeSeriesMCP.{tool_name}", 
                                operation_id=operation_id,
                                error=str(e))
                self._track_tool_call("time_series", tool_name, duration_ms, False)
                raise

        

    def assess_risk(self, symbol: str, price_data: List[Dict[str, Any]], portfolio: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate the Value at Risk (VaR) and Expected Shortfall (ES) for a stock
        based on its historical price data.
        
        Args:
            symbol: Stock symbol
            price_data: List of price data dictionaries with at least 'close' prices
            portfolio: Optional dictionary of portfolio weights
            
        Returns:
            Dictionary with risk metrics including VaR, ES, volatility, and beta
        """
        try:
            self.logger.info(f"Assessing risk for {symbol}")
            
            # Extract close prices and calculate returns
            if not price_data or len(price_data) < 2:
                return {"error": "Insufficient price data for risk assessment", "symbol": symbol}
            
            # Extract close prices
            close_prices = [bar.get('close', 0) for bar in price_data if bar.get('close') is not None]
            if len(close_prices) < 2:
                return {"error": "Insufficient valid close prices for risk assessment", "symbol": symbol}
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(close_prices)):
                if close_prices[i-1] > 0:  # Avoid division by zero
                    daily_return = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                    returns.append(daily_return)
            
            if len(returns) < 2:
                return {"error": "Insufficient returns data for risk assessment", "symbol": symbol}
            
            # Calculate volatility (standard deviation of returns)
            volatility = np.std(returns)
            
            # Calculate VaR using historical method
            confidence_level = 0.95  # 95% confidence level
            var_percentile = 1.0 - confidence_level
            var = np.percentile(returns, var_percentile * 100)
            
            # Calculate Expected Shortfall (ES) / Conditional VaR
            # ES is the average of returns beyond VaR
            tail_returns = [r for r in returns if r <= var]
            expected_shortfall = np.mean(tail_returns) if tail_returns else var
            
            # Calculate beta if market data is available
            beta = 1.0  # Default to market beta
            try:
                # Try to get market returns from TimeSeriesMCP
                self.mcp_tool_call_count += 1
                market_data_result = self.financial_data_mcp.call_tool(
                    "get_market_data",
                    {"symbols": ["SPY"], "timeframe": "1d", "limit": len(price_data)}
                )
                
                if market_data_result and not market_data_result.get("error"):
                    market_price_data = market_data_result.get("data", {}).get("SPY", [])
                    if market_price_data and len(market_price_data) >= 2:
                        # Calculate market returns
                        market_close_prices = [bar.get('close', 0) for bar in market_price_data if bar.get('close') is not None]
                        market_returns = []
                        for i in range(1, len(market_close_prices)):
                            if market_close_prices[i-1] > 0:
                                market_return = (market_close_prices[i] - market_close_prices[i-1]) / market_close_prices[i-1]
                                market_returns.append(market_return)
                        
                        # Align the returns (ensure same length)
                        min_length = min(len(returns), len(market_returns))
                        if min_length >= 2:
                            aligned_returns = returns[-min_length:]
                            aligned_market_returns = market_returns[-min_length:]
                            
                            # Calculate covariance and market variance
                            covariance = np.cov(aligned_returns, aligned_market_returns)[0, 1]
                            market_variance = np.var(aligned_market_returns)
                            
                            if market_variance > 0:
                                beta = covariance / market_variance
            except Exception as e:
                self.logger.warning(f"Error calculating beta for {symbol}: {e}")
                # Continue with default beta
            
            # Calculate maximum position size based on risk metrics
            latest_price = close_prices[-1] if close_prices else 0
            max_position_size = self._calculate_max_position_size(var, expected_shortfall, latest_price)
            
            # Calculate dollar VaR for a hypothetical $10,000 position
            position_value = 10000  # $10,000 position
            dollar_var = position_value * abs(var)
            dollar_es = position_value * abs(expected_shortfall)
            
            # Prepare risk assessment result
            risk_assessment = {
                "symbol": symbol,
                "volatility": volatility,
                "var_95": var,
                "expected_shortfall_95": expected_shortfall,
                "beta": beta,
                "dollar_var_10k": dollar_var,
                "dollar_es_10k": dollar_es,
                "max_position_size": max_position_size,
                "risk_rating": self._get_risk_rating(volatility, beta),
                "timestamp": datetime.now().isoformat()
            }
            
            # If portfolio is provided, calculate contribution to portfolio risk
            if portfolio:
                # Calculate portfolio weight for this symbol
                symbol_weight = portfolio.get(symbol, 0)
                
                # Calculate contribution to portfolio risk (simplified)
                # In a real implementation, this would use covariance matrix
                risk_contribution = symbol_weight * volatility * beta
                risk_assessment["portfolio_risk_contribution"] = risk_contribution
            
            return risk_assessment
            
        except Exception as e:
            self.logger.error(f"Error in assess_risk for {symbol}: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}
    
    def _calculate_max_position_size(self, var: float, expected_shortfall: float, price: float) -> float:
        """
        Calculate the maximum position size based on risk metrics and risk per trade limit.
        
        Args:
            var: Value at Risk (as a decimal, e.g., -0.02 for -2%)
            expected_shortfall: Expected Shortfall (as a decimal)
            price: Current price of the asset
            
        Returns:
            Maximum position size in dollars
        """
        try:
            # Default risk parameters
            max_risk_per_trade_pct = 0.01  # 1% of capital per trade
            capital = 100000  # Default capital assumption of $100,000
            
            # Try to get actual capital from Redis if available
            try:
                self.mcp_tool_call_count += 1
                capital_result = self.redis_mcp.call_tool(
                    "get",
                    {"key": "trade:capital_available"}
                )
                
                if capital_result and not capital_result.get("error"):
                    capital_str = capital_result.get("value")
                    if capital_str:
                        try:
                            capital = float(capital_str)
                        except ValueError:
                            self.logger.warning(f"Invalid capital value in Redis: {capital_str}")
            except Exception as e:
                self.logger.warning(f"Error retrieving capital from Redis: {e}")
            
            # Calculate maximum risk amount in dollars
            max_risk_amount = capital * max_risk_per_trade_pct
            
            # Use the worse of VaR or ES for conservative position sizing
            risk_metric = max(abs(var), abs(expected_shortfall))
            
            # Avoid division by zero
            if risk_metric <= 0 or price <= 0:
                return 0
            
            # Calculate maximum position size
            max_position_size = max_risk_amount / risk_metric
            
            # Calculate maximum number of shares
            max_shares = int(max_position_size / price)
            
            # Return the dollar value of the position
            return max_shares * price
            
        except Exception as e:
            self.logger.error(f"Error calculating max position size: {e}", exc_info=True)
            return 0
    
    def _get_risk_rating(self, volatility: float, beta: float) -> str:
        """
        Determine risk rating based on volatility and beta.
        
        Args:
            volatility: Daily volatility
            beta: Market beta
            
        Returns:
            Risk rating as string: "low", "medium", or "high"
        """
        # Convert daily volatility to annualized (approx. 252 trading days)
        annualized_vol = volatility * (252 ** 0.5)
        
        # Base rating on combination of volatility and beta
        if annualized_vol < 0.15 and beta < 0.8:
            return "low"
        elif annualized_vol > 0.30 or beta > 1.5:
            return "high"
        else:
            return "medium"

    def get_market_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get market data for a symbol using FinancialDataMCP.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data (e.g., '1d', '1h', '5m')
            limit: Maximum number of data points to retrieve

        Returns:
            Dictionary with market data
        """
        # Generate a unique operation ID
        operation_id = f"get_market_data.{symbol}.{timeframe}.{int(time.time() * 1000)}"
        
        # Track timing
        start_time = time.time()
        
        self.logger.info("Getting market data", 
                       operation_id=operation_id,
                       symbol=symbol,
                       timeframe=timeframe,
                       limit=limit)
        
        try:
            # Check cache first (track Redis call)
            redis_start_time = time.time()
            cache_key = f"market_data:{symbol}:{timeframe}"
            
            # Use thread-safe tracking for the Redis call
            try:
                cached_data_result = self.redis_mcp.call_tool("get_json", {"key": cache_key})
                redis_duration_ms = (time.time() - redis_start_time) * 1000
                self._track_tool_call("redis", "get_json", redis_duration_ms, 
                                     not cached_data_result.get("error", False))
                
                cached_data = (cached_data_result.get("value") 
                              if cached_data_result and not cached_data_result.get("error") 
                              else None)
            except Exception as e:
                redis_duration_ms = (time.time() - redis_start_time) * 1000
                self._track_tool_call("redis", "get_json", redis_duration_ms, False)
                self.logger.error("Redis cache check failed", 
                                operation_id=operation_id,
                                error=str(e))
                cached_data = None

            # Check cache validity
            if cached_data:
                last_updated = cached_data.get("timestamp")
                if last_updated:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        # Cache expiration logic (e.g., 1 hour for intraday, 1 day for daily)
                        cache_expiry_hours = 1 if timeframe in ["1m", "5m", "15m", "30m", "1h"] else 24
                        if datetime.now() - last_updated_dt < timedelta(hours=cache_expiry_hours):
                            # Record cache hit
                            duration_ms = (time.time() - start_time) * 1000
                            self.logger.info("Returning cached market data", 
                                           operation_id=operation_id,
                                           symbol=symbol,
                                           duration_ms=duration_ms)
                            
                            # Track timing metrics
                            self._track_operation_timing("get_market_data", duration_ms)
                            self.logger.counter("cache_hit.market_data", 1)
                            
                            # Add metadata to result
                            cached_data["_metadata"] = {
                                "operation_id": operation_id,
                                "duration_ms": duration_ms,
                                "cache_hit": True,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            return cached_data
                    except ValueError:
                        self.logger.warning("Invalid timestamp in cached data", 
                                          operation_id=operation_id,
                                          timestamp=last_updated)

            # Cache miss - fetch from FinancialDataMCP
            self.logger.counter("cache_miss.market_data", 1)
            
            # Track the MCP call timing
            mcp_start_time = time.time()
            try:
                market_data_result = self.financial_data_mcp.call_tool(
                    "get_market_data",  # Or "get_bars"
                    {"symbols": [symbol], "timeframe": timeframe, "limit": limit}
                )
                mcp_duration_ms = (time.time() - mcp_start_time) * 1000
                success = not (market_data_result and market_data_result.get("error"))
                self._track_tool_call("financial_data", "get_market_data", mcp_duration_ms, success)
            except Exception as e:
                mcp_duration_ms = (time.time() - mcp_start_time) * 1000
                self._track_tool_call("financial_data", "get_market_data", mcp_duration_ms, False)
                self.logger.error("FinancialDataMCP call failed", 
                                operation_id=operation_id,
                                error=str(e))
                raise

            if market_data_result and not market_data_result.get("error"):
                 # Process and cache the data
                 price_data = market_data_result.get("data", {}).get(symbol, [])
                 
                 data_to_cache = {
                     "symbol": symbol,
                     "timeframe": timeframe,
                     "price_data": price_data,
                     "timestamp": datetime.now().isoformat(),
                 }
                 
                 # Cache the result in Redis (track the call)
                 redis_set_start = time.time()
                 try:
                     redis_result = self.redis_mcp.call_tool(
                         "set_json",
                         {"key": cache_key, "data": data_to_cache, "expiry": 3600}  # 1 hour expiration
                     )
                     redis_set_duration = (time.time() - redis_set_start) * 1000
                     self._track_tool_call("redis", "set_json", redis_set_duration,
                                           not redis_result.get("error", False))
                 except Exception as e:
                     redis_set_duration = (time.time() - redis_set_start) * 1000
                     self._track_tool_call("redis", "set_json", redis_set_duration, False)
                     self.logger.error("Failed to cache market data",
                                       operation_id=operation_id,
                                       error=str(e))
                 
                 # Calculate total duration and record metrics
                 total_duration_ms = (time.time() - start_time) * 1000
                 self._track_operation_timing("get_market_data", total_duration_ms)
                 
                 # Log successful completion
                 self.logger.info("Market data fetched successfully", 
                                operation_id=operation_id,
                                symbol=symbol, 
                                data_points=len(price_data),
                                duration_ms=total_duration_ms)
                 
                 # Prepare result with metadata
                 result = {
                     "symbol": symbol, 
                     "timeframe": timeframe, 
                     "price_data": price_data, 
                     "timestamp": datetime.now().isoformat(),
                     "_metadata": {
                         "operation_id": operation_id,
                         "duration_ms": total_duration_ms,
                         "cache_hit": False,
                         "data_points": len(price_data)
                     }
                 }
                 
                 return result
            else:
                # Record error metrics
                error_msg = market_data_result.get("error", "Unknown error") if market_data_result else "Failed to fetch market data"
                duration_ms = (time.time() - start_time) * 1000
                
                self.logger.error("Error fetching market data", 
                                operation_id=operation_id,
                                symbol=symbol,
                                error=error_msg,
                                duration_ms=duration_ms)
                
                self.logger.counter("error.get_market_data", 1)
                self._track_operation_timing("get_market_data_error", duration_ms)
                
                return {
                    "error": error_msg, 
                    "symbol": symbol,
                    "_metadata": {
                        "operation_id": operation_id,
                        "duration_ms": duration_ms,
                        "error": True
                    }
                }

        except Exception as e:
            # Handle and track exceptions
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error("Exception in get_market_data", 
                            operation_id=operation_id,
                            symbol=symbol,
                            error=str(e),
                            duration_ms=duration_ms)
            
            self.logger.counter("exception.get_market_data", 1)
            self._track_operation_timing("get_market_data_exception", duration_ms)
            
            with self._metrics_lock:
                self.execution_errors += 1
            
            return {
                "error": str(e), 
                "symbol": symbol,
                "_metadata": {
                    "operation_id": operation_id,
                    "duration_ms": duration_ms,
                    "exception": True
                }
            }

    def analyze_symbol(self, symbol: str, timeframe: str = "1d") -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for analysis

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get market data
            market_data = self.get_market_data(symbol, timeframe)
            if market_data.get("error"):
                return {"error": market_data.get("error"), "symbol": symbol}

            price_data = market_data.get("price_data", [])
            if not price_data:
                return {"error": "No price data available", "symbol": symbol}

            # Calculate technical indicators (using TimeSeriesMCP tool)
            self.mcp_tool_call_count += 1
            indicators_result = self.time_series_mcp.call_tool(
                "calculate_indicators",
                {"data": price_data, "indicators": self.default_indicators}
            )
            if indicators_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error calculating indicators for {symbol}: {indicators_result['error']}")
                 indicators_data = {}
            else:
                 indicators_data = indicators_result.get("indicators", {})


            # Detect patterns (using TimeSeriesMCP tool)
            self.mcp_tool_call_count += 1
            patterns_result = self.time_series_mcp.call_tool(
                "detect_patterns",
                {"data": price_data, "patterns": self.default_patterns}
            )
            if patterns_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error detecting patterns for {symbol}: {patterns_result['error']}")
                 patterns_data = []
            else:
                 patterns_data = patterns_result.get("patterns", [])


            # Detect support/resistance levels (using TimeSeriesMCP tool)
            self.mcp_tool_call_count += 1
            support_resistance_result = self.time_series_mcp.call_tool(
                "detect_support_resistance",
                {"data": price_data, "method": "peaks"}
            )
            if support_resistance_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error detecting support/resistance for {symbol}: {support_resistance_result['error']}")
                 support_resistance_data = {}
            else:
                 support_resistance_data = support_resistance_result.get("levels", {})


            # Generate signals based on indicators and patterns
            signals = []
            
            # Add basic technical signals
            if indicators_data.get("RSI", []) and len(indicators_data.get("RSI", [])) > 0:
                latest_rsi = indicators_data["RSI"][-1]
                if latest_rsi < 30:
                    signals.append({"type": "technical", "signal": "bullish", "source": "RSI", "value": latest_rsi})
                elif latest_rsi > 70:
                    signals.append({"type": "technical", "signal": "bearish", "source": "RSI", "value": latest_rsi})
            
            # Add pattern signals
            for pattern in patterns_data:
                pattern_name = pattern.get("pattern")
                if pattern_name in ["double_bottom", "inverse_head_and_shoulders", "ascending_triangle"]:
                    signals.append({"type": "pattern", "signal": "bullish", "source": pattern_name})
                elif pattern_name in ["double_top", "head_and_shoulders", "descending_triangle"]:
                    signals.append({"type": "pattern", "signal": "bearish", "source": pattern_name})
            
            # Combine results
            analysis_result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": indicators_data,
                "patterns": patterns_data,
                "support_resistance": support_resistance_data,
                "signals": signals, # Filled in the signals placeholder
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                "set_json",
                {
                    "key": f"{self.redis_keys['technical_indicators']}{symbol}",
                    "value": analysis_result,
                    "expiry": 86400,  # 1 day expiration
                }
            )
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                 "set",
                 {"key": f"{self.redis_keys['latest_analysis']}{symbol}", "value": analysis_result["timestamp"]}
            )


            # Publish analysis report to a Redis stream for other models (e.g., Decision Model)
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                 "xadd",
                 {
                      "stream": self.redis_keys["market_analysis_reports_stream"],
                      "data": analysis_result
                 }
            )
            self.logger.info(f"Published market analysis report for {symbol} to stream.")


            return analysis_result

        except Exception as e:
            self.logger.error(f"Error analyzing symbol {symbol}: {e}", exc_info=True)
            return {"error": str(e), "symbol": symbol}

    def scan_market(
        self, symbols: List[str], scan_type: str = "technical"
    ) -> Dict[str, Any]:
        """
        Scan the market for trading opportunities.

        Args:
            symbols: List of symbols to scan
            scan_type: Type of scan to perform

        Returns:
            Dictionary with scan results
        """
        start_time = time.time()
        self.logger.info(f"Starting market scan of {len(symbols)} symbols")
        self.scan_count += 1

        # Limit number of symbols to scan
        symbols = symbols[: self.max_scan_symbols]

        # Results containers
        opportunities = []
        errors = []

        # Scan each symbol
        for symbol in symbols:
            try:
                # Analyze the symbol
                analysis = self.analyze_symbol(symbol)

                if analysis.get("error"):
                    errors.append({"symbol": symbol, "error": analysis.get("error")})
                    continue

                # Check for trading opportunities based on scan type
                if scan_type == "technical":
                    opportunity = self._check_technical_opportunity(analysis)
                elif scan_type == "pattern":
                    opportunity = self._check_pattern_opportunity(analysis)
                elif scan_type == "breakout":
                    opportunity = self._check_breakout_opportunity(analysis)
                else:
                    # Default to technical
                    opportunity = self._check_technical_opportunity(analysis)

                if opportunity:
                    opportunities.append(opportunity)

            except Exception as e:
                self.logger.error(f"Error scanning symbol {symbol}: {e}", exc_info=True)
                errors.append({"symbol": symbol, "error": str(e)})

        # Sort opportunities by score (descending)
        opportunities.sort(key=lambda x: x.get("score", 0), reverse=True)

        # Prepare scan results
        scan_results = {
            "scan_type": scan_type,
            "symbols_scanned": len(symbols),
            "opportunities_found": len(opportunities),
            "opportunities": opportunities,
            "errors": errors,
            "processing_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
        }

        # Store in Redis
        self.mcp_tool_call_count += 1
        self.redis_mcp.call_tool(
            "set_json",
            {
                "key": self.redis_keys["market_scan_results"],
                "value": scan_results,
                "expiry": 86400,  # 1 day expiration
            }
        )
        self.logger.info("Stored market scan results in Redis.")


        # Send feedback to selection model (e.g., top opportunities)
        # This could be a separate stream or key depending on how Selection Model consumes it
        self.send_feedback_to_selection(
            {
                "market_scan": {
                    "scan_type": scan_type,
                    "opportunities": opportunities[:10],  # Send top 10 opportunities
                }
            }
        )
        self.logger.info("Sent feedback to selection model.")


        return scan_results

    def _check_technical_opportunity(
        self, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for trading opportunities based on technical indicators.
        """
        symbol = analysis.get("symbol")
        indicators = analysis.get("indicators", {})
        # signals = analysis.get("signals", []) # Signals should be derived or come from TimeSeriesMCP

        # Initialize score and signals
        score = 0
        bullish_signals = []
        bearish_signals = []

        # Check RSI (assuming TimeSeriesMCP returns standard structure)
        if "RSI" in indicators: # Use uppercase "RSI" based on TimeSeriesMCP test
            rsi_values = indicators["RSI"] # Assuming it's a list of values
            if rsi_values and len(rsi_values) > 0:
                # Find the last non-NaN value
                latest_rsi = None
                for val in reversed(rsi_values):
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        latest_rsi = val
                        break

                if latest_rsi is not None:
                    if latest_rsi < 30:
                        score += 1
                        bullish_signals.append(f"RSI oversold ({latest_rsi:.2f})")
                    elif latest_rsi > 70:
                        score -= 1
                        bearish_signals.append(f"RSI overbought ({latest_rsi:.2f})")


        # Check MACD (assuming TimeSeriesMCP returns standard structure)
        if "MACD" in indicators: # Use uppercase "MACD" based on TimeSeriesMCP test
            macd_data = indicators["MACD"] # Assuming it's a dict like {"macd_line": [...], "signal_line": [...]}
            macd_line = macd_data.get("macd_line", [])
            signal_line = macd_data.get("signal_line", [])

            if (
                macd_line
                and signal_line
                and len(macd_line) > 1
                and len(signal_line) > 1
            ):
                 # Find the last non-NaN values for macd and signal lines
                 latest_macd = None
                 for val in reversed(macd_line):
                     if val is not None and not (isinstance(val, float) and np.isnan(val)):
                         latest_macd = val
                         break

                 latest_signal = None
                 for val in reversed(signal_line):
                     if val is not None and not (isinstance(val, float) and np.isnan(val)):
                         latest_signal = val
                         break

                 prev_macd = None
                 for val in reversed(macd_line[:-1]):
                     if val is not None and not (isinstance(val, float) and np.isnan(val)):
                         prev_macd = val
                         break

                 prev_signal = None
                 for val in reversed(signal_line[:-1]):
                     if val is not None and not (isinstance(val, float) and np.isnan(val)):
                         prev_signal = val
                         break


                 if (prev_macd is not None and prev_signal is not None and
                     latest_macd is not None and latest_signal is not None):

                     if prev_macd < prev_signal and latest_macd > latest_signal:
                         score += 2
                         bullish_signals.append("MACD bullish crossover")
                     elif prev_macd > prev_signal and latest_macd < latest_signal:
                         score -= 2
                         bearish_signals.append("MACD bearish crossover")


        # Check Bollinger Bands (assuming TimeSeriesMCP returns standard structure)
        if "BBANDS" in indicators: # Use uppercase "BBANDS" based on TimeSeriesMCP test
            bbands_data = indicators["BBANDS"] # Assuming it's a dict like {"upper_band": [...], "lower_band": [...]}
            upper_band = bbands_data.get("upper_band", [])
            lower_band = bbands_data.get("lower_band", [])
            close_prices = [bar.get("close") for bar in analysis.get("price_data", [])]

            if upper_band and lower_band and close_prices and len(close_prices) > 0:
                latest_close = close_prices[-1]
                latest_upper = upper_band[-1] if upper_band else None
                latest_lower = lower_band[-1] if lower_band else None

                if latest_close is not None:
                    if latest_lower is not None and latest_close <= latest_lower:
                        score += 1
                        bullish_signals.append("Price at lower Bollinger Band")
                    elif latest_upper is not None and latest_close >= latest_upper:
                        score -= 1
                        bearish_signals.append("Price at upper Bollinger Band")

        # Check for pattern signals (assuming TimeSeriesMCP returns standard structure)
        for pattern in analysis.get("patterns", []): # Assuming patterns are a list of dicts
            pattern_name = pattern.get("pattern") # Assuming 'pattern' key exists
            if pattern_name in ["double_bottom", "inverse_head_and_shoulders", "ascending_triangle"]: # Add ascending triangle
                score += 2
                bullish_signals.append(f"{pattern_name} pattern detected")
            elif pattern_name in ["double_top", "head_and_shoulders", "descending_triangle"]: # Add descending triangle
                score -= 2
                bearish_signals.append(f"{pattern_name} pattern detected")

        # Determine opportunity type
        opportunity_type = None
        if score >= 3:
            opportunity_type = "bullish"
        elif score <= -3:
            opportunity_type = "bearish"

        # Return opportunity if found
        if opportunity_type:
            return {
                "symbol": symbol,
                "type": opportunity_type,
                "score": abs(score),
                "signals": bullish_signals
                if opportunity_type == "bullish"
                else bearish_signals,
                "timestamp": datetime.now().isoformat(),
            }

        return None

    def _check_pattern_opportunity(
        self, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for trading opportunities based on chart patterns.
        """
        # This is a simplified implementation
        symbol = analysis.get("symbol")
        patterns = analysis.get("patterns", []) # Assuming patterns are a list of dicts

        if not patterns:
            return None

        # Find the highest confidence pattern
        # Assuming patterns have a 'confidence' key
        best_pattern = max(patterns, key=lambda p: p.get("confidence", 0))
        pattern_name = best_pattern.get("pattern")
        confidence = best_pattern.get("confidence", 0)

        # Only consider high-confidence patterns
        if confidence < 0.7:
            return None

        # Determine if bullish or bearish
        bullish_patterns = [
            "double_bottom",
            "inverse_head_and_shoulders",
            "ascending_triangle",
        ]
        bearish_patterns = ["double_top", "head_and_shoulders", "descending_triangle"]

        if pattern_name in bullish_patterns:
            opportunity_type = "bullish"
        elif pattern_name in bearish_patterns:
            opportunity_type = "bearish"
        else:
            return None

        return {
            "symbol": symbol,
            "type": opportunity_type,
            "score": confidence * 10,  # Scale to 0-10
            "signals": [f"{pattern_name} pattern with {confidence:.2f} confidence"],
            "timestamp": datetime.now().isoformat(),
        }

    def _check_breakout_opportunity(
        self, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for breakout trading opportunities.
        """
        symbol = analysis.get("symbol")
        support_resistance = analysis.get("support_resistance", {}) # Assuming this is a dict like {"support": {"levels": [...]}, "resistance": {"levels": [...]}}
        price_data = analysis.get("price_data", [])

        if not support_resistance or not price_data or len(price_data) < 2:
            return None

        # Get latest price
        latest_price = price_data[-1].get("close")
        prev_price = price_data[-2].get("close")

        if latest_price is None or prev_price is None:
            return None

        # Get resistance levels
        resistance_levels = support_resistance.get("resistance", {}).get("levels", [])
        support_levels = support_resistance.get("support", {}).get("levels", [])

        # Check for breakouts
        for level in resistance_levels:
            if level is not None:
                # Resistance breakout (bullish)
                if prev_price < level and latest_price > level:
                    return {
                        "symbol": symbol,
                        "type": "bullish",
                        "score": 8, # High score for breakout
                        "signals": [f"Resistance breakout at {level:.2f}"],
                        "timestamp": datetime.now().isoformat(),
                    }

        for level in support_levels:
            if level is not None:
                # Support breakdown (bearish)
                if prev_price > level and latest_price < level:
                    return {
                        "symbol": symbol,
                        "type": "bearish",
                        "score": 8, # High score for breakdown
                        "signals": [f"Support breakdown at {level:.2f}"],
                        "timestamp": datetime.now().isoformat(),
                    }

        return None

    def get_scan_results(self) -> Dict[str, Any]:
        """
        Get the latest market scan results from Redis.
        """
        try:
            # Get scan results from Redis using the RedisMCP instance
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                "get_json",
                {"key": self.redis_keys["market_scan_results"]}
            )

            if result and not result.get("error"):
                self.logger.info("Retrieved market scan results")
                return result.get("value", {}) if result else {} # Return the value part
            else:
                self.logger.warning("No market scan results found")
                return {
                    "status": "warning",
                    "message": "No market scan results available",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Error retrieving market scan results: {e}", exc_info=True)
            self.execution_errors += 1
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get data from the Selection Model from Redis.
        """
        try:
            # Get selection data from Redis using the RedisMCP instance
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                "get_json",
                {"key": self.redis_keys["selection_data"]}
            )

            if result and not result.get("error"):
                self.logger.info("Retrieved selection model data")
                return result.get("value", {}) if result else {} # Return the value part
            else:
                self.logger.warning("No selection model data found")
                return {
                    "status": "warning",
                    "message": "No selection model data available",
                    "timestamp": datetime.now().isoformat()
                }

        except Exception as e:
            self.logger.error(f"Error retrieving selection model data: {e}", exc_info=True)
            self.execution_errors += 1
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def send_feedback_to_selection(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Send technical analysis feedback to the Selection Model via Redis.
        """
        try:
            # Add timestamp to the data
            if "timestamp" not in analysis_data:
                analysis_data["timestamp"] = datetime.now().isoformat()

            # Store feedback in Redis using the RedisMCP instance
            # Assuming Selection Model listens to a specific key or stream for feedback
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                 "xadd", # Using stream for feedback
                 {
                      "stream": self.redis_keys["selection_feedback"],
                      "data": analysis_data
                 }
            )

            if result and not result.get("error"):
                self.logger.info("Sent feedback to selection model via stream.")
                return True
            else:
                self.logger.error(f"Failed to send feedback to selection model: {result.get('error') if result else 'Unknown error'}")
                self.mcp_tool_error_count += 1
                return False

        except Exception as e:
            self.logger.error(f"Error sending feedback to selection model: {e}", exc_info=True)
            self.execution_errors += 1
            return False

    def get_health_check(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check on the MarketAnalysisModel.
        
        Returns:
            Dictionary with health status information and scores
        """
        self.logger.info("Running health check")
        start_time = time.time()
        
        # Initialize health data
        health_data = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "overall_health": 0.0,  # Will be calculated
            "components": {},
            "metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        # 1. Check FinancialDataMCP health
        financial_data_health = self._check_financial_data_mcp_health()
        health_data["components"]["financial_data_mcp"] = financial_data_health
        
        # 2. Check TimeSeriesMCP health
        time_series_health = self._check_time_series_mcp_health()
        health_data["components"]["time_series_mcp"] = time_series_health
        
        # 3. Check RedisMCP health
        redis_health = self._check_redis_mcp_health()
        health_data["components"]["redis_mcp"] = redis_health
        
        # 4. Check AutoGen health
        autogen_health = self._check_autogen_health()
        health_data["components"]["autogen"] = autogen_health
        
        # 5. Add performance metrics
        health_data["metrics"] = self._get_performance_metrics()
        
        # 6. Add system metrics if available
        try:
            system_metrics = self._get_system_metrics()
            health_data["system"] = system_metrics
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            health_data["issues"].append(f"Failed to collect system metrics: {e}")
        
        # Calculate overall health score (weighted average of component scores)
        component_weights = {
            "financial_data_mcp": 0.3,
            "time_series_mcp": 0.3,
            "redis_mcp": 0.2,
            "autogen": 0.2
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, weight in component_weights.items():
            if component in health_data["components"]:
                component_score = health_data["components"][component].get("score", 0)
                weighted_score += component_score * weight
                total_weight += weight
        
        health_data["overall_health"] = round(weighted_score / total_weight if total_weight > 0 else 0, 2)
        
        # Add health status classification
        if health_data["overall_health"] >= 80:
            health_data["status"] = "healthy"
        elif health_data["overall_health"] >= 60:
            health_data["status"] = "degraded"
        else:
            health_data["status"] = "unhealthy"
        
        # Generate recommendations based on issues
        if health_data["issues"]:
            health_data["recommendations"] = self._generate_recommendations(health_data["issues"])
        
        # Record health check duration
        duration_ms = (time.time() - start_time) * 1000
        health_data["duration_ms"] = duration_ms
        self.logger.timing("health_check_duration_ms", duration_ms)
        
        # Update last health check timestamp
        self.last_health_check = datetime.now()
        
        return health_data
    
    def _check_financial_data_mcp_health(self) -> Dict[str, Any]:
        """Check the health of the FinancialDataMCP component."""
        health = {
            "status": "unknown",
            "score": 0,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # 1. Test basic connectivity with a simple API call
            start_time = time.time()
            tool_to_call = "get_health" # Ensure we are calling the correct health tool
            test_result = self.financial_data_mcp.call_tool(
                tool_to_call,
                {}
            )
            response_time = (time.time() - start_time) * 1000
            
            # Track the response time
            health["metrics"]["response_time_ms"] = response_time
            
            if test_result and not test_result.get("error"):
                # Extract health information if available
                mcp_health = test_result.get("health", {})
                health["status"] = mcp_health.get("status", "operational")
                health["details"] = mcp_health
                
                # Calculate score based on response time and status
                if health["status"] == "operational":
                    base_score = 90
                elif health["status"] == "degraded":
                    base_score = 60
                else:
                    base_score = 30
                
                # Adjust score based on response time
                if response_time < 100:  # Less than 100ms is excellent
                    time_score = 10
                elif response_time < 500:  # Less than 500ms is good
                    time_score = 5
                elif response_time < 2000:  # Less than 2s is acceptable
                    time_score = 0
                else:  # More than 2s is poor
                    time_score = -10
                    health["issues"].append(f"High response time: {response_time:.2f}ms")
                
                health["score"] = min(100, max(0, base_score + time_score))
            else:
                health["status"] = "error"
                health["issues"].append(f"Health check failed: {test_result.get('error', 'Unknown error')}" if test_result else "No response")
                health["score"] = 0
        except Exception as e:
            health["status"] = "error"
            health["issues"].append(f"Exception during health check: {str(e)}")
            health["score"] = 0
        
        return health
    
    def _check_time_series_mcp_health(self) -> Dict[str, Any]:
        """Check the health of the TimeSeriesMCP component."""
        health = {
            "status": "unknown",
            "score": 0,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Similar to financial data MCP check
            start_time = time.time()
            test_result = self.time_series_mcp.call_tool(
                "get_health",  # Assuming there's a health endpoint
                {}
            )
            response_time = (time.time() - start_time) * 1000
            
            health["metrics"]["response_time_ms"] = response_time
            
            if test_result and not test_result.get("error"):
                mcp_health = test_result.get("health", {})
                health["status"] = mcp_health.get("status", "operational")
                health["details"] = mcp_health
                
                if health["status"] == "operational":
                    base_score = 90
                elif health["status"] == "degraded":
                    base_score = 60
                else:
                    base_score = 30
                
                if response_time < 100:
                    time_score = 10
                elif response_time < 500:
                    time_score = 5
                elif response_time < 2000:
                    time_score = 0
                else:
                    time_score = -10
                    health["issues"].append(f"High response time: {response_time:.2f}ms")
                
                health["score"] = min(100, max(0, base_score + time_score))
            else:
                health["status"] = "error"
                health["issues"].append(f"Health check failed: {test_result.get('error', 'Unknown error')}" if test_result else "No response")
                health["score"] = 0
        except Exception as e:
            health["status"] = "error"
            health["issues"].append(f"Exception during health check: {str(e)}")
            health["score"] = 0
        
        return health
    
    def _check_redis_mcp_health(self) -> Dict[str, Any]:
        """Check the health of the RedisMCP component."""
        health = {
            "status": "unknown",
            "score": 0,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Test Redis connectivity with a ping
            start_time = time.time()
            test_result = self.redis_mcp.call_tool("ping", {})
            response_time = (time.time() - start_time) * 1000
            
            health["metrics"]["response_time_ms"] = response_time
            
            if test_result and not test_result.get("error"):
                # Check if we got a PONG back
                if test_result.get("result") == "PONG":
                    health["status"] = "operational"
                    
                    # Calculate score based on response time
                    if response_time < 10:  # Redis should be fast (<10ms)
                        time_score = 10
                    elif response_time < 50:  # Still good (<50ms)
                        time_score = 5
                    elif response_time < 200:  # Acceptable (<200ms)
                        time_score = 0
                    else:  # Slow Redis (>200ms)
                        time_score = -10
                        health["issues"].append(f"Slow Redis response: {response_time:.2f}ms")
                    
                    health["score"] = min(100, max(0, 90 + time_score))
                else:
                    health["status"] = "error"
                    health["issues"].append("Redis did not respond with PONG")
                    health["score"] = 50
            else:
                health["status"] = "error"
                health["issues"].append(f"Redis ping failed: {test_result.get('error', 'Unknown error')}" if test_result else "No response")
                health["score"] = 0
        except Exception as e:
            health["status"] = "error"
            health["issues"].append(f"Exception during Redis health check: {str(e)}")
            health["score"] = 0
        
        return health
    
    def _check_autogen_health(self) -> Dict[str, Any]:
        """Check the health of the AutoGen components."""
        health = {
            "status": "unknown",
            "score": 0,
            "issues": [],
            "metrics": {}
        }
        
        # For AutoGen, we can't easily do a direct "health check" call,
        # so we check if the agents are initialized and their configurations
        if self.agents and "market_analysis_assistant" in self.agents and "user_proxy" in self.agents:
            health["status"] = "operational"
            health["score"] = 90
            health["metrics"]["agent_count"] = len(self.agents)
        else:
            health["status"] = "error"
            health["issues"].append("AutoGen agents not properly initialized")
            health["score"] = 0
        
        return health
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for health reporting."""
        with self._metrics_lock:
            metrics = {
                "mcp_call_count": self.mcp_tool_call_count,
                "mcp_error_count": self.mcp_tool_error_count,
                "error_rate": (self.mcp_tool_error_count / self.mcp_tool_call_count) * 100 if self.mcp_tool_call_count > 0 else 0,
                "analysis_count": self.analysis_count,
                "scan_count": self.scan_count,
                "execution_errors": self.execution_errors
            }
            
            # Add operation timing metrics (averages)
            timing_metrics = {}
            for operation, times in self.operation_timing.items():
                if operation != "tool_calls" and times:
                    timing_metrics[f"{operation}_avg_ms"] = sum(times) / len(times)
                    timing_metrics[f"{operation}_min_ms"] = min(times)
                    timing_metrics[f"{operation}_max_ms"] = max(times)
            
            # Add tool call timing metrics (summarized)
            tool_metrics = {}
            for tool_key, times in self.operation_timing.get("tool_calls", {}).items():
                if times:
                    tool_metrics[f"{tool_key}_avg_ms"] = sum(times) / len(times)
            
            metrics["timing"] = timing_metrics
            metrics["tool_timing"] = tool_metrics
            
            return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics if available."""
        metrics = {}
        
        if hasattr(self, "metrics_collector") and self.metrics_collector:
            # Get metrics from SystemMetricsCollector if possible
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent()
                metrics["cpu_percent"] = cpu_percent
                
                # Get memory usage
                memory = psutil.virtual_memory()
                metrics["memory_percent"] = memory.percent
                metrics["memory_used_mb"] = memory.used / (1024 * 1024)
                
                # Get disk usage
                disk = psutil.disk_usage('/')
                metrics["disk_percent"] = disk.percent
                
                # Check if resources are constrained
                if cpu_percent > 90:
                    metrics["cpu_constrained"] = True
                if memory.percent > 90:
                    metrics["memory_constrained"] = True
                if disk.percent > 90:
                    metrics["disk_constrained"] = True
            except Exception as e:
                metrics["error"] = f"Failed to collect system metrics: {str(e)}"
        
        return metrics
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """Generate recommendations based on detected issues."""
        recommendations = []
        
        for issue in issues:
            if "response time" in issue.lower() or "slow" in issue.lower():
                recommendations.append("Check network connectivity and latency between components")
            
            if "redis" in issue.lower():
                recommendations.append("Verify Redis server is running and properly configured")
            
            if "error rate" in issue.lower():
                recommendations.append("Investigate MCP tool errors to identify patterns")
            
            if "memory" in issue.lower() and "constrained" in issue.lower():
                recommendations.append("Consider increasing available memory or optimizing memory usage")
            
            if "cpu" in issue.lower() and "constrained" in issue.lower():
                recommendations.append("Check for CPU-intensive processes or consider scaling up CPU resources")
        
        # Add general recommendations if we have issues but no specific recommendations
        if issues and not recommendations:
            recommendations.append("Run diagnostics on all MCP components")
            recommendations.append("Check system resources and network connectivity")
        
        return recommendations
    
    def run_market_analysis(self, query: str) -> Dict[str, Any]:
        """
        Run market analysis using AutoGen agents.

        Args:
            query: Query or instruction for market analysis

        Returns:
            Results of the market analysis
        """
        # Generate operation ID
        operation_id = f"market_analysis.{int(time.time() * 1000)}"
        
        # Track timing
        start_time = time.time()
        
        self.logger.info("Running market analysis", 
                       operation_id=operation_id,
                       query=query)

        market_analysis_assistant = self.agents.get("market_analysis_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not market_analysis_assistant or not user_proxy:
            self.logger.error("AutoGen agents not initialized", operation_id=operation_id)
            return {
                "error": "AutoGen agents not initialized",
                "_metadata": {
                    "operation_id": operation_id,
                    "duration_ms": (time.time() - start_time) * 1000,
                    "error": True
                }
            }

        try:
            # Initiate chat with the market analysis assistant
            self.logger.info("Initiating AutoGen chat", operation_id=operation_id)
            user_proxy.initiate_chat(market_analysis_assistant, message=query)

            # Get the last message from the assistant
            last_message = user_proxy.last_message(market_analysis_assistant)
            content = last_message.get("content", "")

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            self._track_operation_timing("run_market_analysis", duration_ms)
            
            # Log completion
            self.logger.info("Market analysis completed", 
                           operation_id=operation_id,
                           duration_ms=duration_ms,
                           response_size=len(content))

            # Extract structured data if possible
            try:
                # Find JSON blocks in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    result_str = content[json_start:json_end]
                    result = json.loads(result_str)
                    
                    # Add metadata
                    result["_metadata"] = {
                        "operation_id": operation_id,
                        "duration_ms": duration_ms,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    return result
            except json.JSONDecodeError:
                # Continue to return raw content if JSON parsing fails
                self.logger.warning("Failed to parse JSON from response", 
                                  operation_id=operation_id)

            # Return raw content with metadata
            return {
                "analysis": content,
                "_metadata": {
                    "operation_id": operation_id,
                    "duration_ms": duration_ms,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            # Handle exceptions
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error("Error during AutoGen analysis", 
                            operation_id=operation_id,
                            error=str(e),
                            duration_ms=duration_ms)
            
            with self._metrics_lock:
                self.execution_errors += 1
            
            return {
                "error": str(e),
                "_metadata": {
                    "operation_id": operation_id,
                    "duration_ms": duration_ms,
                    "exception": True,
                    "timestamp": datetime.now().isoformat()
                }
            }

    def generate_chart(self, symbol: str, timeframe: str = "1d", 
                   include_volume: bool = True, include_indicators: bool = True,
                   indicators: List[str] = None) -> Dict[str, Any]:
        """
        Generate a chart for a symbol using StockChartGenerator.
        
        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data (1d, 1h, etc.)
            include_volume: Whether to include volume subplot
            include_indicators: Whether to include technical indicators
            indicators: List of specific indicators to include (defaults to model default_indicators)
            
        Returns:
            Dictionary with chart information including file path
        """
        # Generate operation ID
        operation_id = f"generate_chart.{symbol}.{timeframe}.{int(time.time() * 1000)}"
        
        # Track timing
        start_time = time.time()
        
        self.logger.info("Generating chart",
                        operation_id=operation_id,
                        symbol=symbol,
                        timeframe=timeframe,
                        include_indicators=include_indicators)
        
        try:
            # Get market data first
            market_data_result = self.get_market_data(symbol, timeframe)
            if market_data_result.get("error"):
                self.logger.error("Failed to get market data for chart", 
                                operation_id=operation_id,
                                error=market_data_result.get("error"))
                return {
                    "error": f"Failed to get market data: {market_data_result.get('error')}",
                    "_metadata": {
                        "operation_id": operation_id,
                        "duration_ms": (time.time() - start_time) * 1000,
                        "error": True
                    }
                }
            
            # Convert to pandas DataFrame for chart generation
            price_data = market_data_result.get("price_data", [])
            if not price_data:
                self.logger.error("No price data available for chart", 
                                operation_id=operation_id)
                return {
                    "error": "No price data available",
                    "_metadata": {
                        "operation_id": operation_id,
                        "duration_ms": (time.time() - start_time) * 1000,
                        "error": True
                    }
                }
            
            # Prepare data for chart generator
            import pandas as pd
            df = pd.DataFrame(price_data)
            
            # Make sure we have required columns
            required_columns = ["timestamp", "open", "high", "low", "close"]
            if not all(col in df.columns for col in required_columns):
                # Try to map columns if they have different names
                column_mapping = {}
                for req_col in required_columns:
                    for col in df.columns:
                        if req_col.lower() in col.lower():
                            column_mapping[col] = req_col
                            break
                
                # Apply mapping if possible
                if len(column_mapping) == len(required_columns):
                    df = df.rename(columns=column_mapping)
                else:
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    self.logger.error("Missing required columns for chart", 
                                    operation_id=operation_id,
                                    missing_columns=missing_cols)
                    return {
                        "error": f"Missing required columns: {', '.join(missing_cols)}",
                        "_metadata": {
                            "operation_id": operation_id,
                            "duration_ms": (time.time() - start_time) * 1000,
                            "error": True
                        }
                    }
            
            # Calculate indicators if requested
            if include_indicators:
                indicators_to_use = indicators or self.default_indicators
                
                # Calculate indicators with TimeSeriesMCP
                indicator_start = time.time()
                indicator_result = self.time_series_mcp.call_tool(
                    "calculate_indicators",
                    {"data": price_data, "indicators": indicators_to_use}
                )
                indicator_duration = (time.time() - indicator_start) * 1000
                self._track_tool_call("time_series", "calculate_indicators", indicator_duration, 
                                    not indicator_result.get("error", False))
                
                if indicator_result and not indicator_result.get("error"):
                    # Add indicators to the DataFrame
                    indicators_data = indicator_result.get("indicators", {})
                    for indicator_name, values in indicators_data.items():
                        if isinstance(values, list):
                            df[indicator_name] = values
                        elif isinstance(values, dict):
                            for key, vals in values.items():
                                df[f"{indicator_name}_{key}"] = vals
                
                self.logger.info("Indicators calculated for chart", 
                               operation_id=operation_id,
                               indicator_count=len(indicators_to_use),
                               duration_ms=indicator_duration)
            
            # Generate the chart
            chart_start = time.time()
            try:
                output_file = None  # Let the generator create a default filename
                
                # Use the StockChartGenerator to create the chart
                chart_file = self.chart_generator.create_candlestick_chart(
                    df, symbol, timeframe, 
                    include_volume=include_volume,
                    include_indicators=include_indicators,
                    output_file=output_file
                )
                
                chart_duration = (time.time() - chart_start) * 1000
                self.logger.timing("chart_generation_time_ms", chart_duration)
                
                # Log success
                total_duration = (time.time() - start_time) * 1000
                self.logger.info("Chart generated successfully", 
                               operation_id=operation_id,
                               symbol=symbol,
                               chart_file=chart_file,
                               duration_ms=total_duration)
                
                # Return chart information
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "chart_file": chart_file,
                    "data_points": len(price_data),
                    "indicators": indicators_to_use if include_indicators else [],
                    "include_volume": include_volume,
                    "_metadata": {
                        "operation_id": operation_id,
                        "duration_ms": total_duration,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            except Exception as e:
                chart_duration = (time.time() - chart_start) * 1000
                self.logger.error("Failed to generate chart", 
                                operation_id=operation_id,
                                error=str(e),
                                duration_ms=chart_duration)
                raise
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error("Exception in generate_chart", 
                            operation_id=operation_id,
                            symbol=symbol,
                            error=str(e),
                            duration_ms=duration_ms)
            
            with self._metrics_lock:
                self.execution_errors += 1
            
            return {
                "error": str(e),
                "symbol": symbol,
                "_metadata": {
                    "operation_id": operation_id,
                    "duration_ms": duration_ms,
                    "exception": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def generate_comparison_chart(self, symbols: List[str], timeframe: str = "1d", 
                                normalize: bool = True) -> Dict[str, Any]:
        """
        Generate a comparison chart for multiple symbols.
        
        Args:
            symbols: List of stock symbols to compare
            timeframe: Timeframe for data
            normalize: Whether to normalize prices to percentage change (recommended for comparison)
            
        Returns:
            Dictionary with chart information including file path
        """
        # Generate operation ID
        operation_id = f"comparison_chart.{'-'.join(symbols[:3])}.{timeframe}.{int(time.time() * 1000)}"
        
        # Track timing
        start_time = time.time()
        
        self.logger.info("Generating comparison chart", 
                       operation_id=operation_id,
                       symbols=symbols,
                       timeframe=timeframe,
                       normalize=normalize)
        
        try:
            # Limit the number of symbols for performance
            if len(symbols) > 10:
                self.logger.warning("Too many symbols for comparison chart, limiting to 10", 
                                  operation_id=operation_id,
                                  requested=len(symbols))
                symbols = symbols[:10]
            
            # Generate the comparison chart
            chart_start = time.time()
            try:
                # Use the StockChartGenerator to create the comparison chart
                chart_file = self.chart_generator.create_multi_stock_chart(
                    symbols, timeframe=timeframe, normalize=normalize
                )
                
                chart_duration = (time.time() - chart_start) * 1000
                self.logger.timing("comparison_chart_generation_time_ms", chart_duration)
                
                # Log success
                total_duration = (time.time() - start_time) * 1000
                self.logger.info("Comparison chart generated successfully", 
                               operation_id=operation_id,
                               symbols=symbols,
                               chart_file=chart_file,
                               duration_ms=total_duration)
                
                # Return chart information
                return {
                    "symbols": symbols,
                    "timeframe": timeframe,
                    "chart_file": chart_file,
                    "normalized": normalize,
                    "_metadata": {
                        "operation_id": operation_id,
                        "duration_ms": total_duration,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            except Exception as e:
                chart_duration = (time.time() - chart_start) * 1000
                self.logger.error("Failed to generate comparison chart", 
                                operation_id=operation_id,
                                error=str(e),
                                duration_ms=chart_duration)
                raise
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error("Exception in generate_comparison_chart", 
                            operation_id=operation_id,
                            symbols=symbols,
                            error=str(e),
                            duration_ms=duration_ms)
            
            with self._metrics_lock:
                self.execution_errors += 1
            
            return {
                "error": str(e),
                "symbols": symbols,
                "_metadata": {
                    "operation_id": operation_id,
                    "duration_ms": duration_ms,
                    "exception": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def __del__(self):
        """Clean up resources when the object is being destroyed."""
        try:
            # Stop the monitoring components
            if hasattr(self, "metrics_collector") and self.metrics_collector:
                self.metrics_collector.stop()
            
            # Stop health check thread if running
            if hasattr(self, "health_check_running") and self.health_check_running:
                self.health_check_running = False
                if hasattr(self, "health_check_thread") and self.health_check_thread:
                    self.health_check_thread.join(timeout=1.0)
            
            # Log the shutdown
            if hasattr(self, "logger"):
                self.logger.info("MarketAnalysisModel shutdown complete")
        except Exception as e:
            # Just in case we get an exception during cleanup
            if hasattr(self, "logger"):
                try:
                    self.logger.error(f"Error during cleanup: {e}")
                except:
                    print(f"Error during cleanup: {e}")
            else:
                print(f"Error during cleanup: {e}")

# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
