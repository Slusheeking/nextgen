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
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math # Import math for floor function

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
    # FunctionCall is not available in autogen 0.9.0
)

# MCP tools
from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.analysis_mcp.peak_detection_mcp import PeakDetectionMCP
from mcp_tools.analysis_mcp.slippage_analysis_mcp import SlippageAnalysisMCP
from mcp_tools.analysis_mcp.drift_detection_mcp import DriftDetectionMCP

# Forward declarations for type hinting


class TradeExecutor:
    pass


class TradeMonitor:
    pass


class TradePositionManager:
    pass


class TradeAnalytics:
    pass


class TradeModel:
    pass  # Add self-reference for type hinting


class TradeModel:
    """
    Trade Model for executing and monitoring trading decisions.

    This model receives trade decisions from the Decision Model,
    executes orders via Alpaca, and monitors active positions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Trade Model.

        Args:
            config: Optional configuration dictionary with trading parameters
        """
        init_start_time = time.time()
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-trade-model")

        # Initialize StockChartGenerator
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

        # Initialize MCP clients
        self.alpaca_mcp = AlpacaMCP(self.config.get("alpaca_config"))
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.polygon_mcp = PolygonRestMCP(self.config.get("polygon_config"))

        # Initialize analysis MCP tools
        self.peak_detection_mcp = PeakDetectionMCP(
            self.config.get("peak_detection_config")
        )
        self.slippage_analysis_mcp = SlippageAnalysisMCP(
            self.config.get("slippage_analysis_config")
        )
        self.drift_detection_mcp = DriftDetectionMCP(
            self.config.get("drift_detection_config")
        )

        # Initialize model components
        self.executor = TradeExecutor(self)
        self.monitor = TradeMonitor(self)
        self.position_manager = TradePositionManager(self)
        self.analytics = TradeAnalytics(self)

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions(self.agents["user_proxy"])

        self.logger.info("Trade Model initialized")

        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("trade_model.initialization_time_ms", init_duration)


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
                result = self.executor.execute_market_order(symbol, quantity, side)
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
                result = self.executor.execute_limit_order(
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
                result = self.monitor.start_position_monitoring(symbol, order_id)
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
                result = self.monitor.check_exit_conditions(symbol)
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
                result = self.position_manager.get_positions()
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_positions", "status": "success" if result is not None else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_positions"})
                if result is None:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in get_positions: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_positions", "status": "failed"})
                return {"error": str(e)}


        @register_function(
            name="get_position",
            description="Get position for a specific symbol",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def get_position(symbol: str) -> Optional[Dict[str, Any]]:
            start_time = time.time()
            try:
                result = self.position_manager.get_position(symbol)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_position", "status": "success" if result is not None else "not_found"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_position"})
                if result is None:
                    self.execution_errors += 1
                    self.logger.counter("trade_model.execution_errors")
                return result
            except Exception as e:
                self.logger.error(f"Error in get_position: {e}")
                self.execution_errors += 1
                self.logger.counter("trade_model.execution_errors")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_position", "status": "failed"})
                return {"error": str(e)}


        @register_function(
            name="get_portfolio_constraints",
            description="Get current portfolio constraints",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def get_portfolio_constraints() -> Dict[str, Any]:
            start_time = time.time()
            try:
                result = self.position_manager.get_portfolio_constraints()
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.function_call_duration_ms", duration, tags={"function": "get_portfolio_constraints", "status": "success" if result is not None else "failed"})
                self.logger.counter("trade_model.function_call_count", tags={"function": "get_portfolio_constraints"})
                if result is None:
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
                result = self.analytics.get_trade_history(symbol, limit)
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
                return {"error": str(e)}


        @register_function(
            name="calculate_execution_quality",
            description="Calculate execution quality metrics",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def calculate_execution_quality(order_id: str) -> Dict[str, Any]:
            start_time = time.time()
            try:
                result = self.analytics.calculate_execution_quality(order_id)
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

        #
        # Register all functions with the user proxy agent and trade assistant
        # agent
        trade_assistant = self.agents["trade_assistant"]

        # Register execution functions
        user_proxy.register_function(execute_market_order)
        user_proxy.register_function(execute_limit_order)

        # Register monitoring functions
        user_proxy.register_function(start_position_monitoring)
        user_proxy.register_function(check_exit_conditions)

        # Register position management functions
        user_proxy.register_function(get_positions)
        user_proxy.register_function(get_position)
        user_proxy.register_function(get_portfolio_constraints)

        # Register analytics functions
        user_proxy.register_function(get_trade_history)
        user_proxy.register_function(calculate_execution_quality)


    def _register_mcp_tool_access(self, user_proxy: UserProxyAgent):
        """
        Register MCP tool access functions with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register functions with
        """

        # Define MCP tool access functions
        @register_function(
            name="use_alpaca_tool",
            description="Use a tool provided by the Alpaca MCP server",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_alpaca_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            start_time = time.time()
            try:
                result = self.alpaca_mcp.call_tool(tool_name, arguments)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "alpaca", "tool": tool_name, "status": "success"})
                self.mcp_tool_call_count += 1
                self.logger.counter("trade_model.mcp_tool_call_count")
                return result
            except Exception as e:
                self.logger.error(f"Error calling Alpaca tool {tool_name}: {e}", tool=tool_name, error=str(e))
                self.mcp_tool_error_count += 1
                self.logger.counter("trade_model.mcp_tool_error_count")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "alpaca", "tool": tool_name, "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            start_time = time.time()
            try:
                result = self.redis_mcp.call_tool(tool_name, arguments)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "redis", "tool": tool_name, "status": "success"})
                self.mcp_tool_call_count += 1
                self.logger.counter("trade_model.mcp_tool_call_count")
                return result
            except Exception as e:
                self.logger.error(f"Error calling Redis tool {tool_name}: {e}", tool=tool_name, error=str(e))
                self.mcp_tool_error_count += 1
                self.logger.counter("trade_model.mcp_tool_error_count")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "redis", "tool": tool_name, "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="use_peak_detection_tool",
            description="Use a tool provided by the Peak Detection MCP server",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_peak_detection_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            start_time = time.time()
            try:
                result = self.peak_detection_mcp.call_tool(tool_name, arguments)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "peak_detection", "tool": tool_name, "status": "success"})
                self.mcp_tool_call_count += 1
                self.logger.counter("trade_model.mcp_tool_call_count")
                return result
            except Exception as e:
                self.logger.error(f"Error calling Peak Detection tool {tool_name}: {e}", tool=tool_name, error=str(e))
                self.mcp_tool_error_count += 1
                self.logger.counter("trade_model.mcp_tool_error_count")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "peak_detection", "tool": tool_name, "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="use_slippage_analysis_tool",
            description="Use a tool provided by the Slippage Analysis MCP server",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_slippage_analysis_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            start_time = time.time()
            try:
                result = self.slippage_analysis_mcp.call_tool(tool_name, arguments)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "slippage_analysis", "tool": tool_name, "status": "success"})
                self.mcp_tool_call_count += 1
                self.logger.counter("trade_model.mcp_tool_call_count")
                return result
            except Exception as e:
                self.logger.error(f"Error calling Slippage Analysis tool {tool_name}: {e}", tool=tool_name, error=str(e))
                self.mcp_tool_error_count += 1
                self.logger.counter("trade_model.mcp_tool_error_count")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "slippage_analysis", "tool": tool_name, "status": "failed"})
                return {"error": str(e)}

        @register_function(
            name="use_drift_detection_tool",
            description="Use a tool provided by the Drift Detection MCP server",
            caller=self.agents["trade_assistant"],
            executor=user_proxy,
        )
        def use_drift_detection_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            start_time = time.time()
            try:
                result = self.drift_detection_mcp.call_tool(tool_name, arguments)
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "drift_detection", "tool": tool_name, "status": "success"})
                self.mcp_tool_call_count += 1
                self.logger.counter("trade_model.mcp_tool_call_count")
                return result
            except Exception as e:
                self.logger.error(f"Error calling Drift Detection tool {tool_name}: {e}", tool=tool_name, error=str(e))
                self.mcp_tool_error_count += 1
                self.logger.counter("trade_model.mcp_tool_error_count")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("trade_model.mcp_tool_call_duration_ms", duration, tags={"server": "drift_detection", "tool": tool_name, "status": "failed"})
                return {"error": str(e)}

        # Register the MCP tool access functions
        user_proxy.register_function(use_alpaca_tool)
        user_proxy.register_function(use_redis_tool)
        user_proxy.register_function(use_peak_detection_tool)
        user_proxy.register_function(use_slippage_analysis_tool)
        user_proxy.register_function(use_drift_detection_tool)

    def
