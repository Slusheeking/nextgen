"""
NextGen Trader Model

This module implements the trade execution and monitoring component of the NextGen Models system.
It handles order execution, position monitoring, and trade analytics using MCP tools.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import logging
import time
from monitoring.system_monitor import MonitoringManager
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function

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
        # Initialize logging
        self.logger = logging.getLogger("nextgen_models.nextgen_trader")
        self.logger.setLevel(logging.INFO)

        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize monitoring
        self.monitoring = MonitoringManager(
            service_name="trade-model"
        )
        self.monitoring.log_info(
            "TradeModel initialized",
            component="trade_model",
            action="initialization",
        )

        # Initialize configuration
        self.config = config or {}

        # Set daily capital limit ($5000)
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
        if self.monitoring:
            self.monitoring.log_info(
                "Trade Model initialized", component="trade_model", action="init_agents"
            )

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
        def execute_market_order(
            symbol: str, quantity: float, side: str
        ) -> Dict[str, Any]:
            return self.executor.execute_market_order(symbol, quantity, side)

        def execute_limit_order(
            symbol: str, quantity: float, side: str, limit_price: float
        ) -> Dict[str, Any]:
            return self.executor.execute_limit_order(
                symbol, quantity, side, limit_price
            )

        # Define monitoring functions
        def start_position_monitoring(symbol: str, order_id: str) -> Dict[str, Any]:
            return self.monitor.start_position_monitoring(symbol, order_id)

        def check_exit_conditions(symbol: str) -> Dict[str, Any]:
            return self.monitor.check_exit_conditions(symbol)

        # Define position management functions
        def get_positions() -> List[Dict[str, Any]]:
            return self.position_manager.get_positions()

        def get_position(symbol: str) -> Optional[Dict[str, Any]]:
            return self.position_manager.get_position(symbol)

        def get_portfolio_constraints() -> Dict[str, Any]:
            return self.position_manager.get_portfolio_constraints()

        # Define analytics functions
        def get_trade_history(
            symbol: Optional[str] = None, limit: int = 50
        ) -> List[Dict[str, Any]]:
            return self.analytics.get_trade_history(symbol, limit)

        def calculate_execution_quality(order_id: str) -> Dict[str, Any]:
            return self.analytics.calculate_execution_quality(order_id)

        # Register MCP tool access functions
        self._register_mcp_tool_access(user_proxy)

        #
        # Register all functions with the user proxy agent and trade assistant
        # agent
        trade_assistant = self.agents["trade_assistant"]

        # Register execution functions
        register_function(
            execute_market_order,
            caller=trade_assistant,
            executor=user_proxy,
            description="Execute a market order",
        )

        register_function(
            execute_limit_order,
            caller=trade_assistant,
            executor=user_proxy,
            description="Execute a limit order",
        )

        # Register monitoring functions
        register_function(
            start_position_monitoring,
            caller=trade_assistant,
            executor=user_proxy,
            description="Begin monitoring a new position",
        )

        register_function(
            check_exit_conditions,
            caller=trade_assistant,
            executor=user_proxy,
            description="Check if exit conditions are met for a position",
        )

        # Register position management functions
        register_function(
            get_positions,
            caller=trade_assistant,
            executor=user_proxy,
            description="Get all current positions",
        )

        register_function(
            get_position,
            caller=trade_assistant,
            executor=user_proxy,
            description="Get position for a specific symbol",
        )

        register_function(
            get_portfolio_constraints,
            caller=trade_assistant,
            executor=user_proxy,
            description="Get current portfolio constraints",
        )

        # Register analytics functions
        register_function(
            get_trade_history,
            caller=trade_assistant,
            executor=user_proxy,
            description="Get historical trades",
        )

        register_function(
            calculate_execution_quality,
            caller=trade_assistant,
            executor=user_proxy,
            description="Calculate execution quality metrics",
        )

    def _register_mcp_tool_access(self, user_proxy: UserProxyAgent):
        """
        Register MCP tool access functions with the user proxy agent.

        Args:
            user_proxy: The user proxy agent to register functions with
        """

        # Define MCP tool access functions
        def use_alpaca_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.alpaca_mcp.call_tool(tool_name, arguments)

        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

        def use_peak_detection_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.peak_detection_mcp.call_tool(tool_name, arguments)

        def use_slippage_analysis_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.slippage_analysis_mcp.call_tool(tool_name, arguments)

        def use_drift_detection_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.drift_detection_mcp.call_tool(tool_name, arguments)

        # Register the MCP tool access functions
        trade_assistant = self.agents["trade_assistant"]

        register_function(
            use_alpaca_tool,
            caller=trade_assistant,
            executor=user_proxy,
            description="Use a tool provided by the Alpaca MCP server",
        )

        register_function(
            use_redis_tool,
            caller=trade_assistant,
            executor=user_proxy,
            description="Use a tool provided by the Redis MCP server",
        )

        register_function(
            use_peak_detection_tool,
            caller=trade_assistant,
            executor=user_proxy,
            description="Use a tool provided by the Peak Detection MCP server",
        )

        register_function(
            use_slippage_analysis_tool,
            caller=trade_assistant,
            executor=user_proxy,
            description="Use a tool provided by the Slippage Analysis MCP server",
        )

        register_function(
            use_drift_detection_tool,
            caller=trade_assistant,
            executor=user_proxy,
            description="Use a tool provided by the Drift Detection MCP server",
        )

    def start_of_day(self) -> Dict[str, Any]:
        """
        Execute start of day procedures:
        1. Reset daily capital usage
        2. Calculate available trading capital
        3. Notify Decision Model of available capital
        4. Set trading constraints for the day
        
        Returns:
            Dictionary with start of day status and available capital
        """
        self.logger.info("Executing start of day procedures")
        
        try:
            # Reset the daily usage key to 0
            today_key = f"trade:daily_usage:{datetime.now().strftime('%Y-%m-%d')}"
            self.redis_mcp.set_value(today_key, "0")
            
            # Calculate available trading capital
            # First get account info from Alpaca
            account_info = self.alpaca_mcp.get_account_info()
            
            if not account_info:
                self.logger.error("Failed to retrieve account information")
                return {"status": "failed", "reason": "Could not retrieve account information"}
            
            # Determine available capital (use cash, but respect daily limit)
            cash = float(account_info.get("cash", 0))
            available_capital = min(cash, self.daily_capital_limit)
            
            # Store current constraints
            constraints = {
                "daily_capital_limit": self.daily_capital_limit,
                "available_capital": available_capital,
                "no_overnight_positions": self.no_overnight_positions,
                "market_hours": self.alpaca_mcp.get_market_hours(),
                "date": datetime.now().strftime("%Y-%m-%d"),
            }
            
            self.redis_mcp.set_json("trade:daily_constraints", constraints)
            
            # Notify Decision Model of available capital
            self.notify_decision_model(
                "capital_available",
                {
                    "amount": available_capital,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "is_start_of_day": True
                }
            )
            
            self.logger.info(f"Start of day complete. Available capital: ${available_capital:.2f}")
            
            # Return status and available capital
            return {
                "status": "success",
                "available_capital": available_capital,
                "daily_capital_limit": self.daily_capital_limit,
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            self.logger.error(f"Error during start of day procedure: {e}")
            if self.monitoring:
                self.monitoring.log_error(
                    f"Start of day procedure failed: {e}",
                    component="trade_model",
                    action="start_of_day",
                    error=str(e)
                )
            return {"status": "failed", "reason": str(e)}
    
    def execute_trade(self, trade_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trade decision received from the Decision Model.

        Args:
            trade_decision: Decision dictionary containing:
                - symbol: Stock ticker
                - action: 'buy' or 'sell'
                - quantity: Number of shares to trade
                - order_type: 'market' or 'limit'
                - limit_price: Price for limit orders (optional)
                - capital_amount: Amount of capital allocated for this trade (required for buy)

        Returns:
            Dictionary with execution results
        """
        self.logger.info(f"Executing trade for {trade_decision.get('symbol')}")

        # Verify trading hours
        if not self._is_market_open():
            return {"status": "failed", "reason": "Market closed"}

        # Get parameters
        symbol = trade_decision.get("symbol")
        action = trade_decision.get("action")
        quantity = trade_decision.get("quantity")
        order_type = trade_decision.get("order_type", "market")
        limit_price = trade_decision.get("limit_price")
        capital_amount = trade_decision.get("capital_amount")

        # Validate parameters
        if not symbol or not action or not quantity:
            return {"status": "failed", "reason": "Missing required parameters"}
        
        # For buy orders, require capital_amount
        if action == "buy" and not capital_amount:
            return {"status": "failed", "reason": "Missing capital_amount for buy order"}
        
        # Check available capital for buy orders
        if action == "buy":
            daily_usage = self.position_manager.get_daily_usage()
            available_today = self.daily_capital_limit - daily_usage
            
            if capital_amount > available_today:
                return {
                    "status": "failed", 
                    "reason": f"Insufficient daily capital: requested ${capital_amount:.2f}, available ${available_today:.2f}"
                }
            
            # For market orders, ensure quantity matches capital (approximate)
            if order_type == "market" and quantity > 0:
                try:
                    quote = self.alpaca_mcp.get_latest_quote(symbol)
                    if quote and "ask_price" in quote:
                        estimated_cost = float(quote.get("ask_price")) * quantity
                        # Allow 5% buffer for market price fluctuation
                        if estimated_cost > capital_amount * 1.05:
                            self.logger.warning(f"Quantity ({quantity}) may exceed allocated capital (${capital_amount:.2f})")
                            # Adjust quantity to match capital
                            adjusted_quantity = math.floor(capital_amount / float(quote.get("ask_price")))
                            if adjusted_quantity > 0:
                                self.logger.info(f"Adjusting quantity from {quantity} to {adjusted_quantity} to match capital")
                                quantity = adjusted_quantity
                            else:
                                return {"status": "failed", "reason": "Insufficient capital for even one share"}
                except Exception as e:
                    self.logger.warning(f"Could not verify capital vs quantity: {e}")

        # Execute order based on type
        if order_type == "market":
            return self.executor.execute_market_order(symbol, quantity, action)
        elif order_type == "limit" and limit_price:
            return self.executor.execute_limit_order(
                symbol, quantity, action, limit_price
            )
        else:
            return {
                "status": "failed",
                "reason": "Invalid order type or missing limit price",
            }

    def _is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        try:
            clock = self.alpaca_mcp.get_market_hours()
            return clock.get("is_open", False)
        except Exception as e:
            self.logger.error(f"Error checking market hours: {e}")
            if self.monitoring:
                self.monitoring.log_error(
                    f"Error checking market hours: {e}",
                    component="trade_model",
                    action="market_hours_error",
                    error=str(e),
                )
            return False

    def notify_decision_model(
        self, event_type: str, event_data: Dict[str, Any]
    ) -> bool:
        """
        Notify the Decision Model of significant events.

        Args:
            event_type: Type of event (e.g., 'position_closed', 'capital_available')
            event_data: Event data dictionary

        Returns:
            True if notification was successful, False otherwise
        """
        try:
            # Create event record
            event_record = {
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis for the Decision Model to consume
            key = f"trade:event:{int(time.time())}"
            self.redis_mcp.set_json(key, event_record)

            # Add to event stream
            stream_key = "trade:events"
            self.redis_mcp.add_to_stream(stream_key, event_record)

            self.logger.info(f"Notified Decision Model of {event_type} event")
            
            # Log more detailed information for important events
            if event_type == "capital_available":
                amount = event_data.get("amount", 0.0)
                self.logger.info(f"Notified Decision Model of available capital: ${amount:.2f}")
                self.monitoring.log_info(
                    f"Capital available notification sent: ${amount:.2f}",
                    component="trade_model",
                    action="notify_capital_available",
                    amount=amount
                )
            elif event_type == "position_closed":
                symbol = event_data.get("symbol", "Unknown")
                reason = event_data.get("reason", "Unknown")
                self.logger.info(f"Notified Decision Model of closed position: {symbol} - Reason: {reason}")
                
            return True
        except Exception as e:
            self.logger.error(f"Error notifying Decision Model: {e}")
            if self.monitoring:
                self.monitoring.log_error(
                    f"Error notifying Decision Model: {e}",
                    component="trade_model",
                    action="notify_decision_model_error",
                    error=str(e),
                )
            return False

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """
        Run a monitoring cycle for all active positions.

        Returns:
            Dictionary with monitoring results
        """
        self.logger.info("Running monitoring cycle for active positions")

        # Get all active positions
        positions = self.position_manager.get_positions()

        results = {
            "positions_checked": len(positions),
            "exit_signals": 0,
            "positions_closed": 0,
            "errors": 0,
        }

        # Check each position
        for position in positions:
            symbol = position.get("symbol")

            try:
                # Check exit conditions
                exit_result = self.monitor.check_exit_conditions(symbol)

                # If should exit, execute sell order
                if exit_result.get("should_exit", False):
                    results["exit_signals"] += 1

                    # Execute sell
                    sell_decision = {
                        "symbol": symbol,
                        "action": "sell",
                        "quantity": float(position.get("qty", 0)),
                        "order_type": "market",
                    }

                    sell_result = self.execute_trade(sell_decision)

                    if sell_result.get("status") == "filled":
                        results["positions_closed"] += 1

                        # Notify Decision Model of capital availability
                        self.notify_decision_model(
                            "capital_available",
                            {
                                "amount": float(sell_result.get("proceeds", 0)),
                                "symbol": symbol,
                                "reason": exit_result.get("reason"),
                            },
                        )
            except Exception as e:
                self.logger.error(f"Error monitoring position {symbol}: {e}")
                if self.monitoring:
                    self.monitoring.log_error(
                        f"Error monitoring position {symbol}: {e}",
                        component="trade_model",
                        action="monitoring_error",
                        error=str(e),
                        symbol=symbol,
                    )
                results["errors"] += 1

        self.logger.info(f"Monitoring cycle completed: {results}")
        return results


class TradeExecutor:
    """
    Handles order execution for the Trade Model.
    """

    def __init__(self, trade_model: TradeModel):
        """Initialize with reference to parent Trade Model"""
        self.trade_model = trade_model
        self.logger = trade_model.logger

    def execute_market_order(
        self, symbol: str, quantity: float, side: str
    ) -> Dict[str, Any]:
        """
        Execute a market order using Alpaca MCP.

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            side: 'buy' or 'sell'

        Returns:
            Order result dictionary
        """
        self.logger.info(
            f"Executing {side} market order for {quantity} shares of {symbol}"
        )

        try:
            # Call Alpaca MCP to submit market order
            result = self.trade_model.alpaca_mcp.submit_market_order(
                symbol=symbol, qty=quantity, side=side, time_in_force="day"
            )

            # Store order in Redis
            self._store_order_record(result)

            # Begin monitoring if buy order
            if side == "buy" and result.get("status") == "filled":
                self.trade_model.monitor.start_position_monitoring(
                    symbol, result.get("id")
                )

            # Update daily usage if buy order
            if side == "buy" and result.get("status") in ["filled", "partially_filled"]:
                filled_qty = float(result.get("filled_qty", 0))
                filled_price = float(result.get("filled_avg_price", 0))
                cost = filled_qty * filled_price
                self.trade_model.position_manager.update_daily_usage(cost)

            # Calculate proceeds if sell order
            if side == "sell" and result.get("status") in [
                "filled",
                "partially_filled",
            ]:
                filled_qty = float(result.get("filled_qty", 0))
                filled_price = float(result.get("filled_avg_price", 0))
                proceeds = filled_qty * filled_price
                result["proceeds"] = proceeds

                # Update daily usage (negative to free up capital)
                self.trade_model.position_manager.update_daily_usage(-proceeds)

            return result

        except Exception as e:
            self.logger.error(f"Error executing market order: {str(e)}")
            return {"status": "failed", "reason": str(e)}

    def execute_limit_order(
        self, symbol: str, quantity: float, side: str, limit_price: float
    ) -> Dict[str, Any]:
        """
        Execute a limit order using Alpaca MCP.

        Args:
            symbol: Stock symbol
            quantity: Order quantity
            side: 'buy' or 'sell'
            limit_price: Limit price

        Returns:
            Order result dictionary
        """
        self.logger.info(
            f"Executing {side} limit order for {quantity} shares of {symbol} at {limit_price}"
        )

        try:
            # Call Alpaca MCP to submit limit order
            result = self.trade_model.alpaca_mcp.submit_limit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                limit_price=limit_price,
                time_in_force="day",
            )

            # Store order in Redis
            self._store_order_record(result)

            # Update daily usage if buy order (reserve the capital)
            if side == "buy":
                cost = quantity * limit_price
                self.trade_model.position_manager.update_daily_usage(cost)

            return result

        except Exception as e:
            self.logger.error(f"Error executing limit order: {str(e)}")
            return {"status": "failed", "reason": str(e)}

    def _store_order_record(self, order_result: Dict[str, Any]) -> None:
        """
        Store order record in Redis for tracking and analytics.

        Args:
            order_result: Order result from Alpaca
        """
        try:
            # Create order record
            order_record = {
                "id": order_result.get("id"),
                "symbol": order_result.get("symbol"),
                "side": order_result.get("side"),
                "qty": order_result.get("qty"),
                "type": order_result.get("type"),
                "limit_price": order_result.get("limit_price"),
                "status": order_result.get("status"),
                "created_at": order_result.get("created_at"),
                "filled_at": order_result.get("filled_at"),
                "filled_avg_price": order_result.get("filled_avg_price"),
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            key = f"trade:order:{order_result.get('id')}"
            self.trade_model.redis_mcp.set_json(key, order_record)

            # Add to order history
            self.trade_model.redis_mcp.add_to_sorted_set(
                "trade:orders", order_result.get("id"), int(time.time())
            )

        except Exception as e:
            self.logger.error(f"Error storing order record: {str(e)}")


class TradeMonitor:
    """
    Monitors active trades for exit conditions.

    Includes:
    - Peak detection
    - Slippage measurement
    - Drift detection
    - Price momentum analysis
    """

    def __init__(self, trade_model: TradeModel):
        """Initialize with reference to parent Trade Model"""
        self.trade_model = trade_model
        self.logger = trade_model.logger
        self.active_monitors = {}

    def start_position_monitoring(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """
        Begin monitoring a new position.

        Args:
            symbol: Stock symbol
            order_id: Order ID that opened the position

        Returns:
            Monitoring configuration
        """
        # Get position details
        position = self.trade_model.position_manager.get_position(symbol)

        if not position:
            self.logger.warning(f"No position found for {symbol}")
            return {"status": "failed", "reason": "No position found"}

        # Get order details
        try:
            order = self.trade_model.alpaca_mcp.get_order(order_id)
            if not order or "filled_avg_price" not in order:
                self.logger.error(f"Could not get valid order details for {order_id}")
                return {"status": "failed", "reason": "Invalid order details"}
            entry_price = float(order.get("filled_avg_price", 0))
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            return {"status": "failed", "reason": f"Error getting order: {e}"}

        # Set up monitoring parameters
        monitor_config = {
            "symbol": symbol,
            "entry_price": entry_price,
            "quantity": float(position.get("qty", 0)),
            "entry_time": datetime.now().isoformat(),
            "high_since_entry": entry_price,
            "low_since_entry": entry_price,
            "last_update_time": datetime.now().isoformat(),
            "stop_loss_pct": 0.02,  # 2% stop loss
            "take_profit_pct": 0.05,  # 5% take profit
            "trailing_stop_pct": 0.01,  # 1% trailing stop
            "monitoring_interval_sec": 60,  # Check every 60 seconds
        }

        # Store config in Redis
        key = f"trade:monitor:{symbol}"
        try:
            self.trade_model.redis_mcp.set_json(key, monitor_config)
        except Exception as e:
            self.logger.error(
                f"Error storing monitor config for {symbol} in Redis: {e}"
            )
            # Continue even if Redis fails, keep in memory
            pass

        # Add to active monitors
        self.active_monitors[symbol] = monitor_config

        self.logger.info(f"Started monitoring position for {symbol}")
        return monitor_config

    def check_exit_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Check if exit conditions are met for a position.

        Args:
            symbol: Stock symbol

        Returns:
            Exit assessment
        """
        # Get monitoring config
        config = self.active_monitors.get(symbol) or self._load_monitor_config(symbol)

        if not config:
            self.logger.warning(f"No monitoring config found for {symbol}")
            return {"should_exit": False, "reason": "No monitoring config"}

        # Get current price data
        try:
            quote = self.trade_model.alpaca_mcp.get_latest_quote(symbol)
            if not quote or "ask_price" not in quote:
                self.logger.warning(f"Invalid quote received for {symbol}")
                return {"should_exit": False, "reason": "Invalid quote data"}
            current_price = float(quote.get("ask_price"))  # Ensure float
        except Exception as e:
            self.logger.error(f"Error getting quote for {symbol}: {e}")
            return {"should_exit": False, "reason": "Error getting quote"}

        # Update high/low markers
        config["high_since_entry"] = max(
            config.get("high_since_entry", current_price), current_price
        )
        config["low_since_entry"] = min(
            config.get("low_since_entry", current_price), current_price
        )
        config["last_update_time"] = datetime.now().isoformat()

        # Calculate key metrics
        entry_price = config.get("entry_price", 0)
        gain_loss_pct = (
            (current_price - entry_price) / entry_price if entry_price != 0 else 0
        )
        high_since_entry = config["high_since_entry"]
        drawdown_from_peak = (
            (current_price - high_since_entry) / high_since_entry
            if high_since_entry != 0
            else 0
        )

        # Store updated config
        self._update_monitor_config(symbol, config)

        # Check exit conditions
        exit_result = {
            "should_exit": False,
            "reason": None,
            "data": {
                "current_price": current_price,
                "entry_price": entry_price,
                "gain_loss_pct": gain_loss_pct,
                "high_since_entry": high_since_entry,
                "drawdown_from_peak": drawdown_from_peak,
            },
        }

        stop_loss_pct = config.get("stop_loss_pct", 0.02)
        take_profit_pct = config.get("take_profit_pct", 0.05)
        trailing_stop_pct = config.get("trailing_stop_pct", 0.01)

        # Check stop loss
        if gain_loss_pct <= -stop_loss_pct:
            exit_result["should_exit"] = True
            exit_result["reason"] = "stop_loss"
            self.logger.info(f"Stop loss triggered for {symbol}")

        # Check take profit
        elif gain_loss_pct >= take_profit_pct:
            exit_result["should_exit"] = True
            exit_result["reason"] = "take_profit"
            self.logger.info(f"Take profit triggered for {symbol}")

        # Check trailing stop
        elif drawdown_from_peak <= -trailing_stop_pct and gain_loss_pct > 0:
            exit_result["should_exit"] = True
            exit_result["reason"] = "trailing_stop"
            self.logger.info(f"Trailing stop triggered for {symbol}")

        # Check market close approach (if no overnight positions allowed)
        elif (
            self.trade_model.no_overnight_positions
            and self._is_approaching_market_close()
        ):
            exit_result["should_exit"] = True
            exit_result["reason"] = "market_close"
            self.logger.info(f"Market close approaching, closing position for {symbol}")

        # Check for drift detection
        elif self._check_drift_signal(symbol):
            exit_result["should_exit"] = True
            exit_result["reason"] = "drift_signal"
            self.logger.info(f"Drift signal detected for {symbol}")

        return exit_result

    def _load_monitor_config(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load monitoring configuration from Redis.

        Args:
            symbol: Stock symbol

        Returns:
            Monitoring configuration or None if not found
        """
        try:
            key = f"trade:monitor:{symbol}"
            config = self.trade_model.redis_mcp.get_json(key)

            if config:
                # Ensure numeric types are correct after loading from JSON
                config["entry_price"] = float(config.get("entry_price", 0))
                config["quantity"] = float(config.get("quantity", 0))
                config["high_since_entry"] = float(config.get("high_since_entry", 0))
                config["low_since_entry"] = float(config.get("low_since_entry", 0))
                config["stop_loss_pct"] = float(config.get("stop_loss_pct", 0.02))
                config["take_profit_pct"] = float(config.get("take_profit_pct", 0.05))
                config["trailing_stop_pct"] = float(
                    config.get("trailing_stop_pct", 0.01)
                )
                self.active_monitors[symbol] = config

            return config
        except Exception as e:
            self.logger.error(f"Error loading monitor config for {symbol}: {e}")
            return None

    def _update_monitor_config(self, symbol: str, config: Dict[str, Any]) -> bool:
        """
        Update monitoring configuration in Redis.

        Args:
            symbol: Stock symbol
            config: Updated configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"trade:monitor:{symbol}"
            self.trade_model.redis_mcp.set_json(key, config)
            self.active_monitors[symbol] = config  # Update in-memory cache
            return True
        except Exception as e:
            self.logger.error(f"Error updating monitor config for {symbol}: {e}")
            return False

    def _is_approaching_market_close(self) -> bool:
        """
        Check if market close is approaching (within 15 minutes).

        Returns:
            True if market close is within 15 minutes, False otherwise
        """
        try:
            clock = self.trade_model.alpaca_mcp.get_market_hours()

            if not clock or not clock.get("is_open", False):
                return False

            closing_time_str = clock.get("next_close") or clock.get(
                "closing_at"
            )  # Alpaca API might use next_close
            if not closing_time_str:
                self.logger.warning("Could not determine market closing time.")
                return False

            #
            # Handle potential timezone differences if needed, assuming UTC or
            # local timezone consistent with Alpaca
            # Alpaca times are typically US/Eastern
            closing_time = pd.Timestamp(
                closing_time_str, tz="America/New_York"
            ).tz_convert("UTC")
            now = pd.Timestamp.utcnow()  # Use UTC now

            # Check if within 15 minutes of closing
            time_to_close = (closing_time - now).total_seconds()
            #
            # self.logger.debug(f"Time until market close: {time_to_close /
            # 60:.2f} minutes")
            return 0 < time_to_close <= 900  # Ensure it's positive and within 15 mins
        except Exception as e:
            self.logger.error(f"Error checking market close: {e}")
            return False

    def _check_drift_signal(self, symbol: str) -> bool:
        """
        Check for drift signals using the drift detection MCP.

        Args:
            symbol: Stock symbol

        Returns:
            True if drift signal detected, False otherwise
        """
        try:
            # Get recent price data
            bars = self.trade_model.alpaca_mcp.get_bars(
                symbol=symbol,
                timeframe="1Min",
                limit=30,  # Use enough data for the drift window
            )

            if not bars or len(bars) < 10:  # Need at least window_size bars
                self.logger.debug(f"Not enough bars for drift check on {symbol}")
                return False

            # Convert to format expected by drift detection
            prices = [float(bar.get("c", 0)) for bar in bars]

            # Call drift detection MCP
            result = self.trade_model.drift_detection_mcp.detect_drift(
                {
                    "prices": prices,
                    "threshold": 0.02,  # Example threshold - adjust as needed
                    "window_size": 10,  # Example window size - adjust as needed
                }
            )

            drift_detected = result.get("drift_detected", False)
            if drift_detected:
                self.logger.info(f"Drift detected for {symbol}: {result}")
            return drift_detected
        except Exception as e:
            self.logger.error(f"Error checking drift signal for {symbol}: {e}")
            return False


class TradePositionManager:
    """
    Manages active positions and portfolio constraints.
    """

    def __init__(self, trade_model: TradeModel):
        """Initialize with reference to parent Trade Model"""
        self.trade_model = trade_model
        self.logger = trade_model.logger
        self._daily_usage_key = (
            f"trade:daily_usage:{datetime.now().strftime('%Y-%m-%d')}"
        )

        # Redis keys for portfolio data
        self._portfolio_prefix = "portfolio:"
        self._account_info_key = f"{self._portfolio_prefix}account_info"
        self._positions_key = f"{self._portfolio_prefix}positions"
        self._positions_by_symbol_prefix = f"{self._portfolio_prefix}position:"
        self._portfolio_summary_key = f"{self._portfolio_prefix}summary"
        self._last_updated_key = f"{self._portfolio_prefix}last_updated"

        # Default TTL for portfolio data (15 minutes)
        self._portfolio_data_ttl = 900

        # Initialize portfolio data in Redis
        self.sync_portfolio_data()

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions from Redis or Alpaca.

        This method first tries to get positions from Redis.
        If not found or expired, it fetches from Alpaca and updates Redis.

        Returns:
            List of position dictionaries
        """
        try:
            # Try to get from Redis first
            positions = self.get_positions_from_redis()

            # If not found in Redis or expired, fetch from Alpaca
            if not positions:
                self.logger.info("Positions not found in Redis, fetching from Alpaca")
                positions = self.get_positions_from_alpaca()

                if positions:
                    # Store in Redis with TTL
                    self.trade_model.redis_mcp.set_json(self._positions_key, positions)
                    self.trade_model.redis_mcp.expire(
                        self._positions_key, self._portfolio_data_ttl
                    )

                    # Store individual positions by symbol
                    for position in positions:
                        symbol = position.get("symbol")
                        if symbol:
                            key = f"{self._positions_by_symbol_prefix}{symbol}"
                            self.trade_model.redis_mcp.set_json(key, position)
                            self.trade_model.redis_mcp.expire(
                                key, self._portfolio_data_ttl
                            )

            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol from Redis or Alpaca.

        This method first tries to get the position from Redis.
        If not found or expired, it fetches from Alpaca and updates Redis.

        Args:
            symbol: Stock symbol

        Returns:
            Position dictionary or None if not found
        """
        try:
            # Try to get from Redis first
            key = f"{self._positions_by_symbol_prefix}{symbol}"
            position = self.trade_model.redis_mcp.get_json(key)

            # If not found in Redis or expired, fetch from Alpaca
            if not position:
                self.logger.info(
                    f"Position for {symbol} not found in Redis, fetching from Alpaca"
                )
                try:
                    alpaca_position = self.trade_model.alpaca_mcp.get_position(symbol)
                    if alpaca_position:
                        # Ensure numeric types are floats
                        position = {
                            k: float(v)
                            if isinstance(v, (int, str))
                            and k
                            in [
                                "qty",
                                "avg_entry_price",
                                "market_value",
                                "cost_basis",
                                "unrealized_pl",
                                "unrealized_plpc",
                                "current_price",
                            ]
                            else v
                            for k, v in alpaca_position._raw.items()
                        }

                        # Store in Redis with TTL
                        self.trade_model.redis_mcp.set_json(key, position)
                        self.trade_model.redis_mcp.expire(key, self._portfolio_data_ttl)
                except Exception as e:
                    #
                    # Alpaca API might raise an exception if position doesn't
                    # exist
                    if "position not found" in str(e).lower():
                        self.logger.info(f"No position found for symbol {symbol}.")
                        return None
                    raise  # Re-raise for other exceptions

            return position
        except Exception as e:
            self.logger.error(f"Error getting position for {symbol}: {e}")
            return None

    def get_portfolio_constraints(self) -> Dict[str, Any]:
        """
        Get current portfolio constraints and account information.

        This method combines daily capital usage with account information
        from Redis to provide a comprehensive view of portfolio constraints.

        Returns:
            Dictionary with portfolio constraints and account information
        """
        # Get daily usage
        daily_usage = self.get_daily_usage()
        remaining_capital = self.trade_model.daily_capital_limit - daily_usage

        # Get account info from Redis
        account_info = self.get_account_info()

        # Get portfolio summary
        portfolio_summary = (
            self.trade_model.redis_mcp.get_json(self._portfolio_summary_key) or {}
        )

        # Combine all data
        constraints = {
            "daily_capital_limit": self.trade_model.daily_capital_limit,
            "daily_capital_used": daily_usage,
            "remaining_daily_capital": remaining_capital,
            "no_overnight_positions": self.trade_model.no_overnight_positions,
            "last_updated": self.trade_model.redis_mcp.get_value(self._last_updated_key)
            or datetime.now().isoformat(),
        }

        # Add account info
        if account_info:
            constraints["buying_power"] = float(account_info.get("buying_power", 0))
            constraints["portfolio_value"] = float(
                account_info.get("portfolio_value", 0)
            )
            constraints["cash"] = float(account_info.get("cash", 0))
            constraints["equity"] = float(account_info.get("equity", 0))

        # Add portfolio summary
        if portfolio_summary:
            constraints["total_positions"] = portfolio_summary.get("total_positions", 0)
            constraints["total_market_value"] = portfolio_summary.get(
                "total_market_value", 0.0
            )
            constraints["total_unrealized_pl"] = portfolio_summary.get(
                "total_unrealized_pl", 0.0
            )
            constraints["total_unrealized_plpc"] = portfolio_summary.get(
                "total_unrealized_plpc", 0.0
            )

        return constraints

    def update_daily_usage(self, amount: float) -> float:
        """
        Update the daily capital usage in Redis.

        Args:
            amount: Amount to add to the daily usage (can be negative for sells)

        Returns:
            The new daily usage amount.
        """
        try:
            # Use INCRBYFLOAT to atomically update the usage
            new_usage = self.trade_model.redis_mcp.increment_float(
                self._daily_usage_key, amount
            )
            # Set expiry for the key (e.g., 24 hours) to auto-reset daily
            self.trade_model.redis_mcp.expire(
                self._daily_usage_key, 86400
            )  # 24 * 60 * 60
            self.logger.info(
                f"Updated daily capital usage by {amount:.2f}. New usage: {new_usage:.2f}"
            )
            return new_usage
        except Exception as e:
            self.logger.error(f"Error updating daily usage: {e}")
            # Fallback to getting the value if increment fails
            return self.get_daily_usage()

    def get_daily_usage(self) -> float:
        """
        Get the current daily capital usage from Redis.

        Returns:
            Current daily capital usage.
        """
        try:
            usage = self.trade_model.redis_mcp.get(self._daily_usage_key)
            return float(usage) if usage else 0.0
        except Exception as e:
            self.logger.error(f"Error getting daily usage: {e}")
            return 0.0

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from Redis or Alpaca.

        This method first tries to get account information from Redis.
        If not found or expired, it fetches from Alpaca and updates Redis.

        Returns:
            Dictionary with account information
        """
        try:
            # Try to get from Redis first
            account_info = self.trade_model.redis_mcp.get_json(self._account_info_key)

            # If not found in Redis or expired, fetch from Alpaca
            if not account_info:
                self.logger.info(
                    "Account info not found in Redis, fetching from Alpaca"
                )
                account_info = self.trade_model.alpaca_mcp.get_account_info()

                if account_info:
                    # Store in Redis with TTL
                    self.trade_model.redis_mcp.set_json(
                        self._account_info_key, account_info
                    )
                    self.trade_model.redis_mcp.expire(
                        self._account_info_key, self._portfolio_data_ttl
                    )

                    # Also store individual values for easier access
                    if "buying_power" in account_info:
                        self.trade_model.redis_mcp.set_value(
                            f"{self._portfolio_prefix}buying_power",
                            str(account_info["buying_power"]),
                        )
                        self.trade_model.redis_mcp.expire(
                            f"{self._portfolio_prefix}buying_power",
                            self._portfolio_data_ttl,
                        )

                    if "portfolio_value" in account_info:
                        self.trade_model.redis_mcp.set_value(
                            f"{self._portfolio_prefix}portfolio_value",
                            str(account_info["portfolio_value"]),
                        )
                        self.trade_model.redis_mcp.expire(
                            f"{self._portfolio_prefix}portfolio_value",
                            self._portfolio_data_ttl,
                        )

                    if "cash" in account_info:
                        self.trade_model.redis_mcp.set_value(
                            f"{self._portfolio_prefix}cash", str(account_info["cash"])
                        )
                        self.trade_model.redis_mcp.expire(
                            f"{self._portfolio_prefix}cash", self._portfolio_data_ttl
                        )

                    # Update last updated timestamp
                    self.trade_model.redis_mcp.set_value(
                        self._last_updated_key, datetime.now().isoformat()
                    )
                    self.trade_model.redis_mcp.expire(
                        self._last_updated_key, self._portfolio_data_ttl
                    )

            return account_info or {}

        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}

    def sync_portfolio_data(self) -> bool:
        """
        Synchronize all portfolio data from Alpaca to Redis.

        This method fetches account information and positions from Alpaca
        and stores them in Redis for other components to access.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get account information
            account_info = self.trade_model.alpaca_mcp.get_account_info()

            # Get positions
            positions = self.get_positions_from_alpaca()

            # Calculate portfolio summary
            portfolio_summary = self._calculate_portfolio_summary(
                account_info, positions
            )

            # Store in Redis
            if account_info:
                self.trade_model.redis_mcp.set_json(
                    self._account_info_key, account_info
                )
                self.trade_model.redis_mcp.expire(
                    self._account_info_key, self._portfolio_data_ttl
                )

            if positions:
                self.trade_model.redis_mcp.set_json(self._positions_key, positions)
                self.trade_model.redis_mcp.expire(
                    self._positions_key, self._portfolio_data_ttl
                )

                # Store individual positions by symbol
                for position in positions:
                    symbol = position.get("symbol")
                    if symbol:
                        key = f"{self._positions_by_symbol_prefix}{symbol}"
                        self.trade_model.redis_mcp.set_json(key, position)
                        self.trade_model.redis_mcp.expire(key, self._portfolio_data_ttl)

            if portfolio_summary:
                self.trade_model.redis_mcp.set_json(
                    self._portfolio_summary_key, portfolio_summary
                )
                self.trade_model.redis_mcp.expire(
                    self._portfolio_summary_key, self._portfolio_data_ttl
                )

            # Update last updated timestamp
            self.trade_model.redis_mcp.set_value(
                self._last_updated_key, datetime.now().isoformat()
            )
            self.trade_model.redis_mcp.expire(
                self._last_updated_key, self._portfolio_data_ttl
            )

            self.logger.info("Portfolio data synchronized to Redis")
            return True

        except Exception as e:
            self.logger.error(f"Error synchronizing portfolio data: {e}")
            return False

    def get_positions_from_alpaca(self) -> List[Dict[str, Any]]:
        """
        Get positions directly from Alpaca.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trade_model.alpaca_mcp.get_all_positions()
            # Convert Alpaca position objects to dictionaries if necessary
            # Ensure numeric types are floats
            return (
                [
                    {
                        k: float(v)
                        if isinstance(v, (int, str))
                        and k
                        in [
                            "qty",
                            "avg_entry_price",
                            "market_value",
                            "cost_basis",
                            "unrealized_pl",
                            "unrealized_plpc",
                            "current_price",
                        ]
                        else v
                        for k, v in pos._raw.items()
                    }
                    for pos in positions
                ]
                if positions
                else []
            )
        except Exception as e:
            self.logger.error(f"Error getting positions from Alpaca: {e}")
            return []

    def get_positions_from_redis(self) -> List[Dict[str, Any]]:
        """
        Get positions from Redis.

        Returns:
            List of position dictionaries
        """
        try:
            positions = self.trade_model.redis_mcp.get_json(self._positions_key)
            return positions or []
        except Exception as e:
            self.logger.error(f"Error getting positions from Redis: {e}")
            return []

    def _calculate_portfolio_summary(
        self, account_info: Dict[str, Any], positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate portfolio summary metrics.

        Args:
            account_info: Account information dictionary
            positions: List of position dictionaries

        Returns:
            Dictionary with portfolio summary metrics
        """
        summary = {
            "total_positions": len(positions),
            "total_market_value": 0.0,
            "total_cost_basis": 0.0,
            "total_unrealized_pl": 0.0,
            "total_unrealized_plpc": 0.0,
            "positions_by_sector": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Add account info
        if account_info:
            summary["equity"] = float(account_info.get("equity", 0))
            summary["buying_power"] = float(account_info.get("buying_power", 0))
            summary["cash"] = float(account_info.get("cash", 0))

        # Calculate position totals
        for position in positions:
            summary["total_market_value"] += float(position.get("market_value", 0))
            summary["total_cost_basis"] += float(position.get("cost_basis", 0))
            summary["total_unrealized_pl"] += float(position.get("unrealized_pl", 0))

            # Group by sector if available
            sector = position.get("sector", "Unknown")
            if sector not in summary["positions_by_sector"]:
                summary["positions_by_sector"][sector] = 0
            summary["positions_by_sector"][sector] += 1

        # Calculate percentage P&L
        if summary["total_cost_basis"] > 0:
            summary["total_unrealized_plpc"] = (
                summary["total_unrealized_pl"] / summary["total_cost_basis"]
            ) * 100

        return summary


class TradeAnalytics:
    """
    Provides analytics and performance metrics for trades.
    """

    def __init__(self, trade_model: TradeModel):
        """Initialize with reference to parent Trade Model"""
        self.trade_model = trade_model
        self.logger = trade_model.logger

    def get_trade_history(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get historical trades from Redis.

        Args:
            symbol: Optional filter by symbol
            limit: Maximum number of trades to return

        Returns:
            List of trade dictionaries
        """
        try:
            # Get order IDs from sorted set (newest first)
            order_ids = self.trade_model.redis_mcp.get_sorted_set_members(
                "trade:orders", 0, -1, reverse=True
            )  # Get all, newest first

            if not order_ids:
                return []

            # Get order details for each ID
            orders = []
            count = 0
            for order_id in order_ids:
                key = f"trade:order:{order_id}"
                order = self.trade_model.redis_mcp.get_json(key)

                if order and (not symbol or order.get("symbol") == symbol):
                    # Ensure numeric types are correct
                    order["qty"] = float(order.get("qty", 0))
                    order["limit_price"] = (
                        float(order.get("limit_price", 0))
                        if order.get("limit_price") is not None
                        else None
                    )
                    order["filled_avg_price"] = (
                        float(order.get("filled_avg_price", 0))
                        if order.get("filled_avg_price") is not None
                        else None
                    )
                    orders.append(order)
                    count += 1
                    # Stop if we've reached the limit
                    if count >= limit:
                        break

            return orders
        except Exception as e:
            self.logger.error(f"Error getting trade history: {e}")
            return []

    def calculate_execution_quality(self, order_id: str) -> Dict[str, Any]:
        """
        Calculate execution quality metrics for a specific order.

        Args:
            order_id: Order ID to analyze

        Returns:
            Dictionary with execution quality metrics
        """
        try:
            # Get order details
            key = f"trade:order:{order_id}"
            order = self.trade_model.redis_mcp.get_json(key)

            if not order:
                return {"status": "failed", "reason": "Order not found"}

            # Ensure numeric types
            order["qty"] = float(order.get("qty", 0))
            order["limit_price"] = (
                float(order.get("limit_price", 0))
                if order.get("limit_price") is not None
                else None
            )
            order["filled_avg_price"] = (
                float(order.get("filled_avg_price", 0))
                if order.get("filled_avg_price") is not None
                else None
            )

            # Get market data around execution time
            symbol = order.get("symbol")
            side = order.get("side")
            filled_price = order.get("filled_avg_price")
            filled_qty = order.get("qty")
            filled_at = order.get("filled_at")

            if not filled_at or filled_price is None:
                return {
                    "status": "pending",
                    "reason": "Order not filled yet or missing price",
                }

            # Convert filled_at to datetime
            filled_time = pd.to_datetime(filled_at)  # Assumes ISO format with timezone

            #
            # Get market data for 1 minute before and after execution for
            # tighter analysis
            start_time = filled_time - timedelta(minutes=1)
            end_time = filled_time + timedelta(minutes=1)

            #
            # Format times for Alpaca API (ensure UTC or correct timezone
            # handling)
            start_str = start_time.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            end_str = end_time.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

            # Get market data (e.g., minute bars)
            bars = self.trade_model.alpaca_mcp.get_bars(
                symbol=symbol, timeframe="1Min", start=start_str, end=end_str
            )

            if not bars:
                return {
                    "status": "limited",
                    "reason": "No market data available around execution time",
                    "order_details": order,
                }

            # --- Calculate metrics ---
            prices = [float(bar.get("c", 0)) for bar in bars]
            vwaps = [
                float(bar.get("vw", 0)) for bar in bars if bar.get("vw") is not None
            ]  # Alpaca minute bars have VWAP

            # Calculate VWAP during the execution minute(s)
            exec_minute_bars = [
                b
                for b in bars
                if pd.to_datetime(b.get("t")).floor("min") == filled_time.floor("min")
            ]
            exec_vwap = (
                np.mean(
                    [
                        float(b.get("vw", 0))
                        for b in exec_minute_bars
                        if b.get("vw") is not None
                    ]
                )
                if exec_minute_bars
                else None
            )

            # Calculate slippage from execution minute VWAP
            slippage_vwap = None
            if exec_vwap is not None:
                if side == "buy":
                    slippage_vwap = (filled_price - exec_vwap) / exec_vwap * 100
                else:  # sell
                    slippage_vwap = (exec_vwap - filled_price) / exec_vwap * 100

            #
            # Calculate slippage from arrival price (e.g., price at the start
            # of the minute)
            arrival_price = (
                float(bars[0].get("o", 0)) if bars else None
            )  # Opening price of the first bar
            slippage_arrival = None
            if arrival_price is not None:
                if side == "buy":
                    slippage_arrival = (
                        (filled_price - arrival_price) / arrival_price * 100
                    )
                else:  # sell
                    slippage_arrival = (
                        (arrival_price - filled_price) / arrival_price * 100
                    )

            # Use slippage analysis MCP for potentially more advanced metrics
            advanced_metrics = {}
            try:
                advanced_metrics = (
                    self.trade_model.slippage_analysis_mcp.analyze_slippage(
                        {
                            "symbol": symbol,
                            "side": side,
                            "filled_price": filled_price,
                            "filled_qty": filled_qty,
                            "filled_time": filled_at,
                            "market_data": bars,  # Provide relevant market data
                        }
                    )
                )
            except Exception as e:
                self.logger.warning(
                    f"Error calculating advanced slippage metrics via MCP: {e}"
                )

            # Combine all metrics
            return {
                "status": "success",
                "order_id": order_id,
                "symbol": symbol,
                "side": side,
                "filled_price": filled_price,
                "filled_qty": filled_qty,
                "filled_at": filled_at,
                "metrics": {
                    "exec_minute_vwap": exec_vwap,
                    "arrival_price": arrival_price,
                    "slippage_vs_vwap_pct": slippage_vwap,
                    "slippage_vs_arrival_pct": slippage_arrival,
                    **advanced_metrics,  # Include metrics from the MCP tool
                },
            }
        except Exception as e:
            self.logger.error(
                f"Error calculating execution quality for {order_id}: {e}"
            )
            return {"status": "error", "reason": str(e)}

    def calculate_daily_performance(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate daily trading performance metrics.

        Args:
            date: Date string in YYYY-MM-DD format (defaults to today)

        Returns:
            Dictionary with daily performance metrics
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            # Get all trades for the day
            #
            # Increase limit significantly to capture all potential trades for
            # a day
            trades = self.get_trade_history(limit=5000)

            # Filter trades by date (ensure proper date comparison)
            day_trades = [
                trade
                for trade in trades
                if trade.get("filled_at")
                and pd.to_datetime(trade.get("filled_at")).strftime("%Y-%m-%d") == date
            ]

            if not day_trades:
                return {"date": date, "status": "no_trades", "trades_count": 0}

            # Calculate metrics
            buy_trades = [t for t in day_trades if t.get("side") == "buy"]
            sell_trades = [t for t in day_trades if t.get("side") == "sell"]

            buy_volume = sum(
                t.get("qty", 0) * t.get("filled_avg_price", 0)
                for t in buy_trades
                if t.get("filled_avg_price") is not None
            )
            sell_volume = sum(
                t.get("qty", 0) * t.get("filled_avg_price", 0)
                for t in sell_trades
                if t.get("filled_avg_price") is not None
            )

            # Calculate P&L using position tracking
            # Group trades by symbol to track positions properly
            symbol_trades = {}
            for trade in day_trades:
                symbol = trade.get("symbol")
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)
            
            # Calculate P&L for each symbol
            symbol_pnl = {}
            total_pnl = 0
            
            for symbol, trades in symbol_trades.items():
                # Sort trades by filled time
                sorted_trades = sorted(trades, key=lambda t: pd.to_datetime(t.get("filled_at", "1970-01-01")))
                
                # Track position for this symbol
                position = 0
                cost_basis = 0
                symbol_realized_pnl = 0
                
                for trade in sorted_trades:
                    side = trade.get("side")
                    qty = float(trade.get("qty", 0))
                    price = float(trade.get("filled_avg_price", 0))
                    
                    if side == "buy":
                        # Update cost basis with weighted average
                        if position == 0:
                            cost_basis = price
                        else:
                            # Weighted average of existing position and new purchase
                            cost_basis = ((position * cost_basis) + (qty * price)) / (position + qty)
                        position += qty
                    elif side == "sell":
                        if position > 0:
                            # Calculate realized P&L for this sale
                            trade_pnl = qty * (price - cost_basis)
                            symbol_realized_pnl += trade_pnl
                            position -= qty
                            
                            # If position is now zero, reset cost basis
                            if position <= 0:
                                position = 0
                                cost_basis = 0
                
                symbol_pnl[symbol] = symbol_realized_pnl
                total_pnl += symbol_realized_pnl
            
            # Calculate win rate by pairing buy/sell trades
            profitable_trades = 0
            losing_trades = 0
            
            for symbol, trades in symbol_trades.items():
                # Sort trades by filled time
                sorted_trades = sorted(trades, key=lambda t: pd.to_datetime(t.get("filled_at", "1970-01-01")))
                
                # Create a queue of buy trades to match with sells
                buy_queue = []
                
                for trade in sorted_trades:
                    side = trade.get("side")
                    qty = float(trade.get("qty", 0))
                    price = float(trade.get("filled_avg_price", 0))
                    
                    if side == "buy":
                        # Add to buy queue
                        buy_queue.append((qty, price))
                    elif side == "sell" and buy_queue:
                        # Match with buys using FIFO method
                        remaining_sell_qty = qty
                        
                        while remaining_sell_qty > 0 and buy_queue:
                            buy_qty, buy_price = buy_queue[0]
                            
                            # Determine how much of this buy to use
                            match_qty = min(buy_qty, remaining_sell_qty)
                            
                            # Calculate P&L for this match
                            match_pnl = match_qty * (price - buy_price)
                            
                            # Update counters
                            if match_pnl > 0:
                                profitable_trades += 1
                            elif match_pnl < 0:
                                losing_trades += 1
                            
                            # Update remaining quantities
                            remaining_sell_qty -= match_qty
                            buy_qty -= match_qty
                            
                            # Update or remove the buy entry
                            if buy_qty > 0:
                                buy_queue[0] = (buy_qty, buy_price)
                            else:
                                buy_queue.pop(0)
            
            # Calculate win rate
            total_closed_trades = profitable_trades + losing_trades
            win_rate = (profitable_trades / total_closed_trades * 100) if total_closed_trades > 0 else None

            # Get unique symbols traded
            symbols_traded = set(t.get("symbol") for t in day_trades if t.get("symbol"))

            return {
                "date": date,
                "status": "success",
                "trades_count": len(day_trades),
                "buy_trades_count": len(buy_trades),
                "sell_trades_count": len(sell_trades),
                "buy_volume": round(buy_volume, 2),
                "sell_volume": round(sell_volume, 2),
                "estimated_pnl": round(total_pnl, 2),
                "win_rate_pct": win_rate,  # Placeholder
                "profitable_trades": profitable_trades,  # Placeholder
                "losing_trades": losing_trades,  # Placeholder
                "symbols_traded": list(symbols_traded),
                "symbols_count": len(symbols_traded),
            }
        except Exception as e:
            self.logger.error(f"Error calculating daily performance for {date}: {e}")
            return {"date": date, "status": "error", "reason": str(e)}
