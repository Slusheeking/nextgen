"""
NextGen Stock Selection Model

This module implements an LLM-based stock selection model using AutoGen that identifies 
potential trading candidates based on technical criteria, unusual activity, and day trading parameters.
The model uses MCP tools to access market data and trading functionality.
"""
import os
import logging
import json
import time
from monitoring import setup_monitoring
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, TypedDict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import autogen as autogen
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    config_list_from_json,
    register_function
    # FunctionCall is not available in autogen 0.9.0
)

from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWSMCP
from mcp_tools.data_mcp.unusual_whales_mcp import UnusualWhalesMCP

class SelectionModel:
    """
    LLM-based Stock Selection Model for identifying trading candidates.
    
    This model uses AutoGen and LLMs to identify and rank potential trading candidates
    based on technical criteria, unusual activity signals, and day trading parameters.
    It leverages MCP tools to access market data and trading functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Selection Model.
        
        Args:
            config: Optional configuration dictionary. May contain:
                - max_candidates: Maximum number of candidates to select (default: 20)
                - min_price: Minimum stock price (default: 10.0)
                - max_price: Maximum stock price (default: 200.0)
                - min_volume: Minimum average daily volume (default: 500000)
                - min_relative_volume: Minimum relative volume (default: 1.5)
                - max_spread_pct: Maximum bid-ask spread percentage (default: 0.5)
                - redis_prefix: Prefix for Redis keys (default: "selection:")
                - llm_config: Configuration for the LLM
        """
        self.logger = logging.getLogger(__name__)

        # Initialize monitoring
        self.monitor, self.metrics = setup_monitoring(
            service_name="selection-model",
            enable_prometheus=True,
            enable_loki=True,
            default_labels={"component": "selection_model"}
        )
        if self.monitor:
            self.monitor.log_info("SelectionModel initialized", component="selection_model", action="initialization")
        
        # Initialize configuration
        self.config = config or {}
        self.max_candidates = self.config.get("max_candidates", 20)
        self.min_price = self.config.get("min_price", 10.0)
        self.max_price = self.config.get("max_price", 200.0)
        self.min_volume = self.config.get("min_volume", 500000)
        self.min_relative_volume = self.config.get("min_relative_volume", 1.5)
        self.max_spread_pct = self.config.get("max_spread_pct", 0.5)
        self.redis_prefix = self.config.get("redis_prefix", "selection:")
        
        # Initialize MCP clients
        self.alpaca_mcp = AlpacaMCP(self.config.get("alpaca_config"))
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.polygon_rest_mcp = PolygonRestMCP(self.config.get("polygon_rest_config"))
        self.polygon_ws_mcp = PolygonWSMCP(self.config.get("polygon_ws_config"))
        self.unusual_whales_mcp = UnusualWhalesMCP(self.config.get("unusual_whales_config"))
        
        # Initialize LLM configuration
        self.llm_config = self._get_llm_config()
        
        # Initialize AutoGen agents
        self.agents = self._setup_agents()
        
        self.logger.info("SelectionModel initialized with AutoGen agents")
        if self.monitor:
            self.monitor.log_info("SelectionModel initialized with AutoGen agents", component="selection_model", action="init_agents")
    
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
                        "api_version": None
                    },
                    {
                        "model": "meta-llama/llama-3-70b-instruct",
                        "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                        "base_url": "https://openrouter.ai/api/v1",
                        "api_type": "openai",
                        "api_version": None
                    }
                ]
            }
        
        return {
            "config_list": llm_config.get("config_list", []),
            "temperature": llm_config.get("temperature", 0.1),
            "timeout": llm_config.get("timeout", 600),
            "seed": 42  # Adding seed for reproducibility
        }
    
    def _setup_agents(self) -> Dict[str, Agent]:
        """
        Initialize AutoGen agents for stock selection.
        
        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}
        
        # Create the stock selection assistant agent with AG2 configuration
        agents["selection_assistant"] = AssistantAgent(
            name="StockSelectionAssistant",
            system_message="""You are a stock selection specialist. Your role is to identify promising stocks for day trading based on technical indicators, liquidity, and unusual activity.

You have access to the following tools through MCP servers:
1. Alpaca MCP - For account information and trading
2. Redis MCP - For storing and retrieving candidate information
3. Polygon REST MCP - For market data and technical indicators
4. Polygon WS MCP - For real-time market data
5. Unusual Whales MCP - For unusual options activity

When selecting stocks, focus on:
- Technical criteria (momentum, volatility, liquidity)
- Unusual activity signals
- Day trading parameters
- Producing 10-20 quality candidates

Your selection process should include:
1. Applying liquidity filters (volume, spread)
2. Analyzing technical indicators (RSI, moving averages, ATR)
3. Checking for unusual activity
4. Scoring and ranking candidates
5. Ensuring sector diversity

Provide detailed reasoning for each selection and include relevant metrics.""",
            llm_config=self.llm_config,
            description="A specialist in identifying promising stocks for day trading based on technical indicators and market data"
        )
        
        # Create the data analysis assistant agent with AG2 configuration
        agents["data_assistant"] = AssistantAgent(
            name="DataAnalysisAssistant",
            system_message="""You are a financial data analysis specialist. Your role is to analyze market data and technical indicators to support stock selection.

You have access to the following data through MCP servers:
1. Historical price data
2. Volume data
3. Technical indicators
4. Options activity
5. Market sentiment

Provide detailed analysis of:
- Support and resistance levels
- Trend strength and direction
- Volatility patterns
- Volume analysis
- Relative strength compared to market

Your analysis should be quantitative and evidence-based.""",
            llm_config=self.llm_config,
            description="A specialist in analyzing financial data and technical indicators to support stock selection"
        )
        
        # Create a user proxy agent that can execute functions with AG2 configuration
        user_proxy = UserProxyAgent(
            name="SelectionToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that."
        )
        
        # Register functions using AG2's register_function pattern
        self._register_functions(user_proxy)
        
        agents["user_proxy"] = user_proxy
        
        return agents
        
    def _register_functions(self, user_proxy: UserProxyAgent):
        """
        Register functions with the user proxy agent using AG2 patterns.
        
        Args:
            user_proxy: The user proxy agent to register functions with
        """
        # Register data gathering functions
        @register_function(
            name="get_market_data",
            description="Get initial universe of stocks",
            return_type=List[Dict[str, Any]]
        )
        def get_market_data() -> List[Dict[str, Any]]:
            return self.get_market_data()
            
        @register_function(
            name="get_technical_indicators",
            description="Calculate technical indicators for the stocks",
            parameters={
                "stocks": {
                    "type": "array",
                    "description": "List of stock dictionaries"
                }
            },
            return_type=List[Dict[str, Any]]
        )
        def get_technical_indicators(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.get_technical_indicators(stocks)
            
        @register_function(
            name="get_unusual_activity",
            description="Check for unusual activity in the stocks",
            parameters={
                "stocks": {
                    "type": "array",
                    "description": "List of stock dictionaries"
                }
            },
            return_type=List[Dict[str, Any]]
        )
        def get_unusual_activity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.get_unusual_activity(stocks)
            
        # Register filtering and scoring functions
        @register_function(
            name="filter_by_liquidity",
            description="Apply liquidity filters to the universe of stocks",
            parameters={
                "stocks": {
                    "type": "array",
                    "description": "List of stock dictionaries"
                }
            },
            return_type=List[Dict[str, Any]]
        )
        def filter_by_liquidity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.filter_by_liquidity(stocks)
            
        @register_function(
            name="score_candidates",
            description="Score and rank the candidate stocks",
            parameters={
                "stocks": {
                    "type": "array",
                    "description": "List of stock dictionaries"
                }
            },
            return_type=List[Dict[str, Any]]
        )
        def score_candidates(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.score_candidates(stocks)
            
        # Register storage and retrieval functions
        @register_function(
            name="store_candidates",
            description="Store candidates in Redis",
            parameters={
                "candidates": {
                    "type": "array",
                    "description": "List of candidate dictionaries"
                }
            },
            return_type=bool
        )
        def store_candidates(candidates: List[Dict[str, Any]]) -> bool:
            return self.store_candidates(candidates)
            
        @register_function(
            name="get_candidates",
            description="Get the current list of candidates from Redis",
            return_type=List[Dict[str, Any]]
        )
        def get_candidates() -> List[Dict[str, Any]]:
            return self.get_candidates()
            
        @register_function(
            name="get_top_candidates",
            description="Get the top N candidates by score",
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return",
                    "default": 10
                }
            },
            return_type=List[Dict[str, Any]]
        )
        def get_top_candidates(limit: int = 10) -> List[Dict[str, Any]]:
            return self.get_top_candidates(limit)
            
        @register_function(
            name="get_candidate",
            description="Get a specific candidate by symbol",
            parameters={
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol"
                }
            },
            return_type=Optional[Dict[str, Any]]
        )
        def get_candidate(symbol: str) -> Optional[Dict[str, Any]]:
            return self.get_candidate(symbol)
            
        @register_function(
            name="get_market_context",
            description="Get current market context",
            return_type=Dict[str, Any]
        )
        def get_market_context() -> Dict[str, Any]:
            return self._get_market_context()
            
        # Register MCP tool access functions
        self._register_mcp_tool_access(user_proxy)
        
        # Register all functions with the user proxy agent
        user_proxy.register_function(get_market_data)
        user_proxy.register_function(get_technical_indicators)
        user_proxy.register_function(get_unusual_activity)
        user_proxy.register_function(filter_by_liquidity)
        user_proxy.register_function(score_candidates)
        user_proxy.register_function(store_candidates)
        user_proxy.register_function(get_candidates)
        user_proxy.register_function(get_top_candidates)
        user_proxy.register_function(get_candidate)
        user_proxy.register_function(get_market_context)
        
    def _register_mcp_tool_access(self, user_proxy: UserProxyAgent):
        """
        Register MCP tool access functions with the user proxy agent.
        
        Args:
            user_proxy: The user proxy agent to register functions with
        """
        @register_function(
            name="use_alpaca_tool",
            description="Use a tool provided by the Alpaca MCP server",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool"
                }
            },
            return_type=Any
        )
        def use_alpaca_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.alpaca_mcp.call_tool(tool_name, arguments)
            
        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool"
                }
            },
            return_type=Any
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)
            
        @register_function(
            name="use_polygon_rest_tool",
            description="Use a tool provided by the Polygon REST MCP server",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool"
                }
            },
            return_type=Any
        )
        def use_polygon_rest_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.polygon_rest_mcp.call_tool(tool_name, arguments)
            
        @register_function(
            name="use_polygon_ws_tool",
            description="Use a tool provided by the Polygon WS MCP server",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool"
                }
            },
            return_type=Any
        )
        def use_polygon_ws_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.polygon_ws_mcp.call_tool(tool_name, arguments)
            
        @register_function(
            name="use_unusual_whales_tool",
            description="Use a tool provided by the Unusual Whales MCP server",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute"
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool"
                }
            },
            return_type=Any
        )
        def use_unusual_whales_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.unusual_whales_mcp.call_tool(tool_name, arguments)
            
        @register_function(
            name="list_mcp_tools",
            description="List all available tools on an MCP server",
            parameters={
                "server_name": {
                    "type": "string",
                    "description": "Name of the MCP server (alpaca, redis, polygon_rest, polygon_ws, unusual_whales)"
                }
            },
            return_type=List[Dict[str, str]]
        )
        def list_mcp_tools(server_name: str) -> List[Dict[str, str]]:
            if server_name == "alpaca":
                return self.alpaca_mcp.list_tools()
            elif server_name == "redis":
                return self.redis_mcp.list_tools()
            elif server_name == "polygon_rest":
                return self.polygon_rest_mcp.list_tools()
            elif server_name == "polygon_ws":
                return self.polygon_ws_mcp.list_tools()
            elif server_name == "unusual_whales":
                return self.unusual_whales_mcp.list_tools()
            else:
                return [{"error": f"MCP server not found: {server_name}"}]
                
        # Register the MCP tool access functions
        user_proxy.register_function(use_alpaca_tool)
        user_proxy.register_function(use_redis_tool)
        user_proxy.register_function(use_polygon_rest_tool)
        user_proxy.register_function(use_polygon_ws_tool)
        user_proxy.register_function(use_unusual_whales_tool)
        user_proxy.register_function(list_mcp_tools)
    
    def run_selection_cycle(self) -> List[Dict[str, Any]]:
        """
        Run a complete selection cycle to identify trading candidates using AutoGen agents.
        
        Returns:
            List of candidate dictionaries with scores and metadata
        """
        self.logger.info("Starting selection cycle with AutoGen agents")
        
        try:
            # Check if market is open
            if not self.alpaca_mcp.is_market_open():
                self.logger.info("Market is closed, skipping selection cycle")
                return []
            
            # Get initial market context
            market_context = self._get_market_context()
            
            # Create the initial prompt for the agents
            initial_prompt = f"""
            # Stock Selection Task
            
            ## Market Context
            {json.dumps(market_context, indent=2)}
            
            ## Task
            Identify 10-20 promising stocks for day trading based on the following criteria:
            
            1. Price range: ${self.min_price} to ${self.max_price}
            2. Minimum average daily volume: {self.min_volume:,} shares
            3. Minimum relative volume: {self.min_relative_volume}x
            4. Maximum bid-ask spread: {self.max_spread_pct}%
            
            ## Process
            1. First, use the get_market_data() function to get an initial universe of stocks
            2. Then, use filter_by_liquidity() to apply liquidity filters
            3. Next, use get_technical_indicators() to calculate technical indicators
            4. Then, use get_unusual_activity() to check for unusual activity
            5. Use score_candidates() to score and rank the candidates
            6. Finally, store the results using store_candidates()
            
            For each candidate, provide:
            - Symbol
            - Current price
            - Volume metrics
            - Key technical indicators
            - Unusual activity signals
            - Score and ranking
            - Rationale for selection
            
            ## Output
            Provide a final list of 10-20 candidates ranked by score, with detailed analysis for each.
            """
            
            # Run the conversation between agents using AG2 patterns
            selection_assistant = self.agents["selection_assistant"]
            data_assistant = self.agents["data_assistant"]
            user_proxy = self.agents["user_proxy"]
            
            # Start the conversation with AG2 patterns
            user_proxy.initiate_chat(
                selection_assistant,
                message=initial_prompt,
                clear_history=True
            )
            
            # Get the results from Redis
            candidates = self.get_candidates()
            self.logger.info(f"Selection cycle completed with {len(candidates)} candidates")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"Error in selection cycle: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error in selection cycle: {e}", component="selection_model", action="selection_cycle_error", error=str(e))
            return []
    
    def _get_market_context(self) -> Dict[str, Any]:
        """
        Get current market context.
        
        Returns:
            Dictionary with market context information
        """
        context = {}
        
        try:
            # Get S&P 500 data
            spy_data = self.polygon_rest_mcp.get_last_trade("SPY")
            if spy_data and "results" in spy_data:
                context["spy_price"] = spy_data["results"]["p"]
            
            # Get VIX data
            vix_data = self.polygon_rest_mcp.get_last_trade("VIX")
            if vix_data and "results" in vix_data:
                context["vix"] = vix_data["results"]["p"]
            
            # Get market status
            clock = self.alpaca_mcp.get_market_hours()
            context["market_open"] = clock.get("is_open", False)
            
            # Get account information
            account = self.alpaca_mcp.get_account_info()
            context["buying_power"] = float(account.get("buying_power", 0))
            context["portfolio_value"] = float(account.get("portfolio_value", 0))
            
        except Exception as e:
            self.logger.error(f"Error getting market context: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error getting market context: {e}", component="selection_model", action="market_context_error", error=str(e))
        
        return context
    
    # Function map methods for the user proxy agent
    
    def get_market_movers(self, category: str = "gainers", limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top market movers (gainers or losers) using the Polygon REST MCP tool.
        """
        try:
            movers = self.polygon_rest_mcp.call_tool(
                "get_market_movers",
                {"category": category, "limit": limit}
            )
            if isinstance(movers, dict) and "results" in movers:
                return movers["results"]
            return movers if isinstance(movers, list) else []
        except Exception as e:
            self.logger.error(f"Error getting market movers: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error getting market movers: {e}", component="selection_model", action="market_movers_error", error=str(e))
            return []

    def get_market_data(self) -> List[Dict[str, Any]]:
        """
        Get initial universe of stocks, optionally including market movers.
        
        Returns:
            List of stock dictionaries with basic information
        """
        self.logger.info("Getting initial universe of stocks")
        try:
            # Get all tradable assets from Polygon
            assets = self.polygon_rest_mcp.get_all_tickers()
            stocks = [
                asset for asset in assets 
                if asset.get("type") == "CS"
                and self.min_price <= asset.get("price", 0) <= self.max_price
                and asset.get("market") == "stocks"
            ]
            # Optionally include market movers
            if self.config.get("include_market_movers", True):
                movers = self.get_market_movers(category="gainers", limit=10)
                mover_symbols = {m["ticker"] for m in movers if "ticker" in m}
                stocks = [s for s in stocks if s.get("ticker") not in mover_symbols] + movers
            self.logger.info(f"Found {len(stocks)} stocks in initial universe (including market movers)")
            return stocks
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error getting market data: {e}", component="selection_model", action="market_data_error", error=str(e))
            return []
    
    def filter_by_liquidity(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply liquidity filters to the universe of stocks.
        
        Args:
            stocks: List of stock dictionaries
            
        Returns:
            Filtered list of stock dictionaries
        """
        self.logger.info(f"Applying liquidity filters to {len(stocks)} stocks")
        filtered = []
        
        for stock in stocks:
            symbol = stock.get("ticker")
            
            try:
                # Get average daily volume (10-day)
                volume_data = self.polygon_rest_mcp.get_aggregate_bars(
                    symbol=symbol,
                    multiplier=1,
                    timespan="day",
                    from_date=(datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                    to_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if not volume_data or "results" not in volume_data:
                    continue
                    
                # Calculate average volume
                volumes = [bar.get("v", 0) for bar in volume_data.get("results", [])]
                if not volumes:
                    continue
                    
                avg_volume = sum(volumes) / len(volumes)
                
                # Get current day volume
                current_day_volume = volumes[-1] if volumes else 0
                
                # Calculate relative volume
                relative_volume = current_day_volume / avg_volume if avg_volume > 0 else 0
                
                # Get current quote for bid-ask spread
                quote = self.polygon_rest_mcp.get_last_quote(symbol)
                
                if not quote or "results" not in quote:
                    continue
                    
                bid = quote.get("results", {}).get("bp", 0)
                ask = quote.get("results", {}).get("ap", 0)
                
                # Calculate spread percentage
                spread_pct = (ask - bid) / bid * 100 if bid > 0 else float('inf')
                
                # Apply filters
                if (avg_volume >= self.min_volume and 
                    relative_volume >= self.min_relative_volume and 
                    spread_pct <= self.max_spread_pct):
                    
                    # Add liquidity metrics to stock data
                    stock["avg_volume"] = avg_volume
                    stock["current_volume"] = current_day_volume
                    stock["relative_volume"] = relative_volume
                    stock["spread_pct"] = spread_pct
                    
                    filtered.append(stock)
            except Exception as e:
                self.logger.error(f"Error filtering stock {symbol}: {e}")
                if self.monitor:
                    self.monitor.log_error(f"Error filtering stock {symbol}: {e}", component="selection_model", action="liquidity_filter_error", error=str(e), symbol=symbol)
        
        self.logger.info(f"After liquidity filters: {len(filtered)} stocks")
        return filtered
    
    def get_technical_indicators(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate technical indicators for the stocks.
        
        Args:
            stocks: List of stock dictionaries
            
        Returns:
            List of stock dictionaries with technical indicators
        """
        self.logger.info(f"Calculating technical indicators for {len(stocks)} stocks")
        result = []
        
        # Get SPY data for relative strength calculation
        try:
            spy_data = self.polygon_rest_mcp.get_aggregate_bars(
                symbol="SPY",
                multiplier=1,
                timespan="day",
                from_date=(datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
                to_date=datetime.now().strftime("%Y-%m-%d")
            )
            
            if not spy_data or "results" not in spy_data:
                self.logger.error("Failed to get SPY data for relative strength calculation")
                spy_return = 0
            else:
                spy_closes = [bar.get("c", 0) for bar in spy_data.get("results", [])]
                spy_return = (spy_closes[-1] / spy_closes[0] - 1) * 100 if spy_closes and len(spy_closes) > 1 else 0
        except Exception as e:
            self.logger.error(f"Error getting SPY data: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error getting SPY data: {e}", component="selection_model", action="spy_data_error", error=str(e))
            spy_return = 0
        
        for stock in stocks:
            symbol = stock.get("ticker")
            
            try:
                # Get historical data
                bars = self.polygon_rest_mcp.get_aggregate_bars(
                    symbol=symbol,
                    multiplier=1,
                    timespan="day",
                    from_date=(datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
                    to_date=datetime.now().strftime("%Y-%m-%d")
                )
                
                if not bars or "results" not in bars:
                    continue
                    
                results = bars.get("results", [])
                if len(results) < 14:  # Need at least 14 days for RSI
                    continue
                    
                # Extract price data
                closes = [bar.get("c", 0) for bar in results]
                highs = [bar.get("h", 0) for bar in results]
                lows = [bar.get("l", 0) for bar in results]
                
                # Calculate technical indicators
                
                # 1. RSI (14-day)
                deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                gains = [delta if delta > 0 else 0 for delta in deltas]
                losses = [-delta if delta < 0 else 0 for delta in deltas]
                avg_gain = sum(gains[-14:]) / 14
                avg_loss = sum(losses[-14:]) / 14
                rs = avg_gain / avg_loss if avg_loss > 0 else float('inf')
                rsi = 100 - (100 / (1 + rs)) if avg_loss > 0 else 100
                
                # 2. Moving Averages
                sma5 = sum(closes[-5:]) / 5 if len(closes) >= 5 else 0
                sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else 0
                
                # 3. ATR (14-day)
                tr_values = []
                for i in range(1, len(closes)):
                    tr1 = highs[i] - lows[i]
                    tr2 = abs(highs[i] - closes[i-1])
                    tr3 = abs(lows[i] - closes[i-1])
                    tr_values.append(max(tr1, tr2, tr3))
                atr = sum(tr_values[-14:]) / min(14, len(tr_values))
                
                # 4. Relative Strength vs SPY
                stock_return = (closes[-1] / closes[0] - 1) * 100 if closes and len(closes) > 1 else 0
                relative_strength = stock_return - spy_return
                
                # 5. Intraday price range
                if len(results) > 0:
                    latest_bar = results[-1]
                    intraday_range_pct = (latest_bar.get("h", 0) - latest_bar.get("l", 0)) / latest_bar.get("o", 1) * 100
                else:
                    intraday_range_pct = 0
                
                # 6. Volatility (standard deviation of returns)
                returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                volatility = np.std(returns) * 100 if returns else 0
                
                # Add technical indicators to stock data
                stock["rsi"] = rsi
                stock["sma5"] = sma5
                stock["sma20"] = sma20
                stock["atr"] = atr
                stock["relative_strength"] = relative_strength
                stock["intraday_range_pct"] = intraday_range_pct
                stock["volatility"] = volatility
                stock["ma_crossover"] = sma5 > sma20  # True if 5-day SMA is above 20-day SMA
                
                # Check for support/resistance levels
                significant_levels = []
                
                # Add recent highs as resistance
                for i in range(1, len(highs) - 1):
                    if i < len(highs) and highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                        significant_levels.append(highs[i])
                
                # Add recent lows as support
                for i in range(1, len(lows) - 1):
                    if i < len(lows) and lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                        significant_levels.append(lows[i])
                
                # Check if current price is near any significant level (within 2%)
                near_level = False
                for level in significant_levels:
                    if abs(closes[-1] - level) / level < 0.02:  # Within 2% of level
                        near_level = True
                        break
                
                stock["near_support_resistance"] = near_level
                
                result.append(stock)
            except Exception as e:
                self.logger.error(f"Error calculating technical indicators for {symbol}: {e}")
                if self.monitor:
                    self.monitor.log_error(f"Error calculating technical indicators for {symbol}: {e}", component="selection_model", action="technical_indicator_error", error=str(e), symbol=symbol)
        
        self.logger.info(f"Calculated technical indicators for {len(result)} stocks")
        return result
    
    def get_unusual_activity(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Check for unusual activity in the stocks.
        
        Args:
            stocks: List of stock dictionaries
            
        Returns:
            List of stock dictionaries with unusual activity flags
        """
        self.logger.info(f"Checking unusual activity for {len(stocks)} stocks")
        result = []
        
        for stock in stocks:
            symbol = stock.get("ticker")
            
            try:
                # 1. Check for unusual options activity
                options_data = self.unusual_whales_mcp.get_unusual_options(symbol)
                unusual_options = len(options_data) > 0 if options_data else False
                
                # 2. Check for volume spikes (already calculated in liquidity filters)
                volume_spike = stock.get("relative_volume", 0) >= 2.0  # 200% of average
                
                # 3. Check for block trades
                block_trades = self.polygon_rest_mcp.get_trades(
                    symbol=symbol,
                    timestamp=datetime.now().strftime("%Y-%m-%d")
                )
                
                large_blocks = []
                if block_trades and "results" in block_trades:
                    # Filter for large trades (>= 10,000 shares)
                    large_blocks = [
                        trade for trade in block_trades.get("results", [])
                        if trade.get("size", 0) >= 10000
                    ]
                
                # Add unusual activity data to stock
                stock["unusual_options"] = unusual_options
                stock["volume_spike"] = volume_spike
                stock["large_blocks"] = len(large_blocks)
                
                # Flag as having unusual activity if any of the checks are true
                stock["has_unusual_activity"] = unusual_options or volume_spike or len(large_blocks) > 0
                
                result.append(stock)
            except Exception as e:
                self.logger.error(f"Error checking unusual activity for {symbol}: {e}")
                if self.monitor:
                    self.monitor.log_error(f"Error checking unusual activity for {symbol}: {e}", component="selection_model", action="unusual_activity_error", error=str(e), symbol=symbol)
        
        self.logger.info(f"Checked unusual activity for {len(result)} stocks")
        return result
    
    def score_candidates(self, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score and rank the candidate stocks.
        
        Args:
            stocks: List of stock dictionaries
            
        Returns:
            List of stock dictionaries with scores, sorted by score
        """
        self.logger.info(f"Scoring {len(stocks)} candidates")
        
        for stock in stocks:
            try:
                score = 0
                
                # 1. Liquidity score (0-30 points)
                volume_score = min(30, stock.get("relative_volume", 0) * 10)
                score += volume_score
                
                # 2. Technical score (0-40 points)
                # RSI score - highest at 50 (middle)
                rsi = stock.get("rsi", 50)
                rsi_score = 10 - abs(rsi - 50) / 5
                
                # Trend score
                trend_score = 10 if stock.get("ma_crossover", False) else 0
                
                # Volatility score - reward moderate volatility
                volatility = stock.get("volatility", 0)
                volatility_score = 10 if 1.5 <= volatility <= 5 else 5 if 1 <= volatility <= 7 else 0
                
                # Support/resistance score
                sr_score = 10 if stock.get("near_support_resistance", False) else 0
                
                technical_score = rsi_score + trend_score + volatility_score + sr_score
                score += technical_score
                
                # 3. Unusual activity score (0-30 points)
                unusual_score = 0
                if stock.get("unusual_options", False):
                    unusual_score += 10
                if stock.get("volume_spike", False):
                    unusual_score += 10
                if stock.get("large_blocks", 0) > 0:
                    unusual_score += 10
                
                score += unusual_score
                
                # Add score to stock data
                stock["score"] = score
                stock["volume_score"] = volume_score
                stock["technical_score"] = technical_score
                stock["unusual_score"] = unusual_score
            except Exception as e:
                self.logger.error(f"Error scoring stock {stock.get('ticker')}: {e}")
                if self.monitor:
                    self.monitor.log_error(f"Error scoring stock {stock.get('ticker')}: {e}", component="selection_model", action="scoring_error", error=str(e), symbol=stock.get('ticker'))
                stock["score"] = 0
        
        # Sort by score in descending order
        result = sorted(stocks, key=lambda x: x.get("score", 0), reverse=True)
        
        # Apply diversity filter
        if len(result) > self.max_candidates:
            # Group by sector
            sectors = {}
            for candidate in result:
                sector = candidate.get("sector", "Unknown")
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(candidate)
            
            # Take top candidates from each sector
            diverse_result = []
            max_per_sector = max(2, self.max_candidates // len(sectors))
            
            for sector, sector_candidates in sectors.items():
                # Sort by score
                sector_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                # Take top N from each sector
                diverse_result.extend(sector_candidates[:max_per_sector])
            
            # Sort by score and limit to max_candidates
            diverse_result.sort(key=lambda x: x.get("score", 0), reverse=True)
            result = diverse_result[:self.max_candidates]
        else:
            # Just limit to max_candidates
            result = result[:self.max_candidates]
        
        self.logger.info(f"Scored and ranked {len(result)} candidates")
        return result
    
    def store_candidates(self, candidates: List[Dict[str, Any]]) -> bool:
        """
        Store candidates in Redis.
        
        Args:
            candidates: List of candidate dictionaries
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Storing {len(candidates)} candidates in Redis")
        
        try:
            # Store timestamp
            timestamp = datetime.now().isoformat()
            self.redis_mcp.set_value(f"{self.redis_prefix}last_update", timestamp)
            
            # Store candidates as JSON
            self.redis_mcp.set_json(f"{self.redis_prefix}candidates", candidates)
            
            # Store individual candidates with score as sorted set
            for candidate in candidates:
                symbol = candidate.get("ticker")
                score = candidate.get("score", 0)
                
                # Store candidate data
                self.redis_mcp.set_json(f"{self.redis_prefix}candidate:{symbol}", candidate)
                
                # Add to sorted set for ranking
                self.redis_mcp.add_to_sorted_set(f"{self.redis_prefix}ranked_candidates", symbol, score)
            
            return True
        except Exception as e:
            self.logger.error(f"Error storing candidates: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error storing candidates: {e}", component="selection_model", action="store_candidates_error", error=str(e))
            return False
    
    def get_candidates(self) -> List[Dict[str, Any]]:
        """
        Get the current list of candidates from Redis.
        
        Returns:
            List of candidate dictionaries
        """
        return self.redis_mcp.get_json(f"{self.redis_prefix}candidates") or []
    
    def get_candidate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific candidate by symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Candidate dictionary or None if not found
        """
        return self.redis_mcp.get_json(f"{self.redis_prefix}candidate:{symbol}")
    
    def get_top_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N candidates by score.
        
        Args:
            limit: Maximum number of candidates to return
            
        Returns:
            List of candidate dictionaries
        """
        # Get top symbols from sorted set
        top_symbols = self.redis_mcp.get_sorted_set(
            f"{self.redis_prefix}ranked_candidates",
            start=0,
            stop=limit-1,
            reverse=True,
            with_scores=False
        )
        
        # Get candidate data for each symbol
        result = []
        for symbol in top_symbols:
            candidate = self.get_candidate(symbol)
            if candidate:
                result.append(candidate)
        
        return result
        
    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get comprehensive selection data for the orchestrator.
        
        This method provides a convenient way for the AutoGenOrchestrator
        to access all relevant selection data in one call.
        
        Returns:
            Dictionary containing:
            - market_context: Current market context
            - candidates: List of all candidates
            - top_candidates: List of top candidates
            - last_update: Timestamp of last update
        """
        try:
            # Get market context
            market_context = self._get_market_context()
            
            # Get all candidates
            candidates = self.get_candidates()
            
            # Get top candidates
            top_candidates = self.get_top_candidates(limit=10)
            
            # Get last update timestamp
            last_update = self.redis_mcp.get_value(f"{self.redis_prefix}last_update") or ""
            
            return {
                "market_context": market_context,
                "candidates": candidates,
                "top_candidates": top_candidates,
                "last_update": last_update
            }
        except Exception as e:
            self.logger.error(f"Error getting selection data: {e}")
            if self.monitor:
                self.monitor.log_error(f"Error getting selection data: {e}", component="selection_model", action="get_selection_data_error", error=str(e))
            return {
                "market_context": {},
                "candidates": [],
                "top_candidates": [],
                "last_update": ""
            }
