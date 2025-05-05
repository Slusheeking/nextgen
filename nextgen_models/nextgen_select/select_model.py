"""
NextGen Stock Selection Model

This module implements an LLM-based stock selection model using AutoGen that identifies
potential trading candidates based on technical criteria, unusual activity, and day trading parameters.
The model uses MCP tools to access market data and trading functionality.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import autogen as autogen
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
    # FunctionCall is not available in autogen 0.9.0
)

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# MCP tools (Consolidated)
from mcp_tools.trading_mcp.trading_mcp import TradingMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP # Added TimeSeriesMCP for technical indicators/patterns
from mcp_tools.db_mcp.redis_mcp import RedisMCP

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
                - trading_config: Config for TradingMCP
                - financial_data_config: Config for FinancialDataMCP
                - time_series_config: Config for TimeSeriesMCP
                - max_candidates: Maximum number of candidates to select (default: 20)
                - min_price: Minimum stock price (default: 10.0)
                - max_price: Maximum stock price (default: 200.0)
                - min_volume: Minimum average daily volume (default: 500000)
                - min_relative_volume: Minimum relative volume (default: 1.5)
                - max_spread_pct: Maximum bid-ask spread percentage (default: 0.5)
                - max_risk_per_trade_pct: Maximum risk percentage per trade (default: 1.0)
                - default_confidence_level: Default confidence level for VaR (default: 0.95)
                - default_time_horizon: Default time horizon in days for VaR (default: 1)
                - redis_prefix: Prefix for Redis keys (default: "selection:")
                - llm_config: Configuration for the LLM
        """
        init_start_time = time.time()
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-select-model")

        # Initialize StockChartGenerator
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")

        # Initialize counters for selection model metrics
        self.selection_cycles_run = 0
        self.candidates_selected_count = 0
        self.candidates_filtered_count = 0
        self.candidates_stored_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.execution_errors = 0 # Errors during selection process


        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_select", "select_model_config.json")
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
        self.max_candidates = self.config.get("max_candidates", 20)
        self.min_price = self.config.get("min_price", 10.0)
        self.max_price = self.config.get("max_price", 200.0)
        self.min_volume = self.config.get("min_volume", 500000)
        self.min_relative_volume = self.config.get("min_relative_volume", 1.5)
        self.max_spread_pct = self.config.get("max_spread_pct", 0.5)
        self.max_risk_per_trade_pct = self.config.get("max_risk_per_trade_pct", 1.0)
        self.default_confidence_level = self.config.get("default_confidence_level", 0.95)
        self.default_time_horizon = self.config.get("default_time_horizon", 1)
        self.redis_prefix = self.config.get("redis_prefix", "selection:")

        # Default technical indicators to calculate
        self.default_indicators = self.config.get("default_indicators", [
            {"name": "SMA", "params": {"period": 5}},
            {"name": "SMA", "params": {"period": 20}},
            {"name": "RSI", "params": {"period": 14}},
            {"name": "MACD", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}},
            {"name": "ATR", "params": {"period": 14}}
        ])

        # Initialize Consolidated MCP clients
        # TradingMCP handles Alpaca functionality
        self.trading_mcp = TradingMCP(
            self.config.get("trading_config")
        )
        # FinancialDataMCP handles various data sources like Polygon, Unusual Whales
        self.financial_data_mcp = FinancialDataMCP(
            self.config.get("financial_data_config")
        )
        # TimeSeriesMCP handles technical indicators and pattern recognition
        self.time_series_mcp = TimeSeriesMCP(
             self.config.get("time_series_config")
        )

        # Initialize Redis MCP client
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.redis_client = self.redis_mcp # Alias for backward compatibility if needed

        # Redis keys for data access and storage
        self.redis_keys = {
            "selection_candidates": f"{self.redis_prefix}candidates", # Key for the list of selected candidates
            "selection_candidate_prefix": f"{self.redis_prefix}candidate:", # Prefix for individual candidate data
            "selection_requests_stream": "model:selection:requests", # Stream for receiving selection requests
            "selection_candidates_stream": "model:selection:candidates", # Stream for publishing selected candidates
            "selection_feedback_stream_sentiment": "sentiment:selection_feedback", # Stream for sentiment feedback
            "selection_feedback_stream_market": "market_analysis:selection_feedback", # Stream for market analysis feedback
            "selection_feedback_stream_fundamental": "fundamental:selection_feedback", # Stream for fundamental analysis feedback
        }

        # Ensure Redis streams exist (optional, but good practice)
        try:
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_requests_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['selection_requests_stream']}' exists.")
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_candidates_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['selection_candidates_stream']}' exists.")
            # Ensure feedback streams exist (optional)
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_feedback_stream_sentiment"]})
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_feedback_stream_market"]})
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_feedback_stream_fundamental"]})
        except Exception as e:
            self.logger.warning(f"Could not ensure Redis streams exist: {e}")


        # Initialize LLM configuration
        self.llm_config = self._get_llm_config()

        # Initialize AutoGen agents
        self.agents = self._setup_agents()

        self.logger.info("SelectionModel initialized with AutoGen agents")

        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("selection_model.initialization_time_ms", init_duration)


    def _get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration for AutoGen.
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
        Initialize AutoGen agents for stock selection.
        """
        agents = {}

        # Create the stock selection assistant agent with AG2 configuration
        agents["selection_assistant"] = AssistantAgent(
            name="StockSelectionAssistant",
            system_message="""You are a stock selection specialist. Your role is to identify promising stocks for day trading based on technical indicators, liquidity, and unusual activity.

You have access to the following tools through MCP servers:
1. Trading MCP - For account information and trading
2. Financial Data MCP - For market data and unusual activity
3. Time Series MCP - For technical indicators and patterns

When selecting stocks, focus on:
- Technical criteria (momentum, volatility, liquidity)
- Unusual activity signals
- Day trading parameters
- Producing 10-20 quality candidates

Your selection process should include:
1. First, use the get_market_data() function to get an initial universe of stocks.
2. Then, use filter_by_liquidity() to apply liquidity filters.
3. Next, use get_technical_indicators() to calculate technical indicators.
4. Then, use get_unusual_activity() to check for unusual activity.
5. Use score_candidates() to score and rank the candidates.
6. Finally, store the results using store_candidates().

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
""",
            llm_config=self.llm_config,
            description="A specialist in identifying promising stocks for day trading based on technical indicators and market data",
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
            description="A specialist in analyzing financial data and technical indicators to support stock selection",
        )

        #
        # Create a user proxy agent that can execute functions with AG2
        # configuration
        user_proxy = UserProxyAgent(
            name="SelectionToolUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"use_docker": False},
            is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
            default_auto_reply="I'll help you with that.",
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
            return_type=List[Dict[str, Any]],
        )
        def get_market_data() -> List[Dict[str, Any]]:
            return self.get_market_data()

        @register_function(
            name="get_technical_indicators",
            description="Calculate technical indicators for the stocks",
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"}
            },
            return_type=List[Dict[str, Any]],
        )
        def get_technical_indicators(
            stocks: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            return self.get_technical_indicators(stocks)

        @register_function(
            name="get_unusual_activity",
            description="Check for unusual activity in the stocks",
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"}
            },
            return_type=List[Dict[str, Any]],
        )
        def get_unusual_activity(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return self.get_unusual_activity(stocks)

        # Register filtering and scoring functions
        @register_function(
            name="filter_by_liquidity",
            description="Apply liquidity filters to the universe of stocks, considering available capital",
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"},
                "available_capital": {"type": "number", "description": "Amount of capital available for new positions", "default": 0.0}
            },
            return_type=List[Dict[str, Any]],
        )
        def filter_by_liquidity(stocks: List[Dict[str, Any]], available_capital: float = 0.0) -> List[Dict[str, Any]]:
            return self.filter_by_liquidity(stocks, available_capital)

        @register_function(
            name="score_candidates",
            description="Score and rank the candidate stocks, factoring in capital fit",
            parameters={
                "stocks": {"type": "array", "description": "List of stock dictionaries"},
                "available_capital": {"type": "number", "description": "Amount of capital available for new positions", "default": 0.0}
            },
            return_type=List[Dict[str, Any]],
        )
        def score_candidates(stocks: List[Dict[str, Any]], available_capital: float = 0.0) -> List[Dict[str, Any]]:
            return self.score_candidates(stocks, available_capital)

        # Register storage and retrieval functions
        @register_function(
            name="store_candidates",
            description="Store candidates in Redis",
            parameters={
                "candidates": {
                    "type": "array",
                    "description": "List of candidate dictionaries",
                }
            },
            return_type=bool,
        )
        def store_candidates(candidates: List[Dict[str, Any]]) -> bool:
            return self.store_candidates(candidates)

        @register_function(
            name="get_candidates",
            description="Get the current list of candidates from Redis",
            return_type=List[Dict[str, Any]],
        )
        def get_candidates() -> List[Dict[str, Any]]:
            return self.get_candidates()

        @register_function(
            name="get_top_candidates",
            description="Get the top N candidates by score from Redis",
            parameters={
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of candidates to return",
                    "default": 10,
                }
            },
            return_type=List[Dict[str, Any]],
        )
        def get_top_candidates(limit: int = 10) -> List[Dict[str, Any]]:
            return self.get_top_candidates(limit)

        @register_function(
            name="get_candidate",
            description="Get a specific candidate by symbol from Redis",
            parameters={"symbol": {"type": "string", "description": "Stock symbol"}},
            return_type=Optional[Dict[str, Any]],
        )
        def get_candidate(symbol: str) -> Optional[Dict[str, Any]]:
            return self.get_candidate(symbol)

        @register_function(
            name="get_market_context",
            description="Get current market context",
            return_type=Dict[str, Any],
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

        # Define MCP tool access functions for consolidated MCPs
        @register_function(
            name="use_trading_tool",
            description="Use a tool provided by the Trading MCP server (for Alpaca functionality)",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool",
                },
            },
            return_type=Any,
        )
        def use_trading_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.trading_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="use_financial_data_tool",
            description="Use a tool provided by the Financial Data MCP server (for market data, unusual whales)",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool",
                },
            },
            return_type=Any,
        )
        def use_financial_data_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.financial_data_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="use_time_series_tool",
            description="Use a tool provided by the Time Series MCP server (for technical indicators and patterns)",
            parameters={
                "tool_name": {
                    "type": "string",
                    "description": "Name of the tool to execute",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments for the tool",
                },
            },
            return_type=Any,
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

    async def process_selection_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a selection request from the Decision Model.

        Args:
            request: Dictionary containing request details including:
                - available_capital: Amount of capital available for new positions
                - request_id: Unique identifier for this request
                - timestamp: When the request was made

        Returns:
            Dictionary with results of the selection process
        """
        self.logger.info(f"Processing selection request: {request.get('request_id')} with available capital: ${request.get('available_capital', 0):.2f}")
        try:
            # Run the selection cycle
            available_capital = request.get("available_capital", 0.0)
            selected_candidates = await self.run_selection_cycle(available_capital)

            # Prepare response
            response = {
                "request_id": request.get("request_id"),
                "status": "success",
                "candidates": selected_candidates,
                "timestamp": datetime.now().isoformat(),
                "message": f"Selection process completed. Found {len(selected_candidates)} candidates."
            }

            # Publish the selected candidates to a Redis stream for the Decision Model
            await self._publish_selected_candidates(selected_candidates)

            return response

        except Exception as e:
            self.logger.error(f"Error processing selection request {request.get('request_id')}: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return {
                "request_id": request.get("request_id"),
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "message": "Error during selection process."
            }


    async def process_selection_requests(self) -> int:
        """
        Process all pending selection requests from the Decision Model using Redis Stream.

        Returns:
            Number of requests processed
        """
        self.logger.info("Checking for pending selection requests...")
        processed_count = 0
        try:
            # Read from the selection requests stream
            # This would typically be done by a separate monitoring process or async task
            # For demonstration, we'll just attempt to read a few messages
            self.mcp_tool_call_count += 1
            read_result = self.redis_mcp.call_tool(
                "xread", # Assuming xread tool exists for streams
                {
                    "streams": [self.redis_keys["selection_requests_stream"]],
                    "count": 10, # Read up to 10 messages at a time
                    "block": 100 # Block for 0.1 seconds if no messages
                }
            )

            if read_result and not read_result.get("error"):
                messages = read_result.get("messages", [])
                if messages:
                    self.logger.info(f"Received {len(messages)} messages from selection requests stream.")
                    for stream_name, stream_messages in messages:
                        for message_id, message_data in stream_messages:
                            self.logger.info(f"Processing selection request message {message_id} from {stream_name}: {message_data}")
                            # Process the request
                            await self.process_selection_request(message_data)
                            processed_count += 1
                            # Acknowledge processed messages (optional but good practice for streams)
                            # self.redis_mcp.call_tool("xack", {"stream": stream_name, "group": "some_group", "ids": [message_id]})

            elif read_result and read_result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.error(f"Error reading from selection requests stream: {read_result.get('error')}")

            else:
                 self.logger.info("No new messages in selection requests stream.")


        except Exception as e:
            self.logger.error(f"Error processing selection requests: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")

        self.logger.info(f"Processed {processed_count} selection requests.")
        return processed_count


    async def run_selection_cycle(self, available_capital: float = 0.0) -> List[Dict[str, Any]]:
        """
        Run a complete selection cycle to identify trading candidates using AutoGen agents.

        Args:
            available_capital: Optional amount of capital available for new positions.
                                If provided, will be used to constrain selection.

        Returns:
            List of candidate dictionaries with scores and metadata
        """
        self.logger.info(f"Starting selection cycle with AutoGen agents (available capital: ${available_capital:.2f})")
        self.selection_cycles_run += 1
        start_time = time.time()

        try:
            # Check if market is open (using TradingMCP tool)
            self.mcp_tool_call_count += 1
            market_status = self.trading_mcp.call_tool("is_market_open", {})
            if market_status.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error checking market status: {market_status['error']}")
                 # Decide how to handle market status check failure - skip cycle?
                 self.logger.warning("Failed to check market status, skipping selection cycle.")
                 return []

            if not market_status.get("is_open", False):
                self.logger.info("Market is closed, skipping selection cycle")
                return []

            # Get initial market context (using consolidated MCPs)
            market_context = self._get_market_context()

            # Create the initial prompt for the agents
            initial_prompt = f"""
            # Stock Selection Task

            ## Market Context
            {json.dumps(market_context, indent=2)}

            ## Available Capital
            ${available_capital:.2f} is available for new positions. Consider this when selecting candidates.

            ## Task
            Identify 10-20 promising stocks for day trading based on the following criteria:

            1. Price range: ${self.min_price} to ${self.max_price}
            2. Minimum average daily volume: {self.min_volume:,} shares
            3. Minimum relative volume: {self.min_relative_volume}x
            4. Maximum bid-ask spread: {self.max_spread_pct}%
            5. Capital appropriateness: Stocks should be suitable for the available capital (${available_capital:.2f})

            ## Process
            1. First, use the get_market_data() function to get an initial universe of stocks
            2. Then, use filter_by_liquidity() to apply liquidity filters, considering available capital
            3. Next, use get_technical_indicators() to calculate technical indicators
            4. Then, use get_unusual_activity() to check for unusual activity
            5. Use score_candidates() to score and rank the candidates, factoring in capital fit
            6. Finally, store the results using store_candidates()

            For each candidate, provide:
            - Symbol
            - Current price
            - Volume metrics
            - Key technical indicators
            - Unusual activity signals
            - Capital fit assessment (how well the stock fits our available capital)
            - Score and ranking
            - Rationale for selection

            ## Output
            Provide a final list of 10-20 candidates ranked by score, with detailed analysis for each.
            Ensure the selected candidates are appropriate for the available capital of ${available_capital:.2f}.
            """

            # Run the conversation between agents using AG2 patterns
            selection_assistant = self.agents["selection_assistant"]
            data_assistant = self.agents["data_assistant"]
            user_proxy = self.agents["user_proxy"]

            # Start the conversation with AG2 patterns
            user_proxy.initiate_chat(
                selection_assistant, message=initial_prompt, clear_history=True
            )

            # Get the results from Redis
            candidates = self.get_candidates() # This now uses the corrected Redis call

            self.logger.info(
                f"Selection cycle completed with {len(candidates)} candidates"
            )
            self.candidates_selected_count += len(candidates)
            self.logger.gauge("selection_model.candidates_selected", self.candidates_selected_count)


            # Publish the selected candidates to a Redis stream for the Decision Model
            await self._publish_selected_candidates(candidates)


            duration = (time.time() - start_time) * 1000
            self.logger.timing("selection_model.selection_cycle_duration_ms", duration)

            return candidates

        except Exception as e:
            self.logger.error(f"Error in selection cycle: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            # if self.monitor: # Monitor is not initialized in this class
            #     self.monitor.log_error(
            #         f"Error in selection cycle: {e}",
            #         component="selection_model",
            #         action="selection_cycle_error",
            #         error=str(e),
            #     )
            return []

    def _get_market_context(self) -> Dict[str, Any]:
        """
        Get current market context using consolidated MCPs.
        """
        context = {}

        try:
            # Get S&P 500 data (using FinancialDataMCP tool)
            self.mcp_tool_call_count += 1
            spy_data = self.financial_data_mcp.call_tool("get_latest_trade", {"symbol": "SPY"})
            if spy_data and not spy_data.get("error"):
                context["spy_price"] = spy_data.get("price")
            elif spy_data and spy_data.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting SPY data: {spy_data['error']}")


            # Get VIX data (using FinancialDataMCP tool)
            self.mcp_tool_call_count += 1
            vix_data = self.financial_data_mcp.call_tool("get_latest_trade", {"symbol": "VIX"})
            if vix_data and not vix_data.get("error"):
                context["vix"] = vix_data.get("price")
            elif vix_data and vix_data.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting VIX data: {vix_data['error']}")


            # Get market status (using TradingMCP tool)
            self.mcp_tool_call_count += 1
            clock = self.trading_mcp.call_tool("get_clock", {})
            if clock and not clock.get("error"):
                 context["market_open"] = clock.get("is_open", False)
            elif clock and clock.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting market clock: {clock['error']}")


            # Get account information (using TradingMCP tool)
            self.mcp_tool_call_count += 1
            account = self.trading_mcp.call_tool("get_account_info", {})
            if account and not account.get("error"):
                 context["buying_power"] = float(account.get("buying_power", 0))
                 context["portfolio_value"] = float(account.get("portfolio_value", 0))
            elif account and account.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting account info: {account['error']}")


        except Exception as e:
            self.logger.error(f"Error getting market context: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            # if self.monitor: # Monitor is not initialized in this class
            #     self.monitor.log_error(
            #         f"Error getting market context: {e}",
            #         component="selection_model",
            #         action="market_context_error",
            #         error=str(e),
            #     )

        return context

    # Function map methods for the user proxy agent

    def get_market_movers(
        self, category: str = "gainers", limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top market movers (gainers or losers) using the FinancialDataMCP tool.
        """
        try:
            # Use FinancialDataMCP tool
            self.mcp_tool_call_count += 1
            movers_result = self.financial_data_mcp.call_tool(
                "get_market_movers", {"category": category, "limit": limit}
            )
            if movers_result and not movers_result.get("error"):
                 return movers_result.get("movers", []) # Assuming 'movers' key in result
            elif movers_result and movers_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting market movers: {movers_result['error']}")
                 return []
            return [] # Return empty list on failure
        except Exception as e:
            self.logger.error(f"Error getting market movers: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            # if self.monitor: # Monitor is not initialized in this class
            #     self.monitor.log_error(
            #         f"Error getting market movers: {e}",
            #         component="selection_model",
            #         action="market_movers_error",
            #         error=str(e),
            #     )
            return []

    def get_market_data(self) -> List[Dict[str, Any]]:
        """
        Get initial universe of stocks using FinancialDataMCP.

        Returns:
            List of stock dictionaries with basic information
        """
        self.logger.info("Getting initial universe of stocks")
        try:
            # Get all tradable assets from FinancialDataMCP
            # Assuming FinancialDataMCP has a tool like 'get_all_tickers' or 'get_assets'
            self.mcp_tool_call_count += 1
            assets_result = self.financial_data_mcp.call_tool("get_all_tickers", {}) # Or "get_assets"
            if assets_result and not assets_result.get("error"):
                 assets = assets_result.get("tickers", []) # Assuming 'tickers' or 'assets' key
            elif assets_result and assets_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error getting all tickers: {assets_result['error']}")
                 assets = []
            else:
                 assets = [] # Default to empty list on failure


            stocks = [
                asset
                for asset in assets
                if asset.get("type") == "CS" # Assuming 'type' key exists
                and self.min_price <= asset.get("price", 0) <= self.max_price # Assuming 'price' key exists
                and asset.get("market") == "stocks" # Assuming 'market' key exists
            ]
            # Optionally include market movers
            if self.config.get("include_market_movers", True):
                movers = self.get_market_movers(category="gainers", limit=10)
                mover_symbols = {m["ticker"] for m in movers if "ticker" in m} # Assuming 'ticker' key exists
                stocks = [
                    s for s in stocks if s.get("ticker") not in mover_symbols # Assuming 'ticker' key exists
                ] + movers
            self.logger.info(
                f"Found {len(stocks)} stocks in initial universe (including market movers)"
            )
            return stocks
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            # if self.monitor: # Monitor is not initialized in this class
            #     self.monitor.log_error(
            #         f"Error getting market data: {e}",
            #         component="selection_model",
            #         action="market_data_error",
            #         error=str(e),
            #     )
            return []

    def filter_by_liquidity(self, stocks: List[Dict[str, Any]], available_capital: float = 0.0) -> List[Dict[str, Any]]:
        """
        Apply liquidity filters to the universe of stocks using FinancialDataMCP.
        Also considers available capital to filter appropriate candidates.

        Args:
            stocks: List of stock dictionaries
            available_capital: Amount of capital available for new positions

        Returns:
            Filtered list of stock dictionaries
        """
        self.logger.info(f"Applying liquidity filters to {len(stocks)} stocks with available capital: ${available_capital:.2f}")
        filtered = []

        # Calculate capital-based thresholds
        # If available capital is low, we'll be more selective about price ranges
        min_price_adjusted = self.min_price
        max_price_adjusted = self.max_price
        
        # Adjust price range based on available capital
        if available_capital > 0:
            # For very low capital, focus on lower-priced stocks
            if available_capital < 5000:
                max_price_adjusted = min(self.max_price, available_capital / 50)  # Allow for at least 50 shares
                self.logger.info(f"Low capital adjustment: max price set to ${max_price_adjusted:.2f}")
            # For medium capital, use standard range but with a reasonable upper limit
            elif available_capital < 25000:
                max_price_adjusted = min(self.max_price, available_capital / 100)  # Allow for at least 100 shares
                self.logger.info(f"Medium capital adjustment: max price set to ${max_price_adjusted:.2f}")
            # For high capital, we can use the standard range

        for stock in stocks:
            symbol = stock.get("ticker")
            if not symbol: continue # Skip if no symbol

            try:
                # Get historical data for volume and price (using FinancialDataMCP tool)
                self.mcp_tool_call_count += 1
                bars_result = self.financial_data_mcp.call_tool(
                    "get_historical_bars",
                    {
                        "symbols": [symbol],
                        "timeframe": "1d",
                        "limit": 10 # Need at least 10 days for avg volume
                    }
                )

                if not bars_result or bars_result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Error getting historical bars for {symbol}: {bars_result.get('error') if bars_result else 'Unknown error'}")
                    continue # Skip this stock if data fetch fails

                bars = bars_result.get("data", {}).get(symbol, []) # Assuming structure {"data": {symbol: [...]}}
                if len(bars) < 10:
                    self.logger.warning(f"Not enough historical data for {symbol} to calculate average volume.")
                    continue # Need at least 10 days for avg volume

                # Extract volumes and closes
                volumes = [bar.get("volume", 0) for bar in bars]
                closes = [bar.get("close", 0) for bar in bars]

                # Calculate average volume (10-day)
                avg_volume = sum(volumes[-10:]) / 10 if len(volumes) >= 10 else 0

                # Get current day volume
                current_day_volume = volumes[-1] if volumes else 0

                # Calculate relative volume
                relative_volume = (
                    current_day_volume / avg_volume if avg_volume > 0 else 0
                )

                # Get current quote for bid-ask spread (using FinancialDataMCP tool)
                self.mcp_tool_call_count += 1
                quote_result = self.financial_data_mcp.call_tool("get_latest_quote", {"symbol": symbol})

                if not quote_result or quote_result.get("error"):
                    self.mcp_tool_error_count += 1
                    self.logger.error(f"Error getting latest quote for {symbol}: {quote_result.get('error') if quote_result else 'Unknown error'}")
                    continue # Skip this stock if quote fetch fails

                bid = quote_result.get("bid_price", 0) # Assuming keys 'bid_price' and 'ask_price'
                ask = quote_result.get("ask_price", 0)

                # Calculate spread percentage
                spread_pct = (ask - bid) / bid * 100 if bid > 0 else float("inf")

                # Calculate preliminary risk metrics (simplified version since we don't have full price data yet)
                # This is just a basic check before the more detailed risk assessment in get_technical_indicators
                price_volatility = 0
                if len(closes) > 1:
                    returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
                    price_volatility = np.std(returns) if returns else 0
                
                # Conservative VaR estimate based on volatility
                preliminary_var = price_volatility * 1.65  # 95% confidence level approximation
                preliminary_var = max(0.03, min(0.15, preliminary_var))  # Reasonable bounds
                
                # Check if risk per trade is acceptable based on estimated VaR
                # This prevents excessively volatile stocks from being selected
                risk_limit_ok = preliminary_var <= (self.max_risk_per_trade_pct / 100) * 3
                
                # Get current price for capital-based filtering
                current_price = closes[-1] if closes else 0
                
                # Calculate minimum lot size based on available capital
                # This ensures we can buy a reasonable number of shares
                min_lot_size = 0
                capital_appropriate = True
                
                if available_capital > 0 and current_price > 0:
                    # Calculate how many shares we could buy with available capital
                    max_shares = available_capital / current_price
                    
                    # Check if we can buy a reasonable number of shares
                    # For lower-priced stocks, we want more shares to make the trade worthwhile
                    if current_price < 20:
                        min_lot_size = 100  # Standard round lot for low-priced stocks
                    elif current_price < 50:
                        min_lot_size = 50   # Half round lot for medium-priced stocks
                    else:
                        min_lot_size = 10   # Smaller lot for high-priced stocks
                    
                    # Check if we can afford the minimum lot size
                    capital_appropriate = max_shares >= min_lot_size
                    
                    # Store capital-related metrics
                    stock["max_shares_affordable"] = int(max_shares)
                    stock["min_lot_size"] = min_lot_size
                    stock["capital_appropriate"] = capital_appropriate
                
                # Apply filters, including capital-based filters
                if (
                    avg_volume >= self.min_volume
                    and relative_volume >= self.min_relative_volume
                    and spread_pct <= self.max_spread_pct
                    and risk_limit_ok
                    and min_price_adjusted <= current_price <= max_price_adjusted
                    and capital_appropriate
                ):
                    # Add liquidity metrics to stock data
                    stock["avg_volume"] = avg_volume
                    stock["current_volume"] = current_day_volume
                    stock["relative_volume"] = relative_volume
                    stock["spread_pct"] = spread_pct
                    stock["current_price"] = current_price
                    stock["price_data"] = bars # Add price data for indicator calculation

                    filtered.append(stock)
                    self.candidates_filtered_count += 1

            except Exception as e:
                self.logger.error(f"Error filtering stock {symbol}: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("selection_model.execution_errors")
                # if self.monitor: # Monitor is not initialized in this class
                #     self.monitor.log_error(
                #         f"Error filtering stock {symbol}: {e}",
                #         component="selection_model",
                #         action="liquidity_filter_error",
                #         error=str(e),
                #         symbol=symbol,
                #     )

        self.logger.info(f"After liquidity filters (including capital-based): {len(filtered)} stocks")
        return filtered

    def get_technical_indicators(
        self, stocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate technical indicators for the stocks using TimeSeriesMCP.

        Args:
            stocks: List of stock dictionaries (expected to have 'price_data' or similar)

        Returns:
            List of stock dictionaries with technical indicators
        """
        self.logger.info(f"Calculating technical indicators for {len(stocks)} stocks")
        result = []

        # Get SPY data for relative strength calculation (using FinancialDataMCP tool)
        try:
            self.mcp_tool_call_count += 1
            spy_bars_result = self.financial_data_mcp.call_tool(
                "get_historical_bars",
                {
                    "symbols": ["SPY"],
                    "timeframe": "1d",
                    "limit": 20 # Need at least 20 days for relative strength
                }
            )

            if not spy_bars_result or spy_bars_result.get("error"):
                self.mcp_tool_error_count += 1
                self.logger.error(f"Error getting SPY data: {spy_bars_result.get('error') if spy_bars_result else 'Unknown error'}")
                spy_return = 0
            else:
                spy_bars = spy_bars_result.get("data", {}).get("SPY", [])
                spy_closes = [bar.get("close", 0) for bar in spy_bars]
                spy_return = (
                    (spy_closes[-1] / spy_closes[0] - 1) * 100
                    if spy_closes and len(spy_closes) > 1
                    else 0
                )
        except Exception as e:
            self.logger.error(f"Error getting SPY data: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            # if self.monitor: # Monitor is not initialized in this class
            #     self.monitor.log_error(
            #         f"Error getting SPY data: {e}",
            #         component="selection_model",
            #         action="spy_data_error",
            #         error=str(e),
            #     )
            spy_return = 0

        for stock in stocks:
            symbol = stock.get("ticker")
            if not symbol: continue # Skip if no symbol

            try:
                # Get historical data for indicators (using FinancialDataMCP tool)
                # Assuming stock dict already has 'price_data' from filter_by_liquidity or get_market_data
                # If not, need to fetch it here:
                # self.mcp_tool_call_count += 1
                # bars_result = self.financial_data_mcp.call_tool(...)
                # price_data = bars_result.get("data", {}).get(symbol, [])

                # Assuming price_data is already available in the stock dict
                price_data = stock.get("price_data", []) # Need to ensure this is populated earlier

                if not price_data or len(price_data) < 20: # Need enough data for indicators
                    self.logger.warning(f"Not enough price data for {symbol} to calculate indicators.")
                    continue

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


                # Calculate Relative Strength vs SPY (using price data)
                closes = [bar.get("close", 0) for bar in price_data]
                stock_return = (
                    (closes[-1] / closes[0] - 1) * 100
                    if closes and len(closes) > 1
                    else 0
                )
                relative_strength = stock_return - spy_return

                # Calculate Intraday price range (using latest bar)
                intraday_range_pct = 0
                if price_data:
                    latest_bar = price_data[-1]
                    intraday_range_pct = (
                        (latest_bar.get("high", 0) - latest_bar.get("low", 0))
                        / latest_bar.get("open", 1)
                        * 100
                    )

                # Calculate Volatility (standard deviation of returns)
                returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
                volatility = np.std(returns) * 100 if returns else 0

                # Check for MA crossover (assuming SMA5 and SMA20 are in indicators_data)
                sma5 = indicators_data.get("SMA", []) # Assuming list of values
                sma20 = indicators_data.get("SMA", []) # Need to handle different periods if not default
                ma_crossover = False
                if sma5 and sma20 and len(sma5) > 1 and len(sma20) > 1:
                     # Need to ensure correct SMA values are used based on periods
                     # This requires more detailed logic or a specific tool in TimeSeriesMCP
                     # For now, a simplified check assuming default periods are calculated
                     # This part might need refinement based on actual TimeSeriesMCP output structure
                     pass # Placeholder for MA crossover logic


                # Calculate risk metrics
                var, expected_shortfall = self.assess_risk(symbol, price_data)
                
                # Add technical indicators and risk metrics to stock data
                stock["indicators"] = indicators_data
                stock["support_resistance"] = support_resistance_data
                stock["relative_strength"] = relative_strength
                stock["intraday_range_pct"] = intraday_range_pct
                stock["volatility"] = volatility
                stock["ma_crossover"] = ma_crossover # Placeholder
                stock["var"] = var
                stock["expected_shortfall"] = expected_shortfall
                
                # Add risk assessment against trade size
                stock["max_position_size"] = self._calculate_max_position_size(var, expected_shortfall, closes[-1] if closes else 0)

                result.append(stock)

            except Exception as e:
                self.logger.error(
                    f"Error calculating technical indicators for {symbol}: {e}", exc_info=True
                )
                self.execution_errors += 1
                self.logger.counter("selection_model.execution_errors")
                # if self.monitor: # Monitor is not initialized in this class
                #     self.monitor.log_error(
                #         f"Error calculating technical indicators for {symbol}: {e}",
                #         component="selection_model",
                #         action="technical_indicator_error",
                #         error=str(e),
                #         symbol=symbol,
                #     )

        self.logger.info(f"Calculated technical indicators for {len(result)} stocks")
        return result

    def get_unusual_activity(
        self, stocks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check for unusual activity in the stocks using FinancialDataMCP.

        Args:
            stocks: List of stock dictionaries

        Returns:
            List of stock dictionaries with unusual activity flags
        """
        self.logger.info(f"Checking unusual activity for {len(stocks)} stocks")
        result = []

        for stock in stocks:
            symbol = stock.get("ticker")
            if not symbol: continue # Skip if no symbol

            try:
                # 1. Check for unusual options activity (using FinancialDataMCP tool)
                self.mcp_tool_call_count += 1
                options_result = self.financial_data_mcp.call_tool("get_unusual_options", {"symbol": symbol})
                unusual_options = False
                if options_result and not options_result.get("error"):
                     unusual_options = len(options_result.get("options_activity", [])) > 0 # Assuming 'options_activity' key
                elif options_result and options_result.get("error"):
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Error getting unusual options for {symbol}: {options_result['error']}")


                #
                # 2. Check for volume spikes (already calculated in liquidity
                # filters)
                volume_spike = stock.get("relative_volume", 0) >= 2.0  # 200% of average

                # 3. Check for block trades (using FinancialDataMCP tool)
                self.mcp_tool_call_count += 1
                trades_result = self.financial_data_mcp.call_tool(
                    "get_trades",
                    {"symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d")}
                )

                large_blocks = []
                if trades_result and not trades_result.get("error"):
                     # Filter for large trades (>= 10,000 shares)
                     large_blocks = [
                         trade
                         for trade in trades_result.get("trades", []) # Assuming 'trades' key
                         if trade.get("size", 0) >= 10000 # Assuming 'size' key
                     ]
                elif trades_result and trades_result.get("error"):
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Error getting trades for {symbol}: {trades_result['error']}")


                # Add unusual activity data to stock
                stock["unusual_options"] = unusual_options
                stock["volume_spike"] = volume_spike
                stock["large_blocks"] = len(large_blocks)

                # Flag as having unusual activity if any of the checks are true
                stock["has_unusual_activity"] = (
                    unusual_options or volume_spike or len(large_blocks) > 0
                )

                result.append(stock)
            except Exception as e:
                self.logger.error(f"Error checking unusual activity for {symbol}: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("selection_model.execution_errors")
                # if self.monitor: # Monitor is not initialized in this class
                #     self.monitor.log_error(
                #         f"Error checking unusual activity for {symbol}: {e}",
                #         component="selection_model",
                #         action="unusual_activity_error",
                #         error=str(e),
                #         symbol=symbol,
                #     )

        self.logger.info(f"Checked unusual activity for {len(result)} stocks")
        return result

    def score_candidates(self, stocks: List[Dict[str, Any]], available_capital: float = 0.0) -> List[Dict[str, Any]]:
        """
        Score and rank the candidate stocks, incorporating capital awareness.

        Args:
            stocks: List of stock dictionaries (expected to have indicator/activity data)
            available_capital: Amount of capital available for new positions

        Returns:
            List of stock dictionaries with scores, sorted by score
        """
        self.logger.info(f"Scoring {len(stocks)} candidates with available capital: ${available_capital:.2f}")

        for stock in stocks:
            try:
                score = 0

                # 1. Liquidity score (0-30 points)
                volume_score = min(30, stock.get("relative_volume", 0) * 10)
                score += volume_score

                # 2. Technical score (0-40 points)
                indicators = stock.get("indicators", {})
                # RSI score - highest at 50 (middle)
                rsi_values = indicators.get("RSI", []) # Assuming list of values
                rsi = rsi_values[-1] if rsi_values else 50 # Use last value if available
                rsi_score = 10 - abs(rsi - 50) / 5

                # Trend score (using MA crossover)
                trend_score = 10 if stock.get("ma_crossover", False) else 0

                # Volatility score - reward moderate volatility
                volatility = stock.get("volatility", 0)
                volatility_score = (
                    10 if 1.5 <= volatility <= 5 else 5 if 1 <= volatility <= 7 else 0
                )

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
                
                # 4. Risk score (0-20 points)
                # Reward lower risk (VaR and ES) with higher score
                var = stock.get("var", 0.05)  # Default to 5% if not calculated
                es = stock.get("expected_shortfall", 0.07)  # Default to 7% if not calculated
                
                # Score based on VaR - lower is better
                var_score = 10 - min(10, var * 100)  # 0% VaR = 10 points, 10%+ VaR = 0 points
                
                # Score based on ES - lower is better
                es_score = 10 - min(10, es * 50)  # 0% ES = 10 points, 20%+ ES = 0 points
                
                risk_score = max(0, var_score + es_score)
                score += risk_score
                
                # 5. Capital fit score (0-20 points) - NEW
                capital_fit_score = 0
                current_price = stock.get("current_price", 0)
                
                if available_capital > 0 and current_price > 0:
                    # Calculate how many shares we could buy with available capital
                    max_shares = available_capital / current_price
                    
                    # Calculate optimal position size based on risk metrics
                    max_position_size = stock.get("max_position_size", 100)
                    
                    # Calculate capital fit ratio - how well does our available capital match the optimal position size?
                    # A ratio of 1.0 means perfect fit, higher means we can buy more than optimal, lower means we can't buy enough
                    capital_fit_ratio = max_shares / max_position_size if max_position_size > 0 else 0
                    
                    # Score based on capital fit ratio
                    if capital_fit_ratio >= 0.8 and capital_fit_ratio <= 1.5:
                        # Ideal range: can buy 80-150% of optimal position size
                        capital_fit_score = 20
                    elif capital_fit_ratio >= 0.5 and capital_fit_ratio <= 2.0:
                        # Good range: can buy 50-200% of optimal position size
                        capital_fit_score = 15
                    elif capital_fit_ratio >= 0.3 and capital_fit_ratio <= 3.0:
                        # Acceptable range: can buy 30-300% of optimal position size
                        capital_fit_score = 10
                    elif capital_fit_ratio > 0:
                        # At least we can buy some shares
                        capital_fit_score = 5
                    
                    # Store capital fit metrics
                    stock["capital_fit_ratio"] = capital_fit_ratio
                    stock["capital_fit_score"] = capital_fit_score
                    
                    # Add capital fit score to total score
                    score += capital_fit_score
                
                # Add risk score to stock data
                stock["risk_score"] = risk_score

                # Add score to stock data
                stock["score"] = score
                stock["volume_score"] = volume_score
                stock["technical_score"] = technical_score
                stock["unusual_score"] = unusual_score
            except Exception as e:
                self.logger.error(f"Error scoring stock {stock.get('ticker')}: {e}", exc_info=True)
                self.execution_errors += 1
                self.logger.counter("selection_model.execution_errors")
                # if self.monitor: # Monitor is not initialized in this class
                #     self.monitor.log_error(
                #         f"Error scoring stock {stock.get('ticker')}: {e}",
                #         component="selection_model",
                #         action="scoring_error",
                #         error=str(e),
                #         symbol=stock.get("ticker"),
                #     )
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
            result = diverse_result[: self.max_candidates]
        else:
            # Just limit to max_candidates
            result = result[: self.max_candidates]

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
        try:
            # Add timestamp to the data
            timestamp = datetime.now().isoformat()
            data = {
                "candidates": candidates,
                "timestamp": timestamp,
                "count": len(candidates)
            }

            # Store the main list of candidates in Redis
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                "set_json",
                {"key": self.redis_keys["selection_candidates"], "value": data, "expiry": 86400} # 1 day expiration
            )

            # Store individual candidates by symbol for quick lookup
            for candidate in candidates:
                symbol = candidate.get("ticker")
                if symbol:
                    self.mcp_tool_call_count += 1
                    self.redis_mcp.call_tool(
                        "set_json",
                        {"key": f"{self.redis_keys['selection_candidate_prefix']}{symbol}", "value": {**candidate, "timestamp": timestamp}, "expiry": 86400} # 1 day expiration
                    )

            if result and not result.get("error"):
                self.logger.info(f"Stored {len(candidates)} candidates in Redis")
                self.candidates_stored_count += len(candidates)
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.warning(f"Failed to store candidates in Redis: {result.get('error') if result else 'Unknown error'}")
                return False
        except Exception as e:
            self.logger.error(f"Error storing candidates in Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return False

    def get_candidates(self) -> List[Dict[str, Any]]:
        """
        Get the current list of candidates from Redis.

        Returns:
            List of candidate dictionaries
        """
        try:
            self.mcp_tool_call_count += 1
            data_result = self.redis_mcp.call_tool("get_json", {"key": self.redis_keys["selection_candidates"]})
            data = data_result.get("value") if data_result and not data_result.get("error") else None

            if data and "candidates" in data:
                return data["candidates"]
            elif data_result and data_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.warning(f"Failed to get candidates from Redis: {data_result.get('error')}")
                 return []
            else:
                self.logger.warning("No candidates found in Redis")
                return []
        except Exception as e:
            self.logger.error(f"Error getting candidates from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return []

    def get_candidate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific candidate by symbol from Redis.

        Args:
            symbol: Stock symbol

        Returns:
            Candidate dictionary or None if not found
        """
        try:
            self.mcp_tool_call_count += 1
            candidate_result = self.redis_mcp.call_tool("get_json", {"key": f"{self.redis_keys['selection_candidate_prefix']}{symbol}"})
            candidate = candidate_result.get("value") if candidate_result and not candidate_result.get("error") else None

            if candidate:
                return candidate
            elif candidate_result and candidate_result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.warning(f"Failed to get candidate {symbol} from Redis: {candidate_result.get('error')}")
                 return None
            else:
                self.logger.warning(f"Candidate {symbol} not found in Redis")
                return None
        except Exception as e:
            self.logger.error(f"Error getting candidate {symbol} from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return None

    def get_top_candidates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top N candidates by score from Redis.

        Args:
            limit: Maximum number of candidates to return

        Returns:
            List of candidate dictionaries
        """
        try:
            candidates = self.get_candidates() # This uses the corrected Redis call
            if not candidates:
                return []

            # Sort by score in descending order and limit
            sorted_candidates = sorted(
                candidates,
                key=lambda x: x.get("score", 0),
                reverse=True
            )
            return sorted_candidates[:limit]
        except Exception as e:
            self.logger.error(f"Error getting top candidates from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return []

    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get comprehensive selection data for the orchestrator from Redis.

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

            # Get all candidates data from Redis
            self.mcp_tool_call_count += 1
            candidates_data_result = self.redis_mcp.call_tool("get_json", {"key": self.redis_keys["selection_candidates"]})
            candidates_data = candidates_data_result.get("value") if candidates_data_result and not candidates_data_result.get("error") else None

            candidates = candidates_data.get("candidates", []) if candidates_data else []
            last_update = candidates_data.get("timestamp", "") if candidates_data else ""

            # Get top candidates
            top_candidates = self.get_top_candidates(10) # This uses the corrected Redis call

            return {
                "market_context": market_context,
                "candidates": candidates,
                "top_candidates": top_candidates,
                "last_update": last_update
            }
        except Exception as e:
            self.logger.error(f"Error getting selection data from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return {
                "market_context": {},
                "candidates": [],
                "top_candidates": [],
                "last_update": "",
                "error": str(e)
            }

    async def _publish_selected_candidates(self, candidates: List[Dict[str, Any]]) -> bool:
        """
        Publish the list of selected candidates to a Redis stream for the Decision Model.

        Args:
            candidates: List of selected candidate dictionaries

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Publishing {len(candidates)} selected candidates to stream.")
        try:
            # Prepare the data to publish
            data_to_publish = {
                "candidates": candidates,
                "timestamp": datetime.now().isoformat(),
                "count": len(candidates)
            }

            # Publish to the selection candidates stream
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                 "xadd", # Using stream for candidates
                 {
                      "stream": self.redis_keys["selection_candidates_stream"],
                      "data": data_to_publish
                 }
            )

            if result and not result.get("error"):
                self.logger.info("Published selected candidates to stream.")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to publish selected candidates to stream: {result.get('error') if result else 'Unknown error'}")
                return False

        except Exception as e:
            self.logger.error(f"Error publishing selected candidates: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return False


    async def _listen_for_feedback(self):
        """
        Listen for feedback from other models via Redis streams.
        This runs as a separate async task to continuously monitor feedback streams.
        """
        self.logger.info("Starting listener for feedback streams...")
        streams_to_listen = [
            self.redis_keys["selection_feedback_stream_sentiment"],
            self.redis_keys["selection_feedback_stream_market"],
            self.redis_keys["selection_feedback_stream_fundamental"],
        ]
        consumer_group = "selection_model_group"
        consumer_name = f"selection_model_instance_{os.getpid()}"  # Unique consumer name
        
        # Create consumer groups if they don't exist
        for stream in streams_to_listen:
            try:
                self.mcp_tool_call_count += 1
                # Create the consumer group with ID 0-0 (from beginning of stream)
                # If the stream doesn't exist, mkstream=True will create it
                self.redis_mcp.call_tool(
                    "xgroup_create", 
                    {
                        "stream": stream, 
                        "group": consumer_group, 
                        "id": "0-0", 
                        "mkstream": True
                    }
                )
                self.logger.info(f"Created consumer group '{consumer_group}' for stream '{stream}'")
            except Exception as e:
                # Group may already exist, which is fine
                if "BUSYGROUP" not in str(e):
                    self.logger.warning(f"Error creating consumer group for {stream}: {e}")

        try:
            # Main feedback processing loop
            while True:
                try:
                    # Read new messages from streams using XREADGROUP
                    self.mcp_tool_call_count += 1
                    read_result = self.redis_mcp.call_tool(
                        "xreadgroup", 
                        {
                            "group": consumer_group,
                            "consumer": consumer_name,
                            "streams": streams_to_listen,
                            "ids": [">"]*len(streams_to_listen),  # ">" means new messages only
                            "count": 10,  # Process up to 10 messages at a time
                            "block": 5000  # Block for 5 seconds if no messages
                        }
                    )
                    
                    if read_result and not read_result.get("error"):
                        messages = read_result.get("messages", [])
                        if messages:
                            self.logger.info(f"Received {len(messages)} feedback messages")
                            for stream_name, stream_messages in messages:
                                for message_id, message_data in stream_messages:
                                    self.logger.info(f"Processing feedback from {stream_name}, message ID: {message_id}")
                                    
                                    # Process the feedback
                                    await self._process_feedback(stream_name, message_data)
                                    
                                    # Acknowledge the message
                                    self.mcp_tool_call_count += 1
                                    self.redis_mcp.call_tool(
                                        "xack", 
                                        {
                                            "stream": stream_name,
                                            "group": consumer_group,
                                            "ids": [message_id]
                                        }
                                    )
                    elif read_result and read_result.get("error"):
                        self.mcp_tool_error_count += 1
                        self.logger.error(f"Error reading from feedback streams: {read_result.get('error')}")
                
                except Exception as loop_err:
                    self.logger.error(f"Error in feedback processing loop: {loop_err}", exc_info=True)
                    self.execution_errors += 1
                    self.logger.counter("selection_model.feedback_processing_errors")
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            self.logger.info("Feedback listener task was cancelled")
        except Exception as e:
            self.logger.error(f"Fatal error in feedback listener: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")


    def _calculate_max_position_size(self, var: float, expected_shortfall: float, price: float) -> float:
        """
        Calculate the maximum position size based on risk metrics and risk per trade limit.
        
        Args:
            var: Value at Risk percentage (as decimal)
            expected_shortfall: Expected Shortfall percentage (as decimal)
            price: Current price of the asset
            
        Returns:
            Maximum position size in number of shares
        """
        try:
            # Get account information for equity calculation
            self.mcp_tool_call_count += 1
            account_info = self.trading_mcp.call_tool("get_account_info", {})
            
            if account_info and not account_info.get("error"):
                equity = float(account_info.get("portfolio_value", 0))
                
                # Calculate maximum position size based on risk per trade percentage
                max_risk_dollars = equity * (self.max_risk_per_trade_pct / 100)
                
                # Use the more conservative of VaR and ES
                risk_factor = max(var, expected_shortfall / 1.5)
                
                # Size position so that VaR dollar amount = max_risk_dollars
                if risk_factor > 0 and price > 0:
                    max_position_size = max_risk_dollars / (price * risk_factor)
                    
                    self.logger.info(f"Calculated max position size: {max_position_size:.0f} shares at ${price:.2f} (VaR: {var:.2%}, ES: {expected_shortfall:.2%})")
                    return max_position_size
                else:
                    self.logger.warning(f"Invalid risk factor ({risk_factor}) or price ({price}). Using conservative position size.")
                    return equity * 0.01 / price  # Conservative 1% of portfolio
            else:
                self.logger.warning(f"Could not get account info: {account_info.get('error') if account_info else 'No response'}")
                return 100  # Default conservative position size
                
        except Exception as e:
            self.logger.error(f"Error calculating max position size: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            return 100  # Default conservative position size
    
    def assess_risk(self, symbol: str, price_data: List[Dict[str, Any]], portfolio: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Calculate the Value at Risk (VaR) and Expected Shortfall (ES) for a candidate stock.
        
        Args:
            symbol: The stock symbol
            price_data: Historical price data for the stock
            portfolio: Optional portfolio weights for correlation-aware risk assessment
            
        Returns:
            Tuple of (VaR, Expected Shortfall) as percentages
        """
        self.logger.info(f"Assessing risk for {symbol}")
        
        try:
            # Extract close prices and calculate returns
            if not price_data or len(price_data) < 20:  # Need enough data for risk calculations
                self.logger.warning(f"Insufficient price data for {symbol} to calculate risk metrics")
                return 0.05, 0.07  # Default 5% VaR, 7% ES if insufficient data
                
            closes = [bar.get("close", 0) for bar in price_data]
            opens = [bar.get("open", 0) for bar in price_data]
            highs = [bar.get("high", 0) for bar in price_data]
            lows = [bar.get("low", 0) for bar in price_data]
            volumes = [bar.get("volume", 0) for bar in price_data]
            
            # Calculate daily returns
            returns = [closes[i] / closes[i-1] - 1 for i in range(1, len(closes))]
            
            if len(returns) < 10:
                self.logger.warning(f"Insufficient return data for {symbol} to calculate risk metrics")
                return 0.05, 0.07  # Default values
            
            # Calculate additional risk metrics
            # 1. Volatility (annualized)
            daily_volatility = np.std(returns)
            annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days
            
            # 2. Downside deviation (only negative returns)
            negative_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(negative_returns) if negative_returns else daily_volatility
            
            # 3. Maximum drawdown
            cumulative_returns = np.cumprod(np.array(returns) + 1)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns / running_max) - 1
            max_drawdown = abs(min(drawdowns))
            
            # 4. Intraday volatility
            intraday_ranges = [(h - l) / o for h, l, o in zip(highs, lows, opens) if o > 0]
            intraday_volatility = np.mean(intraday_ranges) if intraday_ranges else 0.02
            
            # 5. Volume-weighted price volatility
            if sum(volumes) > 0:
                vw_prices = [c * v for c, v in zip(closes, volumes)]
                vwap = sum(vw_prices) / sum(volumes)
                vw_volatility = np.std([c / vwap - 1 for c in closes])
            else:
                vw_volatility = daily_volatility
                
            # If we have a portfolio, calculate VaR contribution
            if portfolio and len(portfolio) > 0:
                # Create a dictionary with just this symbol having 100% weight
                single_asset_weight = {symbol: 1.0}
                
                # Call risk_analysis_mcp to calculate VaR
                self.mcp_tool_call_count += 1
                var_result = self.time_series_mcp.call_tool(
                    "calculate_portfolio_risk", 
                    {
                        "weights": single_asset_weight,
                        "price_data": {symbol: price_data},
                        "risk_measure": "var",
                        "confidence_level": self.default_confidence_level,
                        "time_horizon": self.default_time_horizon
                    }
                )
                
                # Call risk_analysis_mcp to calculate Expected Shortfall
                self.mcp_tool_call_count += 1
                es_result = self.time_series_mcp.call_tool(
                    "calculate_portfolio_risk", 
                    {
                        "weights": single_asset_weight,
                        "price_data": {symbol: price_data},
                        "risk_measure": "expected_shortfall",
                        "confidence_level": self.default_confidence_level,
                        "time_horizon": self.default_time_horizon
                    }
                )
                
                # Extract results or use fallback calculation
                if var_result and not var_result.get("error"):
                    var = var_result.get("var", 0.05)
                else:
                    self.mcp_tool_error_count += 1
                    self.logger.warning(f"Error calculating VaR for {symbol}: {var_result.get('error') if var_result else 'Unknown error'}")
                    # Fallback: Calculate historical VaR using return data
                    var = abs(np.percentile(returns, (1 - self.default_confidence_level) * 100))
                
                if es_result and not es_result.get("error"):
                    expected_shortfall = es_result.get("expected_shortfall", 0.07)
                else:
                    self.mcp_tool_error_count += 1
                    self.logger.warning(f"Error calculating ES for {symbol}: {es_result.get('error') if es_result else 'Unknown error'}")
                    # Fallback: Calculate historical ES using return data
                    var_threshold = np.percentile(returns, (1 - self.default_confidence_level) * 100)
                    tail_returns = [r for r in returns if r <= var_threshold]
                    expected_shortfall = abs(np.mean(tail_returns)) if tail_returns else 1.5 * var
                    
            else:
                # Enhanced historical VaR calculation using multiple methods
                
                # 1. Historical simulation method
                hist_var = abs(np.percentile(returns, (1 - self.default_confidence_level) * 100))
                
                # 2. Parametric method (assuming normal distribution)
                z_score = abs(np.percentile(np.random.normal(0, 1, 10000), (1 - self.default_confidence_level) * 100))
                param_var = z_score * daily_volatility
                
                # 3. Monte Carlo simulation
                try:
                    # Simple Monte Carlo using historical volatility
                    mean_return = np.mean(returns)
                    sim_returns = np.random.normal(mean_return, daily_volatility, 10000)
                    mc_var = abs(np.percentile(sim_returns, (1 - self.default_confidence_level) * 100))
                except Exception as mc_err:
                    self.logger.warning(f"Monte Carlo simulation failed: {mc_err}")
                    mc_var = hist_var
                
                # 4. EWMA (Exponentially Weighted Moving Average) volatility
                lambda_factor = 0.94  # Standard EWMA decay factor
                ewma_variance = 0
                for i, r in enumerate(reversed(returns[-20:])):  # Use last 20 returns
                    if i == 0:
                        ewma_variance = r * r
                    else:
                        ewma_variance = lambda_factor * ewma_variance + (1 - lambda_factor) * r * r
                ewma_volatility = np.sqrt(ewma_variance)
                ewma_var = z_score * ewma_volatility
                
                # Combine VaR estimates (weighted average)
                var = 0.4 * hist_var + 0.2 * param_var + 0.2 * mc_var + 0.2 * ewma_var
                
                # Enhanced Expected Shortfall calculation
                # 1. Historical ES
                var_threshold = np.percentile(returns, (1 - self.default_confidence_level) * 100)
                tail_returns = [r for r in returns if r <= var_threshold]
                hist_es = abs(np.mean(tail_returns)) if tail_returns else 1.5 * hist_var
                
                # 2. Parametric ES (for normal distribution)
                # ES = φ(Φ⁻¹(α)) / (1-α) * σ, where φ is PDF and Φ⁻¹ is inverse CDF of normal distribution
                from scipy.stats import norm
                try:
                    alpha = 1 - self.default_confidence_level
                    z_alpha = norm.ppf(alpha)
                    param_es = daily_volatility * norm.pdf(z_alpha) / alpha
                except Exception as es_err:
                    self.logger.warning(f"Parametric ES calculation failed: {es_err}")
                    param_es = 1.5 * param_var
                
                # Combine ES estimates
                expected_shortfall = 0.6 * hist_es + 0.4 * param_es
            
            # Create risk metrics dictionary to return
            risk_metrics = {
                "var": var,
                "expected_shortfall": expected_shortfall,
                "daily_volatility": daily_volatility,
                "annualized_volatility": annualized_volatility,
                "downside_deviation": downside_deviation,
                "max_drawdown": max_drawdown,
                "intraday_volatility": intraday_volatility,
                "volume_weighted_volatility": vw_volatility
            }
            
            self.logger.info(f"Risk assessment for {symbol}: VaR={var:.2%}, ES={expected_shortfall:.2%}, Vol={annualized_volatility:.2%}")
            return var, expected_shortfall
            
        except Exception as e:
            self.logger.error(f"Error assessing risk for {symbol}: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            # Return conservative default values
            return 0.05, 0.07  # Default 5% VaR, 7% ES
    
    async def _process_feedback(self, stream_name: str, feedback_data: Dict[str, Any]):
        """
        Process feedback received from other models and adjust selection parameters accordingly.
        
        Args:
            stream_name: The name of the stream the feedback came from
            feedback_data: Dictionary containing feedback information
        """
        self.logger.info(f"Processing feedback from {stream_name}: {feedback_data}")
        
        try:
            # Extract common feedback fields
            feedback_type = feedback_data.get("type", "unknown")
            symbol = feedback_data.get("symbol")
            timestamp = feedback_data.get("timestamp")
            source_model = feedback_data.get("source_model", "unknown")
            
            if not symbol:
                self.logger.warning(f"Received feedback without symbol: {feedback_data}")
                return
                
            # Process based on stream source
            if "sentiment" in stream_name:
                await self._process_sentiment_feedback(symbol, feedback_data)
            elif "market" in stream_name:
                await self._process_market_feedback(symbol, feedback_data)
            elif "fundamental" in stream_name:
                await self._process_fundamental_feedback(symbol, feedback_data)
            else:
                self.logger.warning(f"Unknown feedback stream: {stream_name}")
                
            # Update candidate in Redis if it exists
            candidate = await self._update_candidate_with_feedback(symbol, stream_name, feedback_data)
            
            # Log the feedback processing
            self.logger.info(f"Processed {feedback_type} feedback for {symbol} from {source_model}")
            
        except Exception as e:
            self.logger.error(f"Error processing feedback: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.feedback_processing_errors")
    
    async def _process_sentiment_feedback(self, symbol: str, feedback_data: Dict[str, Any]):
        """Process sentiment analysis feedback."""
        sentiment_score = feedback_data.get("sentiment_score", 0)
        sentiment_label = feedback_data.get("sentiment_label", "neutral")
        confidence = feedback_data.get("confidence", 0)
        
        # Only adjust parameters if confidence is high enough
        if confidence >= 0.7:
            # Store sentiment adjustment factors for scoring
            adjustment_key = f"sentiment_adjustment:{symbol}"
            
            # Calculate adjustment factor based on sentiment
            # Positive sentiment increases score, negative decreases
            if sentiment_label == "positive" and sentiment_score > 0.6:
                adjustment = min(1.2, 0.8 + sentiment_score/2)  # Max 20% boost
            elif sentiment_label == "negative" and sentiment_score < 0.4:
                adjustment = max(0.8, 1.2 - sentiment_score)  # Max 20% reduction
            else:
                adjustment = 1.0  # Neutral sentiment
                
            # Store the adjustment factor in Redis for future scoring
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                "set_json",
                {
                    "key": adjustment_key,
                    "value": {
                        "factor": adjustment,
                        "timestamp": datetime.now().isoformat(),
                        "source": feedback_data.get("source_model", "sentiment")
                    },
                    "expiry": 86400  # 1 day expiration
                }
            )
            
            self.logger.info(f"Set sentiment adjustment factor for {symbol}: {adjustment:.2f}")
    
    async def _process_market_feedback(self, symbol: str, feedback_data: Dict[str, Any]):
        """Process market analysis feedback."""
        # Extract market analysis metrics
        trend = feedback_data.get("trend")
        support_level = feedback_data.get("support_level")
        resistance_level = feedback_data.get("resistance_level")
        volume_analysis = feedback_data.get("volume_analysis")
        
        # Store market context for the symbol
        context_key = f"market_context:{symbol}"
        
        # Create market context object
        market_context = {
            "trend": trend,
            "support_level": support_level,
            "resistance_level": resistance_level,
            "volume_analysis": volume_analysis,
            "timestamp": datetime.now().isoformat(),
            "source": feedback_data.get("source_model", "market_analysis")
        }
        
        # Store in Redis
        self.mcp_tool_call_count += 1
        self.redis_mcp.call_tool(
            "set_json",
            {
                "key": context_key,
                "value": market_context,
                "expiry": 86400  # 1 day expiration
            }
        )
        
        # If near support level, this could be a good entry point
        if trend == "bullish" and support_level and "near_support" in feedback_data:
            # Boost the score for this symbol in future selections
            adjustment_key = f"market_adjustment:{symbol}"
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                "set_json",
                {
                    "key": adjustment_key,
                    "value": {
                        "factor": 1.15,  # 15% boost
                        "reason": "Near support level in bullish trend",
                        "timestamp": datetime.now().isoformat()
                    },
                    "expiry": 43200  # 12 hour expiration
                }
            )
            
        self.logger.info(f"Processed market feedback for {symbol}: trend={trend}")
    
    async def _process_fundamental_feedback(self, symbol: str, feedback_data: Dict[str, Any]):
        """Process fundamental analysis feedback."""
        # Extract fundamental metrics
        pe_ratio = feedback_data.get("pe_ratio")
        eps_growth = feedback_data.get("eps_growth")
        revenue_growth = feedback_data.get("revenue_growth")
        analyst_rating = feedback_data.get("analyst_rating")
        
        # Store fundamental context
        context_key = f"fundamental_context:{symbol}"
        
        # Create fundamental context object
        fundamental_context = {
            "pe_ratio": pe_ratio,
            "eps_growth": eps_growth,
            "revenue_growth": revenue_growth,
            "analyst_rating": analyst_rating,
            "timestamp": datetime.now().isoformat(),
            "source": feedback_data.get("source_model", "fundamental_analysis")
        }
        
        # Store in Redis
        self.mcp_tool_call_count += 1
        self.redis_mcp.call_tool(
            "set_json",
            {
                "key": context_key,
                "value": fundamental_context,
                "expiry": 604800  # 7 day expiration (fundamentals change more slowly)
            }
        )
        
        # Adjust selection parameters based on fundamental analysis
        if analyst_rating and analyst_rating.lower() in ["buy", "strong buy"]:
            # Boost the score for this symbol in future selections
            adjustment_key = f"fundamental_adjustment:{symbol}"
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                "set_json",
                {
                    "key": adjustment_key,
                    "value": {
                        "factor": 1.1,  # 10% boost
                        "reason": f"Positive analyst rating: {analyst_rating}",
                        "timestamp": datetime.now().isoformat()
                    },
                    "expiry": 604800  # 7 day expiration
                }
            )
            
        self.logger.info(f"Processed fundamental feedback for {symbol}: rating={analyst_rating}")
    
    async def _update_candidate_with_feedback(self, symbol: str, stream_name: str, feedback_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a candidate in Redis with feedback information.
        
        Args:
            symbol: The stock symbol
            stream_name: The name of the feedback stream
            feedback_data: The feedback data
            
        Returns:
            Updated candidate dictionary or None if not found
        """
        # Get the candidate from Redis
        candidate = self.get_candidate(symbol)
        if not candidate:
            return None
            
        # Add feedback to the candidate
        if "feedback" not in candidate:
            candidate["feedback"] = {}
            
        # Determine feedback category based on stream name
        if "sentiment" in stream_name:
            category = "sentiment"
        elif "market" in stream_name:
            category = "market"
        elif "fundamental" in stream_name:
            category = "fundamental"
        else:
            category = "other"
            
        # Add feedback to the appropriate category
        candidate["feedback"][category] = {
            "data": feedback_data,
            "timestamp": datetime.now().isoformat(),
            "stream": stream_name
        }
        
        # Update the candidate in Redis
        self.mcp_tool_call_count += 1
        self.redis_mcp.call_tool(
            "set_json",
            {
                "key": f"{self.redis_keys['selection_candidate_prefix']}{symbol}",
                "value": candidate,
                "expiry": 86400  # 1 day expiration
            }
        )
        
        return candidate


    async def start(self):
        """
        Start the SelectionModel, including processing requests and listening for feedback.
        """
        self.logger.info("Starting SelectionModel...")
        # Start the request processing loop (if running as a standalone service)
        # asyncio.create_task(self.process_selection_requests())

        # Start the feedback listener (if implemented)
        # asyncio.create_task(self._listen_for_feedback())

        self.logger.warning("SelectionModel processing loops are not started automatically. Call process_selection_requests() or run_selection_cycle() manually if needed.")


    def run_selection_agent(self, query: str) -> Dict[str, Any]:
        """
        Run stock selection using AutoGen agents.

        Args:
            query: Query or instruction for stock selection

        Returns:
            Results of the stock selection
        """
        self.logger.info(f"Running stock selection with query: {query}")
        start_time = time.time()

        selection_assistant = self.agents.get("selection_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not selection_assistant or not user_proxy:
            self.logger.error("AutoGen agents not initialized for run_selection_agent")
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("selection_model.run_selection_agent_duration_ms", duration, tags={"status": "failed", "reason": "agents_not_initialized"})
            return {"error": "AutoGen agents not initialized"}

        try:
            llm_call_start_time = time.time()
            user_proxy.initiate_chat(selection_assistant, message=query)
            llm_call_duration = (time.time() - llm_call_start_time) * 1000
            self.logger.timing("selection_model.llm_call_duration_ms", llm_call_duration)
            self.llm_api_call_count += 1

            # Get the last message from the assistant
            last_message = user_proxy.last_message(selection_assistant)
            content = last_message.get("content", "")

            # Extract structured data if possible
            try:
                # Find JSON blocks in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    result_str = content[json_start:json_end]
                    result = json.loads(result_str)
                    duration = (time.time() - start_time) * 1000
                    self.logger.timing("selection_model.run_selection_agent_duration_ms", duration, tags={"status": "success"})
                    return result
            except json.JSONDecodeError:
                # Return the raw content if JSON parsing fails
                self.logger.warning("Could not parse JSON analysis from agent response")
                duration = (time.time() - start_time) * 1000
                self.logger.timing("selection_model.run_selection_agent_duration_ms", duration, tags={"status": "success_non_json"})
                pass

            duration = (time.time() - start_time) * 1000
            self.logger.timing("selection_model.run_selection_agent_duration_ms", duration, tags={"status": "success_non_json"})
            return {"analysis": content}

        except Exception as e:
            self.logger.error(f"Error during AutoGen chat: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("selection_model.execution_errors")
            duration = (time.time() - start_time) * 1000
            self.logger.timing("selection_model.run_selection_agent_duration_ms", duration, tags={"status": "failed", "reason": "autogen_chat_error"})
            return {"error": str(e)}


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
