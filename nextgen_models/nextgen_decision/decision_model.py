"""
Decision Model for NextGen Trading System

This module implements the decision-making component of the NextGen Models system.
It aggregates data from all analysis models, applies risk management rules,
and makes final trading decisions.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import logging
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
)

# MCP tools
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.analysis_mcp.decision_analytics_mcp import DecisionAnalyticsMCP
from mcp_tools.analysis_mcp.portfolio_optimization_mcp import PortfolioOptimizationMCP
from mcp_tools.analysis_mcp.drift_detection_mcp import DriftDetectionMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.analysis_mcp.embeddings_mcp import EmbeddingsMCP
from mcp_tools.analysis_mcp.vector_db_mcp import VectorDBMCP
from mcp_tools.analysis_mcp.document_retrieval_mcp import DocumentRetrievalMCP




class PortfolioAnalyzer:
    pass


class RiskManager:
    pass


class MarketStateAnalyzer:
    pass


class DecisionModel:
    pass  # Add self-reference for type hinting


class DecisionModel:
    """
    Decision Model for the NextGen Trading System.

    Aggregates data from all analysis models, considers portfolio constraints
    and market conditions, applies risk management rules, and makes final
    trading decisions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Decision Model.

        Args:
            config: Optional configuration dictionary
        """
        # Initialize logging
        self.logger = logging.getLogger("nextgen_models.nextgen_decision")
        self.logger.setLevel(logging.INFO)

        # Add console handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize configuration
        self.config = config or {}

        # Initialize MCP clients
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.decision_analytics_mcp = DecisionAnalyticsMCP(
            self.config.get("decision_analytics_config")
        )
        self.portfolio_optimization_mcp = PortfolioOptimizationMCP(
            self.config.get("portfolio_optimization_config")
        )
        self.drift_detection_mcp = DriftDetectionMCP(
            self.config.get("drift_detection_config")
        )
        self.polygon_rest_mcp = PolygonRestMCP(self.config.get("polygon_rest_config"))

        # Initialize RAG-related MCP clients
        self.embeddings_mcp = EmbeddingsMCP(self.config.get("embeddings_config"))
        self.vector_db_mcp = VectorDBMCP(self.config.get("vector_db_config"))
        self.document_retrieval_mcp = DocumentRetrievalMCP(
            self.config.get(
                "document_retrieval_config",
                {
                    "embeddings_mcp_client": self.embeddings_mcp,
                    "vector_db_mcp_client": self.vector_db_mcp,
                },
            )
        )

        # Initialize model components
        self.portfolio_analyzer = PortfolioAnalyzer(self)
        self.risk_manager = RiskManager(self)
        self.market_state_analyzer = MarketStateAnalyzer(self)

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        # Redis keys for data access
        self.redis_keys = {
            "selection_data": "selection:data",
            "finnlp_data": "finnlp:data",
            "forecaster_data": "forecaster:data",
            "rag_data": "rag:data",
            "fundamental_data": "fundamental:data:",  # Prefix for fundamental data per symbol
            "portfolio_data": "portfolio:data",  # Key for portfolio summary
            "account_info": "portfolio:account_info",  # Key for account info
            "positions": "portfolio:positions",  # Key for list of positions
            "decisions": "decision:decisions",
            "decision_history": "decision:history",
            "position_updates": "trade:events",
        }

        # Decision thresholds
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.max_position_size_pct = self.config.get("max_position_size_pct", 5.0)
        self.risk_per_trade_pct = self.config.get("risk_per_trade_pct", 1.0)

        self.logger.info("Decision Model initialized")

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
        Initialize AutoGen agents for decision making.

        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the decision assistant agent
        agents["decision_assistant"] = AssistantAgent(
            name="DecisionAssistantAgent",
            system_message="""You are a trading decision specialist. Your role is to:
            1. Analyze data from multiple sources (selection, sentiment, forecasting, fundamental analysis, etc.)
            2. Consider portfolio constraints and risk management rules
            3. Make final trading decisions with confidence levels
            4. Provide clear reasoning for each decision

            You have tools for calculating confidence scores, optimizing position sizes,
            and analyzing portfolio impact. Always consider risk management first and
            provide detailed reasoning for your decisions.

            Pay special attention to fundamental analysis data, which provides insights on:
            - Financial health scores
            - Growth quality and sustainability
            - Valuation metrics
            - Sector comparisons""",
            llm_config=self.llm_config,
            description="A specialist in trading decision making",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="DecisionToolUser",
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
        decision_assistant = self.agents["decision_assistant"]

        # Define data access functions
        @register_function(
            name="get_selection_data",
            description="Get data from the Selection Model",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_selection_data() -> Dict[str, Any]:
            return self.get_selection_data()

        @register_function(
            name="get_finnlp_data",
            description="Get data from the FinNLP Model",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_finnlp_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            return self.get_finnlp_data(symbol)

        @register_function(
            name="get_forecaster_data",
            description="Get data from the Forecaster Model",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_forecaster_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            return self.get_forecaster_data(symbol)

        @register_function(
            name="get_rag_data",
            description="Get data from the RAG Model",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_rag_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            return self.get_rag_data(symbol)

        @register_function(
            name="get_fundamental_data",
            description="Get data from the Fundamental Analysis Model",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_fundamental_data(symbol: Optional[str] = None) -> Dict[str, Any]:
            return self.get_fundamental_data(symbol)

        @register_function(
            name="get_portfolio_data",
            description="Get current portfolio data",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_portfolio_data() -> Dict[str, Any]:
            return self.portfolio_analyzer.get_portfolio_data()

        # Define decision-making functions
        @register_function(
            name="calculate_confidence_score",
            description="Calculate confidence score from multiple signals",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def calculate_confidence_score(
            signals: Dict[str, Any], weights: Optional[Dict[str, float]] = None
        ) -> Dict[str, Any]:
            return self.calculate_confidence_score(signals, weights)

        @register_function(
            name="calculate_optimal_position_size",
            description="Calculate optimal position size based on confidence and portfolio constraints",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def calculate_optimal_position_size(
            symbol: str, confidence: float
        ) -> Dict[str, Any]:
            return self.calculate_optimal_position_size(symbol, confidence)

        @register_function(
            name="calculate_portfolio_impact",
            description="Calculate the impact of a new position on the portfolio",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def calculate_portfolio_impact(
            symbol: str, position_size: float
        ) -> Dict[str, Any]:
            return self.calculate_portfolio_impact(symbol, position_size)

        @register_function(
            name="evaluate_market_conditions",
            description="Evaluate current market conditions",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def evaluate_market_conditions() -> Dict[str, Any]:
            return self.market_state_analyzer.evaluate_market_conditions()

        @register_function(
            name="check_risk_limits",
            description="Check if a trade meets risk management criteria",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def check_risk_limits(
            symbol: str, position_size: float, action: str
        ) -> Dict[str, Any]:
            return self.risk_manager.check_risk_limits(symbol, position_size, action)

        @register_function(
            name="make_trade_decision",
            description="Make final trade decision with confidence levels",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def make_trade_decision(
            symbol: str, analysis_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            return self.make_trade_decision(symbol, analysis_data)

        @register_function(
            name="store_decision",
            description="Store a trading decision in Redis",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def store_decision(decision: Dict[str, Any]) -> bool:
            return self.store_decision(decision)

        @register_function(
            name="get_recent_decisions",
            description="Get recent trading decisions",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def get_recent_decisions(limit: int = 10) -> List[Dict[str, Any]]:
            return self.get_recent_decisions(limit)

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        decision_assistant = self.agents["decision_assistant"]

        # Define MCP tool access functions
        @register_function(
            name="use_decision_analytics_tool",
            description="Use a tool provided by the Decision Analytics MCP server",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_decision_analytics_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.decision_analytics_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_portfolio_optimization_tool",
            description="Use a tool provided by the Portfolio Optimization MCP server",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_portfolio_optimization_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.portfolio_optimization_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_drift_detection_tool",
            description="Use a tool provided by the Drift Detection MCP server",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_drift_detection_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.drift_detection_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

        # Register RAG-related MCP tool access functions
        @register_function(
            name="use_embeddings_tool",
            description="Use a tool provided by the Embeddings MCP server to generate vector embeddings for text",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_embeddings_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.embeddings_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_vector_db_tool",
            description="Use a tool provided by the Vector DB MCP server to store or query vector embeddings",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_vector_db_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.vector_db_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_document_retrieval_tool",
            description="Use a tool provided by the Document Retrieval MCP server to find relevant documents",
            caller=decision_assistant,
            executor=user_proxy,
        )
        def use_document_retrieval_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.document_retrieval_mcp.call_tool(tool_name, arguments)

    def process_analysis_results(
        self,
        selection_data: Dict[str, Any],
        finnlp_data: Dict[str, Any],
        forecaster_data: Dict[str, Any],
        rag_data: Dict[str, Any],
        fundamental_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process analysis results from all models to make decisions.

        Args:
            selection_data: Data from the Selection Model
            finnlp_data: Data from the FinNLP Model
            forecaster_data: Data from the Forecaster Model
            rag_data: Data from the RAG Model
            fundamental_data: Data from the Fundamental Analysis Model (optional)

        Returns:
            List of trading decisions
        """
        self.logger.info("Processing analysis results from all models")

        # Get candidates from selection data
        candidates = selection_data.get("candidates", [])
        if not candidates:
            self.logger.warning("No candidates found in selection data")
            return []

        # Get market conditions
        market_conditions = self.market_state_analyzer.evaluate_market_conditions()

        # Get portfolio constraints
        portfolio_data = self.portfolio_analyzer.get_portfolio_data()

        # Process each candidate
        decisions = []
        for candidate in candidates:
            symbol = candidate.get("symbol")
            if not symbol:
                continue

            # Gather all analysis data for this symbol
            symbol_analysis = {
                "selection": candidate,
                "finnlp": self._get_symbol_data(finnlp_data, symbol),
                "forecaster": self._get_symbol_data(forecaster_data, symbol),
                "rag": self._get_symbol_data(rag_data, symbol),
                "fundamental": self.get_fundamental_data(symbol)
                if fundamental_data is None
                else self._get_symbol_data(fundamental_data, symbol),
                "market_conditions": market_conditions,
            }

            # Make decision for this symbol
            decision = self.make_trade_decision(symbol, symbol_analysis)

            # Apply risk management
            if decision.get("action") != "hold":
                risk_check = self.risk_manager.check_risk_limits(
                    symbol, decision.get("position_size", 0), decision.get("action", "")
                )

                if not risk_check.get("approved", False):
                    self.logger.warning(
                        f"Trade for {symbol} rejected by risk management: {risk_check.get('reason')}"
                    )
                    decision["action"] = "hold"
                    decision["reason"] = f"Risk management: {risk_check.get('reason')}"
                else:
                    # Ensure position size is valid after risk check
                    if decision.get("position_size", 0) <= 0:
                        self.logger.warning(
                            f"Invalid position size (<=0) for {symbol} after risk check."
                        )
                        decision["action"] = "hold"
                        decision["reason"] = "Invalid position size after risk check"

            # Store the decision (even if 'hold')
            if decision:
                self.store_decision(decision)
                # Only add actionable decisions to the list to be executed
                if decision.get("action") != "hold":
                    decisions.append(decision)

        self.logger.info(f"Made {len(decisions)} actionable trading decisions")
        return decisions

    def _get_symbol_data(
        self, model_data: Dict[str, Any], symbol: str
    ) -> Dict[str, Any]:
        """
        Extract symbol-specific data from model data.

        Args:
            model_data: Data from a model
            symbol: Symbol to extract data for

        Returns:
            Symbol-specific data
        """
        # Check if model_data is a dictionary with symbols as keys
        if isinstance(model_data, dict) and "symbols" in model_data:
            symbols_data = model_data.get("symbols", {})
            return symbols_data.get(symbol, {})

        # Check if model_data is a list of dictionaries with symbol field
        elif isinstance(model_data, list):
            for item in model_data:
                if isinstance(item, dict) and item.get("symbol") == symbol:
                    return item

        return {}

    def make_trade_decision(
        self, symbol: str, analysis_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make a trade decision for a specific symbol using AutoGen agent.

        Args:
            symbol: Stock symbol
            analysis_data: Combined analysis data from all models

        Returns:
            Trading decision
        """
        self.logger.info(f"Making trade decision for {symbol} using AutoGen agent")

        decision_assistant = self.agents["decision_assistant"]
        user_proxy = self.agents["user_proxy"]

        # Prepare the prompt for the decision assistant
        prompt = f"""
        Analyze the following data for symbol {symbol} and make a trading decision (buy, sell, or hold).
        Provide confidence level (0-1) and reasoning. If buying, suggest an optimal position size.

        Analysis Data:
        {json.dumps(analysis_data, indent=2)}

        Use the available tools to:
        1. Calculate a combined confidence score.
        2. Evaluate market conditions.
        3. Analyze fundamental data (financial health, growth quality, valuation).
        4. If confidence is high enough, determine the action (buy/sell).
        5. If buying, calculate the optimal position size considering risk and portfolio constraints.
        6. Check risk limits for the proposed trade.
        7. Format the final decision clearly.

        Example Decision Format:
        {{
            "symbol": "{symbol}",
            "action": "buy/sell/hold",
            "confidence": 0.85,
            "position_size": 1000.0, // Only if action is 'buy'
            "reasoning": "Detailed explanation...",
            "timestamp": "YYYY-MM-DDTHH:MM:SS.ffffff"
        }}
        """

        # Initiate chat with the decision assistant
        try:
            user_proxy.initiate_chat(decision_assistant, message=prompt)
            # Get the last message from the assistant
            last_message = user_proxy.last_message(decision_assistant)
            content = last_message.get("content", "")

            # Attempt to parse the decision from the content
            try:
                # Find the JSON block in the response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start != -1 and json_end != -1:
                    decision_str = content[json_start:json_end]
                    decision = json.loads(decision_str)
                    # Add timestamp if missing
                    if "timestamp" not in decision:
                        decision["timestamp"] = datetime.now().isoformat()
                    # Add symbol if missing
                    if "symbol" not in decision:
                        decision["symbol"] = symbol
                    # Ensure position size is float
                    if "position_size" in decision:
                        decision["position_size"] = float(
                            decision.get("position_size", 0.0)
                        )
                    else:
                        # Default position size to 0 if not buying
                        if decision.get("action") != "buy":
                            decision["position_size"] = 0.0

                    self.logger.info(
                        f"Decision made for {symbol}: {decision.get('action')}"
                    )
                    return decision
                else:
                    self.logger.warning(
                        f"Could not parse JSON decision from agent response for {symbol}. Content: {content}"
                    )
                    return {
                        "symbol": symbol,
                        "action": "hold",
                        "reasoning": "Failed to parse agent decision",
                        "timestamp": datetime.now().isoformat(),
                    }

            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Error decoding JSON decision for {symbol}: {e}. Content: {content}"
                )
                return {
                    "symbol": symbol,
                    "action": "hold",
                    "reasoning": f"JSON decode error: {e}",
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            self.logger.error(f"Error during AutoGen chat for {symbol}: {e}")
            return {
                "symbol": symbol,
                "action": "hold",
                "reasoning": f"AutoGen chat error: {e}",
                "timestamp": datetime.now().isoformat(),
            }

    def calculate_confidence_score(
        self, signals: Dict[str, Any], weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate a confidence score from multiple signals.

        Args:
            signals: Dictionary of signals from different models
            weights: Optional dictionary of weights for each signal source

        Returns:
            Confidence score and breakdown
        """
        # Use the DecisionAnalyticsMCP to calculate confidence score
        return self.decision_analytics_mcp.calculate_confidence_score(
            signals, weights, self.confidence_threshold
        )

    def calculate_optimal_position_size(
        self, symbol: str, confidence: float
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on confidence and portfolio constraints.

        Args:
            symbol: Stock symbol
            confidence: Decision confidence (0-1)

        Returns:
            Optimal position size and reasoning
        """
        # Get portfolio data
        portfolio_data = self.portfolio_analyzer.get_portfolio_data()

        # Use the PortfolioOptimizationMCP to calculate position size
        return self.portfolio_optimization_mcp.calculate_optimal_position_size(
            symbol=symbol,
            confidence=confidence,
            portfolio_data=portfolio_data,
            max_position_pct=self.max_position_size_pct,
            risk_per_trade_pct=self.risk_per_trade_pct,
        )

    def calculate_portfolio_impact(
        self, symbol: str, position_size: float
    ) -> Dict[str, Any]:
        """
        Calculate the impact of a new position on the portfolio.

        Args:
            symbol: Stock symbol
            position_size: Proposed position size

        Returns:
            Portfolio impact metrics
        """
        # Get portfolio data
        portfolio_data = self.portfolio_analyzer.get_portfolio_data()

        # Get correlation data if available
        correlation_data = self.portfolio_analyzer.get_correlation_data()

        # Use the PortfolioOptimizationMCP to calculate portfolio impact
        return self.portfolio_optimization_mcp.calculate_portfolio_impact(
            symbol=symbol,
            position_size=position_size,
            portfolio_data=portfolio_data,
            correlation_data=correlation_data,
        )

    def store_decision(self, decision: Dict[str, Any]) -> bool:
        """
        Store a trading decision in Redis.

        Args:
            decision: Trading decision dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate a unique ID if not present
            if "id" not in decision:
                decision["id"] = f"decision:{decision.get('symbol')}:{int(time.time())}"

            # Store in Redis
            key = f"decision:{decision.get('id')}"
            self.redis_mcp.set_json(key, decision)

            # Add to decision history
            self.redis_mcp.add_to_sorted_set(
                "decision:history", decision.get("id"), int(time.time())
            )

            # Add to symbol-specific history
            symbol = decision.get("symbol")
            if symbol:
                self.redis_mcp.add_to_sorted_set(
                    f"decision:history:{symbol}", decision.get("id"), int(time.time())
                )

            return True
        except Exception as e:
            self.logger.error(f"Error storing decision: {e}")
            return False

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent trading decisions from Redis.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of recent trading decisions
        """
        try:
            # Get decision IDs from sorted set (newest first)
            decision_ids = self.redis_mcp.get_sorted_set_members(
                "decision:history", 0, limit - 1, reverse=True
            )

            if not decision_ids:
                return []

            # Get decision details for each ID
            decisions = []
            for decision_id in decision_ids:
                key = f"decision:{decision_id}"
                decision = self.redis_mcp.get_json(key)
                if decision:
                    decisions.append(decision)

            return decisions
        except Exception as e:
            self.logger.error(f"Error getting recent decisions: {e}")
            return []

    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get data from the Selection Model.

        Returns:
            Selection Model data
        """
        try:
            return self.redis_mcp.get_json(self.redis_keys["selection_data"]) or {}
        except Exception as e:
            self.logger.error(f"Error getting selection data: {e}")
            return {}

    def get_finnlp_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data from the FinNLP Model.

        Args:
            symbol: Optional symbol to filter data for

        Returns:
            FinNLP Model data
        """
        try:
            data = self.redis_mcp.get_json(self.redis_keys["finnlp_data"]) or {}

            if symbol and "symbols" in data:
                symbol_data = data.get("symbols", {}).get(symbol, {})
                return {"symbols": {symbol: symbol_data}} if symbol_data else {}

            return data
        except Exception as e:
            self.logger.error(f"Error getting FinNLP data: {e}")
            return {}

    def get_forecaster_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data from the Forecaster Model.

        Args:
            symbol: Optional symbol to filter data for

        Returns:
            Forecaster Model data
        """
        try:
            data = self.redis_mcp.get_json(self.redis_keys["forecaster_data"]) or {}

            if symbol and "symbols" in data:
                symbol_data = data.get("symbols", {}).get(symbol, {})
                return {"symbols": {symbol: symbol_data}} if symbol_data else {}

            return data
        except Exception as e:
            self.logger.error(f"Error getting Forecaster data: {e}")
            return {}

    def get_rag_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data from the RAG Model.

        Args:
            symbol: Optional symbol to filter data for

        Returns:
            RAG Model data
        """
        try:
            # First try to get data from Redis
            data = self.redis_mcp.get_json(self.redis_keys["rag_data"]) or {}

            if symbol and "symbols" in data:
                symbol_data = data.get("symbols", {}).get(symbol, {})
                if symbol_data:
                    return {"symbols": {symbol: symbol_data}}

            #
            # If no data in Redis or no symbol-specific data, try to retrieve
            # from DocumentRetrievalMCP
            if symbol and hasattr(self.document_retrieval_mcp, "retrieve_by_text"):
                try:
                    # Query for relevant documents about this symbol
                    query = f"latest information about {symbol} stock"
                    retrieval_result = self.document_retrieval_mcp.retrieve_by_text(
                        query_text=query,
                        top_k=5,
                        where_filter={"symbol": symbol} if symbol else None,
                    )

                    if retrieval_result and not retrieval_result.get("error"):
                        #
                        # Format the results for consistency with other data
                        # sources
                        results = retrieval_result.get("results", [])
                        context_data = {
                            "symbol": symbol,
                            "context_snippets": [
                                r.get("text_snippet")
                                for r in results
                                if r.get("text_snippet")
                            ],
                            "sources": [
                                r.get("metadata", {}).get("source")
                                for r in results
                                if r.get("metadata")
                            ],
                            "relevance_scores": [r.get("score") for r in results],
                            "timestamp": datetime.now().isoformat(),
                        }

                        # Store in Redis for future use
                        if not data.get("symbols"):
                            data["symbols"] = {}
                        data["symbols"][symbol] = context_data
                        self.redis_mcp.set_json(self.redis_keys["rag_data"], data)

                        return {"symbols": {symbol: context_data}}
                except Exception as e:
                    self.logger.error(
                        f"Error retrieving RAG data from DocumentRetrievalMCP: {e}"
                    )
                    # Fall back to Redis data or empty dict

            return data
        except Exception as e:
            self.logger.error(f"Error getting RAG data: {e}")
            return {}

    def handle_position_update(self, update_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle position updates (trades executed, positions closed).

        Args:
            update_data: Position update data

        Returns:
            Response with action taken
        """
        self.logger.info(f"Handling position update: {update_data.get('event_type')}")

        event_type = update_data.get("event_type")
        event_data = update_data.get("event_data", {})

        response = {
            "event_type": event_type,
            "action_taken": None,
            "timestamp": datetime.now().isoformat(),
        }

        # Handle different event types
        if event_type == "position_closed":
            symbol = event_data.get("symbol")
            reason = event_data.get("reason")

            self.logger.info(f"Position closed for {symbol}: {reason}")
            response["action_taken"] = "position_recorded"

        elif event_type == "capital_available":
            amount = event_data.get("amount", 0.0)

            if amount > 0:
                # Request new selections if significant capital is available
                if amount >= 1000:  # Minimum threshold for new selections
                    self.request_new_selections(amount)
                    response["action_taken"] = "new_selections_requested"
                else:
                    response["action_taken"] = "capital_noted"

        # Store the response in Redis
        try:
            key = f"decision:response:{int(time.time())}"
            self.redis_mcp.set_json(key, response)
        except Exception as e:
            self.logger.error(f"Error storing position update response: {e}")

        return response

    def request_new_selections(self, available_capital: float) -> Dict[str, Any]:
        """
        Request new stock selections when capital becomes available.

        Args:
            available_capital: Amount of capital available for new positions

        Returns:
            Request details
        """
        self.logger.info(
            f"Requesting new selections with {available_capital:.2f} available capital"
        )

        # Create selection request
        request = {
            "type": "selection_request",
            "available_capital": float(available_capital),
            "timestamp": datetime.now().isoformat(),
            "request_id": f"selection_request:{int(time.time())}",
        }

        # Store in Redis for the Selection Model to pick up
        try:
            key = f"selection:request:{int(time.time())}"
            self.redis_mcp.set_json(key, request)

            # Add to request stream
            self.redis_mcp.add_to_stream("selection:requests", request)

            return request
        except Exception as e:
            self.logger.error(f"Error requesting new selections: {e}")
            return {"error": str(e)}

    def get_fundamental_data(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get data from the Fundamental Analysis Model.

        Args:
            symbol: Optional symbol to filter data for

        Returns:
            Fundamental Analysis Model data
        """
        try:
            if symbol:
                # Get symbol-specific fundamental data
                key = f"{self.redis_keys['fundamental_data']}{symbol}"
                data = self.redis_mcp.get_json(key)
                if data:
                    return data

            #
            # If no symbol-specific data or no symbol provided, get all
            # fundamental data
            # This would require scanning Redis for keys with the prefix
            if symbol:
                self.logger.warning(f"No fundamental data found for symbol {symbol}")
                return {}

            # For getting all fundamental data (not implemented yet)
            self.logger.warning("Retrieving all fundamental data not implemented")
            return {}

        except Exception as e:
            self.logger.error(f"Error getting fundamental data: {e}")
            return {}


# --- Helper Component Classes ---


class PortfolioAnalyzer:
    """Analyzes current portfolio state."""

    def __init__(self, decision_model: DecisionModel):
        self.decision_model = decision_model
        self.logger = decision_model.logger

    def get_portfolio_data(self) -> Dict[str, Any]:
        """Get current portfolio data from Redis."""
        try:
            account_info = (
                self.decision_model.redis_mcp.get_json(
                    self.decision_model.redis_keys["account_info"]
                )
                or {}
            )
            positions = (
                self.decision_model.redis_mcp.get_json(
                    self.decision_model.redis_keys["positions"]
                )
                or []
            )
            portfolio_summary = (
                self.decision_model.redis_mcp.get_json(
                    self.decision_model.redis_keys["portfolio_data"]
                )
                or {}
            )

            # Combine relevant data
            portfolio_data = {
                "portfolio_value": float(account_info.get("portfolio_value", 0.0)),
                "buying_power": float(account_info.get("buying_power", 0.0)),
                "cash": float(account_info.get("cash", 0.0)),
                "equity": float(account_info.get("equity", 0.0)),
                "positions": positions,
                "sector_exposure": portfolio_summary.get("positions_by_sector", {}),
                "total_positions": portfolio_summary.get("total_positions", 0),
                "last_updated": portfolio_summary.get("timestamp"),
            }
            return portfolio_data
        except Exception as e:
            self.logger.error(f"Error getting portfolio data: {e}")
            return {}

    def get_correlation_data(self) -> Dict[str, Any]:
        """Get asset correlation data (placeholder)."""
        #
        # In a real implementation, this would fetch or calculate correlation
        # data
        self.logger.warning("Correlation data retrieval is not fully implemented.")
        return {}


class RiskManager:
    """Applies risk management rules."""

    def __init__(self, decision_model: DecisionModel):
        self.decision_model = decision_model
        self.logger = decision_model.logger
        self.max_position_pct = decision_model.max_position_size_pct
        self.risk_per_trade_pct = decision_model.risk_per_trade_pct

    def check_risk_limits(
        self, symbol: str, position_size: float, action: str
    ) -> Dict[str, Any]:
        """Check if a proposed trade meets risk limits."""
        portfolio_data = self.decision_model.portfolio_analyzer.get_portfolio_data()
        portfolio_value = portfolio_data.get("portfolio_value", 0.0)
        buying_power = portfolio_data.get("buying_power", 0.0)

        if portfolio_value <= 0:
            return {"approved": False, "reason": "Invalid portfolio value"}

        # Check max position size
        max_position_value = portfolio_value * (self.max_position_pct / 100.0)
        if position_size > max_position_value:
            return {
                "approved": False,
                "reason": f"Position size ({position_size:.2f}) exceeds max limit ({max_position_value:.2f})",
            }

        # Check buying power
        if action == "buy" and position_size > buying_power:
            return {
                "approved": False,
                "reason": f"Position size ({position_size:.2f}) exceeds buying power ({buying_power:.2f})",
            }

        #
        # Check risk per trade (simplified - assumes stop loss is set
        # elsewhere)
        # A more robust check would involve volatility and stop-loss distance
        max_risk_amount = portfolio_value * (self.risk_per_trade_pct / 100.0)
        # Assuming risk is proportional to position size for now
        if (
            position_size > max_risk_amount * 20
        ):  # Example: Max size is 20x max risk amount (5% risk)
            return {
                "approved": False,
                "reason": f"Position size ({position_size:.2f}) implies risk exceeding limit ({max_risk_amount:.2f})",
            }

        return {"approved": True, "reason": "Within risk limits"}


class MarketStateAnalyzer:
    """Evaluates overall market conditions."""

    def __init__(self, decision_model: DecisionModel):
        self.decision_model = decision_model
        self.logger = decision_model.logger

    def evaluate_market_conditions(self) -> Dict[str, Any]:
        """Evaluate market conditions using Polygon REST MCP."""
        try:
            # Get market status
            market_status = self.decision_model.polygon_rest_mcp.get_market_status()

            # Get VIX (using a proxy like VXX or ^VIX if available via Polygon)
            # Placeholder: Assume VIX is fetched somehow
            vix_value = 15.0  # Placeholder

            # Get SPY price (proxy for market direction)
            spy_data = self.decision_model.polygon_rest_mcp.get_stock_price("SPY")
            spy_price = (
                spy_data.get("results", [{}])[0].get("c")
                if spy_data.get("results")
                else None
            )

            # Determine volatility state
            volatility = "normal"
            if vix_value > 25:
                volatility = "high"
            elif vix_value < 15:
                volatility = "low"

            # Determine market state (simplified)
            market_state = "neutral"
            # Could use drift detection on SPY here
            #
            # drift_result =
            # self.decision_model.drift_detection_mcp.detect_ma_drift(...)
            #
            # if drift_result.get("trend") == "uptrend" or
            # drift_result.get("trend") == "downtrend":
            #     market_state = "trending"

            return {
                "is_market_open": market_status.get("market", "closed") == "open",
                "volatility_index": vix_value,
                "volatility_state": volatility,
                "market_state": market_state,
                "spy_price": spy_price,
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error evaluating market conditions: {e}")
            return {"error": str(e)}
