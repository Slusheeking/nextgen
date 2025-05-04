"""
Market Analysis Model

This module defines the MarketAnalysisModel, responsible for technical analysis
of market data using indicators, pattern recognition, and market scanning capabilities.
It integrates with AutoGen for advanced analysis and decision making.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import monitoring
from monitoring.system_monitor import MonitoringManager

# For Redis integration
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# MCP tools for technical analysis
from mcp_tools.analysis_mcp.technical_indicators_mcp import TechnicalIndicatorsMCP
from mcp_tools.analysis_mcp.pattern_recognition_mcp import PatternRecognitionMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWSMCP

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function


class MarketAnalysisModel:
    """
    Analyzes market data using technical indicators and pattern recognition.

    This model integrates TechnicalIndicatorsMCP and PatternRecognitionMCP to provide
    comprehensive market analysis, including indicator calculations, pattern detection,
    and market scanning capabilities.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the MarketAnalysisModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - technical_indicators_config: Config for TechnicalIndicatorsMCP
                - pattern_recognition_config: Config for PatternRecognitionMCP
                - redis_config: Config for RedisMCP
                - polygon_rest_config: Config for PolygonRestMCP
                - polygon_ws_config: Config for PolygonWSMCP
                - llm_config: Configuration for AutoGen LLM
                - scan_interval: Interval for market scanning in seconds (default: 3600)
                - max_scan_symbols: Maximum number of symbols to scan (default: 100)
                - default_indicators: List of default indicators to calculate
                - default_patterns: List of default patterns to detect
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Initialize logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Initialize monitoring
        self.monitor = MonitoringManager(
            service_name="market-analysis-model"
        )
        self.monitor.log_info(
            "MarketAnalysisModel initialized",
            component="market_analysis",
            action="initialization",
        )

        # Initialize MCP clients
        self.technical_indicators_mcp = TechnicalIndicatorsMCP(
            self.config.get("technical_indicators_config")
        )
        self.pattern_recognition_mcp = PatternRecognitionMCP(
            self.config.get("pattern_recognition_config")
        )
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.polygon_rest_mcp = PolygonRestMCP(self.config.get("polygon_rest_config"))
        self.polygon_ws_mcp = PolygonWSMCP(self.config.get("polygon_ws_config"))

        # Configuration parameters
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

        # Redis keys for data storage
        self.redis_keys = {
            "market_analysis_data": "market_analysis:data",  # Overall market analysis data
            "technical_indicators": "market_analysis:indicators:",  # Prefix for indicator data
            "pattern_recognition": "market_analysis:patterns:",  # Prefix for pattern data
            "market_scan_results": "market_analysis:scan_results",  # Market scan results
            "latest_analysis": "market_analysis:latest_analysis",  # Latest analysis timestamp
            "selection_data": "selection:data",  # Selection model data
            "selection_feedback": "market_analysis:selection_feedback",  # Feedback to selection model
        }

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        self.logger.info("MarketAnalysisModel initialized.")

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

        Returns:
            Dictionary of AutoGen agents
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

    def _register_functions(self):
        """
        Register functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        market_analysis_assistant = self.agents["market_analysis_assistant"]

        # Register technical indicator functions
        @register_function(
            name="calculate_indicators",
            description="Calculate technical indicators for price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def calculate_indicators(
            price_data: List[Dict[str, Any]], indicators: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            indicators = indicators or self.default_indicators
            return self.technical_indicators_mcp.calculate_indicators(
                price_data, indicators
            )

        @register_function(
            name="calculate_moving_averages",
            description="Calculate various types of moving averages",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def calculate_moving_averages(
            price_data: List[Dict[str, Any]], ma_types: List[str], periods: List[int]
        ) -> Dict[str, Any]:
            return self.technical_indicators_mcp.moving_averages(
                price_data, ma_types, periods
            )

        @register_function(
            name="calculate_momentum_oscillators",
            description="Calculate momentum oscillators like RSI, Stochastic, MACD",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def calculate_momentum_oscillators(
            price_data: List[Dict[str, Any]], oscillators: List[str]
        ) -> Dict[str, Any]:
            return self.technical_indicators_mcp.momentum_oscillators(
                price_data, oscillators
            )

        # Register pattern recognition functions
        @register_function(
            name="detect_patterns",
            description="Detect technical chart patterns in price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def detect_patterns(
            price_data: List[Dict[str, Any]], patterns: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            patterns = patterns or self.default_patterns
            return self.pattern_recognition_mcp.detect_patterns(price_data, patterns)

        @register_function(
            name="detect_candlestick_patterns",
            description="Detect candlestick patterns in price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def detect_candlestick_patterns(
            price_data: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            return self.pattern_recognition_mcp.detect_candlestick_patterns(price_data)

        @register_function(
            name="detect_support_resistance",
            description="Detect support and resistance levels in price data",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def detect_support_resistance(
            price_data: List[Dict[str, Any]], method: str = "peaks"
        ) -> Dict[str, Any]:
            return self.pattern_recognition_mcp.detect_support_resistance(
                price_data, method
            )

        # Register market data functions
        @register_function(
            name="get_market_data",
            description="Get market data for a symbol",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def get_market_data(
            symbol: str, timeframe: str = "1d", limit: int = 100
        ) -> Dict[str, Any]:
            return self.get_market_data(symbol, timeframe, limit)

        # Register market scanning functions
        @register_function(
            name="scan_market",
            description="Scan the market for trading opportunities",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def scan_market(
            symbols: List[str], scan_type: str = "technical"
        ) -> Dict[str, Any]:
            return self.scan_market(symbols, scan_type)

        @register_function(
            name="get_scan_results",
            description="Get the latest market scan results",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def get_scan_results() -> Dict[str, Any]:
            return self.get_scan_results()

        # Register selection model integration functions
        @register_function(
            name="get_selection_data",
            description="Get data from the Selection Model",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def get_selection_data() -> Dict[str, Any]:
            return self.get_selection_data()

        @register_function(
            name="send_feedback_to_selection",
            description="Send technical analysis feedback to the Selection Model",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def send_feedback_to_selection(analysis_data: Dict[str, Any]) -> bool:
            return self.send_feedback_to_selection(analysis_data)

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        market_analysis_assistant = self.agents["market_analysis_assistant"]

        # Define MCP tool access functions
        @register_function(
            name="use_technical_indicators_tool",
            description="Use a tool provided by the Technical Indicators MCP server",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_technical_indicators_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.technical_indicators_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_pattern_recognition_tool",
            description="Use a tool provided by the Pattern Recognition MCP server",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_pattern_recognition_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.pattern_recognition_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_polygon_rest_tool",
            description="Use a tool provided by the Polygon REST MCP server",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_polygon_rest_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.polygon_rest_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_polygon_ws_tool",
            description="Use a tool provided by the Polygon WebSocket MCP server",
            caller=market_analysis_assistant,
            executor=user_proxy,
        )
        def use_polygon_ws_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.polygon_ws_mcp.call_tool(tool_name, arguments)

    def get_market_data(
        self, symbol: str, timeframe: str = "1d", limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get market data for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Timeframe for data (e.g., '1d', '1h', '5m')
            limit: Maximum number of data points to retrieve

        Returns:
            Dictionary with market data
        """
        try:
            # Check if we have cached data in Redis
            cache_key = f"market_data:{symbol}:{timeframe}"
            cached_data = self.redis_mcp.get_json(cache_key)

            if cached_data:
                # Check if data is fresh (less than 1 hour old for daily data)
                last_updated = cached_data.get("timestamp")
                if last_updated:
                    last_updated_dt = datetime.fromisoformat(last_updated)
                    if datetime.now() - last_updated_dt < timedelta(hours=1):
                        return cached_data

            # Fetch data from Polygon
            if timeframe == "1d":
                result = self.polygon_rest_mcp.get_daily_bars(
                    ticker=symbol, limit=limit
                )
            elif timeframe == "1h":
                result = self.polygon_rest_mcp.get_hourly_bars(
                    ticker=symbol, limit=limit
                )
            else:
                # Default to daily
                result = self.polygon_rest_mcp.get_daily_bars(
                    ticker=symbol, limit=limit
                )

            if result and not result.get("error"):
                # Format data for technical analysis
                price_data = []
                for bar in result.get("results", []):
                    price_data.append(
                        {
                            "date": datetime.fromtimestamp(
                                bar.get("t") / 1000
                            ).isoformat()
                            if "t" in bar
                            else None,
                            "open": bar.get("o"),
                            "high": bar.get("h"),
                            "low": bar.get("l"),
                            "close": bar.get("c"),
                            "volume": bar.get("v"),
                        }
                    )

                # Cache in Redis
                data_to_cache = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "price_data": price_data,
                    "timestamp": datetime.now().isoformat(),
                }
                self.redis_mcp.set_json(
                    cache_key, data_to_cache, ex=3600
                )  # 1 hour expiration

                return data_to_cache
            else:
                self.logger.error(
                    f"Error fetching market data for {symbol}: {result.get('error')}"
                )
                return {"error": result.get("error"), "symbol": symbol}

        except Exception as e:
            self.logger.error(
                f"Error in get_market_data for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

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

            # Calculate technical indicators
            indicators_result = self.technical_indicators_mcp.calculate_indicators(
                price_data=price_data, indicators=self.default_indicators
            )

            # Detect patterns
            patterns_result = self.pattern_recognition_mcp.detect_patterns(
                price_data=price_data, patterns=self.default_patterns
            )

            # Detect support/resistance levels
            support_resistance = self.pattern_recognition_mcp.detect_support_resistance(
                price_data=price_data, method="peaks"
            )

            # Combine results
            analysis_result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "indicators": indicators_result.get("indicators", {}),
                "patterns": patterns_result.get("patterns", []),
                "support_resistance": support_resistance.get("levels", {}),
                "signals": indicators_result.get("signals", [])
                + patterns_result.get("patterns", []),
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            self.redis_mcp.set_json(
                f"{self.redis_keys['technical_indicators']}{symbol}",
                analysis_result,
                ex=86400,  # 1 day expiration
            )

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
        self.redis_mcp.set_json(
            self.redis_keys["market_scan_results"],
            scan_results,
            ex=86400,  # 1 day expiration
        )

        # Send feedback to selection model
        self.send_feedback_to_selection(
            {
                "market_scan": {
                    "scan_type": scan_type,
                    "opportunities": opportunities[:10],  # Send top 10 opportunities
                }
            }
        )

        return scan_results

    def _check_technical_opportunity(
        self, analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Check for trading opportunities based on technical indicators.

        Args:
            analysis: Symbol analysis results

        Returns:
            Dictionary with opportunity details or None
        """
        symbol = analysis.get("symbol")
        indicators = analysis.get("indicators", {})
        signals = analysis.get("signals", [])

        # Initialize score and signals
        score = 0
        bullish_signals = []
        bearish_signals = []

        # Check RSI
        if "rsi" in indicators:
            rsi_values = indicators["rsi"].get("values", [])
            if rsi_values and len(rsi_values) > 0:
                latest_rsi = rsi_values[-1]
                if latest_rsi < 30:
                    score += 1
                    bullish_signals.append(f"RSI oversold ({latest_rsi:.2f})")
                elif latest_rsi > 70:
                    score -= 1
                    bearish_signals.append(f"RSI overbought ({latest_rsi:.2f})")

        # Check MACD
        if "macd" in indicators:
            macd_line = indicators["macd"].get("macd_line", [])
            signal_line = indicators["macd"].get("signal_line", [])
            if (
                macd_line
                and signal_line
                and len(macd_line) > 1
                and len(signal_line) > 1
            ):
                if macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]:
                    score += 2
                    bullish_signals.append("MACD bullish crossover")
                elif (
                    macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]
                ):
                    score -= 2
                    bearish_signals.append("MACD bearish crossover")

        # Check Bollinger Bands
        if "bbands" in indicators:
            upper_band = indicators["bbands"].get("upper_band", [])
            lower_band = indicators["bbands"].get("lower_band", [])
            close_prices = [bar.get("close") for bar in analysis.get("price_data", [])]

            if upper_band and lower_band and close_prices and len(close_prices) > 0:
                latest_close = close_prices[-1]
                if latest_close <= lower_band[-1]:
                    score += 1
                    bullish_signals.append("Price at lower Bollinger Band")
                elif latest_close >= upper_band[-1]:
                    score -= 1
                    bearish_signals.append("Price at upper Bollinger Band")

        # Check for pattern signals
        for pattern in analysis.get("patterns", []):
            pattern_name = pattern.get("pattern")
            if pattern_name in ["double_bottom", "inverse_head_and_shoulders"]:
                score += 2
                bullish_signals.append(f"{pattern_name} pattern detected")
            elif pattern_name in ["double_top", "head_and_shoulders"]:
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

        Args:
            analysis: Symbol analysis results

        Returns:
            Dictionary with opportunity details or None
        """
        # This is a simplified implementation
        symbol = analysis.get("symbol")
        patterns = analysis.get("patterns", [])

        if not patterns:
            return None

        # Find the highest confidence pattern
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

        Args:
            analysis: Symbol analysis results

        Returns:
            Dictionary with opportunity details or None
        """
        symbol = analysis.get("symbol")
        support_resistance = analysis.get("support_resistance", {})
        price_data = analysis.get("price_data", [])

        if not support_resistance or not price_data or len(price_data) < 2:
            return None

        # Get latest price
        latest_price = price_data[-1].get("close")
        prev_price = price_data[-2].get("close")

        if not latest_price or not prev_price:
            return None

        # Get resistance levels
        resistance_levels = support_resistance.get("resistance", {}).get("levels", [])
        support_levels = support_resistance.get("support", {}).get("levels", [])

        # Check for breakouts
        for level in resistance_levels:
            # Resistance breakout (bullish)
            if prev_price < level and latest_price > level:
                return {
                    "symbol": symbol,
                    "type": "bullish",
                    "score": 8,
                    "signals": [f"Resistance breakout at {level:.2f}"],
                    "timestamp": datetime.now().isoformat(),
                }

        for level in support_levels:
            # Support breakdown (bearish)
            if prev_price > level and latest_price < level:
                return {
                    "symbol": symbol,
                    "type": "bearish",
                    "score": 8,
                    "signals": [f"Support breakdown at {level:.2f}"],
                    "timestamp": datetime.now().isoformat(),
                }

        return None

    def get_scan_results(self) -> Dict[str, Any]:
        """
        Get the latest market scan results.

        Returns:
            Dictionary with scan results
        """
        try:
            results = self.redis_mcp.get_json(self.redis_keys["market_scan_results"])
            if not results:
                return {"error": "No scan results available"}

            return results
        except Exception as e:
            self.logger.error(f"Error getting scan results: {e}", exc_info=True)
            return {"error": str(e)}

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

    def send_feedback_to_selection(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Send technical analysis feedback to the Selection Model.

        Args:
            analysis_data: Technical analysis data to send

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add timestamp and source
            feedback = {
                "source": "market_analysis",
                "timestamp": datetime.now().isoformat(),
                "data": analysis_data,
            }

            # Store in Redis for the Selection Model to pick up
            self.redis_mcp.set_json(self.redis_keys["selection_feedback"], feedback)

            # Add to feedback stream
            self.redis_mcp.add_to_stream("market_analysis:feedback", feedback)

            return True
        except Exception as e:
            self.logger.error(f"Error sending feedback to selection model: {e}")
            return False

    def run_market_analysis(self, query: str) -> Dict[str, Any]:
        """
        Run market analysis using AutoGen agents.

        Args:
            query: Query or instruction for market analysis

        Returns:
            Results of the market analysis
        """
        self.logger.info(f"Running market analysis with query: {query}")

        market_analysis_assistant = self.agents.get("market_analysis_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not market_analysis_assistant or not user_proxy:
            return {"error": "AutoGen agents not initialized"}

        try:
            # Initiate chat with the market analysis assistant
            user_proxy.initiate_chat(market_analysis_assistant, message=query)

            # Get the last message from the assistant
            last_message = user_proxy.last_message(market_analysis_assistant)
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
            return {"error": str(e)}
