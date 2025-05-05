"""
Fundamental Analysis Model

This module defines the FundamentalAnalysisModel, responsible for analyzing
financial statements, calculating key ratios, and evaluating company fundamentals.
It integrates with AutoGen for advanced analysis and decision making.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# For Redis integration
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# MCP tools for fundamental analysis
from mcp_tools.analysis_mcp.fundamental_scoring_mcp import FundamentalScoringMCP
from mcp_tools.analysis_mcp.growth_analysis_mcp import GrowthAnalysisMCP
from mcp_tools.analysis_mcp.risk_metrics_mcp import RiskMetricsMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function




class FundamentalAnalysisModel:
    """
    Analyzes company fundamentals using financial statements and key metrics.

    This model integrates FundamentalScoringMCP and GrowthAnalysisMCP to provide
    comprehensive fundamental analysis, including ratio calculation, financial
    health scoring, growth analysis, and earnings report processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FundamentalAnalysisModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - fundamental_scoring_config: Config for FundamentalScoringMCP
                - growth_analysis_config: Config for GrowthAnalysisMCP
                - risk_metrics_config: Config for RiskMetricsMCP
                - redis_config: Config for RedisMCP
                - polygon_rest_config: Config for PolygonRestMCP
                - yahoo_finance_config: Config for YahooFinanceMCP
                - llm_config: Configuration for AutoGen LLM
                - analysis_interval: Interval for periodic analysis in seconds (default: 86400)
                - max_companies: Maximum number of companies to analyze (default: 50)
        """
        init_start_time = time.time()
        # Initialize NetdataLogger for monitoring and logging
        self.logger = NetdataLogger(component_name="nextgen-fundamental-analysis-model")

        # Initialize StockChartGenerator
        self.chart_generator = StockChartGenerator()
        self.logger.info("StockChartGenerator initialized")

        # Initialize counters for fundamental analysis metrics
        self.companies_analyzed_count = 0
        self.analysis_errors_count = 0
        self.llm_api_call_count = 0
        self.mcp_tool_call_count = 0
        self.mcp_tool_error_count = 0
        self.total_analysis_cycles = 0 # Total times run_periodic_analysis completes a cycle


        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_fundamental_analysis", "fundamental_analysis_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}")
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}")
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}")
                self.config = {}
        else:
            self.config = config

        # Initialize MCP clients
        self.fundamental_scoring_mcp = FundamentalScoringMCP(
            self.config.get("fundamental_scoring_config")
        )
        self.growth_analysis_mcp = GrowthAnalysisMCP(
            self.config.get("growth_analysis_config")
        )
        self.risk_metrics_mcp = RiskMetricsMCP(self.config.get("risk_metrics_config"))
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.polygon_rest_mcp = PolygonRestMCP(self.config.get("polygon_rest_config"))
        self.yahoo_finance_mcp = YahooFinanceMCP(
            self.config.get("yahoo_finance_config")
        )

        # Configuration parameters
        self.analysis_interval = self.config.get(
            "analysis_interval", 86400
        )  # Default: 1 day
        self.max_companies = self.config.get("max_companies", 50)

        # Redis keys for data storage
        self.redis_keys = {
            "fundamental_data": "fundamental:data:",  # Prefix for overall fundamental data per symbol
            "financial_statements": "fundamental:statements:",  # Prefix for financial statements
            "financial_ratios": "fundamental:ratios:",  # Prefix for financial ratios
            "financial_health": "fundamental:health:",  # Prefix for financial health scores
            "growth_analysis": "fundamental:growth:",  # Prefix for growth analysis
            "earnings_reports": "fundamental:earnings:",  # Prefix for earnings reports
            "latest_analysis": "fundamental:latest_analysis",  # Latest analysis timestamp
            "selection_data": "selection:data",  # Selection model data
            "selection_feedback": "fundamental:selection_feedback",  # Feedback to selection model
            "decision_data": "decision:data",  # Decision model data
        }

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        self.logger.info("FundamentalAnalysisModel initialized.")

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
        Initialize AutoGen agents for fundamental analysis.

        Returns:
            Dictionary of AutoGen agents
        """
        agents = {}

        # Create the fundamental analysis assistant agent
        agents["fundamental_assistant"] = AssistantAgent(
            name="FundamentalAnalysisAssistant",
            system_message="""You are a fundamental analysis specialist. Your role is to:
            1. Analyze financial statements (income statement, balance sheet, cash flow)
            2. Calculate and interpret financial ratios and metrics
            3. Evaluate company growth and financial health
            4. Process earnings reports and assess their implications
            5. Provide insights on company valuation and investment potential

            You have tools for calculating financial ratios, scoring financial health,
            analyzing growth trends, and processing earnings reports. Always provide clear
            reasoning for your analysis and recommendations.""",
            llm_config=self.llm_config,
            description="A specialist in fundamental financial analysis",
        )

        # Create a user proxy agent that can execute functions
        user_proxy = UserProxyAgent(
            name="FundamentalToolUser",
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
        fundamental_assistant = self.agents["fundamental_assistant"]

        # Register financial data retrieval functions
        @register_function(
            name="get_financial_statements",
            description="Get financial statements for a company",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def get_financial_statements(
            symbol: str, statement_type: str = "all", period: str = "annual"
        ) -> Dict[str, Any]:
            return self.get_financial_statements(symbol, statement_type, period)

        @register_function(
            name="get_market_data",
            description="Get market data for a company",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def get_market_data(symbol: str) -> Dict[str, Any]:
            return self.get_market_data(symbol)

        # Register fundamental scoring functions
        @register_function(
            name="calculate_financial_ratios",
            description="Calculate financial ratios from financial statements",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def calculate_financial_ratios(
            financial_data: Dict[str, Any], ratios: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            return self.fundamental_scoring_mcp.calculate_ratios(financial_data, ratios)

        @register_function(
            name="score_financial_health",
            description="Score financial health based on ratios",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def score_financial_health(
            ratios: Dict[str, float], sector: Optional[str] = None
        ) -> Dict[str, Any]:
            return self.fundamental_scoring_mcp.score_financial_health(ratios, sector)

        @register_function(
            name="calculate_value_metrics",
            description="Calculate valuation metrics",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def calculate_value_metrics(
            market_data: Dict[str, Any], financial_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            return self.fundamental_scoring_mcp.calculate_value_metrics(
                market_data, financial_data
            )

        @register_function(
            name="compare_to_sector",
            description="Compare company's ratios to sector benchmarks",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def compare_to_sector(ratios: Dict[str, float], sector: str) -> Dict[str, Any]:
            return self.fundamental_scoring_mcp.compare_to_sector(ratios, sector)

        # Register growth analysis functions
        @register_function(
            name="calculate_growth_rates",
            description="Calculate growth rates from historical financial data",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def calculate_growth_rates(
            financial_data: Dict[str, Any], metrics: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            return self.growth_analysis_mcp.calculate_growth_rates(
                financial_data, metrics
            )

        @register_function(
            name="analyze_growth_trends",
            description="Analyze growth trends and patterns",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def analyze_growth_trends(growth_rates: Dict[str, Any]) -> Dict[str, Any]:
            return self.growth_analysis_mcp.analyze_growth_trends(growth_rates)

        @register_function(
            name="score_growth_quality",
            description="Score the quality and sustainability of growth",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def score_growth_quality(
            growth_rates: Dict[str, Any],
            trends: Optional[Dict[str, Any]] = None,
            sector: Optional[str] = None,
        ) -> Dict[str, Any]:
            return self.growth_analysis_mcp.score_growth_quality(
                growth_rates, trends, sector
            )

        @register_function(
            name="compare_to_sector_growth",
            description="Compare company's growth to sector benchmarks",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def compare_to_sector_growth(
            growth_rates: Dict[str, Any], sector: str
        ) -> Dict[str, Any]:
            return self.growth_analysis_mcp.compare_to_sector_growth(
                growth_rates, sector
            )

        # Register earnings report functions
        @register_function(
            name="analyze_earnings_report",
            description="Analyze an earnings report",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def analyze_earnings_report(
            report_data: Dict[str, Any],
            historical_data: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            return self.analyze_earnings_report(report_data, historical_data)

        @register_function(
            name="get_latest_earnings",
            description="Get the latest earnings report for a company",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def get_latest_earnings(symbol: str) -> Dict[str, Any]:
            return self.get_latest_earnings(symbol)

        # Register comprehensive analysis functions
        @register_function(
            name="analyze_company",
            description="Perform comprehensive fundamental analysis of a company",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def analyze_company(
            symbol: str, sector: Optional[str] = None
        ) -> Dict[str, Any]:
            return self.analyze_company(symbol, sector)

        # Register selection model integration functions
        @register_function(
            name="get_selection_data",
            description="Get data from the Selection Model",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def get_selection_data() -> Dict[str, Any]:
            return self.get_selection_data()

        @register_function(
            name="send_feedback_to_selection",
            description="Send fundamental analysis feedback to the Selection Model",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def send_feedback_to_selection(analysis_data: Dict[str, Any]) -> bool:
            return self.send_feedback_to_selection(analysis_data)

        # Register decision model integration functions
        @register_function(
            name="send_analysis_to_decision",
            description="Send fundamental analysis to the Decision Model",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def send_analysis_to_decision(analysis_data: Dict[str, Any]) -> bool:
            return self.send_analysis_to_decision(analysis_data)

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        fundamental_assistant = self.agents["fundamental_assistant"]

        # Define MCP tool access functions
        @register_function(
            name="use_fundamental_scoring_tool",
            description="Use a tool provided by the Fundamental Scoring MCP server",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_fundamental_scoring_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.fundamental_scoring_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_growth_analysis_tool",
            description="Use a tool provided by the Growth Analysis MCP server",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_growth_analysis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.growth_analysis_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_risk_metrics_tool",
            description="Use a tool provided by the Risk Metrics MCP server",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_risk_metrics_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.risk_metrics_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_polygon_rest_tool",
            description="Use a tool provided by the Polygon REST MCP server",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_polygon_rest_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.polygon_rest_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_yahoo_finance_tool",
            description="Use a tool provided by the Yahoo Finance MCP server",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_yahoo_finance_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.yahoo_finance_mcp.call_tool(tool_name, arguments)

    def get_financial_statements(
        self, symbol: str, statement_type: str = "all", period: str = "annual"
    ) -> Dict[str, Any]:
        """
        Get financial statements for a company.

        Args:
            symbol: Stock symbol
            statement_type: Type of statement ('income', 'balance', 'cash_flow', or 'all')
            period: Period ('annual' or 'quarterly')

        Returns:
            Dictionary with financial statements
        """
        try:
            # Check if we have cached data in Redis
            cache_key = f"{self.redis_keys['financial_statements']}{symbol}:{period}"
            cached_data = self.redis_mcp.get_json(cache_key)

            if cached_data:
                #
                # Check if data is fresh (less than 1 day old for annual, 6
                # hours for quarterly)
                last_updated = cached_data.get("timestamp")
                if last_updated:
                    last_updated_dt = datetime.fromisoformat(last_updated)
                    max_age = (
                        timedelta(days=1) if period == "annual" else timedelta(hours=6)
                    )
                    if datetime.now() - last_updated_dt < max_age:
                        # Filter by statement type if needed
                        if statement_type != "all":
                            return {statement_type: cached_data.get(statement_type, {})}
                        return cached_data

            # Fetch data from Yahoo Finance
            statements = {}

            if statement_type in ["income", "all"]:
                income_statement = self.yahoo_finance_mcp.get_income_statement(
                    symbol, period
                )
                if income_statement and not income_statement.get("error"):
                    statements["income_statement"] = income_statement

            if statement_type in ["balance", "all"]:
                balance_sheet = self.yahoo_finance_mcp.get_balance_sheet(symbol, period)
                if balance_sheet and not balance_sheet.get("error"):
                    statements["balance_sheet"] = balance_sheet

            if statement_type in ["cash_flow", "all"]:
                cash_flow = self.yahoo_finance_mcp.get_cash_flow(symbol, period)
                if cash_flow and not cash_flow.get("error"):
                    statements["cash_flow"] = cash_flow

            if statements:
                # Cache in Redis
                statements["timestamp"] = datetime.now().isoformat()
                statements["symbol"] = symbol
                statements["period"] = period

                self.redis_mcp.set_json(
                    cache_key, statements, ex=86400
                )  # 1 day expiration

                # Filter by statement type if needed
                if statement_type != "all":
                    return {statement_type: statements.get(statement_type, {})}
                return statements
            else:
                self.logger.error(f"Error fetching financial statements for {symbol}")
                return {
                    "error": "Failed to fetch financial statements",
                    "symbol": symbol,
                }

        except Exception as e:
            self.logger.error(
                f"Error in get_financial_statements for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a company.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with market data
        """
        try:
            # Fetch data from Yahoo Finance
            quote = self.yahoo_finance_mcp.get_quote(symbol)

            if quote and not quote.get("error"):
                market_data = {
                    "symbol": symbol,
                    "price": quote.get("regularMarketPrice"),
                    "market_cap": quote.get("marketCap"),
                    "shares_outstanding": quote.get("sharesOutstanding"),
                    "pe_ratio": quote.get("trailingPE"),
                    "forward_pe": quote.get("forwardPE"),
                    "dividend_yield": quote.get("dividendYield"),
                    "timestamp": datetime.now().isoformat(),
                }
                return market_data
            else:
                self.logger.error(f"Error fetching market data for {symbol}")
                return {"error": "Failed to fetch market data", "symbol": symbol}

        except Exception as e:
            self.logger.error(
                f"Error in get_market_data for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    def analyze_earnings_report(
        self,
        report_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an earnings report.

        Args:
            report_data: Dictionary containing earnings report data
            historical_data: Optional dictionary with historical financial data for comparison

        Returns:
            Dictionary with earnings report analysis
        """
        try:
            # Use GrowthAnalysisMCP to analyze the earnings report
            analysis = self.growth_analysis_mcp.analyze_earnings_report(
                report_data, historical_data
            )

            # Enhance with additional fundamental analysis
            if historical_data:
                # Calculate financial ratios for the current report
                current_ratios = self.fundamental_scoring_mcp.calculate_ratios(
                    report_data
                )

                # Calculate financial ratios for historical data
                historical_ratios = self.fundamental_scoring_mcp.calculate_ratios(
                    historical_data
                )

                # Compare ratios
                ratio_changes = {}
                for ratio, value in current_ratios.get("ratios", {}).items():
                    historical_value = historical_ratios.get("ratios", {}).get(ratio)
                    if value is not None and historical_value is not None:
                        ratio_changes[ratio] = {
                            "current": value,
                            "previous": historical_value,
                            "change": value - historical_value,
                            "change_pct": ((value / historical_value) - 1) * 100
                            if historical_value != 0
                            else None,
                        }

                # Add ratio changes to analysis
                analysis["ratio_changes"] = ratio_changes

            # Store in Redis
            if "symbol" in report_data:
                symbol = report_data["symbol"]
                cache_key = f"{self.redis_keys['earnings_reports']}{symbol}"
                self.redis_mcp.set_json(
                    cache_key, analysis, ex=86400
                )  # 1 day expiration

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing earnings report: {e}", exc_info=True)
            return {"error": str(e)}

    def get_latest_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest earnings report for a company.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with the latest earnings report
        """
        try:
            # Check if we have cached data in Redis
            cache_key = f"{self.redis_keys['earnings_reports']}{symbol}"
            cached_data = self.redis_mcp.get_json(cache_key)

            if cached_data:
                # Check if data is fresh (less than 1 day old)
                last_updated = cached_data.get("timestamp")
                if last_updated:
                    last_updated_dt = datetime.fromisoformat(last_updated)
                    if datetime.now() - last_updated_dt < timedelta(days=1):
                        return cached_data

            # Fetch data from Yahoo Finance
            earnings = self.yahoo_finance_mcp.get_earnings(symbol)

            if earnings and not earnings.get("error"):
                # Format the earnings data
                latest_earnings = {
                    "symbol": symbol,
                    "date": earnings.get("earningsDate", [None])[0],
                    "eps": earnings.get("eps", None),
                    "eps_estimate": earnings.get("epsEstimate", None),
                    "eps_surprise": earnings.get("epsSurprise", None),
                    "revenue": earnings.get("revenue", None),
                    "revenue_estimate": earnings.get("revenueEstimate", None),
                    "revenue_surprise": earnings.get("revenueSurprise", None),
                    "timestamp": datetime.now().isoformat(),
                }

                # Cache in Redis
                self.redis_mcp.set_json(
                    cache_key, latest_earnings, ex=86400
                )  # 1 day expiration

                return latest_earnings
            else:
                self.logger.error(f"Error fetching earnings for {symbol}")
                return {"error": "Failed to fetch earnings", "symbol": symbol}

        except Exception as e:
            self.logger.error(
                f"Error in get_latest_earnings for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    def analyze_company(
        self, symbol: str, sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis of a company.

        Args:
            symbol: Stock symbol
            sector: Optional industry sector

        Returns:
            Dictionary with comprehensive analysis
        """
        try:
            # Get financial statements
            financial_statements = self.get_financial_statements(
                symbol, "all", "annual"
            )
            if financial_statements.get("error"):
                return {"error": financial_statements.get("error"), "symbol": symbol}

            # Get market data
            market_data = self.get_market_data(symbol)
            if market_data.get("error"):
                return {"error": market_data.get("error"), "symbol": symbol}

            # Calculate financial ratios
            ratios_result = self.fundamental_scoring_mcp.calculate_ratios(
                financial_statements
            )

            # Score financial health
            health_score = self.fundamental_scoring_mcp.score_financial_health(
                ratios_result.get("ratios", {}), sector
            )

            # Calculate value metrics
            value_metrics = self.fundamental_scoring_mcp.calculate_value_metrics(
                market_data, financial_statements
            )

            # Compare to sector
            sector_comparison = None
            if sector:
                sector_comparison = self.fundamental_scoring_mcp.compare_to_sector(
                    ratios_result.get("ratios", {}), sector
                )

            # Calculate growth rates
            growth_rates = self.growth_analysis_mcp.calculate_growth_rates(
                financial_statements
            )

            # Analyze growth trends
            growth_trends = self.growth_analysis_mcp.analyze_growth_trends(
                growth_rates.get("growth_rates", {})
            )

            # Score growth quality
            growth_score = self.growth_analysis_mcp.score_growth_quality(
                growth_rates.get("growth_rates", {}),
                growth_trends.get("trends", {}),
                sector,
            )

            # Get latest earnings
            latest_earnings = self.get_latest_earnings(symbol)

            # Combine all analysis
            comprehensive_analysis = {
                "symbol": symbol,
                "sector": sector,
                "market_data": market_data,
                "financial_ratios": ratios_result.get("ratios", {}),
                "financial_health": {
                    "overall_score": health_score.get("overall_score"),
                    "category_scores": health_score.get("category_scores", {}),
                },
                "value_metrics": value_metrics.get("metrics", {}),
                "growth_analysis": {
                    "growth_rates": growth_rates.get("growth_rates", {}),
                    "growth_trends": growth_trends.get("trends", {}),
                    "growth_score": growth_score.get("overall_score"),
                    "metric_scores": growth_score.get("metric_scores", {}),
                },
                "latest_earnings": latest_earnings,
                "sector_comparison": sector_comparison.get("comparison")
                if sector_comparison
                else None,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            cache_key = f"{self.redis_keys['fundamental_data']}{symbol}"
            self.redis_mcp.set_json(
                cache_key, comprehensive_analysis, ex=86400
            )  # 1 day expiration

            # Send feedback to selection model
            self.send_feedback_to_selection(
                {
                    "symbol": symbol,
                    "fundamental_score": health_score.get("overall_score"),
                    "growth_score": growth_score.get("overall_score"),
                    "value_metrics": value_metrics.get("metrics", {}),
                }
            )

            # Send analysis to decision model
            self.send_analysis_to_decision(comprehensive_analysis)

            return comprehensive_analysis

        except Exception as e:
            self.logger.error(
                f"Error in analyze_company for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

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
        Send fundamental analysis feedback to the Selection Model.

        Args:
            analysis_data: Fundamental analysis data to send

        Returns:
            True if successful, False otherwise
        """
        try:
            # Publish feedback to a Redis channel or add to a list
            self.redis_mcp.publish(
                self.redis_keys["selection_feedback"], json.dumps(analysis_data)
            )
            self.logger.info(
                f"Sent feedback for {analysis_data.get('symbol')} to Selection Model"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending feedback to Selection Model: {e}")
            return False

    def send_analysis_to_decision(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Send fundamental analysis to the Decision Model.

        Args:
            analysis_data: Fundamental analysis data to send

        Returns:
            True if successful, False otherwise
        """
        try:
            # Publish analysis to a Redis channel or add to a list
            self.redis_mcp.publish(
                self.redis_keys["decision_data"], json.dumps(analysis_data)
            )
            self.logger.info(
                f"Sent analysis for {analysis_data.get('symbol')} to Decision Model"
            )
            return True
        except Exception as e:
            self.logger.error(f"Error sending analysis to Decision Model: {e}")
            return False

    async def run_periodic_analysis(self):
        """
        Run fundamental analysis periodically.
        """
        while True:
            try:
                self.logger.info("Starting periodic fundamental analysis...")

                # Get list of companies from Selection Model
                selection_data = self.get_selection_data()
                companies_to_analyze = selection_data.get("selected_companies", [])[
                    : self.max_companies
                ]

                if not companies_to_analyze:
                    self.logger.info("No companies selected for analysis.")
                else:
                    self.logger.info(
                        f"Analyzing {len(companies_to_analyze)} companies: {companies_to_analyze}"
                    )

                    for company in companies_to_analyze:
                        symbol = company.get("symbol")
                        sector = company.get("sector")
                        if symbol:
                            self.logger.info(f"Analyzing {symbol}...")
                            analysis_result = self.analyze_company(symbol, sector)
                            if analysis_result.get("error"):
                                self.logger.error(
                                    f"Failed to analyze {symbol}: {analysis_result['error']}"
                                )
                            else:
                                self.logger.info(f"Successfully analyzed {symbol}")

                # Update latest analysis timestamp
                self.redis_mcp.set(
                    self.redis_keys["latest_analysis"], datetime.now().isoformat()
                )
                self.logger.info("Periodic fundamental analysis complete.")

            except Exception as e:
                self.logger.error(f"Error during periodic analysis: {e}", exc_info=True)

            # Wait for the next interval
            await asyncio.sleep(self.analysis_interval)

    def start(self):
        """
        Start the periodic analysis loop.
        """
        self.logger.info("Starting FundamentalAnalysisModel...")
        asyncio.run(self.run_periodic_analysis())


# For running as a standalone model
if __name__ == "__main__":
    # Set up configuration (replace with actual config loading)
    config = {
        "fundamental_scoring_config": {},
        "growth_analysis_config": {},
        "risk_metrics_config": {},
        "redis_config": {"host": "localhost", "port": 6379, "db": 0},
        "polygon_rest_config": {},
        "yahoo_finance_config": {},
        "llm_config": {},  # Use default LLM config
        "analysis_interval": 3600,  # Analyze every hour for testing
        "max_companies": 10,
    }

    # Initialize the model
    model = FundamentalAnalysisModel(config)

    # Example: Analyze a single company
    print("Analyzing AAPL...")
    aapl_analysis = model.analyze_company("AAPL", "Technology")
    print(json.dumps(aapl_analysis, indent=2))

    # Start the periodic analysis (this will run indefinitely)
    # model.start()
