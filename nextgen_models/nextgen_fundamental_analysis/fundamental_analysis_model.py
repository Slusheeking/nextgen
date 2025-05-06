"""
Fundamental Analysis Model

fiThis module defines the FundamentalAnalysisModel, responsible for analyzing
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

# Import base agent
from nextgen_models.base_mcp_agent import BaseMCPAgent

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.stock_charts import StockChartGenerator

# MCP tools (Consolidated)
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# AutoGen imports
from autogen import Agent, AssistantAgent, UserProxyAgent, register_function


class FundamentalAnalysisModel(BaseMCPAgent):
    """
    Analyzes company fundamentals using financial statements and key metrics.

    This model integrates FinancialDataMCP and RiskAnalysisMCP to provide
    comprehensive fundamental analysis, including data retrieval, ratio calculation,
    financial health scoring, growth analysis, and earnings report processing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the FundamentalAnalysisModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - financial_data_config: Config for FinancialDataMCP
                - risk_analysis_config: Config for RiskAnalysisMCP
                - llm_config: Configuration for AutoGen LLM
                - analysis_interval: Interval for periodic analysis in seconds (default: 86400)
                - max_companies: Maximum number of companies to analyze (default: 50)
                - risk_config: Configuration for risk assessment, including:
                    - risk_per_trade: Maximum risk percentage per trade (default: 0.005 or 0.5%)
                    - var_confidence_level: VaR confidence level (default: 0.95 or 95%)
                    - historical_window: Number of days for historical VaR (default: 252)
                    - risk_free_rate: Annual risk-free rate (default: 0.02 or 2%)
        """
        init_start_time = time.time()

        # Initialize BaseMCPAgent first
        super().__init__(config=config)

        try:
            # Initialize NetdataLogger
            self.logger = NetdataLogger("nextgen_fundamental_analysis")

            # Initialize StockChartGenerator
            self.chart_generator = StockChartGenerator("stock-charts")
            self.logger.info("StockChartGenerator initialized")

            # Initialize counters for fundamental analysis metrics
            self.companies_analyzed_count = 0
            self.analysis_errors_count = 0
            self.llm_api_call_count = 0
            self.mcp_tool_call_count = 0
            self.mcp_tool_error_count = 0
            self.total_analysis_cycles = 0 # Total times run_periodic_analysis completes a cycle
        except Exception as e:
            print(f"Error initializing FundamentalAnalysisModel: {e}")
            raise



        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "nextgen_fundamental_analysis", "fundamental_analysis_model_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration successfully loaded from {config_path}")
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
            self.logger.info("Configuration successfully loaded from provided config dictionary")

        # Ensure critical configuration parameters are set
        financial_data_config_path = self.config.get("financial_data_config_path")
        if not financial_data_config_path:
            default_path = os.path.join("config", "financial_data_mcp", "financial_data_mcp_config.json")
            self.logger.warning(f"Missing financial_data_config_path in configuration. Using default: {default_path}")
            financial_data_config_path = default_path

        # Load financial data config from path
        financial_data_config = {}
        if financial_data_config_path and os.path.exists(financial_data_config_path):
            try:
                with open(financial_data_config_path, 'r') as f:
                    financial_data_config = json.load(f)
                self.logger.info(f"Financial data configuration successfully loaded from {financial_data_config_path}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing financial data configuration file {financial_data_config_path}: {e}")
            except Exception as e:
                self.logger.error(f"Error loading financial data configuration file {financial_data_config_path}: {e}")
        elif financial_data_config_path:
             self.logger.warning(f"Financial data configuration file not found at {financial_data_config_path}")
        else:
             self.logger.warning("financial_data_config_path not specified in configuration.")


        # Initialize Consolidated MCP clients
        # FinancialDataMCP handles data retrieval (Polygon, Yahoo)
        self.financial_data_mcp = FinancialDataMCP(financial_data_config)

        # RiskAnalysisMCP handles risk metrics and potentially other analysis
        risk_analysis_config = self.config.get("risk_analysis_mcp_config") # Look for the correct key
        if not risk_analysis_config:
            self.logger.error("Missing risk_analysis_mcp_config in configuration") # Update error message
            raise ValueError("Missing risk_analysis_mcp_config in configuration") # Update error message

        self.risk_analysis_mcp = RiskAnalysisMCP(risk_analysis_config)

        # Initialize Redis MCP client
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))
        self.redis_client = self.redis_mcp # Alias for backward compatibility if needed


        # Configuration parameters
        self.analysis_interval = self.config.get(
            "analysis_interval", 86400
        )  # Default: 1 day
        self.max_companies = self.config.get("max_companies", 50)

        # Redis keys for data storage and inter-model communication
        self.redis_keys = {
            "fundamental_data": "fundamental:data:",  # Prefix for overall fundamental data per symbol
            "financial_statements": "fundamental:statements:",  # Prefix for financial statements
            "financial_ratios": "fundamental:ratios:",  # Prefix for financial ratios
            "financial_health": "fundamental:health:",  # Prefix for financial health scores
            "growth_analysis": "fundamental:growth:",  # Prefix for growth analysis
            "earnings_reports": "fundamental:earnings:",  # Prefix for earnings reports
            "latest_analysis_timestamp": "fundamental:latest_analysis_timestamp",  # Latest analysis timestamp (single key)
            "selection_candidates": "selection:candidates",  # Selection model candidates (list or set)
            "selection_feedback_stream": "fundamental:selection_feedback",  # Feedback to selection model (stream)
            "fundamental_analysis_reports_stream": "model:fundamental:insights",  # Stream for publishing fundamental analysis reports
        }

        # Ensure Redis streams exist (optional, but good practice)
        try:
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["selection_feedback_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['selection_feedback_stream']}' exists.")
            self.redis_mcp.call_tool("create_stream", {"stream": self.redis_keys["fundamental_analysis_reports_stream"]})
            self.logger.info(f"Ensured Redis stream '{self.redis_keys['fundamental_analysis_reports_stream']}' exists.")
        except Exception as e:
            self.logger.warning(f"Could not ensure Redis streams exist: {e}")


        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        self.logger.info("FundamentalAnalysisModel initialized.")
        init_duration = (time.time() - init_start_time) * 1000
        self.logger.timing("fundamental_analysis_model.initialization_time_ms", init_duration)


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

        # Register fundamental scoring functions (now part of RiskAnalysisMCP)
        @register_function(
            name="calculate_financial_ratios",
            description="Calculate financial ratios from financial statements",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def calculate_financial_ratios(
            financial_data: Dict[str, Any], ratios: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("calculate_financial_ratios", {"financial_data": financial_data, "ratios": ratios})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="score_financial_health",
            description="Score financial health based on ratios",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def score_financial_health(
            ratios: Dict[str, float], sector: Optional[str] = None
        ) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("score_financial_health", {"ratios": ratios, "sector": sector})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="calculate_value_metrics",
            description="Calculate valuation metrics",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def calculate_value_metrics(
            market_data: Dict[str, Any], financial_data: Dict[str, Any]
        ) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("calculate_value_metrics", {"market_data": market_data, "financial_data": financial_data})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="compare_to_sector",
            description="Compare company's ratios to sector benchmarks",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def compare_to_sector(ratios: Dict[str, float], sector: str) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("compare_to_sector", {"ratios": ratios, "sector": sector})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        # Register growth analysis functions (now part of RiskAnalysisMCP)
        @register_function(
            name="calculate_growth_rates",
            description="Calculate growth rates from historical financial data",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def calculate_growth_rates(
            financial_data: Dict[str, Any], metrics: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("calculate_growth_rates", {"financial_data": financial_data, "metrics": metrics})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="analyze_growth_trends",
            description="Analyze growth trends and patterns",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def analyze_growth_trends(growth_rates: Dict[str, Any]) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("analyze_growth_trends", {"growth_rates": growth_rates})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

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
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("score_growth_quality", {"growth_rates": growth_rates, "trends": trends, "sector": sector})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="compare_to_sector_growth",
            description="Compare company's growth to sector benchmarks",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def compare_to_sector_growth(
            growth_rates: Dict[str, Any], sector: str
        ) -> Dict[str, Any]:
            # Call the tool on RiskAnalysisMCP
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool("compare_to_sector_growth", {"growth_rates": growth_rates, "sector": sector})
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

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

        # Define MCP tool access functions for consolidated MCPs
        @register_function(
            name="use_financial_data_tool",
            description="Use a tool provided by the Financial Data MCP server (for financial statements, market data, earnings)",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_financial_data_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            self.mcp_tool_call_count += 1
            result = self.financial_data_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        @register_function(
            name="use_risk_analysis_tool",
            description="Use a tool provided by the Risk Analysis MCP server (for ratios, health, growth, value metrics)",
            caller=fundamental_assistant,
            executor=user_proxy,
        )
        def use_risk_analysis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            self.mcp_tool_call_count += 1
            result = self.risk_analysis_mcp.call_tool(tool_name, arguments)
            if result and result.get("error"):
                 self.mcp_tool_error_count += 1
            return result

        # Old MCP tool access functions removed

    def get_financial_statements(
        self, symbol: str, statement_type: str = "all", period: str = "annual"
    ) -> Dict[str, Any]:
        """
        Get financial statements for a company using FinancialDataMCP and cache in Redis.

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
            self.mcp_tool_call_count += 1
            cached_data_result = self.redis_mcp.call_tool("get_json", {"key": cache_key})
            cached_data = cached_data_result.get("value") if cached_data_result and not cached_data_result.get("error") else None

            if cached_data:
                last_updated = cached_data.get("timestamp")
                if last_updated:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        # Cache expiration logic (1 day for annual, 6 hours for quarterly)
                        max_age = (
                            timedelta(days=1) if period == "annual" else timedelta(hours=6)
                        )
                        if datetime.now() - last_updated_dt < max_age:
                            self.logger.info(f"Returning cached financial statements for {symbol}:{period}")
                            if statement_type != "all":
                                return {statement_type: cached_data.get(statement_type, {})}
                            return cached_data
                    except ValueError:
                        self.logger.warning(f"Invalid timestamp format in cached data for {symbol}:{period}")


            # Fetch data using FinancialDataMCP
            # Assuming FinancialDataMCP has a tool like 'get_financial_statements'
            self.mcp_tool_call_count += 1
            statements_result = self.financial_data_mcp.call_tool(
                "get_financial_statements",
                {"symbol": symbol, "statement_type": statement_type, "period": period}
            )

            if statements_result and not statements_result.get("error"):
                statements = statements_result.get("statements", {}) # Assuming 'statements' key in result

                # Cache in Redis
                if statements:
                    statements["timestamp"] = datetime.now().isoformat()
                    statements["symbol"] = symbol
                    statements["period"] = period
                    self.mcp_tool_call_count += 1
                    self.redis_mcp.call_tool(
                        "set_json",
                        {"key": cache_key, "value": statements, "expiry": 86400} # 1 day expiration for simplicity, could be dynamic
                    )

                # Filter by statement type if needed (already handled by tool?)
                # if statement_type != "all":
                #     return {statement_type: statements.get(statement_type, {})}
                return statements
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Error fetching financial statements for {symbol}: {statements_result.get('error') if statements_result else 'Unknown error'}")
                return {
                    "error": statements_result.get('error', 'Failed to fetch financial statements') if statements_result else 'Failed to fetch financial statements',
                    "symbol": symbol,
                }

        except Exception as e:
            self.mcp_tool_error_count += 1
            self.logger.error(
                f"Error in get_financial_statements for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get market data for a company using FinancialDataMCP.
        """
        try:
            # Fetch data using FinancialDataMCP
            # Assuming FinancialDataMCP has a tool like 'get_market_data' or 'get_quote'
            self.mcp_tool_call_count += 1
            market_data_result = self.financial_data_mcp.call_tool("get_market_data", {"symbol": symbol}) # Or "get_quote"

            if market_data_result and not market_data_result.get("error"):
                 # Assuming the result structure is compatible or can be mapped
                 # Example mapping if needed:
                 # market_data = {
                 #     "symbol": symbol,
                 #     "price": market_data_result.get("price"),
                 #     "market_cap": market_data_result.get("market_cap"),
                 #     # ... other fields
                 #     "timestamp": datetime.now().isoformat(),
                 # }
                 return market_data_result # Return directly if structure is fine
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Error fetching market data for {symbol}: {market_data_result.get('error') if market_data_result else 'Unknown error'}")
                return {"error": market_data_result.get('error', 'Failed to fetch market data') if market_data_result else 'Failed to fetch market data', "symbol": symbol}

        except Exception as e:
            self.mcp_tool_error_count += 1
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
        Analyze an earnings report using RiskAnalysisMCP and store in Redis.

        Args:
            report_data: Dictionary containing earnings report data
            historical_data: Optional dictionary with historical financial data for comparison

        Returns:
            Dictionary with earnings report analysis
        """
        try:
            # Use RiskAnalysisMCP to analyze the earnings report
            # Assuming RiskAnalysisMCP has a tool like 'analyze_earnings_report'
            self.mcp_tool_call_count += 1
            analysis_result = self.risk_analysis_mcp.call_tool(
                "analyze_earnings_report",
                {"report_data": report_data, "historical_data": historical_data}
            )

            # Enhance with additional fundamental analysis (now handled by RiskAnalysisMCP tool)
            # if historical_data:
            #     current_ratios = self.fundamental_scoring_mcp.calculate_ratios(report_data)
            #     historical_ratios = self.fundamental_scoring_mcp.calculate_ratios(historical_data)
            #     ratio_changes = {}
            #     for ratio, value in current_ratios.get("ratios", {}).items():
            #         historical_value = historical_ratios.get("ratios", {}).get(ratio)
            #         if value is not None and historical_value is not None:
            #             ratio_changes[ratio] = {
            #                 "current": value, "previous": historical_value, "change": value - historical_value,
            #                 "change_pct": ((value / historical_value) - 1) * 100 if historical_value != 0 else None,
            #             }
            #     analysis_result["ratio_changes"] = ratio_changes


            # Store in Redis
            if analysis_result and "symbol" in analysis_result: # Use analysis_result as it should contain symbol
                symbol = analysis_result["symbol"]
                cache_key = f"{self.redis_keys['earnings_reports']}{symbol}"
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool(
                    "set_json",
                    {"key": cache_key, "value": analysis_result, "expiry": 86400} # 1 day expiration
                )
                self.logger.info(f"Stored earnings report analysis for {symbol} in Redis.")


            if analysis_result and not analysis_result.get("error"):
                 return analysis_result
            else:
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error analyzing earnings report: {analysis_result.get('error') if analysis_result else 'Unknown error'}")
                 return {"error": analysis_result.get('error', 'Failed to analyze earnings report') if analysis_result else 'Failed to analyze earnings report'}


        except Exception as e:
            self.mcp_tool_error_count += 1
            self.logger.error(f"Error analyzing earnings report: {e}", exc_info=True)
            return {"error": str(e)}


    def get_latest_earnings(self, symbol: str) -> Dict[str, Any]:
        """
        Get the latest earnings report for a company using FinancialDataMCP and cache in Redis.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with the latest earnings report
        """
        try:
            # Check if we have cached data in Redis
            cache_key = f"{self.redis_keys['earnings_reports']}{symbol}"
            self.mcp_tool_call_count += 1
            cached_data_result = self.redis_mcp.call_tool("get_json", {"key": cache_key})
            cached_data = cached_data_result.get("value") if cached_data_result and not cached_data_result.get("error") else None

            if cached_data:
                last_updated = cached_data.get("timestamp")
                if last_updated:
                    try:
                        last_updated_dt = datetime.fromisoformat(last_updated)
                        if datetime.now() - last_updated_dt < timedelta(days=1): # Cache for 1 day
                            self.logger.info(f"Returning cached earnings report for {symbol}")
                            return cached_data
                    except ValueError:
                        self.logger.warning(f"Invalid timestamp format in cached earnings data for {symbol}")


            # Fetch data using FinancialDataMCP
            # Assuming FinancialDataMCP has a tool like 'get_earnings'
            self.mcp_tool_call_count += 1
            earnings_result = self.financial_data_mcp.call_tool("get_earnings", {"symbol": symbol})

            if earnings_result and not earnings_result.get("error"):
                 # Assuming the result structure is compatible or can be mapped
                 # Example mapping if needed:
                 # latest_earnings = {
                 #     "symbol": symbol,
                 #     "date": earnings_result.get("earningsDate", [None])[0],
                 #     # ... other fields
                 #     "timestamp": datetime.now().isoformat(),
                 # }

                 # Cache in Redis
                 if earnings_result:
                     self.mcp_tool_call_count += 1
                     self.redis_mcp.call_tool(
                         "set_json",
                         {"key": cache_key, "value": earnings_result, "expiry": 86400} # 1 day expiration
                     )
                     self.logger.info(f"Cached latest earnings report for {symbol}.")


                 return earnings_result # Return directly if structure is fine
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Error fetching earnings for {symbol}: {earnings_result.get('error') if earnings_result else 'Unknown error'}")
                return {"error": earnings_result.get('error', 'Failed to fetch earnings') if earnings_result else 'Failed to fetch earnings', "symbol": symbol}

        except Exception as e:
            self.mcp_tool_error_count += 1
            self.logger.error(
                f"Error in get_latest_earnings for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}

    def analyze_company(
        self, symbol: str, sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fundamental analysis of a company and store in Redis.

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
                self.analysis_errors_count += 1
                return {"error": financial_statements.get("error"), "symbol": symbol}

            # Get market data
            market_data = self.get_market_data(symbol)
            if market_data.get("error"):
                self.analysis_errors_count += 1
                return {"error": market_data.get("error"), "symbol": symbol}

            # Calculate financial ratios (using RiskAnalysisMCP tool)
            ratios_result = self.risk_analysis_mcp.call_tool(
                "calculate_financial_ratios", {"financial_data": financial_statements}
            )
            if ratios_result.get("error"):
                 self.analysis_errors_count += 1
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error calculating ratios for {symbol}: {ratios_result['error']}")
                 # Continue analysis if possible, but log error
                 ratios_data = {}
            else:
                 ratios_data = ratios_result.get("ratios", {})


            # Score financial health (using RiskAnalysisMCP tool)
            health_score_result = self.risk_analysis_mcp.call_tool(
                "score_financial_health", {"ratios": ratios_data, "sector": sector}
            )
            if health_score_result.get("error"):
                 self.analysis_errors_count += 1
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error scoring financial health for {symbol}: {health_score_result['error']}")
                 health_score_data = {}
            else:
                 health_score_data = health_score_result
            


            # Calculate value metrics (using RiskAnalysisMCP tool)
            value_metrics_result = self.risk_analysis_mcp.call_tool(
                "calculate_value_metrics", {"market_data": market_data, "financial_data": financial_statements}
            )
            if value_metrics_result.get("error"):
                 self.analysis_errors_count += 1
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error calculating value metrics for {symbol}: {value_metrics_result['error']}")
                 value_metrics_data = {}
            else:
                 value_metrics_data = value_metrics_result.get("metrics", {})


            # Compare to sector (using RiskAnalysisMCP tool)
            sector_comparison_data = None
            if sector:
                sector_comparison_result = self.risk_analysis_mcp.call_tool(
                    "compare_to_sector", {"ratios": ratios_data, "sector": sector}
                )
                if sector_comparison_result.get("error"):
                     self.analysis_errors_count += 1
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Error comparing to sector for {symbol}: {sector_comparison_result['error']}")
                else:
                     sector_comparison_data = sector_comparison_result.get("comparison")


            # Calculate growth rates (using RiskAnalysisMCP tool)
            growth_rates_result = self.risk_analysis_mcp.call_tool(
                "calculate_growth_rates", {"financial_data": financial_statements}
            )
            if growth_rates_result.get("error"):
                 self.analysis_errors_count += 1
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Error calculating growth rates for {symbol}: {growth_rates_result['error']}")
                 growth_rates_data = {}
            else:
                 growth_rates_data = growth_rates_result.get("growth_rates", {})


            # Analyze growth trends (using RiskAnalysisMCP tool)
            growth_trends_data = {}
            if growth_rates_data: # Only analyze trends if rates were calculated
                growth_trends_result = self.risk_analysis_mcp.call_tool(
                    "analyze_growth_trends", {"growth_rates": growth_rates_data}
                )
                if growth_trends_result.get("error"):
                     self.analysis_errors_count += 1
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Error analyzing growth trends for {symbol}: {growth_trends_result['error']}")
                else:
                     growth_trends_data = growth_trends_result.get("trends", {})


            # Score growth quality (using RiskAnalysisMCP tool)
            growth_score_data = {}
            if growth_rates_data: # Only score growth if rates were calculated
                growth_score_result = self.risk_analysis_mcp.call_tool(
                    "score_growth_quality",
                    {"growth_rates": growth_rates_data, "trends": growth_trends_data, "sector": sector}
                )
                if growth_score_result.get("error"):
                     self.analysis_errors_count += 1
                     self.mcp_tool_error_count += 1
                     self.logger.error(f"Error scoring growth quality for {symbol}: {growth_score_result['error']}")
                else:
                     growth_score_data = growth_score_result


            # Get latest earnings (using FinancialDataMCP tool)
            latest_earnings = self.get_latest_earnings(symbol)
            if latest_earnings.get("error"):
                 self.analysis_errors_count += 1
                 self.logger.error(f"Error getting latest earnings for {symbol}: {latest_earnings['error']}")
                 latest_earnings_data = {}
            else:
                 latest_earnings_data = latest_earnings


            # Combine all analysis
            comprehensive_analysis = {
                "symbol": symbol,
                "sector": sector,
                "market_data": market_data,
                "financial_ratios": ratios_data,
                "financial_health": {
                    "overall_score": health_score_data.get("overall_score"),
                    "category_scores": health_score_data.get("category_scores", {}),
                },
                "value_metrics": value_metrics_data,
                "growth_analysis": {
                    "growth_rates": growth_rates_data,
                    "growth_trends": growth_trends_data,
                    "growth_score": growth_score_data.get("overall_score"),
                    "metric_scores": growth_score_data.get("metric_scores", {}),
                },
                "latest_earnings": latest_earnings_data,
                "sector_comparison": sector_comparison_data,
                "timestamp": datetime.now().isoformat(),
            }

            # Store in Redis
            cache_key = f"{self.redis_keys['fundamental_data']}{symbol}"
            self.mcp_tool_call_count += 1
            self.redis_mcp.call_tool(
                "set_json",
                {"key": cache_key, "value": comprehensive_analysis, "expiry": 86400} # 1 day expiration
            )
            self.logger.info(f"Stored comprehensive fundamental analysis for {symbol} in Redis.")

            # Track accuracy metrics
            if health_score_data.get("overall_score") is not None:
                self.logger.gauge("fundamental_analysis_health_score", health_score_data.get("overall_score"), symbol=symbol)
            
            if growth_score_data.get("overall_score") is not None:
                self.logger.gauge("fundamental_analysis_growth_score", growth_score_data.get("overall_score"), symbol=symbol)
            
            # Track overall prediction confidence/accuracy
            if health_score_data.get("overall_score") is not None and growth_score_data.get("overall_score") is not None:
                overall_score = (health_score_data.get("overall_score", 0) + growth_score_data.get("overall_score", 0)) / 2
                self.logger.gauge("fundamental_analysis_overall_score", overall_score, symbol=symbol)
                self.logger.gauge("fundamental_analysis_confidence", overall_score / 100.0 if overall_score else 0.0, symbol=symbol)
            
            # Generate and store chart for visualization of fundamental analysis results
            try:
                chart_data = {
                    "financial_health_score": health_score_data.get("overall_score", 0),
                    "growth_score": growth_score_data.get("overall_score", 0),
                    "value_metrics": value_metrics_data
                }
                chart_file = self.chart_generator.generate_fundamental_analysis_chart(chart_data, symbol)
                self.logger.info(f"Generated fundamental analysis chart for {symbol}", chart_file=chart_file)
                
                # Add chart file path to comprehensive analysis
                comprehensive_analysis["chart_file"] = chart_file
            except Exception as chart_error:
                self.logger.error(f"Error generating chart for {symbol}: {chart_error}")
                comprehensive_analysis["chart_file"] = None

            # Send feedback to selection model (e.g., key scores and metrics)
            feedback_data = {
                "symbol": symbol,
                "fundamental_score": health_score_data.get("overall_score"),
                "growth_score": growth_score_data.get("overall_score"),
                "value_metrics": value_metrics_data,
                "timestamp": datetime.now().isoformat(),
            }
            self.send_feedback_to_selection(feedback_data)
            self.logger.info(f"Sent feedback to selection model for {symbol}.")


            # Send analysis to decision model
            self.send_analysis_to_decision(comprehensive_analysis)
            self.logger.info(f"Sent analysis to decision model for {symbol}.")


            self.companies_analyzed_count += 1
            return comprehensive_analysis

        except Exception as e:
            self.analysis_errors_count += 1
            self.logger.error(
                f"Error in analyze_company for {symbol}: {e}", exc_info=True
            )
            return {"error": str(e), "symbol": symbol}


    def get_selection_data(self) -> Dict[str, Any]:
        """
        Get data from the Selection Model using Redis.

        Returns:
            Selection Model data (or empty dict/error if not available)
        """
        self.logger.info("Getting selection data from Redis...")
        try:
            # Assuming Selection Model stores candidates in a list or set at a known key
            self.mcp_tool_call_count += 1
            # Example: Get members of a set
            result = self.redis_mcp.call_tool("smembers", {"key": self.redis_keys["selection_candidates"]})

            if result and not result.get("error"):
                candidates = result.get("members", [])
                self.logger.info(f"Retrieved {len(candidates)} selection candidates.")
                # Assuming candidates are just symbols, format as expected by analyze_company caller
                return {"selected_companies": [{"symbol": c} for c in candidates]}
            elif result and result.get("error"):
                 self.mcp_tool_error_count += 1
                 self.logger.error(f"Failed to get selection data from Redis: {result.get('error')}")
                 return {"error": result.get('error', 'Failed to get selection data from Redis')}
            else:
                 self.logger.warning("No selection data found in Redis.")
                 return {"selected_companies": []} # Return empty list if no data

        except Exception as e:
            self.logger.error(f"Error getting selection data from Redis: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("fundamental_analysis_model.execution_errors")
            return {"error": str(e)}

    def send_feedback_to_selection(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Send fundamental analysis feedback to the Selection Model using Redis Stream.

        Args:
            analysis_data: Fundamental analysis data to send

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Sending feedback to selection model for symbol: {analysis_data.get('symbol', 'unknown')}")
        try:
            # Add timestamp to the data if not already present
            if "timestamp" not in analysis_data:
                analysis_data["timestamp"] = datetime.now().isoformat()

            # Publish feedback to the selection feedback stream
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                 "xadd", # Using stream for feedback
                 {
                      "stream": self.redis_keys["selection_feedback_stream"],
                      "data": analysis_data
                 }
            )

            if result and not result.get("error"):
                self.logger.info("Sent feedback to selection model via stream.")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to send feedback to selection model: {result.get('error') if result else 'Unknown error'}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending feedback to selection model: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("fundamental_analysis_model.execution_errors")
            return False

    def send_analysis_to_decision(self, analysis_data: Dict[str, Any]) -> bool:
        """
        Send fundamental analysis to the Decision Model using Redis Stream.

        Args:
            analysis_data: Fundamental analysis data to send

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Sending analysis to decision model for symbol: {analysis_data.get('symbol', 'unknown')}")
        try:
            # Add timestamp to the data if not already present
            if "timestamp" not in analysis_data:
                analysis_data["timestamp"] = datetime.now().isoformat()

            # Publish analysis report to the fundamental analysis reports stream
            self.mcp_tool_call_count += 1
            result = self.redis_mcp.call_tool(
                 "xadd", # Using stream for reports
                 {
                      "stream": self.redis_keys["fundamental_analysis_reports_stream"],
                      "data": analysis_data
                 }
            )

            if result and not result.get("error"):
                self.logger.info("Sent analysis to decision model via stream.")
                return True
            else:
                self.mcp_tool_error_count += 1
                self.logger.error(f"Failed to send analysis to decision model: {result.get('error') if result else 'Unknown error'}")
                return False

        except Exception as e:
            self.logger.error(f"Error sending analysis to decision model: {e}", exc_info=True)
            self.execution_errors += 1
            self.logger.counter("fundamental_analysis_model.execution_errors")
            return False

    async def run_periodic_analysis(self):
        """
        Run fundamental analysis periodically.
        """
        while True:
            try:
                self.logger.info("Starting periodic fundamental analysis...")

                # Get list of companies from Selection Model using Redis
                selection_data = self.get_selection_data()
                companies_to_analyze = selection_data.get("selected_companies", [])[
                    : self.max_companies
                ]

                if not companies_to_analyze:
                    self.logger.info("No companies selected for analysis. Skipping analysis cycle.")
                else:
                    self.logger.info(
                        f"Analyzing {len(companies_to_analyze)} companies: {[c.get('symbol') for c in companies_to_analyze]}"
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

                # Update latest analysis timestamp in Redis
                self.mcp_tool_call_count += 1
                self.redis_mcp.call_tool(
                    "set",
                    {"key": self.redis_keys["latest_analysis_timestamp"], "value": datetime.now().isoformat()}
                )
                self.logger.info("Updated latest fundamental analysis timestamp in Redis.")

                self.total_analysis_cycles += 1
                self.logger.info(f"Periodic fundamental analysis cycle {self.total_analysis_cycles} complete.")

            except Exception as e:
                self.logger.error(f"Error during periodic analysis: {e}", exc_info=True)
                self.analysis_errors_count += 1
                self.logger.counter("fundamental_analysis_model.analysis_errors_count")


            # Wait for the next interval
            self.logger.info(f"Waiting for {self.analysis_interval} seconds until next analysis cycle.")
            await asyncio.sleep(self.analysis_interval)

    def start(self):
        """
        Start the periodic analysis loop.
        """
        self.logger.info("Starting FundamentalAnalysisModel...")
        # Running the periodic analysis requires an event loop, typically managed externally.
        # For standalone testing, you might uncomment the asyncio.run line.
        # asyncio.run(self.run_periodic_analysis())
        self.logger.warning("Periodic analysis loop is not started automatically. Call run_periodic_analysis() manually if needed.")
