"""
Risk Assessment Model

This module defines the RiskAssessmentModel, responsible for evaluating portfolio risk
using scenario generation, risk attribution, and stress testing capabilities.
It integrates with various MCP tools to provide comprehensive risk analysis.
"""

import logging
import json
import time
import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import monitoring
from monitoring.system_monitor import MonitoringManager

# For Redis integration
from mcp_tools.db_mcp.redis_mcp import RedisMCP

# MCP tools
from mcp_tools.analysis_mcp.risk_metrics_mcp import RiskMetricsMCP
from mcp_tools.analysis_mcp.scenario_generation_mcp import ScenarioGenerationMCP
from mcp_tools.analysis_mcp.risk_attribution_mcp import RiskAttributionMCP
from mcp_tools.analysis_mcp.portfolio_optimization_mcp import PortfolioOptimizationMCP

# AutoGen imports
from autogen import (
    Agent,
    AssistantAgent,
    UserProxyAgent,
    register_function,
)


class RiskAssessmentModel:
    """
    Evaluates portfolio risk using scenario generation, risk attribution, and stress testing.
    
    Acts as a central processing hub that:
    1. Collects and analyzes reports from all processing models
    2. Integrates data into comprehensive risk-assessed packages
    3. Provides consolidated reports to the Decision Model
    4. Coordinates with MCP tools to evaluate portfolio and position risk
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RiskAssessmentModel.

        Args:
            config: Configuration dictionary, expected to contain:
                - scenario_generation_config: Config for ScenarioGenerationMCP
                - risk_attribution_config: Config for RiskAttributionMCP
                - risk_metrics_config: Config for RiskMetricsMCP
                - portfolio_optimization_config: Config for PortfolioOptimizationMCP
                - redis_config: Config for RedisMCP
                - default_confidence_level: Default confidence level for risk metrics (default: 0.95)
                - default_time_horizon: Default time horizon in days (default: 20)
                - risk_data_ttl: Time-to-live for risk data in seconds (default: 86400 - 1 day)
                - llm_config: Configuration for AutoGen LLM
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
            service_name="nextgen_risk_assessment-risk-assessment-model"
        )
        self.monitor.log_info(
            "RiskAssessmentModel initialized",
            component="risk_assessment",
            action="initialization",
        )

        # Initialize MCP clients
        self.scenario_generation_mcp = ScenarioGenerationMCP(
            self.config.get("scenario_generation_config")
        )
        self.risk_attribution_mcp = RiskAttributionMCP(
            self.config.get("risk_attribution_config")
        )
        self.risk_metrics_mcp = RiskMetricsMCP(self.config.get("risk_metrics_config"))
        self.portfolio_optimization_mcp = PortfolioOptimizationMCP(
            self.config.get("portfolio_optimization_config")
        )
        self.redis_mcp = RedisMCP(self.config.get("redis_config"))

        # Configuration parameters
        self.default_confidence_level = self.config.get(
            "default_confidence_level", 0.95
        )
        self.default_time_horizon = self.config.get(
            "default_time_horizon", 20
        )  # 20 trading days (approx. 1 month)
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
            
            # Keys for accessing other model reports
            "sentiment_data": "sentiment:data",  # Sentiment analysis reports
            "fundamental_data": "fundamental:data:",  # Prefix for fundamental data per symbol
            "technical_data": "technical:data",  # Technical analysis reports
            "market_data": "market:data",  # Market data reports
            "selection_data": "selection:data",  # Selection Model data
            
            # Keys for storing consolidated packages
            "consolidated_package": "risk:consolidated_package:",  # Prefix for consolidated risk packages
            "package_history": "risk:package_history",  # History of package IDs
            
            # Trade model interaction keys
            "trade_events": "trade:events",  # Trade event stream
            "capital_available": "trade:capital_available",  # Available capital
        }

        # Initialize AutoGen integration
        self.llm_config = self._get_llm_config()
        self.agents = self._setup_agents()

        # Register functions with the agents
        self._register_functions()

        self.logger.info("RiskAssessmentModel initialized.")

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

        # Register scenario generation functions
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
            result = self.scenario_generation_mcp.generate_historical_scenario(
                event_name=event_name,
                asset_returns=asset_returns,
                lookback_days=lookback_days,
            )
            return result or {"error": "Historical scenario generation failed"}

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
            result = self.scenario_generation_mcp.generate_monte_carlo_scenario(
                asset_returns=asset_returns,
                correlation_matrix=correlation_matrix,
                num_scenarios=num_scenarios,
                time_horizon_days=time_horizon_days or self.default_time_horizon,
                confidence_level=confidence_level or self.default_confidence_level,
            )
            return result or {"error": "Monte Carlo scenario generation failed"}

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
            result = self.scenario_generation_mcp.generate_custom_scenario(
                asset_returns=asset_returns,
                shock_factors=shock_factors,
                correlation_matrix=correlation_matrix,
                propagate_shocks=propagate_shocks,
            )
            return result or {"error": "Custom scenario generation failed"}

        # Register risk attribution functions
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
            result = self.risk_attribution_mcp.calculate_risk_contributions(
                portfolio_weights=portfolio_weights,
                asset_returns=asset_returns,
                use_correlation=use_correlation,
            )
            return result or {"error": "Risk contribution calculation failed"}

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
            result = self.risk_attribution_mcp.perform_factor_analysis(
                portfolio_returns=portfolio_returns,
                factor_returns=factor_returns,
                factors=factors,
            )
            return result or {"error": "Factor analysis failed"}

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
            result = self.risk_attribution_mcp.decompose_asset_risk(
                asset_returns=asset_returns,
                factor_returns=factor_returns,
                factors=factors,
            )
            return result or {"error": "Asset risk decomposition failed"}

        # Register risk metrics functions
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
            result = self.risk_metrics_mcp.calculate_var(
                returns=portfolio_returns,
                confidence_level=confidence_level or self.default_confidence_level,
                method=method,
            )
            return result or {"error": "VaR calculation failed"}

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
            result = self.risk_metrics_mcp.calculate_expected_shortfall(
                returns=portfolio_returns,
                confidence_level=confidence_level or self.default_confidence_level,
                method=method,
            )
            return result or {"error": "Expected Shortfall calculation failed"}

        # Register portfolio optimization functions
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
            result = self.portfolio_optimization_mcp.optimize_portfolio(
                asset_returns=asset_returns,
                optimization_objective=optimization_objective,
                constraints=constraints,
                risk_aversion=risk_aversion,
            )
            return result or {"error": "Portfolio optimization failed"}

        # Register scenario impact function
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
            result = self.scenario_generation_mcp.calculate_scenario_impact(
                portfolio=portfolio, scenario=scenario, initial_value=initial_value
            )
            return result or {"error": "Scenario impact calculation failed"}

        # Register risk limit functions
        @register_function(
            name="set_risk_limits",
            description="Set risk limits for a portfolio",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def set_risk_limits(
            portfolio_id: str, risk_limits: Dict[str, Any]
        ) -> Dict[str, Any]:
            return await self.set_risk_limits(portfolio_id, risk_limits)

        @register_function(
            name="check_risk_limits",
            description="Check if portfolio risk metrics exceed defined limits",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def check_risk_limits(
            portfolio_id: str, risk_metrics: Dict[str, Any]
        ) -> Dict[str, Any]:
            return await self.check_risk_limits(portfolio_id, risk_metrics)

        # Register data storage and retrieval functions
        @register_function(
            name="store_risk_analysis",
            description="Store risk analysis results",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def store_risk_analysis(
            portfolio_id: str, analysis_results: Dict[str, Any]
        ) -> Dict[str, Any]:
            return await self.store_risk_analysis(portfolio_id, analysis_results)

        @register_function(
            name="get_risk_analysis",
            description="Retrieve stored risk analysis results",
            caller=risk_assistant,
            executor=user_proxy,
        )
        async def get_risk_analysis(portfolio_id: str) -> Dict[str, Any]:
            return await self.get_risk_analysis(portfolio_id)

        # Register MCP tool access functions
        self._register_mcp_tool_access()

    def _register_mcp_tool_access(self):
        """
        Register MCP tool access functions with the user proxy agent.
        """
        user_proxy = self.agents["user_proxy"]
        risk_assistant = self.agents["risk_assistant"]

        # Define MCP tool access functions
        @register_function(
            name="use_scenario_generation_tool",
            description="Use a tool provided by the Scenario Generation MCP server",
            caller=risk_assistant,
            executor=user_proxy,
        )
        def use_scenario_generation_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.scenario_generation_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_risk_attribution_tool",
            description="Use a tool provided by the Risk Attribution MCP server",
            caller=risk_assistant,
            executor=user_proxy,
        )
        def use_risk_attribution_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.risk_attribution_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_risk_metrics_tool",
            description="Use a tool provided by the Risk Metrics MCP server",
            caller=risk_assistant,
            executor=user_proxy,
        )
        def use_risk_metrics_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.risk_metrics_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_portfolio_optimization_tool",
            description="Use a tool provided by the Portfolio Optimization MCP server",
            caller=risk_assistant,
            executor=user_proxy,
        )
        def use_portfolio_optimization_tool(
            tool_name: str, arguments: Dict[str, Any]
        ) -> Any:
            return self.portfolio_optimization_mcp.call_tool(tool_name, arguments)

        @register_function(
            name="use_redis_tool",
            description="Use a tool provided by the Redis MCP server",
            caller=risk_assistant,
            executor=user_proxy,
        )
        def use_redis_tool(tool_name: str, arguments: Dict[str, Any]) -> Any:
            return self.redis_mcp.call_tool(tool_name, arguments)

    async def analyze_portfolio_risk(
        self,
        portfolio: Dict[str, float],
        asset_returns: Dict[str, List[float]],
        portfolio_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive risk analysis on a portfolio.

        Args:
            portfolio: Dictionary mapping asset identifiers to their weights
            asset_returns: Dictionary mapping asset identifiers to their historical returns
            portfolio_id: Optional identifier for the portfolio

        Returns:
            Dictionary with comprehensive risk analysis results
        """
        self.logger.info(f"Analyzing portfolio risk for {len(portfolio)} assets")

        if not portfolio_id:
            portfolio_id = f"portfolio_{int(time.time())}"

        start_time = time.time()

        try:
            # 1. Calculate basic risk metrics
            portfolio_returns = self._calculate_portfolio_returns(
                portfolio, asset_returns
            )

            var_result = self.risk_metrics_mcp.calculate_var(
                returns=portfolio_returns,
                confidence_level=self.default_confidence_level,
                method="historical",
            )

            es_result = self.risk_metrics_mcp.calculate_expected_shortfall(
                returns=portfolio_returns,
                confidence_level=self.default_confidence_level,
                method="historical",
            )

            # 2. Calculate risk contributions
            risk_contrib_result = (
                self.risk_attribution_mcp.calculate_risk_contributions(
                    portfolio_weights=portfolio,
                    asset_returns=asset_returns,
                    use_correlation=True,
                )
            )

            # 3. Generate historical scenario (2008 financial crisis)
            historical_scenario = (
                self.scenario_generation_mcp.generate_historical_scenario(
                    event_name="2008_financial_crisis", asset_returns=asset_returns
                )
            )

            # 4. Calculate scenario impact
            scenario_impact = None
            if historical_scenario and "scenario" in historical_scenario:
                scenario_impact = (
                    self.scenario_generation_mcp.calculate_scenario_impact(
                        portfolio=portfolio,
                        scenario=historical_scenario["scenario"],
                        initial_value=1000000,  # Assume $1M portfolio
                    )
                )

            # 5. Compile results
            analysis_results = {
                "portfolio_id": portfolio_id,
                "timestamp": datetime.now().isoformat(),
                "portfolio_size": len(portfolio),
                "risk_metrics": {
                    "var": var_result.get("var") if var_result else None,
                    "expected_shortfall": es_result.get("expected_shortfall")
                    if es_result
                    else None,
                    "portfolio_volatility": risk_contrib_result.get(
                        "portfolio_volatility"
                    )
                    if risk_contrib_result
                    else None,
                    "diversification_ratio": risk_contrib_result.get(
                        "diversification_ratio"
                    )
                    if risk_contrib_result
                    else None,
                },
                "risk_contributions": risk_contrib_result.get(
                    "percentage_contributions"
                )
                if risk_contrib_result
                else {},
                "historical_scenario": {
                    "name": "2008_financial_crisis",
                    "impact": scenario_impact.get("portfolio_return")
                    if scenario_impact
                    else None,
                    "value_after_scenario": scenario_impact.get("value_after_scenario")
                    if scenario_impact
                    else None,
                },
                "processing_time": time.time() - start_time,
            }

            # 6. Store results
            await self.store_risk_analysis(portfolio_id, analysis_results)

            # 7. Check risk limits
            limits_check = await self.check_risk_limits(
                portfolio_id, analysis_results["risk_metrics"]
            )
            analysis_results["limits_check"] = limits_check

            return analysis_results

        except Exception as e:
            self.logger.error(f"Error analyzing portfolio risk: {e}")
            return {
                "error": str(e),
                "portfolio_id": portfolio_id,
                "processing_time": time.time() - start_time,
            }

    def _calculate_portfolio_returns(
        self, portfolio: Dict[str, float], asset_returns: Dict[str, List[float]]
    ) -> List[float]:
        """Calculate historical returns for a portfolio based on asset weights and returns."""
        # Normalize weights to sum to 1
        total_weight = sum(portfolio.values())
        normalized_weights = {k: v / total_weight for k, v in portfolio.items()}

        # Find common assets between portfolio and returns
        common_assets = set(normalized_weights.keys()).intersection(
            set(asset_returns.keys())
        )

        if not common_assets:
            raise ValueError("No common assets between portfolio and returns data")

        # Find the shortest return series length
        min_length = min(len(asset_returns[asset]) for asset in common_assets)

        # Calculate portfolio returns for each period
        portfolio_returns = [0] * min_length
        for asset in common_assets:
            weight = normalized_weights[asset]
            returns = asset_returns[asset][:min_length]

            for i in range(min_length):
                portfolio_returns[i] += weight * returns[i]

        return portfolio_returns

    async def set_risk_limits(
        self, portfolio_id: str, risk_limits: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Set risk limits for a portfolio.

        Args:
            portfolio_id: Identifier for the portfolio
            risk_limits: Dictionary with risk limits (e.g., max_var, max_es, max_volatility)

        Returns:
            Status of the operation
        """
        try:
            # Validate risk limits
            valid_limit_types = [
                "max_var",
                "max_es",
                "max_volatility",
                "max_drawdown",
                "max_concentration",
            ]
            validated_limits = {}

            for limit_type, limit_value in risk_limits.items():
                if limit_type in valid_limit_types:
                    validated_limits[limit_type] = float(limit_value)

            # Add timestamp
            validated_limits["timestamp"] = datetime.now().isoformat()
            validated_limits["portfolio_id"] = portfolio_id

            # Store in Redis
            limits_key = f"{self.redis_keys['risk_limits']}{portfolio_id}"
            self.redis_mcp.set_json(limits_key, validated_limits)

            return {
                "status": "success",
                "portfolio_id": portfolio_id,
                "limits_set": validated_limits,
            }

        except Exception as e:
            self.logger.error(f"Error setting risk limits: {e}")
            return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

    async def check_risk_limits(
        self, portfolio_id: str, risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if portfolio risk metrics exceed defined limits.

        Args:
            portfolio_id: Identifier for the portfolio
            risk_metrics: Dictionary with current risk metrics

        Returns:
            Dictionary with limit check results
        """
        try:
            # Get risk limits from Redis
            limits_key = f"{self.redis_keys['risk_limits']}{portfolio_id}"
            risk_limits = self.redis_mcp.get_json(limits_key) or {}

            if not risk_limits:
                return {
                    "status": "no_limits",
                    "portfolio_id": portfolio_id,
                    "message": "No risk limits defined for this portfolio",
                }

            # Check each limit
            limit_checks = {}
            alerts = []

            if "max_var" in risk_limits and "var" in risk_metrics:
                limit_checks["var"] = {
                    "limit": risk_limits["max_var"],
                    "current": risk_metrics["var"],
                    "exceeded": risk_metrics["var"] > risk_limits["max_var"],
                }
                if limit_checks["var"]["exceeded"]:
                    alerts.append(
                        f"VaR limit exceeded: {risk_metrics['var']:.2%} > {risk_limits['max_var']:.2%}"
                    )

            if "max_es" in risk_limits and "expected_shortfall" in risk_metrics:
                limit_checks["expected_shortfall"] = {
                    "limit": risk_limits["max_es"],
                    "current": risk_metrics["expected_shortfall"],
                    "exceeded": risk_metrics["expected_shortfall"]
                    > risk_limits["max_es"],
                }
                if limit_checks["expected_shortfall"]["exceeded"]:
                    alerts.append(
                        f"Expected Shortfall limit exceeded: {risk_metrics['expected_shortfall']:.2%} > {risk_limits['max_es']:.2%}"
                    )

            if (
                "max_volatility" in risk_limits
                and "portfolio_volatility" in risk_metrics
            ):
                limit_checks["portfolio_volatility"] = {
                    "limit": risk_limits["max_volatility"],
                    "current": risk_metrics["portfolio_volatility"],
                    "exceeded": risk_metrics["portfolio_volatility"]
                    > risk_limits["max_volatility"],
                }
                if limit_checks["portfolio_volatility"]["exceeded"]:
                    alerts.append(
                        f"Volatility limit exceeded: {risk_metrics['portfolio_volatility']:.2%} > {risk_limits['max_volatility']:.2%}"
                    )

            # Store alerts if any
            if alerts:
                alert_data = {
                    "portfolio_id": portfolio_id,
                    "timestamp": datetime.now().isoformat(),
                    "alerts": alerts,
                }
                self.redis_mcp.add_to_list(
                    self.redis_keys["risk_alerts"], json.dumps(alert_data)
                )

            return {
                "status": "completed",
                "portfolio_id": portfolio_id,
                "limit_checks": limit_checks,
                "alerts": alerts,
                "limits_exceeded": any(
                    check["exceeded"] for check in limit_checks.values()
                ),
            }

        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

    async def store_risk_analysis(
        self, portfolio_id: str, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store risk analysis results in Redis.

        Args:
            portfolio_id: Identifier for the portfolio
            analysis_results: Risk analysis results to store

        Returns:
            Status of the operation
        """
        try:
            # Store full analysis
            analysis_key = f"{self.redis_keys['portfolio_risk']}{portfolio_id}"
            self.redis_mcp.set_json(
                analysis_key, analysis_results, ex=self.risk_data_ttl
            )

            # Store risk metrics separately for easier access
            if "risk_metrics" in analysis_results:
                metrics_key = f"{self.redis_keys['risk_metrics']}{portfolio_id}"
                self.redis_mcp.set_json(
                    metrics_key, analysis_results["risk_metrics"], ex=self.risk_data_ttl
                )

            # Store risk attribution data
            if "risk_contributions" in analysis_results:
                attribution_key = f"{self.redis_keys['risk_attribution']}{portfolio_id}"
                self.redis_mcp.set_json(
                    attribution_key,
                    analysis_results["risk_contributions"],
                    ex=self.risk_data_ttl,
                )

            # Store scenario results
            if "historical_scenario" in analysis_results:
                scenario_key = f"{self.redis_keys['scenario_results']}{portfolio_id}"
                self.redis_mcp.set_json(
                    scenario_key,
                    analysis_results["historical_scenario"],
                    ex=self.risk_data_ttl,
                )

            # Update latest analysis timestamp
            self.redis_mcp.set_json(
                self.redis_keys["latest_analysis"],
                {"timestamp": datetime.now().isoformat(), "portfolio_id": portfolio_id},
            )

            return {
                "status": "success",
                "portfolio_id": portfolio_id,
                "stored_at": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error storing risk analysis: {e}")
            return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

    async def get_risk_analysis(self, portfolio_id: str) -> Dict[str, Any]:
        """
        Retrieve stored risk analysis results from Redis.

        Args:
            portfolio_id: Identifier for the portfolio

        Returns:
            Dictionary with risk analysis results or error
        """
        try:
            analysis_key = f"{self.redis_keys['portfolio_risk']}{portfolio_id}"
            analysis_results = self.redis_mcp.get_json(analysis_key)

            if analysis_results:
                return {
                    "status": "success",
                    "portfolio_id": portfolio_id,
                    "analysis_results": analysis_results,
                }
            else:
                return {
                    "status": "not_found",
                    "portfolio_id": portfolio_id,
                    "message": "Risk analysis results not found for this portfolio",
                }

        except Exception as e:
            self.logger.error(f"Error retrieving risk analysis: {e}")
            return {"status": "error", "portfolio_id": portfolio_id, "error": str(e)}

    async def collect_model_reports(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Collect reports from all processing models for the specified symbols.
        
        Args:
            symbols: List of symbols to collect reports for
            
        Returns:
            Dictionary containing all collected reports
        """
        self.logger.info(f"Collecting model reports for {len(symbols)} symbols")
        
        start_time = time.time()
        collected_reports = {
            "timestamp": datetime.now().isoformat(),
            "symbols_requested": symbols,
            "model_reports": {},
        }
        
        try:
            # Collect Sentiment Analysis reports
            sentiment_data = self.redis_mcp.get_json(self.redis_keys["sentiment_data"]) or {}
            if sentiment_data and "symbols" in sentiment_data:
                collected_reports["model_reports"]["sentiment"] = {
                    "model_id": "sentiment_model",
                    "timestamp": sentiment_data.get("timestamp", datetime.now().isoformat()),
                    "symbols": {
                        symbol: data 
                        for symbol, data in sentiment_data["symbols"].items()
                        if symbol in symbols
                    }
                }
            
            # Collect Technical Analysis reports
            technical_data = self.redis_mcp.get_json(self.redis_keys["technical_data"]) or {}
            if technical_data and "symbols" in technical_data:
                collected_reports["model_reports"]["technical"] = {
                    "model_id": "technical_model",
                    "timestamp": technical_data.get("timestamp", datetime.now().isoformat()),
                    "symbols": {
                        symbol: data 
                        for symbol, data in technical_data["symbols"].items()
                        if symbol in symbols
                    }
                }
            
            # Collect Fundamental Analysis reports
            fundamental_reports = {}
            for symbol in symbols:
                key = f"{self.redis_keys['fundamental_data']}{symbol}"
                symbol_data = self.redis_mcp.get_json(key)
                if symbol_data:
                    fundamental_reports[symbol] = symbol_data
            
            if fundamental_reports:
                collected_reports["model_reports"]["fundamental"] = {
                    "model_id": "fundamental_model",
                    "timestamp": datetime.now().isoformat(),
                    "symbols": fundamental_reports
                }
            
            # Collect Market Data reports
            market_data = self.redis_mcp.get_json(self.redis_keys["market_data"]) or {}
            if market_data:
                collected_reports["model_reports"]["market"] = {
                    "model_id": "market_data",
                    "timestamp": market_data.get("timestamp", datetime.now().isoformat()),
                    "data": market_data
                }
            
            # Add processing time
            collected_reports["processing_time"] = time.time() - start_time
            collected_reports["symbols_found"] = list(set(
                symbol for model in collected_reports["model_reports"].values() 
                for symbol in model.get("symbols", {}).keys()
            ))
            
            return collected_reports
            
        except Exception as e:
            self.logger.error(f"Error collecting model reports: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "symbols_requested": symbols
            }
    
    async def create_consolidated_package(
        self, 
        symbols: List[str], 
        request_id: Optional[str] = None,
        available_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Create a consolidated package of reports for the Decision Model.
        
        This is the central function that:
        1. Collects reports from all processing models
        2. Performs risk assessment for each symbol
        3. Adds position sizing recommendations
        4. Creates a package with market context
        
        Args:
            symbols: List of symbols to analyze
            request_id: Optional request identifier
            available_capital: Optional amount of available capital
            
        Returns:
            Consolidated package with all analysis
        """
        self.logger.info(f"Creating consolidated package for {len(symbols)} symbols")
        
        if not request_id:
            request_id = f"req_{int(time.time())}"
            
        package_id = f"risk_pkg_{int(time.time())}"
        start_time = time.time()
        
        try:
            # 1. Collect all model reports
            reports = await self.collect_model_reports(symbols)
            
            if "error" in reports:
                return {"error": reports["error"], "request_id": request_id}
                
            # 2. Get market conditions
            market_context = {}
            if "market" in reports["model_reports"]:
                market_context = reports["model_reports"]["market"].get("data", {})
            
            # 3. Process each symbol
            symbols_analysis = {}
            for symbol in symbols:
                symbol_analysis = await self.process_symbol_for_decision(
                    symbol, 
                    reports["model_reports"],
                    available_capital
                )
                if symbol_analysis:
                    symbols_analysis[symbol] = symbol_analysis
            
            # 4. Create the consolidated package
            package = {
                "package_id": package_id,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "symbols": symbols_analysis,
                "market_context": {
                    "volatility_regime": market_context.get("volatility_state", "normal"),
                    "liquidity_conditions": market_context.get("liquidity_state", "normal"),
                    "correlation_regime": market_context.get("correlation_state", "normal"),
                    "is_market_open": market_context.get("is_market_open", True),
                },
                "available_capital": available_capital,
                "processing_time": time.time() - start_time
            }
            
            # 5. Store the package in Redis
            package_key = f"{self.redis_keys['consolidated_package']}{package_id}"
            self.redis_mcp.set_json(package_key, package, ex=self.risk_data_ttl)
            
            # 6. Add to package history
            self.redis_mcp.add_to_sorted_set(
                self.redis_keys["package_history"], 
                package_id, 
                int(time.time())
            )
            
            self.logger.info(f"Created consolidated package {package_id} with {len(symbols_analysis)} symbols")
            return package
            
        except Exception as e:
            self.logger.error(f"Error creating consolidated package: {e}")
            return {
                "error": str(e),
                "request_id": request_id,
                "package_id": package_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_symbol_for_decision(
        self, 
        symbol: str, 
        model_reports: Dict[str, Any],
        available_capital: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process all data for a single symbol to prepare it for decision-making.
        
        Args:
            symbol: Stock symbol to process
            model_reports: Dictionary of reports from all models
            available_capital: Optional amount of available capital
            
        Returns:
            Dictionary with comprehensive symbol analysis
        """
        self.logger.info(f"Processing symbol {symbol} for decision")
        
        try:
            # Extract all reports for this symbol
            symbol_data = {
                "model_reports": {},
                "risk_assessment": {},
                "position_recommendation": {},
                "portfolio_impact": {}
            }
            
            # Extract data from each model's report
            for model_name, model_report in model_reports.items():
                if model_name == "market":
                    continue  # Market data is handled separately
                    
                if "symbols" in model_report and symbol in model_report["symbols"]:
                    symbol_data["model_reports"][model_name] = model_report["symbols"][symbol]
            
            # If we have no data for this symbol, skip it
            if not symbol_data["model_reports"]:
                self.logger.warning(f"No model reports found for symbol {symbol}")
                return None
            
            # Calculate risk assessment
            risk_score = 0.5  # Default neutral risk score
            volatility_risk = 0.5
            correlation_risk = 0.5
            liquidity_risk = 0.5
            
            # If we have technical data, use it for volatility risk
            if "technical" in symbol_data["model_reports"]:
                tech_data = symbol_data["model_reports"]["technical"]
                if "volatility" in tech_data:
                    volatility = tech_data["volatility"]
                    # Higher volatility = higher risk (0-1 scale)
                    volatility_risk = min(1.0, volatility / 0.3)  # Normalize: 30% annualized vol = 1.0 risk
            
            # If we have fundamental data, adjust risk score
            if "fundamental" in symbol_data["model_reports"]:
                fund_data = symbol_data["model_reports"]["fundamental"]
                if "health_score" in fund_data:
                    # Higher health score = lower risk (0-1 scale, inverted)
                    liquidity_risk = 1.0 - fund_data["health_score"]
            
            # Calculate overall risk score (weighted average)
            risk_score = (0.4 * volatility_risk) + (0.3 * correlation_risk) + (0.3 * liquidity_risk)
            
            # Add risk assessment to symbol data
            symbol_data["risk_assessment"] = {
                "overall_risk_score": risk_score,
                "volatility_risk": volatility_risk,
                "correlation_risk": correlation_risk,
                "liquidity_risk": liquidity_risk
            }
            
            # Calculate position recommendation if capital is available
            if available_capital and available_capital > 0:
                # Base size on risk score - lower risk allows larger position
                risk_factor = 1.0 - risk_score  # Invert risk score: higher = better
                
                # Determine confidence score from model reports
                confidence_score = 0.5  # Default neutral confidence
                
                # If we have sentiment data, use it for confidence adjustment
                if "sentiment" in symbol_data["model_reports"]:
                    sent_data = symbol_data["model_reports"]["sentiment"]
                    if "overall_score" in sent_data:
                        # Sentiment score directly contributes to confidence
                        confidence_score = sent_data["overall_score"]
                
                # If we have technical data, use it for confidence adjustment
                if "technical" in symbol_data["model_reports"]:
                    tech_data = symbol_data["model_reports"]["technical"]
                    if "signal_strength" in tech_data:
                        # Technical signals contribute to confidence
                        tech_confidence = tech_data["signal_strength"]
                        # Combine with existing confidence (weighted average)
                        confidence_score = (0.6 * confidence_score) + (0.4 * tech_confidence)
                
                # Calculate position size based on confidence and risk
                max_position_size = min(available_capital * 0.1, 25000)  # Max 10% of capital or $25k
                recommended_size = max_position_size * confidence_score * risk_factor
                
                # Add position recommendation to symbol data
                symbol_data["position_recommendation"] = {
                    "max_position_size": float(max_position_size),
                    "recommended_size": float(recommended_size),
                    "sizing_confidence": float(confidence_score * risk_factor)
                }
                
                # Add portfolio impact assessment
                symbol_data["portfolio_impact"] = {
                    "diversification_impact": 0.15,  # Placeholder
                    "var_contribution": 0.08  # Placeholder
                }
            
            return symbol_data
            
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol} for decision: {e}")
            return {
                "error": str(e),
                "symbol": symbol
            }
    
    async def notify_decision_model(self, package_id: str) -> Dict[str, Any]:
        """
        Notify the Decision Model about a new consolidated package.
        
        Args:
            package_id: ID of the consolidated package
            
        Returns:
            Status of the notification
        """
        try:
            # Get the package
            package_key = f"{self.redis_keys['consolidated_package']}{package_id}"
            package = self.redis_mcp.get_json(package_key)
            
            if not package:
                return {
                    "status": "error",
                    "message": f"Package {package_id} not found"
                }
            
            # Create notification
            notification = {
                "notification_type": "risk_package_ready",
                "package_id": package_id,
                "timestamp": datetime.now().isoformat(),
                "symbol_count": len(package.get("symbols", {})),
                "request_id": package.get("request_id")
            }
            
            # Store in Redis stream for Decision Model
            stream_key = "decision:notifications"
            self.redis_mcp.add_to_stream(stream_key, notification)
            
            self.logger.info(f"Notified Decision Model about package {package_id}")
            return {
                "status": "success",
                "notification": notification
            }
            
        except Exception as e:
            self.logger.error(f"Error notifying Decision Model: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def monitor_selection_responses(self) -> Dict[str, Any]:
        """
        Monitor for new selection responses and process them.
        
        This method:
        1. Checks the selection:responses stream for new responses
        2. Creates consolidated packages for each response
        3. Notifies the Decision Model about the packages
        
        Returns:
            Status of the monitoring operation
        """
        try:
            # Read from the selection responses stream
            stream_key = "selection:responses"
            responses = self.redis_mcp.read_from_stream(stream_key, count=5)
            
            if not responses:
                return {"status": "no_responses"}
            
            results = []
            for response_id, response_data in responses:
                self.logger.info(f"Processing selection response: {response_data.get('request_id')}")
                
                # Only process successful responses
                if response_data.get("status") == "success":
                    # Get candidate symbols
                    candidates = response_data.get("candidates", [])
                    symbols = [c.get("symbol") for c in candidates if c.get("symbol")]
                    
                    # Get available capital
                    available_capital = response_data.get("available_capital")
                    
                    if symbols:
                        # Create consolidated package
                        package = await self.create_consolidated_package(
                            symbols=symbols,
                            request_id=response_data.get("request_id"),
                            available_capital=available_capital
                        )
                        
                        # Notify Decision Model
                        if package and "error" not in package:
                            notification = await self.notify_decision_model(package["package_id"])
                            results.append({
                                "request_id": response_data.get("request_id"),
                                "package_id": package["package_id"],
                                "notification": notification.get("status")
                            })
                
                # Acknowledge the message
                self.redis_mcp.acknowledge_from_stream(stream_key, response_id)
            
            return {
                "status": "success",
                "processed_count": len(results),
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring selection responses: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def monitor_trade_events(self) -> Dict[str, Any]:
        """
        Monitor trade events stream for position updates and capital availability.
        
        Returns:
            Status of the monitoring operation
        """
        try:
            # Read from the trade events stream
            stream_key = self.redis_keys["trade_events"]
            events = self.redis_mcp.read_from_stream(stream_key, count=5)
            
            if not events:
                return {"status": "no_events"}
            
            results = []
            for event_id, event_data in events:
                event_type = event_data.get("event_type")
                
                # Process capital available events
                if event_type == "capital_available":
                    amount = event_data.get("event_data", {}).get("amount", 0.0)
                    
                    # Store capital available information
                    self.redis_mcp.set_value(
                        self.redis_keys["capital_available"],
                        str(amount)
                    )
                    
                    results.append({
                        "event_type": event_type,
                        "amount": amount,
                        "action": "capital_recorded"
                    })
                
                # Acknowledge the message
                self.redis_mcp.acknowledge_from_stream(stream_key, event_id)
            
            return {
                "status": "success",
                "processed_count": len(results),
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring trade events: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def run_risk_analysis_agent(self, query: str) -> Dict[str, Any]:
        """
        Run risk analysis using AutoGen agents.

        Args:
            query: Query or instruction for risk analysis

        Returns:
            Results of the risk analysis
        """
        self.logger.info(f"Running risk analysis with query: {query}")

        risk_assistant = self.agents.get("risk_assistant")
        user_proxy = self.agents.get("user_proxy")

        if not risk_assistant or not user_proxy:
            return {"error": "AutoGen agents not initialized"}

        try:
            # Initiate chat with the risk assistant
            user_proxy.initiate_chat(risk_assistant, message=query)

            # Get the last message from the assistant
            last_message = user_proxy.last_message(risk_assistant)
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
