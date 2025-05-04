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

    This model integrates with ScenarioGenerationMCP, RiskAttributionMCP, and other MCP tools
    to provide comprehensive risk analysis for investment portfolios.
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

        # Redis keys for data storage
        self.redis_keys = {
            "portfolio_risk": "risk:portfolio:",  # Prefix for portfolio risk data
            "scenario_results": "risk:scenarios:",  # Prefix for scenario results
            "risk_attribution": "risk:attribution:",  # Prefix for risk attribution data
            "risk_metrics": "risk:metrics:",  # Prefix for risk metrics
            "optimization_results": "risk:optimization:",  # Prefix for optimization results
            "latest_analysis": "risk:latest_analysis",  # Latest analysis timestamp
            "risk_limits": "risk:limits:",  # Prefix for risk limits
            "risk_alerts": "risk:alerts",  # Risk alerts
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


# Note: The example usage block (main function and __main__ guard)
# has been removed to ensure this file contains only production code.
# Testing should be done via separate test scripts or integration tests.
