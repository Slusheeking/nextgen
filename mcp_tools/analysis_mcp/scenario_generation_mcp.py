"""
Scenario Generation MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
scenario generation capabilities for stress testing financial portfolios.
"""

import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-scenario-generation",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class ScenarioGenerationMCP(BaseMCPServer):
    """
    MCP server for generating market scenarios for stress testing.

    This tool generates various types of scenarios including historical scenario recreation,
    Monte Carlo simulations, and custom scenarios for risk assessment and stress testing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Scenario Generation MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - monte_carlo_simulations: Number of simulations for Monte Carlo methods (default: 10000)
                - historical_lookback_days: Default lookback period for historical scenarios (default: 252)
                - cache_dir: Directory for caching intermediate results
                - random_seed: Seed for random number generation (default: 42)
                - default_confidence_level: Default confidence level for scenarios (default: 0.95)
        """
        super().__init__(name="scenario_generation_mcp", config=config)

        # Set default configurations
        self.monte_carlo_simulations = self.config.get("monte_carlo_simulations", 10000)
        self.historical_lookback_days = self.config.get(
            "historical_lookback_days", 252
        )  # Default to 1 year
        self.cache_dir = self.config.get("cache_dir", "./scenario_cache")
        self.random_seed = self.config.get("random_seed", 42)
        self.default_confidence_level = self.config.get(
            "default_confidence_level", 0.95
        )

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "generate_historical_scenario",
            self.generate_historical_scenario,
            "Generate a scenario based on a historical market event",
            {
                "event_name": {
                    "type": "string",
                    "description": "Name of the historical event (e.g., '2008_financial_crisis', 'covid_crash', 'dot_com_bubble')",
                },
                "asset_returns": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their historical returns time series",
                },
                "lookback_days": {
                    "type": "integer",
                    "description": "Number of days to look back for the historical event",
                    "default": self.historical_lookback_days,
                },
            },
            {
                "type": "object",
                "properties": {
                    "scenario": {"type": "object"},  # Asset returns under the scenario
                    "event_details": {
                        "type": "object"
                    },  # Details about the historical event
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "generate_monte_carlo_scenario",
            self.generate_monte_carlo_scenario,
            "Generate scenarios using Monte Carlo simulation",
            {
                "asset_returns": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their historical returns time series",
                },
                "correlation_matrix": {
                    "type": "array",
                    "description": "Optional correlation matrix between assets. If not provided, will be calculated from historical returns",
                    "required": False,
                },
                "num_scenarios": {
                    "type": "integer",
                    "description": "Number of scenarios to generate",
                    "default": self.monte_carlo_simulations,
                },
                "time_horizon_days": {
                    "type": "integer",
                    "description": "Time horizon for scenarios in days",
                    "default": 20,  # Default to 1 month (20 trading days)
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for scenario statistics",
                    "default": self.default_confidence_level,
                },
            },
            {
                "type": "object",
                "properties": {
                    "scenarios": {"type": "array"},  # Array of scenario dictionaries
                    "statistics": {
                        "type": "object"
                    },  # Statistical summary of scenarios
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "generate_custom_scenario",
            self.generate_custom_scenario,
            "Generate a custom scenario based on specified market movements",
            {
                "asset_returns": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their historical returns time series",
                },
                "shock_factors": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers or market factors to their shock values (e.g., {'SPY': -0.15} for 15% drop in SPY)",
                },
                "correlation_matrix": {
                    "type": "array",
                    "description": "Optional correlation matrix between assets. If not provided, will be calculated from historical returns",
                    "required": False,
                },
                "propagate_shocks": {
                    "type": "boolean",
                    "description": "Whether to propagate shocks to correlated assets",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "scenario": {"type": "object"},  # Asset returns under the scenario
                    "shock_details": {
                        "type": "object"
                    },  # Details about the applied shocks
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_scenario_impact",
            self.calculate_scenario_impact,
            "Calculate the impact of a scenario on a portfolio",
            {
                "portfolio": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their weights in the portfolio",
                },
                "scenario": {
                    "type": "object",
                    "description": "Scenario dictionary mapping asset identifiers to their returns under the scenario",
                },
                "initial_value": {
                    "type": "number",
                    "description": "Initial portfolio value",
                    "default": 1000000,  # Default to $1M
                },
            },
            {
                "type": "object",
                "properties": {
                    "portfolio_return": {
                        "type": "number"
                    },  # Overall portfolio return under the scenario
                    "asset_contributions": {
                        "type": "object"
                    },  # Contribution of each asset to the portfolio return
                    "value_after_scenario": {
                        "type": "number"
                    },  # Portfolio value after the scenario
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "list_historical_events",
            self.list_historical_events,
            "List available historical market events for scenario generation",
            {},
            {
                "type": "object",
                "properties": {
                    "events": {"type": "array"},  # List of available historical events
                    "processing_time": {"type": "number"},
                },
            },
        )

    def generate_historical_scenario(
        self,
        event_name: str,
        asset_returns: Dict[str, List[float]],
        lookback_days: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a scenario based on a historical market event.

        Args:
            event_name: Name of the historical event
            asset_returns: Dictionary mapping asset identifiers to their historical returns
            lookback_days: Number of days to look back for the historical event

        Returns:
            Dictionary with scenario details
        """
        start_time = time.time()
        lookback_days = lookback_days or self.historical_lookback_days

        # Define historical events with their characteristics
        historical_events = {
            "2008_financial_crisis": {
                "start_date": "2008-09-15",  # Lehman Brothers bankruptcy
                "end_date": "2009-03-09",  # Market bottom
                "description": "Global financial crisis triggered by the collapse of Lehman Brothers",
                "peak_drawdown": -0.56,  # S&P 500 peak drawdown
                "duration_days": 517,  # From peak to recovery
                "volatility_factor": 3.2,  # Increase in volatility
            },
            "covid_crash": {
                "start_date": "2020-02-19",  # Market peak before crash
                "end_date": "2020-03-23",  # Market bottom
                "description": "Market crash due to the COVID-19 pandemic",
                "peak_drawdown": -0.34,  # S&P 500 peak drawdown
                "duration_days": 33,  # From peak to bottom
                "volatility_factor": 4.5,  # Increase in volatility
            },
            "dot_com_bubble": {
                "start_date": "2000-03-10",  # NASDAQ peak
                "end_date": "2002-10-09",  # Market bottom
                "description": "Collapse of the dot-com bubble",
                "peak_drawdown": -0.78,  # NASDAQ peak drawdown
                "duration_days": 929,  # From peak to bottom
                "volatility_factor": 2.8,  # Increase in volatility
            },
            "black_monday": {
                "start_date": "1987-10-19",  # Black Monday
                "end_date": "1987-10-19",  # Single day event
                "description": "Stock market crash on October 19, 1987",
                "peak_drawdown": -0.22,  # S&P 500 single day drop
                "duration_days": 1,  # Single day
                "volatility_factor": 7.0,  # Extreme volatility spike
            },
            "2018_q4_selloff": {
                "start_date": "2018-10-01",  # Start of Q4 selloff
                "end_date": "2018-12-24",  # Christmas Eve bottom
                "description": "Q4 2018 market selloff due to Fed rate hikes and trade tensions",
                "peak_drawdown": -0.20,  # S&P 500 drawdown
                "duration_days": 84,  # Duration of selloff
                "volatility_factor": 2.5,  # Increase in volatility
            },
        }

        # Check if the event exists
        if event_name not in historical_events:
            return {
                "error": f"Historical event '{event_name}' not found. Available events: {list(historical_events.keys())}",
                "processing_time": time.time() - start_time,
            }

        event_details = historical_events[event_name]

        # Generate scenario based on the historical event characteristics
        scenario = {}

        try:
            # For each asset, apply the historical event characteristics
            for asset_id, returns in asset_returns.items():
                if not returns or len(returns) < lookback_days:
                    self.logger.warning(
                        f"Insufficient historical data for {asset_id}. Skipping."
                    )
                    continue

                # Calculate asset volatility
                asset_volatility = np.std(returns)

                # Apply the event's volatility factor and drawdown characteristics
                # Calculate expected drawdown for this asset based on its volatility relative to market
                asset_drawdown = event_details["peak_drawdown"] * (
                    asset_volatility / 0.01
                )  # Assuming 1% is typical daily vol

                # Generate a scenario return for this asset
                scenario[asset_id] = asset_drawdown

            return {
                "scenario": scenario,
                "event_details": event_details,
                "processing_time": time.time() - start_time,
            }
        except Exception as e:
            self.logger.error(f"Error generating historical scenario: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
        except Exception as e:
            self.logger.error(f"Error generating historical scenario: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def generate_monte_carlo_scenario(
        self,
        asset_returns: Dict[str, List[float]],
        correlation_matrix: Optional[List[List[float]]] = None,
        num_scenarios: int = None,
        time_horizon_days: int = 20,
        confidence_level: float = None,
    ) -> Dict[str, Any]:
        """
        Generate scenarios using Monte Carlo simulation.

        Args:
            asset_returns: Dictionary mapping asset identifiers to their historical returns
            correlation_matrix: Optional correlation matrix between assets
            num_scenarios: Number of scenarios to generate
            time_horizon_days: Time horizon for scenarios in days
            confidence_level: Confidence level for scenario statistics

        Returns:
            Dictionary with generated scenarios and statistics
        """
        start_time = time.time()
        num_scenarios = num_scenarios or self.monte_carlo_simulations
        confidence_level = confidence_level or self.default_confidence_level

        try:
            # Convert asset returns to a DataFrame for easier manipulation
            returns_df = pd.DataFrame(asset_returns)

            # Calculate mean returns and covariance matrix
            mean_returns = returns_df.mean().values
            cov_matrix = returns_df.cov().values

            # If correlation matrix is provided, use it to adjust the covariance matrix
            if correlation_matrix is not None:
                # Convert to numpy array if it's not already
                corr_matrix = np.array(correlation_matrix)

                # Calculate standard deviations
                std_devs = np.sqrt(np.diag(cov_matrix))

                # Reconstruct covariance matrix using the provided correlation matrix
                cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

            # Generate random scenarios
            scenarios = []
            scenario_returns = np.random.multivariate_normal(
                mean_returns * time_horizon_days,
                cov_matrix * time_horizon_days,
                size=num_scenarios,
            )

            # Convert scenarios to dictionaries
            asset_ids = list(asset_returns.keys())
            for i in range(num_scenarios):
                scenario = {
                    asset_ids[j]: float(scenario_returns[i, j])
                    for j in range(len(asset_ids))
                }
                scenarios.append(scenario)

            # Calculate statistics
            scenario_array = scenario_returns
            var_index = int((1 - confidence_level) * num_scenarios)
            sorted_returns = np.sort(scenario_array, axis=0)

            var_values = sorted_returns[var_index, :]
            cvar_values = sorted_returns[:var_index, :].mean(axis=0)

            statistics = {
                "mean": mean_returns.tolist(),
                "var": var_values.tolist(),
                "cvar": cvar_values.tolist(),
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon_days,
            }

            # Map statistics back to asset IDs
            statistics["mean"] = {
                asset_ids[i]: statistics["mean"][i] for i in range(len(asset_ids))
            }
            statistics["var"] = {
                asset_ids[i]: statistics["var"][i] for i in range(len(asset_ids))
            }
            statistics["cvar"] = {
                asset_ids[i]: statistics["cvar"][i] for i in range(len(asset_ids))
            }

            return {
                "scenarios": scenarios,
                "statistics": statistics,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error generating Monte Carlo scenarios: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def generate_custom_scenario(
        self,
        asset_returns: Dict[str, List[float]],
        shock_factors: Dict[str, float],
        correlation_matrix: Optional[List[List[float]]] = None,
        propagate_shocks: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a custom scenario based on specified market movements.

        Args:
            asset_returns: Dictionary mapping asset identifiers to their historical returns
            shock_factors: Dictionary mapping asset identifiers or market factors to their shock values
            correlation_matrix: Optional correlation matrix between assets
            propagate_shocks: Whether to propagate shocks to correlated assets

        Returns:
            Dictionary with scenario details
        """
        start_time = time.time()

        try:
            # Convert asset returns to a DataFrame for easier manipulation
            returns_df = pd.DataFrame(asset_returns)
            asset_ids = list(asset_returns.keys())

            # Initialize scenario with zeros
            scenario = {asset_id: 0.0 for asset_id in asset_ids}

            # Apply direct shocks
            for asset_id, shock in shock_factors.items():
                if asset_id in scenario:
                    scenario[asset_id] = shock

            # If propagate_shocks is True, propagate shocks to correlated assets
            if propagate_shocks:
                # Calculate correlation matrix if not provided
                if correlation_matrix is None:
                    correlation_matrix = returns_df.corr().values
                else:
                    correlation_matrix = np.array(correlation_matrix)

                # For each shocked asset, propagate to correlated assets
                for i, asset_id in enumerate(asset_ids):
                    if asset_id in shock_factors:
                        shock = shock_factors[asset_id]
                        for j, target_asset in enumerate(asset_ids):
                            if (
                                target_asset != asset_id
                                and target_asset not in shock_factors
                            ):
                                # Propagate shock based on correlation
                                corr = correlation_matrix[i, j]
                                propagated_shock = (
                                    shock * corr * 0.5
                                )  # Dampen the propagated shock
                                scenario[target_asset] += propagated_shock

            # Prepare shock details
            shock_details = {
                "direct_shocks": shock_factors,
                "propagated": propagate_shocks,
                "timestamp": datetime.now().isoformat(),
            }

            return {
                "scenario": scenario,
                "shock_details": shock_details,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error generating custom scenario: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def calculate_scenario_impact(
        self,
        portfolio: Dict[str, float],
        scenario: Dict[str, float],
        initial_value: float = 1000000,
    ) -> Dict[str, Any]:
        """
        Calculate the impact of a scenario on a portfolio.

        Args:
            portfolio: Dictionary mapping asset identifiers to their weights in the portfolio
            scenario: Scenario dictionary mapping asset identifiers to their returns under the scenario
            initial_value: Initial portfolio value

        Returns:
            Dictionary with impact details
        """
        start_time = time.time()

        try:
            # Calculate portfolio return under the scenario
            portfolio_return = 0.0
            asset_contributions = {}

            for asset_id, weight in portfolio.items():
                if asset_id in scenario:
                    asset_return = scenario[asset_id]
                    contribution = weight * asset_return
                    portfolio_return += contribution
                    asset_contributions[asset_id] = {
                        "weight": weight,
                        "scenario_return": asset_return,
                        "contribution": contribution,
                    }
                else:
                    self.logger.warning(
                        f"Asset {asset_id} in portfolio not found in scenario. Assuming zero return."
                    )
                    asset_contributions[asset_id] = {
                        "weight": weight,
                        "scenario_return": 0.0,
                        "contribution": 0.0,
                    }

            # Calculate portfolio value after the scenario
            value_after_scenario = initial_value * (1 + portfolio_return)

            return {
                "portfolio_return": portfolio_return,
                "asset_contributions": asset_contributions,
                "value_after_scenario": value_after_scenario,
                "initial_value": initial_value,
                "processing_time": time.time() - start_time,
            }

        except Exception as e:
            self.logger.error(f"Error calculating scenario impact: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def list_historical_events(self) -> Dict[str, Any]:
        """
        List available historical market events for scenario generation.

        Returns:
            Dictionary with list of available events
        """
        start_time = time.time()

        # Define historical events with their characteristics
        historical_events = [
            {
                "id": "2008_financial_crisis",
                "name": "2008 Financial Crisis",
                "start_date": "2008-09-15",
                "end_date": "2009-03-09",
                "description": "Global financial crisis triggered by the collapse of Lehman Brothers",
                "peak_drawdown": -0.56,
            },
            {
                "id": "covid_crash",
                "name": "COVID-19 Market Crash",
                "start_date": "2020-02-19",
                "end_date": "2020-03-23",
                "description": "Market crash due to the COVID-19 pandemic",
                "peak_drawdown": -0.34,
            },
            {
                "id": "dot_com_bubble",
                "name": "Dot-Com Bubble Burst",
                "start_date": "2000-03-10",
                "end_date": "2002-10-09",
                "description": "Collapse of the dot-com bubble",
                "peak_drawdown": -0.78,
            },
            {
                "id": "black_monday",
                "name": "Black Monday",
                "start_date": "1987-10-19",
                "end_date": "1987-10-19",
                "description": "Stock market crash on October 19, 1987",
                "peak_drawdown": -0.22,
            },
            {
                "id": "2018_q4_selloff",
                "name": "Q4 2018 Market Selloff",
                "start_date": "2018-10-01",
                "end_date": "2018-12-24",
                "description": "Q4 2018 market selloff due to Fed rate hikes and trade tensions",
                "peak_drawdown": -0.20,
            },
        ]

        return {
            "events": historical_events,
            "processing_time": time.time() - start_time,
        }
