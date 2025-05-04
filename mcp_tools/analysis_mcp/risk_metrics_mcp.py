"""
Risk Metrics MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
risk metric calculations for financial assets and portfolios.
"""

import os
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from scipy.stats import norm

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-risk-metrics",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class RiskMetricsMCP(BaseMCPServer):
    """
    MCP server for calculating financial risk metrics.

    This tool calculates metrics like Value at Risk (VaR), Conditional VaR (CVaR),
    volatility, Sharpe ratio, Sortino ratio, and performs stress tests.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Risk Metrics MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - default_confidence_level: Default confidence level for VaR (e.g., 0.95)
                - default_time_horizon_days: Default time horizon for VaR in days
                - monte_carlo_simulations: Number of simulations for Monte Carlo methods
                - cache_dir: Directory for caching intermediate results
        """
        super().__init__(name="risk_metrics_mcp", config=config)

        # Set default configurations
        self.default_confidence_level = self.config.get(
            "default_confidence_level", 0.95
        )
        self.default_time_horizon_days = self.config.get("default_time_horizon_days", 1)
        self.monte_carlo_simulations = self.config.get("monte_carlo_simulations", 10000)
        self.cache_dir = self.config.get("cache_dir", "./risk_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "calculate_volatility",
            self.calculate_volatility,
            "Calculate historical volatility for a series of returns",
            {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns (e.g., daily returns)",
                },
                "window": {
                    "type": "integer",
                    "description": "Rolling window size for volatility calculation (optional)",
                    "required": False,
                },
                "annualization_factor": {
                    "type": "number",
                    "description": "Factor to annualize volatility (e.g., 252 for daily returns)",
                    "default": 252,
                },
            },
            {
                "type": "object",
                "properties": {
                    "volatility": {"type": "number"},  # Annualized volatility
                    "rolling_volatility": {"type": "array"},  # If window is provided
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_var",
            self.calculate_var,
            "Calculate Value at Risk (VaR) for a series of returns",
            {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns",
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for VaR (e.g., 0.95, 0.99)",
                    "default": self.default_confidence_level,
                },
                "method": {
                    "type": "string",
                    "description": "Method for VaR calculation: 'historical', 'parametric', 'monte_carlo'",
                    "default": "historical",
                },
                "time_horizon_days": {
                    "type": "integer",
                    "description": "Time horizon for VaR in days",
                    "default": self.default_time_horizon_days,
                },
            },
            {
                "type": "object",
                "properties": {
                    "var": {"type": "number"},  # Value at Risk as a negative number
                    "confidence_level": {"type": "number"},
                    "time_horizon_days": {"type": "integer"},
                    "method": {"type": "string"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_cvar",
            self.calculate_cvar,
            "Calculate Conditional Value at Risk (CVaR) / Expected Shortfall",
            {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns",
                },
                "confidence_level": {
                    "type": "number",
                    "description": "Confidence level for CVaR (e.g., 0.95, 0.99)",
                    "default": self.default_confidence_level,
                },
                "method": {
                    "type": "string",
                    "description": "Method for CVaR calculation: 'historical', 'parametric', 'monte_carlo'",
                    "default": "historical",
                },
                "time_horizon_days": {
                    "type": "integer",
                    "description": "Time horizon for CVaR in days",
                    "default": self.default_time_horizon_days,
                },
            },
            {
                "type": "object",
                "properties": {
                    "cvar": {"type": "number"},  # Conditional VaR as a negative number
                    "confidence_level": {"type": "number"},
                    "time_horizon_days": {"type": "integer"},
                    "method": {"type": "string"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_sharpe_ratio",
            self.calculate_sharpe_ratio,
            "Calculate the Sharpe ratio for a series of returns",
            {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns",
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": "Annualized risk-free rate (e.g., 0.02 for 2%)",
                    "default": 0.0,
                },
                "periods_per_year": {
                    "type": "integer",
                    "description": "Number of return periods per year (e.g., 252 for daily, 12 for monthly)",
                    "default": 252,
                },
            },
            {
                "type": "object",
                "properties": {
                    "sharpe_ratio": {"type": "number"},
                    "annualized_return": {"type": "number"},
                    "annualized_volatility": {"type": "number"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_sortino_ratio",
            self.calculate_sortino_ratio,
            "Calculate the Sortino ratio (uses downside deviation)",
            {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns",
                },
                "risk_free_rate": {
                    "type": "number",
                    "description": "Annualized risk-free rate",
                    "default": 0.0,
                },
                "target_return": {
                    "type": "number",
                    "description": "Target return for downside deviation calculation (optional, defaults to risk-free rate)",
                    "required": False,
                },
                "periods_per_year": {
                    "type": "integer",
                    "description": "Number of return periods per year",
                    "default": 252,
                },
            },
            {
                "type": "object",
                "properties": {
                    "sortino_ratio": {"type": "number"},
                    "annualized_return": {"type": "number"},
                    "downside_deviation": {"type": "number"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_max_drawdown",
            self.calculate_max_drawdown,
            "Calculate the maximum drawdown for a series of returns or prices",
            {
                "data": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns or prices",
                },
                "is_returns": {
                    "type": "boolean",
                    "description": "True if input data is returns, False if prices",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "max_drawdown": {"type": "number"},  # As a negative percentage
                    "peak_index": {"type": "integer"},
                    "trough_index": {"type": "integer"},
                    "recovery_index": {"type": "integer"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "stress_test",
            self.stress_test,
            "Perform stress tests on returns based on historical scenarios",
            {
                "returns": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Time series of asset returns",
                },
                "scenarios": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of stress test scenarios (e.g., {'name': '2008_crash', 'factor_shocks': {...}})",
                },
            },
            {
                "type": "object",
                "properties": {
                    "results": {"type": "array"},  # List of results per scenario
                    "processing_time": {"type": "number"},
                },
            },
        )

    def calculate_volatility(
        self,
        returns: List[float],
        window: Optional[int] = None,
        annualization_factor: int = 252,
    ) -> Dict[str, Any]:
        """Calculate historical volatility."""
        start_time = time.time()

        if not returns:
            return {
                "volatility": None,
                "rolling_volatility": None,
                "error": "Input returns list is empty",
                "processing_time": 0,
            }

        returns_series = pd.Series(returns)

        # Calculate overall volatility
        volatility = returns_series.std() * np.sqrt(annualization_factor)

        # Calculate rolling volatility if window is provided
        rolling_volatility = None
        if window and window > 0 and window <= len(returns_series):
            rolling_vol = returns_series.rolling(window=window).std() * np.sqrt(
                annualization_factor
            )
            rolling_volatility = (
                rolling_vol.replace([np.inf, -np.inf], np.nan).fillna(0).tolist()
            )  # Handle NaN/inf
        elif window:
            self.logger.warning(
                f"Window size {window} is invalid for data length {len(returns_series)}"
            )

        return {
            "volatility": float(volatility) if not np.isnan(volatility) else None,
            "rolling_volatility": rolling_volatility,
            "processing_time": time.time() - start_time,
        }

    def calculate_var(
        self,
        returns: List[float],
        confidence_level: float = None,
        method: str = "historical",
        time_horizon_days: int = None,
    ) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR)."""
        start_time = time.time()

        if not returns:
            return {
                "var": None,
                "error": "Input returns list is empty",
                "processing_time": 0,
            }

        confidence_level = confidence_level or self.default_confidence_level
        time_horizon_days = time_horizon_days or self.default_time_horizon_days

        returns_array = np.array(returns)
        var = None

        try:
            if method == "historical":
                # Historical Simulation VaR
                if len(returns_array) > 0:
                    var_percentile = np.percentile(
                        returns_array, (1 - confidence_level) * 100
                    )
                    # Scale VaR for the time horizon
                    var = var_percentile * np.sqrt(time_horizon_days)
                else:
                    var = np.nan

            elif method == "parametric":
                # Parametric (Variance-Covariance) VaR - assumes normality
                mean_return = np.mean(returns_array)
                std_dev = np.std(returns_array)
                # Z-score for the confidence level
                z_score = norm.ppf(1 - confidence_level)
                # Calculate VaR
                var_daily = mean_return + z_score * std_dev
                # Scale VaR for the time horizon
                var = var_daily * np.sqrt(time_horizon_days)

            elif method == "monte_carlo":
                # Monte Carlo Simulation VaR
                mean_return = np.mean(returns_array)
                std_dev = np.std(returns_array)

                # Simulate future returns
                simulated_returns = np.random.normal(
                    mean_return * time_horizon_days,
                    std_dev * np.sqrt(time_horizon_days),
                    self.monte_carlo_simulations,
                )

                # Calculate VaR from simulated returns
                var = np.percentile(simulated_returns, (1 - confidence_level) * 100)

            else:
                return {
                    "var": None,
                    "error": f"Unsupported VaR method: {method}",
                    "processing_time": time.time() - start_time,
                }

        except Exception as e:
            self.logger.error(f"Error calculating VaR ({method}): {e}")
            return {
                "var": None,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

        return {
            "var": float(var) if not np.isnan(var) else None,
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon_days,
            "method": method,
            "processing_time": time.time() - start_time,
        }

    def calculate_cvar(
        self,
        returns: List[float],
        confidence_level: float = None,
        method: str = "historical",
        time_horizon_days: int = None,
    ) -> Dict[str, Any]:
        """Calculate Conditional Value at Risk (CVaR) / Expected Shortfall."""
        start_time = time.time()

        if not returns:
            return {
                "cvar": None,
                "error": "Input returns list is empty",
                "processing_time": 0,
            }

        confidence_level = confidence_level or self.default_confidence_level
        time_horizon_days = time_horizon_days or self.default_time_horizon_days

        returns_array = np.array(returns)
        cvar = None

        try:
            # First, calculate the VaR for the given method
            var_result = self.calculate_var(
                returns, confidence_level, method, time_horizon_days
            )
            var_value = var_result.get("var")

            if var_value is None:
                # If VaR calculation failed, CVaR cannot be calculated
                return {
                    "cvar": None,
                    "error": var_result.get(
                        "error", "Failed to calculate VaR for CVaR"
                    ),
                    "processing_time": time.time() - start_time,
                }

            if method == "historical":
                # Historical CVaR: Average of returns worse than VaR
                tail_returns = returns_array[
                    returns_array <= var_value / np.sqrt(time_horizon_days)
                ]  # Compare with daily VaR equivalent
                if len(tail_returns) > 0:
                    cvar_daily = np.mean(tail_returns)
                    cvar = cvar_daily * np.sqrt(time_horizon_days)
                else:
                    cvar = var_value  # If no returns are worse, CVaR is VaR

            elif method == "parametric":
                # Parametric CVaR (assuming normality)
                mean_return = np.mean(returns_array)
                std_dev = np.std(returns_array)
                z_score = norm.ppf(1 - confidence_level)
                # Formula for CVaR under normality
                cvar_daily = mean_return - std_dev * (
                    norm.pdf(z_score) / (1 - confidence_level)
                )
                cvar = cvar_daily * np.sqrt(time_horizon_days)

            elif method == "monte_carlo":
                # Monte Carlo CVaR
                mean_return = np.mean(returns_array)
                std_dev = np.std(returns_array)

                # Simulate future returns (same simulation as in VaR)
                simulated_returns = np.random.normal(
                    mean_return * time_horizon_days,
                    std_dev * np.sqrt(time_horizon_days),
                    self.monte_carlo_simulations,
                )

                # Calculate CVaR from simulated returns worse than VaR
                tail_simulated_returns = simulated_returns[
                    simulated_returns <= var_value
                ]
                if len(tail_simulated_returns) > 0:
                    cvar = np.mean(tail_simulated_returns)
                else:
                    cvar = var_value  # If no simulated returns are worse, CVaR is VaR

            else:
                return {
                    "cvar": None,
                    "error": f"Unsupported CVaR method: {method}",
                    "processing_time": time.time() - start_time,
                }

        except Exception as e:
            self.logger.error(f"Error calculating CVaR ({method}): {e}")
            return {
                "cvar": None,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

        return {
            "cvar": float(cvar) if not np.isnan(cvar) else None,
            "confidence_level": confidence_level,
            "time_horizon_days": time_horizon_days,
            "method": method,
            "processing_time": time.time() - start_time,
        }

    def calculate_sharpe_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """Calculate the Sharpe ratio."""
        start_time = time.time()

        if not returns or len(returns) < 2:
            return {
                "sharpe_ratio": None,
                "error": "Not enough return data points",
                "processing_time": 0,
            }

        returns_series = pd.Series(returns)

        # Calculate excess returns per period
        # Adjust risk-free rate to the period frequency
        periodic_risk_free_rate = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns_series - periodic_risk_free_rate

        # Calculate annualized mean excess return
        mean_excess_return = excess_returns.mean()
        annualized_return = (1 + mean_excess_return) ** periods_per_year - 1

        #
        # Calculate annualized volatility of excess returns (or just returns,
        # common practice)
        volatility = returns_series.std() * np.sqrt(periods_per_year)

        # Calculate Sharpe Ratio
        if volatility is not None and volatility != 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility
        else:
            sharpe_ratio = np.nan  # Undefined if volatility is zero

        # Also calculate annualized return of the original series
        mean_return = returns_series.mean()
        total_annualized_return = (1 + mean_return) ** periods_per_year - 1

        return {
            "sharpe_ratio": float(sharpe_ratio) if not np.isnan(sharpe_ratio) else None,
            "annualized_return": float(total_annualized_return)
            if not np.isnan(total_annualized_return)
            else None,
            "annualized_volatility": float(volatility)
            if not np.isnan(volatility)
            else None,
            "processing_time": time.time() - start_time,
        }

    def calculate_sortino_ratio(
        self,
        returns: List[float],
        risk_free_rate: float = 0.0,
        target_return: Optional[float] = None,
        periods_per_year: int = 252,
    ) -> Dict[str, Any]:
        """Calculate the Sortino ratio."""
        start_time = time.time()

        if not returns or len(returns) < 2:
            return {
                "sortino_ratio": None,
                "error": "Not enough return data points",
                "processing_time": 0,
            }

        returns_series = pd.Series(returns)

        # Use risk-free rate as target return if not specified
        if target_return is None:
            target_return = risk_free_rate

        # Adjust target return to the period frequency
        periodic_target_return = (1 + target_return) ** (1 / periods_per_year) - 1

        # Calculate excess returns over target
        excess_returns = returns_series - periodic_target_return

        # Calculate annualized mean excess return
        mean_excess_return = excess_returns.mean()
        annualized_excess_return = (1 + mean_excess_return) ** periods_per_year - 1

        # Calculate downside deviation
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation_periodic = np.sqrt(np.mean(downside_returns**2))
            downside_deviation_annualized = downside_deviation_periodic * np.sqrt(
                periods_per_year
            )
        else:
            downside_deviation_annualized = 0.0  # No downside returns

        # Calculate Sortino Ratio
        if (
            downside_deviation_annualized is not None
            and downside_deviation_annualized != 0
        ):
            sortino_ratio = annualized_excess_return / downside_deviation_annualized
        else:
            #
            # If downside deviation is zero, ratio is technically infinite if
            # return > target, 0 otherwise
            sortino_ratio = np.inf if annualized_excess_return > 0 else 0.0

        # Calculate annualized return of the original series
        mean_return = returns_series.mean()
        total_annualized_return = (1 + mean_return) ** periods_per_year - 1

        return {
            "sortino_ratio": float(sortino_ratio)
            if np.isfinite(sortino_ratio)
            else None,  # Return None for Inf
            "annualized_return": float(total_annualized_return)
            if not np.isnan(total_annualized_return)
            else None,
            "downside_deviation": float(downside_deviation_annualized)
            if not np.isnan(downside_deviation_annualized)
            else None,
            "processing_time": time.time() - start_time,
        }

    def calculate_max_drawdown(
        self, data: List[float], is_returns: bool = True
    ) -> Dict[str, Any]:
        """Calculate the maximum drawdown."""
        start_time = time.time()

        if not data or len(data) < 2:
            return {
                "max_drawdown": None,
                "error": "Not enough data points",
                "processing_time": 0,
            }

        if is_returns:
            # Convert returns to cumulative wealth index
            returns_series = pd.Series(data)
            cumulative_wealth = (1 + returns_series).cumprod()
            # Start wealth index at 1
            cumulative_wealth.iloc[0] = 1
            prices = cumulative_wealth
        else:
            # Data is already prices
            prices = pd.Series(data)

        if len(prices) < 2:
            return {
                "max_drawdown": None,
                "error": "Not enough data points after conversion",
                "processing_time": 0,
            }

        # Calculate rolling maximum
        rolling_max = prices.cummax()

        # Calculate drawdown series
        drawdown = (prices - rolling_max) / rolling_max

        # Find maximum drawdown
        max_drawdown = drawdown.min()

        # Find indices
        try:
            trough_index = drawdown.idxmin()
            peak_index = rolling_max.iloc[
                : trough_index + 1
            ].idxmax()  # Peak before the trough

            #
            # Find recovery index (first time price exceeds the peak after the
            # trough)
            recovery_index = None
            if trough_index < len(prices) - 1:
                prices_after_trough = prices.iloc[trough_index + 1 :]
                peak_value = prices.iloc[peak_index]
                recovery_candidates = prices_after_trough[
                    prices_after_trough >= peak_value
                ]
                if not recovery_candidates.empty:
                    recovery_index = recovery_candidates.index[0]

        except Exception as e:
            self.logger.warning(
                f"Could not determine peak/trough/recovery indices: {e}"
            )
            peak_index, trough_index, recovery_index = None, None, None

        return {
            "max_drawdown": float(max_drawdown) if not np.isnan(max_drawdown) else None,
            "peak_index": int(peak_index) if peak_index is not None else None,
            "trough_index": int(trough_index) if trough_index is not None else None,
            "recovery_index": int(recovery_index)
            if recovery_index is not None
            else None,
            "processing_time": time.time() - start_time,
        }

    def stress_test(
        self, returns: List[float], scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform stress tests based on historical scenarios."""
        start_time = time.time()

        if not returns:
            return {
                "results": [],
                "error": "Input returns list is empty",
                "processing_time": 0,
            }

        # This is a placeholder implementation. A real implementation would:
        #
        # 1. Define historical scenarios (e.g., 2008 crash, COVID drop) with
        # factor shocks.
        #
        # 2. Have a factor model linking asset returns to market factors (e.g.,
        # market beta, size, value).
        #
        # 3. Apply the factor shocks from the scenario to the asset's factor
        # exposures.
        # 4. Calculate the estimated impact on the asset's return.

        results = []
        for scenario in scenarios:
            scenario_name = scenario.get("name", "Unnamed Scenario")
            # Placeholder: Apply a simple shock based on scenario name
            shock_factor = -0.1  # Default 10% drop
            if "crash" in scenario_name.lower():
                shock_factor = -0.25  # 25% drop for crashes
            elif "correction" in scenario_name.lower():
                shock_factor = -0.15  # 15% drop for corrections

            # Apply shock to average return (very simplified)
            avg_return = np.mean(returns)
            stressed_return_estimate = avg_return + shock_factor

            results.append(
                {
                    "scenario_name": scenario_name,
                    "estimated_impact": float(shock_factor),  # The shock applied
                    "estimated_stressed_return": float(stressed_return_estimate),
                    "details": "Simplified stress test based on scenario name",
                }
            )

        return {"results": results, "processing_time": time.time() - start_time}


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {}

    # Create and start the server
    server = RiskMetricsMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("RiskMetricsMCP server started")

    # Example usage
    # Generate some sample daily returns
    np.random.seed(42)
    sample_returns = np.random.normal(
        loc=0.0005, scale=0.015, size=500
    )  # ~12.6% annual return, ~23.8% vol

    # Calculate Volatility
    vol_result = server.calculate_volatility(sample_returns.tolist())
    print(f"\nVolatility: {json.dumps(vol_result, indent=2)}")

    # Calculate VaR (Historical)
    var_hist_result = server.calculate_var(
        sample_returns.tolist(), confidence_level=0.95, method="historical"
    )
    print(f"\nHistorical VaR (95%): {json.dumps(var_hist_result, indent=2)}")

    # Calculate VaR (Parametric)
    var_param_result = server.calculate_var(
        sample_returns.tolist(), confidence_level=0.99, method="parametric"
    )
    print(f"\nParametric VaR (99%): {json.dumps(var_param_result, indent=2)}")

    # Calculate CVaR (Historical)
    cvar_hist_result = server.calculate_cvar(
        sample_returns.tolist(), confidence_level=0.95, method="historical"
    )
    print(f"\nHistorical CVaR (95%): {json.dumps(cvar_hist_result, indent=2)}")

    # Calculate Sharpe Ratio
    sharpe_result = server.calculate_sharpe_ratio(
        sample_returns.tolist(), risk_free_rate=0.02
    )
    print(f"\nSharpe Ratio: {json.dumps(sharpe_result, indent=2)}")

    # Calculate Sortino Ratio
    sortino_result = server.calculate_sortino_ratio(
        sample_returns.tolist(), risk_free_rate=0.02
    )
    print(f"\nSortino Ratio: {json.dumps(sortino_result, indent=2)}")

    # Calculate Max Drawdown
    mdd_result = server.calculate_max_drawdown(sample_returns.tolist(), is_returns=True)
    print(f"\nMax Drawdown: {json.dumps(mdd_result, indent=2)}")

    # Stress Test
    stress_scenarios = [
        {"name": "Market Correction", "factor_shocks": {"market": -0.15}},
        {
            "name": "2008-like Crash",
            "factor_shocks": {"market": -0.5, "credit_spread": 0.05},
        },
    ]
    stress_result = server.stress_test(sample_returns.tolist(), stress_scenarios)
    print(f"\nStress Test Results: {json.dumps(stress_result, indent=2)}")
