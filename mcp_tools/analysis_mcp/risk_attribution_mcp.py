"""
Risk Attribution MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
risk attribution capabilities for decomposing portfolio risk into factor components.
"""

import os
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from scipy import stats
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-risk-attribution",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class RiskAttributionMCP(BaseMCPServer):
    """
    MCP server for decomposing portfolio risk into factor components.

    This tool calculates risk contributions and performs factor analysis to
    attribute portfolio risk to various factors and individual assets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Risk Attribution MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - cache_dir: Directory for caching intermediate results
                - default_factors: List of default factors to use for attribution
                - factor_data_source: Source for factor data (e.g., 'internal', 'external')
                - correlation_threshold: Threshold for correlation significance (default: 0.3)
                - risk_free_rate: Annual risk-free rate (default: 0.02)
        """
        super().__init__(name="risk_attribution_mcp", config=config)

        # Set default configurations
        self.cache_dir = self.config.get("cache_dir", "./risk_attribution_cache")
        self.default_factors = self.config.get(
            "default_factors", ["Market", "Size", "Value", "Momentum", "Quality"]
        )
        self.factor_data_source = self.config.get("factor_data_source", "internal")
        self.correlation_threshold = self.config.get("correlation_threshold", 0.3)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "calculate_risk_contributions",
            self.calculate_risk_contributions,
            "Calculate risk contributions of assets in a portfolio",
            {
                "portfolio_weights": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their weights in the portfolio",
                },
                "asset_returns": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their historical returns time series",
                },
                "use_correlation": {
                    "type": "boolean",
                    "description": "Whether to use correlation-based risk contributions",
                    "default": True,
                },
            },
            {
                "type": "object",
                "properties": {
                    "portfolio_volatility": {
                        "type": "number"
                    },  # Overall portfolio volatility
                    "risk_contributions": {
                        "type": "object"
                    },  # Risk contribution of each asset
                    "percentage_contributions": {
                        "type": "object"
                    },  # Percentage contribution of each asset
                    "diversification_ratio": {
                        "type": "number"
                    },  # Portfolio diversification ratio
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "perform_factor_analysis",
            self.perform_factor_analysis,
            "Perform factor analysis on portfolio returns",
            {
                "portfolio_returns": {
                    "type": "array",
                    "description": "Time series of portfolio returns",
                },
                "factor_returns": {
                    "type": "object",
                    "description": "Dictionary mapping factor names to their historical returns time series",
                },
                "factors": {
                    "type": "array",
                    "description": "List of factors to include in the analysis (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "factor_exposures": {
                        "type": "object"
                    },  # Exposure (beta) to each factor
                    "factor_contributions": {
                        "type": "object"
                    },  # Risk contribution of each factor
                    "r_squared": {"type": "number"},  # R-squared of the factor model
                    "specific_risk": {
                        "type": "number"
                    },  # Specific (idiosyncratic) risk
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "decompose_asset_risk",
            self.decompose_asset_risk,
            "Decompose risk of individual assets into factor components",
            {
                "asset_returns": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their historical returns time series",
                },
                "factor_returns": {
                    "type": "object",
                    "description": "Dictionary mapping factor names to their historical returns time series",
                },
                "factors": {
                    "type": "array",
                    "description": "List of factors to include in the analysis (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "asset_decompositions": {
                        "type": "object"
                    },  # Risk decomposition for each asset
                    "factor_exposures": {
                        "type": "object"
                    },  # Factor exposures for each asset
                    "specific_risks": {
                        "type": "object"
                    },  # Specific risk for each asset
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_marginal_risk_contributions",
            self.calculate_marginal_risk_contributions,
            "Calculate marginal risk contributions for portfolio optimization",
            {
                "portfolio_weights": {
                    "type": "object",
                    "description": "Dictionary mapping asset identifiers to their weights in the portfolio",
                },
                "covariance_matrix": {
                    "type": "array",
                    "description": "Covariance matrix of asset returns",
                },
                "asset_ids": {
                    "type": "array",
                    "description": "List of asset identifiers corresponding to the covariance matrix rows/columns",
                },
            },
            {
                "type": "object",
                "properties": {
                    "marginal_contributions": {
                        "type": "object"
                    },  # Marginal risk contribution of each asset
                    "risk_budget": {"type": "object"},  # Current risk budget allocation
                    "risk_concentration": {
                        "type": "number"
                    },  # Risk concentration measure
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "analyze_risk_factors",
            self.analyze_risk_factors,
            "Analyze risk factors and their correlations",
            {
                "factor_returns": {
                    "type": "object",
                    "description": "Dictionary mapping factor names to their historical returns time series",
                },
                "lookback_period": {
                    "type": "integer",
                    "description": "Number of periods to look back for analysis",
                    "default": 252,  # Default to 1 year of daily data
                },
            },
            {
                "type": "object",
                "properties": {
                    "factor_volatilities": {
                        "type": "object"
                    },  # Volatility of each factor
                    "factor_correlations": {
                        "type": "object"
                    },  # Correlation matrix between factors
                    "factor_trends": {"type": "object"},  # Trend analysis of factors
                    "processing_time": {"type": "number"},
                },
            },
        )

    def calculate_risk_contributions(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: Dict[str, List[float]],
        use_correlation: bool = True,
    ) -> Dict[str, Any]:
        """
        Calculate risk contributions of assets in a portfolio.

        Args:
            portfolio_weights: Dictionary mapping asset identifiers to their weights
            asset_returns: Dictionary mapping asset identifiers to their historical returns
            use_correlation: Whether to use correlation-based risk contributions

        Returns:
            Dictionary with risk contribution details
        """
        start_time = time.time()

        try:
            # Convert inputs to numpy arrays and pandas DataFrames
            weights = []
            assets = []
            returns_data = []

            for asset_id, weight in portfolio_weights.items():
                if asset_id in asset_returns and asset_returns[asset_id]:
                    assets.append(asset_id)
                    weights.append(weight)
                    returns_data.append(asset_returns[asset_id])

            if not assets:
                return {
                    "error": "No valid assets found in both portfolio weights and returns data",
                    "processing_time": time.time() - start_time,
                }

            # Create returns DataFrame and weight array
            returns_df = pd.DataFrame(dict(zip(assets, returns_data)))
            weights_array = np.array(weights)

            # Normalize weights to sum to 1
            weights_array = weights_array / np.sum(weights_array)

            # Calculate covariance matrix
            cov_matrix = returns_df.cov().values

            # Calculate portfolio volatility
            portfolio_variance = weights_array.T @ cov_matrix @ weights_array
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Calculate risk contributions
            if use_correlation:
                # Calculate using correlation-weighted approach
                std_devs = np.sqrt(np.diag(cov_matrix))
                corr_matrix = cov_matrix / np.outer(std_devs, std_devs)

                # Calculate marginal contributions
                marginal_contributions = cov_matrix @ weights_array

                # Calculate component contributions
                risk_contributions = weights_array * marginal_contributions

                # Normalize to sum to portfolio variance
                risk_contributions = (
                    risk_contributions / np.sum(risk_contributions) * portfolio_variance
                )

                # Convert to volatility contributions
                risk_contributions = np.sqrt(risk_contributions)
            else:
                # Direct volatility contribution calculation
                marginal_contributions = cov_matrix @ weights_array
                risk_contributions = (
                    weights_array * marginal_contributions / portfolio_volatility
                )

            # Calculate percentage contributions
            percentage_contributions = risk_contributions / np.sum(risk_contributions)

            # Calculate diversification ratio
            weighted_volatilities = weights_array * np.sqrt(np.diag(cov_matrix))
            sum_weighted_vol = np.sum(weighted_volatilities)
            diversification_ratio = sum_weighted_vol / portfolio_volatility

            # Prepare results
            result = {
                "portfolio_volatility": float(portfolio_volatility),
                "risk_contributions": dict(zip(assets, risk_contributions.tolist())),
                "percentage_contributions": dict(
                    zip(assets, percentage_contributions.tolist())
                ),
                "diversification_ratio": float(diversification_ratio),
                "processing_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error calculating risk contributions: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def perform_factor_analysis(
        self,
        portfolio_returns: List[float],
        factor_returns: Dict[str, List[float]],
        factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform factor analysis on portfolio returns.

        Args:
            portfolio_returns: Time series of portfolio returns
            factor_returns: Dictionary mapping factor names to their historical returns
            factors: List of factors to include in the analysis (optional)

        Returns:
            Dictionary with factor analysis results
        """
        start_time = time.time()

        try:
            # Select factors to use
            if factors is None:
                factors = list(factor_returns.keys())
            else:
                # Filter to only include available factors
                factors = [f for f in factors if f in factor_returns]

            if not factors:
                return {
                    "error": "No valid factors available for analysis",
                    "processing_time": time.time() - start_time,
                }

            # Create factor returns DataFrame
            factor_data = []
            for factor in factors:
                if factor in factor_returns and len(factor_returns[factor]) >= len(
                    portfolio_returns
                ):
                    # Ensure factor returns match portfolio returns length
                    factor_data.append(factor_returns[factor][: len(portfolio_returns)])
                else:
                    self.logger.warning(
                        f"Factor {factor} has insufficient data and will be excluded"
                    )
                    factors.remove(factor)

            if not factors:
                return {
                    "error": "No factors with sufficient data for analysis",
                    "processing_time": time.time() - start_time,
                }

            # Create X (factors) and y (portfolio returns) for regression
            X = np.column_stack(factor_data)
            y = np.array(portfolio_returns)

            # Add constant for intercept
            X_with_const = np.column_stack([np.ones(X.shape[0]), X])

            # Perform linear regression
            beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

            # Extract alpha and factor exposures
            alpha = beta[0]
            factor_exposures = dict(zip(factors, beta[1:].tolist()))

            # Calculate model fit
            y_pred = X_with_const @ beta
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            # Calculate factor contributions to risk
            portfolio_variance = np.var(y)
            factor_variances = np.var(X, axis=0)
            factor_contributions = {}

            # Calculate covariance matrix of factors
            factor_cov = np.cov(X, rowvar=False)

            # Calculate factor contributions
            for i, factor in enumerate(factors):
                # Direct contribution from factor variance
                direct_contrib = (beta[i + 1] ** 2) * factor_variances[i]

                # Contributions from covariances with other factors
                cov_contrib = 0
                for j, other_factor in enumerate(factors):
                    if i != j:
                        cov_contrib += beta[i + 1] * beta[j + 1] * factor_cov[i, j]

                factor_contributions[factor] = direct_contrib + cov_contrib

            # Calculate specific risk (unexplained variance)
            specific_risk = portfolio_variance - sum(factor_contributions.values())
            specific_risk = max(0, specific_risk)  # Ensure non-negative

            # Normalize factor contributions to sum to explained variance
            explained_variance = portfolio_variance - specific_risk
            if explained_variance > 0:
                for factor in factor_contributions:
                    factor_contributions[factor] /= explained_variance

            # Prepare results
            result = {
                "alpha": float(alpha),
                "factor_exposures": factor_exposures,
                "factor_contributions": factor_contributions,
                "r_squared": float(r_squared),
                "specific_risk": float(specific_risk),
                "portfolio_volatility": float(np.sqrt(portfolio_variance)),
                "processing_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error performing factor analysis: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def decompose_asset_risk(
        self,
        asset_returns: Dict[str, List[float]],
        factor_returns: Dict[str, List[float]],
        factors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Decompose risk of individual assets into factor components.

        Args:
            asset_returns: Dictionary mapping asset identifiers to their historical returns
            factor_returns: Dictionary mapping factor names to their historical returns
            factors: List of factors to include in the analysis (optional)

        Returns:
            Dictionary with risk decomposition results
        """
        start_time = time.time()

        try:
            # Select factors to use
            if factors is None:
                factors = list(factor_returns.keys())
            else:
                # Filter to only include available factors
                factors = [f for f in factors if f in factor_returns]

            if not factors:
                return {
                    "error": "No valid factors available for analysis",
                    "processing_time": time.time() - start_time,
                }

            # Create factor returns DataFrame
            factor_data = {}
            min_length = float("inf")

            # Find minimum length across all data series
            for factor in factors:
                if factor in factor_returns:
                    factor_data[factor] = factor_returns[factor]
                    min_length = min(min_length, len(factor_returns[factor]))

            for asset_id in asset_returns:
                min_length = min(min_length, len(asset_returns[asset_id]))

            # Truncate all series to the same length
            for factor in factor_data:
                factor_data[factor] = factor_data[factor][:min_length]

            asset_data = {}
            for asset_id in asset_returns:
                asset_data[asset_id] = asset_returns[asset_id][:min_length]

            # Create factor returns matrix
            X = np.column_stack([factor_data[factor] for factor in factors])

            # Add constant for intercept
            X_with_const = np.column_stack([np.ones(X.shape[0]), X])

            # Calculate factor covariance matrix
            factor_cov = np.cov(X, rowvar=False)

            # Perform regression for each asset
            asset_decompositions = {}
            factor_exposures = {}
            specific_risks = {}

            for asset_id, returns in asset_data.items():
                y = np.array(returns)

                # Perform linear regression
                beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)

                # Extract alpha and factor exposures
                alpha = beta[0]
                asset_factor_exposures = dict(zip(factors, beta[1:].tolist()))
                factor_exposures[asset_id] = asset_factor_exposures

                # Calculate model fit
                y_pred = X_with_const @ beta
                ss_total = np.sum((y - np.mean(y)) ** 2)
                ss_residual = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)

                # Calculate asset variance
                asset_variance = np.var(y)

                # Calculate factor contributions to risk
                factor_contributions = {}
                beta_factors = beta[1:]  # Skip alpha

                # Calculate factor contributions using matrix multiplication
                factor_var_contrib = beta_factors @ factor_cov @ beta_factors

                # Calculate specific risk (unexplained variance)
                specific_risk = asset_variance - factor_var_contrib
                specific_risk = max(0, specific_risk)  # Ensure non-negative
                specific_risks[asset_id] = float(specific_risk)

                # Calculate individual factor contributions
                decomposition = {}
                for i, factor in enumerate(factors):
                    # Direct contribution from factor variance
                    direct_contrib = (beta[i + 1] ** 2) * factor_cov[i, i]

                    # Contributions from covariances with other factors
                    cov_contrib = 0
                    for j, other_factor in enumerate(factors):
                        if i != j:
                            cov_contrib += beta[i + 1] * beta[j + 1] * factor_cov[i, j]

                    decomposition[factor] = float(direct_contrib + cov_contrib)

                # Add specific risk to decomposition
                decomposition["Specific"] = float(specific_risk)

                # Store decomposition
                asset_decompositions[asset_id] = {
                    "decomposition": decomposition,
                    "r_squared": float(r_squared),
                    "alpha": float(alpha),
                    "total_risk": float(asset_variance),
                }

            # Prepare results
            result = {
                "asset_decompositions": asset_decompositions,
                "factor_exposures": factor_exposures,
                "specific_risks": specific_risks,
                "processing_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error decomposing asset risk: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def calculate_marginal_risk_contributions(
        self,
        portfolio_weights: Dict[str, float],
        covariance_matrix: List[List[float]],
        asset_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Calculate marginal risk contributions for portfolio optimization.

        Args:
            portfolio_weights: Dictionary mapping asset identifiers to their weights
            covariance_matrix: Covariance matrix of asset returns
            asset_ids: List of asset identifiers corresponding to the covariance matrix rows/columns

        Returns:
            Dictionary with marginal risk contribution details
        """
        start_time = time.time()

        try:
            # Convert inputs to numpy arrays
            cov_matrix = np.array(covariance_matrix)

            # Create weight array in the same order as asset_ids
            weights = np.zeros(len(asset_ids))
            for i, asset_id in enumerate(asset_ids):
                if asset_id in portfolio_weights:
                    weights[i] = portfolio_weights[asset_id]

            # Normalize weights to sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)

            # Calculate portfolio volatility
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)

            # Calculate marginal risk contributions (partial derivatives)
            marginal_contributions = (cov_matrix @ weights) / portfolio_volatility

            # Calculate component risk contributions
            component_contributions = weights * marginal_contributions

            # Calculate risk budget (percentage of risk contributed by each asset)
            risk_budget = component_contributions / np.sum(component_contributions)

            # Calculate risk concentration (Herfindahl index of risk contributions)
            risk_concentration = np.sum(risk_budget**2)

            # Prepare results
            result = {
                "marginal_contributions": dict(
                    zip(asset_ids, marginal_contributions.tolist())
                ),
                "component_contributions": dict(
                    zip(asset_ids, component_contributions.tolist())
                ),
                "risk_budget": dict(zip(asset_ids, risk_budget.tolist())),
                "risk_concentration": float(risk_concentration),
                "portfolio_volatility": float(portfolio_volatility),
                "processing_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error calculating marginal risk contributions: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def analyze_risk_factors(
        self, factor_returns: Dict[str, List[float]], lookback_period: int = 252
    ) -> Dict[str, Any]:
        """
        Analyze risk factors and their correlations.

        Args:
            factor_returns: Dictionary mapping factor names to their historical returns
            lookback_period: Number of periods to look back for analysis

        Returns:
            Dictionary with factor analysis results
        """
        start_time = time.time()

        try:
            # Extract factors and ensure they have sufficient data
            factors = []
            factor_data = []

            for factor, returns in factor_returns.items():
                if len(returns) >= lookback_period:
                    factors.append(factor)
                    factor_data.append(returns[:lookback_period])
                else:
                    self.logger.warning(
                        f"Factor {factor} has insufficient data and will be excluded"
                    )

            if not factors:
                return {
                    "error": "No factors with sufficient data for analysis",
                    "processing_time": time.time() - start_time,
                }

            # Create factor returns DataFrame
            factor_df = pd.DataFrame(dict(zip(factors, factor_data)))

            # Calculate factor volatilities
            factor_volatilities = {}
            for factor in factors:
                factor_volatilities[factor] = float(
                    factor_df[factor].std() * np.sqrt(252)
                )  # Annualized

            # Calculate factor correlations
            factor_correlations = factor_df.corr().to_dict()

            # Analyze factor trends
            factor_trends = {}
            for factor in factors:
                # Calculate rolling volatility (30-day window)
                rolling_vol = factor_df[factor].rolling(
                    window=30
                ).std().dropna() * np.sqrt(252)

                # Calculate trend using linear regression on recent data
                x = np.arange(len(rolling_vol))
                y = rolling_vol.values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                # Determine trend direction
                if p_value < 0.05:  # Statistically significant
                    if slope > 0:
                        trend = "Increasing"
                    else:
                        trend = "Decreasing"
                else:
                    trend = "Stable"

                factor_trends[factor] = {
                    "trend": trend,
                    "slope": float(slope),
                    "p_value": float(p_value),
                    "current_volatility": float(rolling_vol.iloc[-1])
                    if len(rolling_vol) > 0
                    else None,
                    "mean_volatility": float(rolling_vol.mean())
                    if len(rolling_vol) > 0
                    else None,
                }

            # Prepare results
            result = {
                "factor_volatilities": factor_volatilities,
                "factor_correlations": factor_correlations,
                "factor_trends": factor_trends,
                "analysis_period": lookback_period,
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time,
            }

            return result

        except Exception as e:
            self.logger.error(f"Error analyzing risk factors: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

        except Exception as e:
            self.logger.error(f"Error analyzing risk factors: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}
