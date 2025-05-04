"""
Correlation Analysis MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
correlation analysis capabilities for financial assets and portfolios.
"""

import os
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-correlation-analysis",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class CorrelationAnalysisMCP(BaseMCPServer):
    """
    MCP server for calculating correlations and analyzing portfolio effects.

    This tool calculates correlation matrices, detects changes in correlation regimes,
    and assesses the diversification benefits or concentration risks within a portfolio.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Correlation Analysis MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - default_correlation_window: Default window size for rolling correlations
                - regime_change_threshold: Threshold for detecting correlation regime changes
                - cache_dir: Directory for caching intermediate results
        """
        super().__init__(name="correlation_analysis_mcp", config=config)

        # Set default configurations
        self.default_correlation_window = self.config.get(
            "default_correlation_window", 60
        )  # e.g., 60 days
        self.regime_change_threshold = self.config.get(
            "regime_change_threshold", 0.2
        )  # e.g., 20% change
        self.cache_dir = self.config.get("cache_dir", "./correlation_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "calculate_correlation_matrix",
            self.calculate_correlation_matrix,
            "Calculate the correlation matrix for multiple asset return series",
            {
                "returns_data": {
                    "type": "object",
                    "description": "Dictionary where keys are asset symbols and values are lists of returns",
                },
                "window": {
                    "type": "integer",
                    "description": "Rolling window size for correlation calculation (optional, calculates static if not provided)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "correlation_matrix": {
                        "type": "object"
                    },  # Matrix as nested dict or list of lists
                    "is_rolling": {"type": "boolean"},
                    "window": {"type": "integer"},
                    "assets": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "analyze_portfolio_correlation",
            self.analyze_portfolio_correlation,
            "Analyze the average correlation within a portfolio",
            {
                "returns_data": {
                    "type": "object",
                    "description": "Dictionary of asset returns in the portfolio",
                },
                "weights": {
                    "type": "object",
                    "description": "Dictionary of asset weights in the portfolio (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "average_correlation": {"type": "number"},
                    "weighted_average_correlation": {
                        "type": "number"
                    },  # If weights provided
                    "diversification_benefit": {"type": "number"},  # e.g., 1 - avg_corr
                    "risk_concentration_index": {
                        "type": "number"
                    },  # e.g., Herfindahl index on correlation contributions
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "detect_correlation_regime_change",
            self.detect_correlation_regime_change,
            "Detect significant changes in correlation patterns",
            {
                "returns_data": {
                    "type": "object",
                    "description": "Dictionary of asset returns",
                },
                "window1": {
                    "type": "integer",
                    "description": "First window size for comparison",
                    "default": self.default_correlation_window // 2,
                },
                "window2": {
                    "type": "integer",
                    "description": "Second window size for comparison",
                    "default": self.default_correlation_window,
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold for detecting significant change",
                    "default": self.regime_change_threshold,
                },
            },
            {
                "type": "object",
                "properties": {
                    "regime_change_detected": {"type": "boolean"},
                    "change_magnitude": {
                        "type": "number"
                    },  # Average absolute change in correlations
                    "changed_pairs": {
                        "type": "array"
                    },  # List of pairs with significant changes
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_asset_portfolio_correlation",
            self.calculate_asset_portfolio_correlation,
            "Calculate the correlation of a single asset with the rest of the portfolio",
            {
                "asset_symbol": {
                    "type": "string",
                    "description": "Symbol of the asset to analyze",
                },
                "returns_data": {
                    "type": "object",
                    "description": "Dictionary of asset returns including the target asset and portfolio assets",
                },
                "portfolio_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols representing the rest of the portfolio",
                },
            },
            {
                "type": "object",
                "properties": {
                    "asset_symbol": {"type": "string"},
                    "average_correlation_with_portfolio": {"type": "number"},
                    "individual_correlations": {
                        "type": "object"
                    },  # Correlation with each portfolio asset
                    "processing_time": {"type": "number"},
                },
            },
        )

    def _prepare_returns_dataframe(
        self, returns_data: Dict[str, List[float]]
    ) -> pd.DataFrame:
        """
        Convert returns data dictionary to a pandas DataFrame.
        Aligns series by padding with NaN if lengths differ.

        Args:
            returns_data: Dictionary where keys are symbols and values are lists of returns.

        Returns:
            Pandas DataFrame with symbols as columns and returns as rows.
        """
        try:
            # Find the maximum length
            max_len = 0
            for symbol, returns in returns_data.items():
                if returns:  # Check if list is not empty
                    max_len = max(max_len, len(returns))

            if max_len == 0:
                self.logger.warning("All return series are empty.")
                return pd.DataFrame()

            # Create aligned dictionary
            aligned_data = {}
            for symbol, returns in returns_data.items():
                if not returns:  # Handle empty list
                    aligned_data[symbol] = [np.nan] * max_len
                    continue

                current_len = len(returns)
                if current_len < max_len:
                    # Pad with NaN at the beginning
                    aligned_data[symbol] = [np.nan] * (max_len - current_len) + returns
                else:
                    aligned_data[symbol] = returns[:max_len]  # Ensure all are max_len

            df = pd.DataFrame(aligned_data)
            # Convert to numeric, coercing errors
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception as e:
            self.logger.error(f"Error preparing returns DataFrame: {e}")
            return pd.DataFrame()

    def calculate_correlation_matrix(
        self, returns_data: Dict[str, List[float]], window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Calculate the correlation matrix for multiple asset returns."""
        start_time = time.time()

        if not returns_data:
            return {
                "correlation_matrix": None,
                "error": "Input returns_data is empty",
                "processing_time": 0,
            }

        df = self._prepare_returns_dataframe(returns_data)

        if df.empty or len(df.columns) < 2:
            return {
                "correlation_matrix": None,
                "error": "Not enough valid asset data to calculate correlation",
                "processing_time": time.time() - start_time,
            }

        correlation_matrix = None
        is_rolling = False

        try:
            if window and window > 0 and window <= len(df):
                # Calculate rolling correlation matrix
                #
                # Pandas rolling corr requires pairwise calculation or specific
                # structure
                #
                # For simplicity, calculate the correlation matrix for the last
                # 'window' periods
                correlation_matrix = df.iloc[-window:].corr()
                is_rolling = True
                window_used = window
            else:
                # Calculate static correlation matrix over the entire period
                correlation_matrix = df.corr()
                is_rolling = False
                window_used = len(df)

            # Convert matrix to nested dictionary for JSON compatibility
            matrix_dict = (
                correlation_matrix.replace([np.inf, -np.inf], None)
                .fillna(None)
                .to_dict()
            )

            # Ensure values are floats or None
            for col, values in matrix_dict.items():
                matrix_dict[col] = {
                    idx: (float(v) if v is not None else None)
                    for idx, v in values.items()
                }

        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return {
                "correlation_matrix": None,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

        return {
            "correlation_matrix": matrix_dict,
            "is_rolling": is_rolling,
            "window": window_used if is_rolling else None,
            "assets": df.columns.tolist(),
            "processing_time": time.time() - start_time,
        }

    def analyze_portfolio_correlation(
        self,
        returns_data: Dict[str, List[float]],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Analyze the average correlation within a portfolio."""
        start_time = time.time()

        if not returns_data or len(returns_data) < 2:
            return {
                "average_correlation": None,
                "error": "Requires at least two assets",
                "processing_time": 0,
            }

        # Calculate the static correlation matrix first
        corr_result = self.calculate_correlation_matrix(returns_data)
        corr_matrix_dict = corr_result.get("correlation_matrix")

        if corr_matrix_dict is None:
            return {
                "average_correlation": None,
                "error": corr_result.get(
                    "error", "Failed to calculate correlation matrix"
                ),
                "processing_time": time.time() - start_time,
            }

        # Convert dict back to DataFrame for easier manipulation
        corr_matrix = pd.DataFrame(corr_matrix_dict)

        # Calculate average pairwise correlation (excluding diagonal 1s)
        num_assets = len(corr_matrix)
        # Get upper triangle values excluding diagonal
        corr_values = corr_matrix.mask(
            np.tril(np.ones(corr_matrix.shape, dtype=bool))
        ).stack()
        average_correlation = corr_values.mean()

        # Calculate weighted average correlation if weights are provided
        weighted_avg_corr = None
        if weights:
            valid_weights = {
                sym: w for sym, w in weights.items() if sym in corr_matrix.columns
            }
            total_weight = sum(valid_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    sym: w / total_weight for sym, w in valid_weights.items()
                }

                weighted_sum = 0
                weight_product_sum = 0
                assets = corr_matrix.columns

                for i in range(num_assets):
                    for j in range(i + 1, num_assets):
                        sym1, sym2 = assets[i], assets[j]
                        w1 = normalized_weights.get(sym1, 0)
                        w2 = normalized_weights.get(sym2, 0)
                        corr = corr_matrix.loc[sym1, sym2]
                        if corr is not None:
                            weight_product = w1 * w2
                            weighted_sum += weight_product * corr
                            weight_product_sum += weight_product

                if weight_product_sum > 0:
                    weighted_avg_corr = weighted_sum / weight_product_sum

        # Diversification benefit (simple measure)
        diversification_benefit = (
            1.0 - average_correlation if average_correlation is not None else None
        )

        # Risk concentration (Herfindahl index on weights - simpler proxy)
        risk_concentration_index = None
        if weights:
            valid_weights = {
                sym: w for sym, w in weights.items() if sym in corr_matrix.columns
            }
            total_weight = sum(valid_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    sym: w / total_weight for sym, w in valid_weights.items()
                }
                risk_concentration_index = sum(
                    w**2 for w in normalized_weights.values()
                )

        return {
            "average_correlation": float(average_correlation)
            if average_correlation is not None
            else None,
            "weighted_average_correlation": float(weighted_avg_corr)
            if weighted_avg_corr is not None
            else None,
            "diversification_benefit": float(diversification_benefit)
            if diversification_benefit is not None
            else None,
            "risk_concentration_index": float(risk_concentration_index)
            if risk_concentration_index is not None
            else None,
            "processing_time": time.time() - start_time,
        }

    def detect_correlation_regime_change(
        self,
        returns_data: Dict[str, List[float]],
        window1: int = None,
        window2: int = None,
        threshold: float = None,
    ) -> Dict[str, Any]:
        """Detect significant changes in correlation patterns."""
        start_time = time.time()

        if not returns_data or len(returns_data) < 2:
            return {
                "regime_change_detected": False,
                "error": "Requires at least two assets",
                "processing_time": 0,
            }

        window1 = window1 or self.default_correlation_window // 2
        window2 = window2 or self.default_correlation_window
        threshold = threshold or self.regime_change_threshold

        df = self._prepare_returns_dataframe(returns_data)

        if df.empty or len(df.columns) < 2 or len(df) < window2:
            return {
                "regime_change_detected": False,
                "error": "Not enough data or assets for regime change detection",
                "processing_time": time.time() - start_time,
            }

        try:
            # Calculate correlation matrix for the first (shorter) window
            corr1_df = df.iloc[-window1:].corr()

            # Calculate correlation matrix for the second (longer) window
            corr2_df = df.iloc[-window2:].corr()

            # Calculate the difference matrix
            corr_diff = (corr1_df - corr2_df).abs()

            # Calculate average absolute change (excluding diagonal)
            diff_values = corr_diff.mask(
                np.tril(np.ones(corr_diff.shape, dtype=bool))
            ).stack()
            avg_change = diff_values.mean()

            # Find pairs with changes above the threshold
            changed_pairs = []
            assets = corr_diff.columns
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    sym1, sym2 = assets[i], assets[j]
                    change = corr_diff.loc[sym1, sym2]
                    if change is not None and change >= threshold:
                        changed_pairs.append(
                            {
                                "pair": f"{sym1}-{sym2}",
                                "change": float(change),
                                "corr_window1": float(corr1_df.loc[sym1, sym2]),
                                "corr_window2": float(corr2_df.loc[sym1, sym2]),
                            }
                        )

            regime_change_detected = avg_change >= threshold or len(changed_pairs) > 0

        except Exception as e:
            self.logger.error(f"Error detecting correlation regime change: {e}")
            return {
                "regime_change_detected": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

        return {
            "regime_change_detected": regime_change_detected,
            "change_magnitude": float(avg_change) if avg_change is not None else None,
            "changed_pairs": changed_pairs,
            "threshold": threshold,
            "window1": window1,
            "window2": window2,
            "processing_time": time.time() - start_time,
        }

    def calculate_asset_portfolio_correlation(
        self,
        asset_symbol: str,
        returns_data: Dict[str, List[float]],
        portfolio_symbols: List[str],
    ) -> Dict[str, Any]:
        """Calculate the correlation of a single asset with the rest of the
        portfolio.
        """
        start_time = time.time()

        if asset_symbol not in returns_data:
            return {
                "average_correlation_with_portfolio": None,
                "error": f"Asset {asset_symbol} not found in returns data",
                "processing_time": 0,
            }

        #
        # Filter portfolio symbols that are in returns_data and different from
        # the target asset
        valid_portfolio_symbols = [
            sym
            for sym in portfolio_symbols
            if sym in returns_data and sym != asset_symbol
        ]

        if not valid_portfolio_symbols:
            return {
                "average_correlation_with_portfolio": None,
                "error": "No valid portfolio symbols provided or found in data",
                "processing_time": 0,
            }

        # Include the target asset in the data for correlation calculation
        symbols_to_correlate = [asset_symbol] + valid_portfolio_symbols
        subset_returns_data = {
            sym: returns_data[sym]
            for sym in symbols_to_correlate
            if sym in returns_data
        }

        # Calculate the correlation matrix for the subset
        corr_result = self.calculate_correlation_matrix(subset_returns_data)
        corr_matrix_dict = corr_result.get("correlation_matrix")

        if corr_matrix_dict is None or asset_symbol not in corr_matrix_dict:
            return {
                "average_correlation_with_portfolio": None,
                "error": corr_result.get(
                    "error", "Failed to calculate correlation matrix"
                ),
                "processing_time": time.time() - start_time,
            }

        # Extract correlations of the target asset with portfolio assets
        asset_correlations = corr_matrix_dict[asset_symbol]

        individual_correlations = {}
        portfolio_corrs_list = []
        for sym in valid_portfolio_symbols:
            corr_value = asset_correlations.get(sym)
            individual_correlations[sym] = (
                float(corr_value) if corr_value is not None else None
            )
            if corr_value is not None:
                portfolio_corrs_list.append(corr_value)

        # Calculate average correlation
        average_correlation = (
            np.mean(portfolio_corrs_list) if portfolio_corrs_list else None
        )

        return {
            "asset_symbol": asset_symbol,
            "average_correlation_with_portfolio": float(average_correlation)
            if average_correlation is not None
            else None,
            "individual_correlations": individual_correlations,
            "processing_time": time.time() - start_time,
        }


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {}

    # Create and start the server
    server = CorrelationAnalysisMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("CorrelationAnalysisMCP server started")

    # Example usage
    np.random.seed(42)
    # Generate sample returns for 3 assets
    returns_asset1 = np.random.normal(loc=0.0005, scale=0.015, size=200)
    returns_asset2 = 0.6 * returns_asset1 + np.random.normal(
        loc=0.0002, scale=0.01, size=200
    )  # Correlated
    returns_asset3 = np.random.normal(
        loc=0.0003, scale=0.02, size=200
    )  # Less correlated

    sample_returns_data = {
        "AAPL": returns_asset1.tolist(),
        "MSFT": returns_asset2.tolist(),
        "GOOGL": returns_asset3.tolist(),
    }

    # Calculate Correlation Matrix
    corr_matrix_result = server.calculate_correlation_matrix(sample_returns_data)
    print(f"\nCorrelation Matrix: {json.dumps(corr_matrix_result, indent=2)}")

    # Analyze Portfolio Correlation
    portfolio_weights = {"AAPL": 0.4, "MSFT": 0.4, "GOOGL": 0.2}
    portfolio_corr_result = server.analyze_portfolio_correlation(
        sample_returns_data, weights=portfolio_weights
    )
    print(
        f"\nPortfolio Correlation Analysis: {json.dumps(portfolio_corr_result, indent=2)}"
    )

    # Detect Regime Change
    # Add a period of higher correlation for testing
    returns_asset1_regime = np.concatenate(
        [returns_asset1, np.random.normal(loc=0.0005, scale=0.015, size=50)]
    )
    returns_asset2_regime = np.concatenate(
        [
            returns_asset2,
            0.9 * returns_asset1_regime[-50:]
            + np.random.normal(loc=0.0001, scale=0.005, size=50),
        ]
    )
    returns_asset3_regime = np.concatenate(
        [returns_asset3, np.random.normal(loc=0.0003, scale=0.02, size=50)]
    )
    regime_returns_data = {
        "AAPL": returns_asset1_regime.tolist(),
        "MSFT": returns_asset2_regime.tolist(),
        "GOOGL": returns_asset3_regime.tolist(),
    }
    regime_change_result = server.detect_correlation_regime_change(
        regime_returns_data, window1=50, window2=100
    )
    print(
        f"\nCorrelation Regime Change Detection: {json.dumps(regime_change_result, indent=2)}"
    )

    # Calculate Asset-Portfolio Correlation
    asset_portfolio_corr_result = server.calculate_asset_portfolio_correlation(
        asset_symbol="AAPL",
        returns_data=sample_returns_data,
        portfolio_symbols=["MSFT", "GOOGL"],
    )
    print(
        f"\nAsset-Portfolio Correlation (AAPL vs MSFT, GOOGL): {json.dumps(asset_portfolio_corr_result, indent=2)}"
    )
