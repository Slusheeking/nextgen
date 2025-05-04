"""
Portfolio Optimization MCP Server

This module implements a Model Context Protocol (MCP) server for portfolio optimization
functions used by the Decision Model to optimize position sizing and portfolio allocation.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy import optimize
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-portfolio-optimization",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class PortfolioOptimizationMCP(BaseMCPServer):
    """
    MCP server for portfolio optimization functions.

    This server provides tools for calculating optimal position sizes,
    portfolio allocations, and risk-adjusted position sizing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Portfolio Optimization MCP server.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(name="portfolio_optimization_mcp", config=config)

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        self.logger.info("PortfolioOptimizationMCP initialized")

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "calculate_optimal_position_size": {
                "description": "Calculate optimal position size based on portfolio constraints",
                "category": "position_sizing",
                "required_params": ["symbol", "confidence", "portfolio_data"],
                "optional_params": ["max_position_pct", "risk_per_trade_pct"],
                "handler": self._handle_calculate_optimal_position_size,
            },
            "calculate_portfolio_impact": {
                "description": "Calculate the impact of a new position on the portfolio",
                "category": "portfolio_analysis",
                "required_params": ["symbol", "position_size", "portfolio_data"],
                "optional_params": ["correlation_data"],
                "handler": self._handle_calculate_portfolio_impact,
            },
            "optimize_portfolio_allocation": {
                "description": "Optimize portfolio allocation using modern portfolio theory",
                "category": "portfolio_optimization",
                "required_params": ["assets", "returns_data"],
                "optional_params": ["risk_free_rate", "target_return", "constraints"],
                "handler": self._handle_optimize_portfolio_allocation,
            },
            "calculate_kelly_criterion": {
                "description": "Calculate position size using Kelly Criterion",
                "category": "position_sizing",
                "required_params": ["win_rate", "win_loss_ratio"],
                "optional_params": ["max_allocation", "fractional_kelly"],
                "handler": self._handle_calculate_kelly_criterion,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Portfolio Optimization MCP."""
        self.register_tool(self.calculate_optimal_position_size)
        self.register_tool(self.calculate_portfolio_impact)
        self.register_tool(self.optimize_portfolio_allocation)
        self.register_tool(self.calculate_kelly_criterion)

    # Handler methods for specific endpoints

    def _handle_calculate_optimal_position_size(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle calculate_optimal_position_size endpoint."""
        symbol = params.get("symbol", "")
        confidence = float(params.get("confidence", 0.5))
        portfolio_data = params.get("portfolio_data", {})
        max_position_pct = float(params.get("max_position_pct", 5.0))
        risk_per_trade_pct = float(params.get("risk_per_trade_pct", 1.0))

        if not symbol:
            return {"error": "No symbol provided"}

        if not portfolio_data:
            return {"error": "No portfolio data provided"}

        try:
            # Calculate optimal position size
            result = self._calculate_optimal_position_size(
                symbol, confidence, portfolio_data, max_position_pct, risk_per_trade_pct
            )
            return result
        except Exception as e:
            self.logger.error(f"Error calculating optimal position size: {e}")
            return {"error": f"Failed to calculate optimal position size: {str(e)}"}

    def _handle_calculate_portfolio_impact(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle calculate_portfolio_impact endpoint."""
        symbol = params.get("symbol", "")
        position_size = float(params.get("position_size", 0.0))
        portfolio_data = params.get("portfolio_data", {})
        correlation_data = params.get("correlation_data", {})

        if not symbol:
            return {"error": "No symbol provided"}

        if position_size <= 0:
            return {
                "error": f"Invalid position size: {position_size}. Must be positive."
            }

        if not portfolio_data:
            return {"error": "No portfolio data provided"}

        try:
            # Calculate portfolio impact
            result = self._calculate_portfolio_impact(
                symbol, position_size, portfolio_data, correlation_data
            )
            return result
        except Exception as e:
            self.logger.error(f"Error calculating portfolio impact: {e}")
            return {"error": f"Failed to calculate portfolio impact: {str(e)}"}

    def _handle_optimize_portfolio_allocation(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle optimize_portfolio_allocation endpoint."""
        assets = params.get("assets", [])
        returns_data = params.get("returns_data", {})
        risk_free_rate = float(params.get("risk_free_rate", 0.0))
        target_return = params.get("target_return", None)
        constraints = params.get("constraints", {})

        if not assets:
            return {"error": "No assets provided"}

        if not returns_data:
            return {"error": "No returns data provided"}

        try:
            # Optimize portfolio allocation
            result = self._optimize_portfolio_allocation(
                assets, returns_data, risk_free_rate, target_return, constraints
            )
            return result
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio allocation: {e}")
            return {"error": f"Failed to optimize portfolio allocation: {str(e)}"}

    def _handle_calculate_kelly_criterion(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle calculate_kelly_criterion endpoint."""
        win_rate = float(params.get("win_rate", 0.0))
        win_loss_ratio = float(params.get("win_loss_ratio", 0.0))
        max_allocation = float(params.get("max_allocation", 1.0))
        fractional_kelly = float(params.get("fractional_kelly", 0.5))

        if win_rate <= 0 or win_rate >= 1:
            return {"error": f"Invalid win rate: {win_rate}. Must be between 0 and 1."}

        if win_loss_ratio <= 0:
            return {
                "error": f"Invalid win/loss ratio: {win_loss_ratio}. Must be positive."
            }

        try:
            # Calculate Kelly criterion
            result = self._calculate_kelly_criterion(
                win_rate, win_loss_ratio, max_allocation, fractional_kelly
            )
            return result
        except Exception as e:
            self.logger.error(f"Error calculating Kelly criterion: {e}")
            return {"error": f"Failed to calculate Kelly criterion: {str(e)}"}

    # Core analysis methods

    def _calculate_optimal_position_size(
        self,
        symbol: str,
        confidence: float,
        portfolio_data: Dict[str, Any],
        max_position_pct: float = 5.0,
        risk_per_trade_pct: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on portfolio constraints.

        Args:
            symbol: Stock symbol
            confidence: Decision confidence (0-1)
            portfolio_data: Current portfolio state
            max_position_pct: Maximum position size as percentage of portfolio
            risk_per_trade_pct: Risk per trade as percentage of portfolio

        Returns:
            Optimal position size and reasoning
        """
        # Extract portfolio data
        portfolio_value = float(portfolio_data.get("portfolio_value", 0.0))
        buying_power = float(portfolio_data.get("buying_power", 0.0))
        positions = portfolio_data.get("positions", [])

        # Validate portfolio data
        if portfolio_value <= 0:
            return {"error": "Invalid portfolio value"}

        if buying_power <= 0:
            return {"error": "Insufficient buying power"}

        # Calculate position size based on portfolio value and risk per trade
        base_size = portfolio_value * (risk_per_trade_pct / 100.0)

        # Adjust for confidence (0.5 to 1.0 scale)
        confidence_factor = (
            max(0, min(confidence - 0.5, 0.5)) * 2
        )  # Scale 0.5-1.0 to 0-1.0
        adjusted_size = base_size * (0.5 + confidence_factor)  # 50-100% of base size

        # Apply maximum position size limit
        max_position = portfolio_value * (max_position_pct / 100.0)
        final_size = min(adjusted_size, max_position)

        # Ensure we don't exceed buying power
        final_size = min(final_size, buying_power * 0.95)  # Use 95% of buying power max

        # Check if we already have a position in this symbol
        existing_position = next(
            (p for p in positions if p.get("symbol") == symbol), None
        )
        existing_position_value = (
            float(existing_position.get("market_value", 0.0))
            if existing_position
            else 0.0
        )

        # Adjust for existing position
        if existing_position_value > 0:
            # Calculate how much more we can add to this position
            remaining_allocation = max_position - existing_position_value
            final_size = min(final_size, remaining_allocation)

            if final_size <= 0:
                return {
                    "symbol": symbol,
                    "optimal_position_size": 0.0,
                    "existing_position_value": existing_position_value,
                    "reasoning": "Position already at or exceeding maximum allocation",
                    "constraints_applied": {
                        "max_position_pct": max_position_pct,
                        "risk_per_trade_pct": risk_per_trade_pct,
                        "confidence_factor": confidence_factor,
                        "max_position": max_position,
                        "buying_power": buying_power,
                    },
                }

        # Calculate percentage of portfolio
        position_pct = (
            (final_size / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
        )

        # Determine the limiting factor
        limiting_factor = "risk_per_trade"
        if final_size >= max_position:
            limiting_factor = "max_position_size"
        elif final_size >= buying_power * 0.95:
            limiting_factor = "buying_power"

        return {
            "symbol": symbol,
            "optimal_position_size": float(final_size),
            "position_percentage": float(position_pct),
            "existing_position_value": float(existing_position_value),
            "limiting_factor": limiting_factor,
            "reasoning": f"Position sized based on {risk_per_trade_pct}% risk per trade, adjusted for {confidence:.2f} confidence",
            "constraints_applied": {
                "max_position_pct": float(max_position_pct),
                "risk_per_trade_pct": float(risk_per_trade_pct),
                "confidence_factor": float(confidence_factor),
                "max_position": float(max_position),
                "buying_power": float(buying_power),
            },
        }

    def _calculate_portfolio_impact(
        self,
        symbol: str,
        position_size: float,
        portfolio_data: Dict[str, Any],
        correlation_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the impact of a new position on the portfolio.

        Args:
            symbol: Stock symbol
            position_size: Proposed position size
            portfolio_data: Current portfolio state
            correlation_data: Optional correlation data between assets

        Returns:
            Portfolio impact metrics
        """
        # Extract portfolio data
        portfolio_value = float(portfolio_data.get("portfolio_value", 0.0))
        positions = portfolio_data.get("positions", [])
        sector_exposure = portfolio_data.get("sector_exposure", {})

        # Get symbol information
        symbol_info = next((p for p in positions if p.get("symbol") == symbol), {})
        symbol_sector = symbol_info.get("sector", "Unknown")

        # Calculate position percentage
        position_pct = (
            (position_size / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
        )

        # Calculate sector impact
        current_sector_exposure = float(sector_exposure.get(symbol_sector, 0.0))
        new_sector_exposure = current_sector_exposure + position_size
        sector_exposure_pct = (
            (new_sector_exposure / portfolio_value) * 100.0
            if portfolio_value > 0
            else 0.0
        )
        sector_impact = (
            position_size / new_sector_exposure if new_sector_exposure > 0 else 0.0
        )

        # Calculate diversification impact
        total_positions = len(positions)
        current_diversification = 1.0 / total_positions if total_positions > 0 else 0.0
        new_diversification = (
            1.0 / (total_positions + 1)
            if symbol not in [p.get("symbol") for p in positions]
            else current_diversification
        )
        diversification_impact = new_diversification - current_diversification

        # Calculate correlation impact if correlation data is provided
        correlation_impact = 0.0
        weighted_correlation = 0.0

        if correlation_data and symbol in correlation_data:
            symbol_correlations = correlation_data[symbol]
            total_weight = 0.0

            for position in positions:
                pos_symbol = position.get("symbol")
                pos_weight = float(position.get("weight", 0.0))

                if pos_symbol in symbol_correlations:
                    correlation = float(symbol_correlations[pos_symbol])
                    weighted_correlation += correlation * pos_weight
                    total_weight += pos_weight

            if total_weight > 0:
                weighted_correlation /= total_weight
                correlation_impact = weighted_correlation

        # Calculate risk metrics
        # Higher correlation impact means higher portfolio risk
        risk_impact = 0.5 + (correlation_impact / 2.0) if correlation_data else 0.5

        # Calculate expected portfolio metrics after adding the position
        new_portfolio_value = portfolio_value + position_size
        new_position_count = total_positions + (
            1 if symbol not in [p.get("symbol") for p in positions] else 0
        )

        return {
            "symbol": symbol,
            "position_size": float(position_size),
            "position_percentage": float(position_pct),
            "sector_impact": {
                "sector": symbol_sector,
                "current_exposure": float(current_sector_exposure),
                "new_exposure": float(new_sector_exposure),
                "exposure_percentage": float(sector_exposure_pct),
                "sector_impact": float(sector_impact),
            },
            "diversification_impact": {
                "current_positions": total_positions,
                "new_positions": new_position_count,
                "current_diversification": float(current_diversification),
                "new_diversification": float(new_diversification),
                "diversification_impact": float(diversification_impact),
            },
            "correlation_impact": {
                "weighted_correlation": float(weighted_correlation),
                "correlation_impact": float(correlation_impact),
            },
            "risk_metrics": {"risk_impact": float(risk_impact)},
            "portfolio_metrics": {
                "current_portfolio_value": float(portfolio_value),
                "new_portfolio_value": float(new_portfolio_value),
            },
        }

    def _optimize_portfolio_allocation(
        self,
        assets: List[str],
        returns_data: Dict[str, List[float]],
        risk_free_rate: float = 0.0,
        target_return: Optional[float] = None,
        constraints: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation using modern portfolio theory.

        Args:
            assets: List of asset symbols
            returns_data: Dictionary mapping assets to their historical returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            target_return: Optional target return for optimization
            constraints: Optional constraints for optimization

        Returns:
            Optimized portfolio allocation
        """
        # Validate inputs
        if not assets:
            return {"error": "No assets provided"}

        if not returns_data:
            return {"error": "No returns data provided"}

        # Convert returns data to numpy arrays
        returns_arrays = {}
        for asset in assets:
            if asset in returns_data:
                returns_arrays[asset] = np.array(returns_data[asset])

        # Create returns matrix
        returns_matrix = np.array([returns_arrays[asset] for asset in assets])

        # Calculate expected returns and covariance matrix
        expected_returns = np.mean(returns_matrix, axis=1)
        cov_matrix = np.cov(returns_matrix)

        # Default constraints if none provided
        if constraints is None:
            constraints = {
                "min_weight": 0.0,  # No short selling
                "max_weight": 1.0,  # No leverage
                "sum_weights": 1.0,  # Fully invested
            }

        min_weight = constraints.get("min_weight", 0.0)
        max_weight = constraints.get("max_weight", 1.0)
        sum_weights = constraints.get("sum_weights", 1.0)

        # Define optimization functions
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        def portfolio_return(weights):
            return np.sum(expected_returns * weights)

        def negative_sharpe_ratio(weights):
            p_ret = portfolio_return(weights)
            p_vol = portfolio_volatility(weights)
            return -(p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0

        # Define constraints
        constraints_list = [
            {
                "type": "eq",
                "fun": lambda x: np.sum(x) - sum_weights,
            }  # Sum of weights = sum_weights
        ]

        # Define bounds
        bounds = tuple((min_weight, max_weight) for _ in range(len(assets)))

        # Initial guess (equal weights)
        initial_weights = np.array([1.0 / len(assets)] * len(assets))

        # Optimization results
        optimization_results = {}

        # Optimize for minimum volatility
        min_vol_result = optimize.minimize(
            portfolio_volatility,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
        )

        min_vol_weights = min_vol_result["x"]
        min_vol_return = portfolio_return(min_vol_weights)
        min_vol_volatility = portfolio_volatility(min_vol_weights)
        min_vol_sharpe = (
            (min_vol_return - risk_free_rate) / min_vol_volatility
            if min_vol_volatility > 0
            else 0
        )

        optimization_results["min_volatility"] = {
            "weights": {
                asset: float(weight) for asset, weight in zip(assets, min_vol_weights)
            },
            "expected_return": float(min_vol_return),
            "volatility": float(min_vol_volatility),
            "sharpe_ratio": float(min_vol_sharpe),
        }

        # Optimize for maximum Sharpe ratio
        max_sharpe_result = optimize.minimize(
            negative_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list,
        )

        max_sharpe_weights = max_sharpe_result["x"]
        max_sharpe_return = portfolio_return(max_sharpe_weights)
        max_sharpe_volatility = portfolio_volatility(max_sharpe_weights)
        max_sharpe_ratio = (
            (max_sharpe_return - risk_free_rate) / max_sharpe_volatility
            if max_sharpe_volatility > 0
            else 0
        )

        optimization_results["max_sharpe"] = {
            "weights": {
                asset: float(weight)
                for asset, weight in zip(assets, max_sharpe_weights)
            },
            "expected_return": float(max_sharpe_return),
            "volatility": float(max_sharpe_volatility),
            "sharpe_ratio": float(max_sharpe_ratio),
        }

        # Optimize for target return if provided
        if target_return is not None:
            target_return_constraint = {
                "type": "eq",
                "fun": lambda x: portfolio_return(x) - target_return,
            }

            target_return_constraints = constraints_list + [target_return_constraint]

            target_return_result = optimize.minimize(
                portfolio_volatility,
                initial_weights,
                method="SLSQP",
                bounds=bounds,
                constraints=target_return_constraints,
            )

            if target_return_result["success"]:
                target_return_weights = target_return_result["x"]
                target_return_volatility = portfolio_volatility(target_return_weights)
                target_return_sharpe = (
                    (target_return - risk_free_rate) / target_return_volatility
                    if target_return_volatility > 0
                    else 0
                )

                optimization_results["target_return"] = {
                    "weights": {
                        asset: float(weight)
                        for asset, weight in zip(assets, target_return_weights)
                    },
                    "expected_return": float(target_return),
                    "volatility": float(target_return_volatility),
                    "sharpe_ratio": float(target_return_sharpe),
                }

        # Generate efficient frontier points
        if len(assets) > 1:
            target_returns = np.linspace(min_vol_return, max(expected_returns), 20)
            efficient_frontier = []

            for target_ret in target_returns:
                target_ret_constraint = {
                    "type": "eq",
                    "fun": lambda x: portfolio_return(x) - target_ret,
                }

                target_ret_constraints = constraints_list + [target_ret_constraint]

                target_ret_result = optimize.minimize(
                    portfolio_volatility,
                    initial_weights,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=target_ret_constraints,
                )

                if target_ret_result["success"]:
                    target_ret_weights = target_ret_result["x"]
                    target_ret_volatility = portfolio_volatility(target_ret_weights)

                    efficient_frontier.append(
                        {
                            "return": float(target_ret),
                            "volatility": float(target_ret_volatility),
                        }
                    )

            optimization_results["efficient_frontier"] = efficient_frontier

        return {
            "assets": assets,
            "risk_free_rate": float(risk_free_rate),
            "expected_returns": {
                asset: float(ret) for asset, ret in zip(assets, expected_returns)
            },
            "optimization_results": optimization_results,
            "constraints_applied": constraints,
        }

    def _calculate_kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float,
        max_allocation: float = 1.0,
        fractional_kelly: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss
            max_allocation: Maximum allocation as fraction of capital
            fractional_kelly: Fraction of Kelly to use (0-1)

        Returns:
            Kelly criterion allocation
        """
        # Calculate full Kelly criterion
        # Kelly % = W - (1-W)/R where W is win rate and R is win/loss ratio
        full_kelly = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fractional Kelly
        fractional_kelly_value = full_kelly * fractional_kelly

        # Apply maximum allocation constraint
        final_allocation = min(fractional_kelly_value, max_allocation)

        # Handle negative Kelly (negative expectation)
        if full_kelly <= 0:
            final_allocation = 0.0
            recommendation = "Do not trade - negative expectation"
        else:
            recommendation = f"Allocate {final_allocation:.2%} of capital"

        return {
            "win_rate": float(win_rate),
            "win_loss_ratio": float(win_loss_ratio),
            "full_kelly": float(full_kelly),
            "fractional_kelly": float(fractional_kelly_value),
            "final_allocation": float(final_allocation),
            "allocation_percentage": float(final_allocation * 100),
            "recommendation": recommendation,
            "parameters": {
                "max_allocation": float(max_allocation),
                "fractional_kelly_factor": float(fractional_kelly),
            },
        }

    # Public API methods

    def calculate_optimal_position_size(
        self,
        symbol: str,
        confidence: float,
        portfolio_data: Dict[str, Any],
        max_position_pct: float = 5.0,
        risk_per_trade_pct: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on portfolio constraints.

        Args:
            symbol: Stock symbol
            confidence: Decision confidence (0-1)
            portfolio_data: Current portfolio state
            max_position_pct: Maximum position size as percentage of portfolio
            risk_per_trade_pct: Risk per trade as percentage of portfolio

        Returns:
            Dictionary with optimal position size and reasoning
        """
        params = {
            "symbol": symbol,
            "confidence": confidence,
            "portfolio_data": portfolio_data,
            "max_position_pct": max_position_pct,
            "risk_per_trade_pct": risk_per_trade_pct,
        }
        return self._handle_calculate_optimal_position_size(params)

    def calculate_portfolio_impact(
        self,
        symbol: str,
        position_size: float,
        portfolio_data: Dict[str, Any],
        correlation_data: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Calculate the impact of a new position on the portfolio.

        Args:
            symbol: Stock symbol
            position_size: Proposed position size
            portfolio_data: Current portfolio state
            correlation_data: Optional correlation data between assets

        Returns:
            Dictionary with portfolio impact metrics
        """
        params = {
            "symbol": symbol,
            "position_size": position_size,
            "portfolio_data": portfolio_data,
            "correlation_data": correlation_data,
        }
        return self._handle_calculate_portfolio_impact(params)

    def optimize_portfolio_allocation(
        self,
        assets: List[str],
        returns_data: Dict[str, List[float]],
        risk_free_rate: float = 0.0,
        target_return: Optional[float] = None,
        constraints: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation using modern portfolio theory.

        Args:
            assets: List of asset symbols
            returns_data: Dictionary mapping assets to their historical returns
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            target_return: Optional target return for optimization
            constraints: Optional constraints for optimization

        Returns:
            Dictionary with optimized portfolio allocation
        """
        params = {
            "assets": assets,
            "returns_data": returns_data,
            "risk_free_rate": risk_free_rate,
            "target_return": target_return,
            "constraints": constraints,
        }
        return self._handle_optimize_portfolio_allocation(params)

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        win_loss_ratio: float,
        max_allocation: float = 1.0,
        fractional_kelly: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Calculate position size using Kelly Criterion.

        Args:
            win_rate: Probability of winning (0-1)
            win_loss_ratio: Ratio of average win to average loss
            max_allocation: Maximum allocation as fraction of capital
            fractional_kelly: Fraction of Kelly to use (0-1)

        Returns:
            Dictionary with Kelly criterion allocation
        """
        params = {
            "win_rate": win_rate,
            "win_loss_ratio": win_loss_ratio,
            "max_allocation": max_allocation,
            "fractional_kelly": fractional_kelly,
        }
        return self._handle_calculate_kelly_criterion(params)
