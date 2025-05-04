"""
Slippage Analysis MCP Server

This module implements a Model Context Protocol (MCP) server for analyzing
execution quality, slippage, and market impact of trades.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# GPU acceleration imports
try:
    import cupy as cp

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False
    cp = np  # Fallback to numpy if CUDA not available

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring


# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis_mcp-slippage-analysis-mcp",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class SlippageAnalysisMCP(BaseMCPServer):
    """
    MCP server for execution quality analysis.

    This server provides tools for measuring execution slippage,
    analyzing market impact, and evaluating execution timing.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Slippage Analysis MCP server.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(name="slippage_analysis_mcp", config=config)

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        # Configure GPU usage
        self.use_gpu = self.config.get("use_gpu", HAVE_GPU)

        if self.use_gpu and HAVE_GPU:
            gpu_device = self.config.get("gpu_device", 0)
            try:
                # Set the active CUDA device
                cp.cuda.Device(gpu_device).use()
                self.logger.info(f"Using GPU device {gpu_device} for slippage analysis")

                # Check if device is A100
                device_attributes = cp.cuda.runtime.getDeviceProperties(gpu_device)
                if "A100" in device_attributes["name"]:
                    self.logger.info(f"Detected A100 GPU: {device_attributes['name']}")
                    # Could enable specific A100 optimizations here
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize GPU: {e}. Falling back to CPU."
                )
                self.use_gpu = False
        else:
            if not HAVE_GPU and self.config.get("use_gpu", False):
                self.logger.warning(
                    "GPU usage requested but CuPy/cuDF not available. Using CPU instead."
                )
            else:
                self.logger.info("Using CPU for slippage analysis")
            self.use_gpu = False

        self.logger.info("SlippageAnalysisMCP initialized")

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "calculate_slippage": {
                "description": "Calculate execution slippage",
                "category": "execution_quality",
                "required_params": ["expected_price", "executed_price", "side"],
                "optional_params": ["quantity", "timestamp"],
                "handler": self._handle_calculate_slippage,
            },
            "analyze_market_impact": {
                "description": "Analyze market impact of a trade",
                "category": "execution_quality",
                "required_params": [
                    "pre_trade_prices",
                    "post_trade_prices",
                    "trade_size",
                    "average_volume",
                ],
                "optional_params": ["time_window"],
                "handler": self._handle_analyze_market_impact,
            },
            "evaluate_execution_timing": {
                "description": "Evaluate execution timing optimality",
                "category": "execution_quality",
                "required_params": [
                    "execution_price",
                    "price_series",
                    "execution_index",
                ],
                "optional_params": ["window_size"],
                "handler": self._handle_evaluate_execution_timing,
            },
            "analyze_liquidity_impact": {
                "description": "Analyze impact on market liquidity",
                "category": "execution_quality",
                "required_params": ["pre_trade_bid_ask", "post_trade_bid_ask", "side"],
                "optional_params": ["time_window"],
                "handler": self._handle_analyze_liquidity_impact,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Slippage Analysis MCP."""
        self.register_tool(self.calculate_slippage)
        self.register_tool(self.analyze_market_impact)
        self.register_tool(self.evaluate_execution_timing)
        self.register_tool(self.analyze_liquidity_impact)

    # Handler methods for specific endpoints

    def _handle_calculate_slippage(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle calculate_slippage endpoint."""
        expected_price = params.get("expected_price")
        executed_price = params.get("executed_price")
        side = params.get("side")
        quantity = params.get("quantity", 1.0)
        timestamp = params.get("timestamp")

        if expected_price is None or executed_price is None or not side:
            return {"error": "Missing required parameters"}

        try:
            # Calculate slippage
            slippage_amount, slippage_percent = self._calculate_slippage(
                expected_price, executed_price, side
            )

            # Calculate cost impact
            cost_impact = slippage_amount * quantity

            result = {
                "expected_price": float(expected_price),
                "executed_price": float(executed_price),
                "side": side,
                "slippage_amount": float(slippage_amount),
                "slippage_percent": float(slippage_percent),
                "cost_impact": float(cost_impact),
            }

            if timestamp:
                result["timestamp"] = timestamp

            return result
        except Exception as e:
            self.logger.error(f"Error calculating slippage: {e}")
            return {"error": f"Failed to calculate slippage: {str(e)}"}

    def _handle_analyze_market_impact(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analyze_market_impact endpoint."""
        pre_trade_prices = params.get("pre_trade_prices", [])
        post_trade_prices = params.get("post_trade_prices", [])
        trade_size = params.get("trade_size")
        average_volume = params.get("average_volume")
        time_window = params.get("time_window", 5)  # minutes

        if (
            not pre_trade_prices
            or not post_trade_prices
            or trade_size is None
            or average_volume is None
        ):
            return {"error": "Missing required parameters"}

        try:
            # Convert to numpy arrays if needed
            if not isinstance(pre_trade_prices, np.ndarray):
                pre_trade_prices = np.array(pre_trade_prices)
            if not isinstance(post_trade_prices, np.ndarray):
                post_trade_prices = np.array(post_trade_prices)

            # Calculate market impact
            impact_result = self._analyze_market_impact(
                pre_trade_prices,
                post_trade_prices,
                trade_size,
                average_volume,
                time_window,
            )

            return impact_result
        except Exception as e:
            self.logger.error(f"Error analyzing market impact: {e}")
            return {"error": f"Failed to analyze market impact: {str(e)}"}

    def _handle_evaluate_execution_timing(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle evaluate_execution_timing endpoint."""
        execution_price = params.get("execution_price")
        price_series = params.get("price_series", [])
        execution_index = params.get("execution_index")
        window_size = params.get("window_size", 20)

        if execution_price is None or not price_series or execution_index is None:
            return {"error": "Missing required parameters"}

        try:
            # Convert to numpy array if needed
            if not isinstance(price_series, np.ndarray):
                price_series = np.array(price_series)

            # Evaluate execution timing
            timing_result = self._evaluate_execution_timing(
                execution_price, price_series, execution_index, window_size
            )

            return timing_result
        except Exception as e:
            self.logger.error(f"Error evaluating execution timing: {e}")
            return {"error": f"Failed to evaluate execution timing: {str(e)}"}

    def _handle_analyze_liquidity_impact(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle analyze_liquidity_impact endpoint."""
        pre_trade_bid_ask = params.get("pre_trade_bid_ask", {})
        post_trade_bid_ask = params.get("post_trade_bid_ask", {})
        side = params.get("side")
        time_window = params.get("time_window", 5)  # minutes

        if not pre_trade_bid_ask or not post_trade_bid_ask or not side:
            return {"error": "Missing required parameters"}

        try:
            # Analyze liquidity impact
            liquidity_result = self._analyze_liquidity_impact(
                pre_trade_bid_ask, post_trade_bid_ask, side, time_window
            )

            return liquidity_result
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity impact: {e}")
            return {"error": f"Failed to analyze liquidity impact: {str(e)}"}

    # Core analysis methods

    def _calculate_slippage(
        self, expected_price: float, executed_price: float, side: str
    ) -> Tuple[float, float]:
        """
        Calculate execution slippage.

        Args:
            expected_price: Expected execution price
            executed_price: Actual execution price
            side: Trade side ('buy' or 'sell')

        Returns:
            Tuple of (slippage_amount, slippage_percent)
        """
        if side.lower() == "buy":
            slippage_amount = executed_price - expected_price
        else:  # sell
            slippage_amount = expected_price - executed_price

        slippage_percent = slippage_amount / expected_price * 100

        return slippage_amount, slippage_percent

    def _analyze_market_impact(
        self,
        pre_trade_prices: np.ndarray,
        post_trade_prices: np.ndarray,
        trade_size: float,
        average_volume: float,
        time_window: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze market impact of a trade.

        Args:
            pre_trade_prices: Price series before trade
            post_trade_prices: Price series after trade
            trade_size: Size of the executed trade
            average_volume: Average trading volume
            time_window: Time window in minutes

        Returns:
            Market impact analysis results
        """
        if self.use_gpu and HAVE_GPU:
            # Convert numpy arrays to cupy arrays for GPU computation
            try:
                pre_trade_gpu = cp.asarray(pre_trade_prices)
                post_trade_gpu = cp.asarray(post_trade_prices)

                # Calculate pre-trade price statistics on GPU
                pre_trade_mean = cp.mean(pre_trade_gpu)
                pre_trade_std = cp.std(pre_trade_gpu)

                # Calculate post-trade price statistics on GPU
                post_trade_mean = cp.mean(post_trade_gpu)
                post_trade_std = cp.std(post_trade_gpu)

                # Calculate price change
                price_change = post_trade_mean - pre_trade_mean
                price_change_percent = price_change / pre_trade_mean * 100

                # Calculate volatility change
                volatility_change = post_trade_std - pre_trade_std
                volatility_change_percent = (
                    volatility_change / pre_trade_std * 100 if pre_trade_std > 0 else 0
                )

                # Calculate relative trade size
                relative_size = trade_size / average_volume

                # Calculate estimated market impact
                estimated_impact = relative_size * price_change_percent

                # Convert back to CPU for return
                price_change = float(cp.asnumpy(price_change))
                price_change_percent = float(cp.asnumpy(price_change_percent))
                volatility_change = float(cp.asnumpy(volatility_change))
                volatility_change_percent = float(cp.asnumpy(volatility_change_percent))
                estimated_impact = float(cp.asnumpy(estimated_impact))

                #
                # Use CPU for simple calculation to avoid unnecessary GPU
                # transfers
                relative_size = float(relative_size)

            except Exception as e:
                self.logger.warning(f"GPU computation failed: {e}, falling back to CPU")
                # Fallback to CPU implementation
                return self._analyze_market_impact_cpu(
                    pre_trade_prices,
                    post_trade_prices,
                    trade_size,
                    average_volume,
                    time_window,
                )
        else:
            # Use CPU implementation
            return self._analyze_market_impact_cpu(
                pre_trade_prices,
                post_trade_prices,
                trade_size,
                average_volume,
                time_window,
            )

        return {
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "volatility_change": volatility_change,
            "volatility_change_percent": volatility_change_percent,
            "relative_trade_size": relative_size,
            "estimated_impact": estimated_impact,
            "time_window_minutes": time_window,
            "computed_on": "gpu",
        }

    def _analyze_market_impact_cpu(
        self,
        pre_trade_prices: np.ndarray,
        post_trade_prices: np.ndarray,
        trade_size: float,
        average_volume: float,
        time_window: int = 5,
    ) -> Dict[str, Any]:
        """CPU implementation of market impact analysis."""
        # Calculate pre-trade price statistics
        pre_trade_mean = np.mean(pre_trade_prices)
        pre_trade_std = np.std(pre_trade_prices)

        # Calculate post-trade price statistics
        post_trade_mean = np.mean(post_trade_prices)
        post_trade_std = np.std(post_trade_prices)

        # Calculate price change
        price_change = post_trade_mean - pre_trade_mean
        price_change_percent = price_change / pre_trade_mean * 100

        # Calculate volatility change
        volatility_change = post_trade_std - pre_trade_std
        volatility_change_percent = (
            volatility_change / pre_trade_std * 100 if pre_trade_std > 0 else 0
        )

        # Calculate relative trade size
        relative_size = trade_size / average_volume

        # Calculate estimated market impact
        estimated_impact = relative_size * price_change_percent

        return {
            "price_change": float(price_change),
            "price_change_percent": float(price_change_percent),
            "volatility_change": float(volatility_change),
            "volatility_change_percent": float(volatility_change_percent),
            "relative_trade_size": float(relative_size),
            "estimated_impact": float(estimated_impact),
            "time_window_minutes": time_window,
            "computed_on": "cpu",
        }

    def _evaluate_execution_timing(
        self,
        execution_price: float,
        price_series: np.ndarray,
        execution_index: int,
        window_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Evaluate execution timing optimality.

        Args:
            execution_price: Price at which the trade was executed
            price_series: Full price series around execution
            execution_index: Index in the series where execution occurred
            window_size: Window size for analysis

        Returns:
            Execution timing evaluation results
        """
        if self.use_gpu and HAVE_GPU and len(price_series) > 1000:
            # Use GPU for larger arrays where there's more benefit
            try:
                return self._evaluate_execution_timing_gpu(
                    execution_price, price_series, execution_index, window_size
                )
            except Exception as e:
                self.logger.warning(
                    f"GPU execution timing evaluation failed: {e}, falling back to CPU"
                )
                # Fallback to CPU implementation

        # CPU implementation (default)
        # Extract window around execution
        start_idx = max(0, execution_index - window_size)
        end_idx = min(len(price_series), execution_index + window_size)
        window = price_series[start_idx:end_idx]

        # Calculate optimal prices
        if len(window) > 0:
            min_price = np.min(window)
            max_price = np.max(window)
            mean_price = np.mean(window)
        else:
            min_price = max_price = mean_price = execution_price

        # Calculate timing metrics
        min_diff = execution_price - min_price
        max_diff = max_price - execution_price
        mean_diff = execution_price - mean_price

        # Calculate percentile of execution
        if len(window) > 1:
            sorted_prices = np.sort(window)
            percentile = (
                np.searchsorted(sorted_prices, execution_price)
                / len(sorted_prices)
                * 100
            )
        else:
            percentile = 50.0

        # Calculate timing score (0-100, higher is better)
        if len(window) > 1:
            if min_price == max_price:
                timing_score = 100.0  # No price variation
            else:
                # For buys, lower is better; for sells, higher is better
                # We'll use a neutral score based on distance from mean
                timing_score = 100.0 - (
                    abs(execution_price - mean_price) / (max_price - min_price) * 100.0
                )
        else:
            timing_score = 50.0  # Neutral score if not enough data

        return {
            "execution_price": float(execution_price),
            "min_price": float(min_price),
            "max_price": float(max_price),
            "mean_price": float(mean_price),
            "min_diff": float(min_diff),
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "percentile": float(percentile),
            "timing_score": float(timing_score),
            "window_size": window_size,
            "computed_on": "cpu",
        }

    def _evaluate_execution_timing_gpu(
        self,
        execution_price: float,
        price_series: np.ndarray,
        execution_index: int,
        window_size: int = 20,
    ) -> Dict[str, Any]:
        """GPU implementation of execution timing evaluation."""
        # Transfer data to GPU
        price_series_gpu = cp.asarray(price_series)

        # Extract window around execution on GPU
        start_idx = max(0, execution_index - window_size)
        end_idx = min(len(price_series), execution_index + window_size)
        window_gpu = price_series_gpu[start_idx:end_idx]

        # Calculate optimal prices on GPU
        if len(window_gpu) > 0:
            min_price = float(cp.min(window_gpu).get())
            max_price = float(cp.max(window_gpu).get())
            mean_price = float(cp.mean(window_gpu).get())
        else:
            min_price = max_price = mean_price = execution_price

        # Calculate timing metrics on CPU (simple operations)
        min_diff = execution_price - min_price
        max_diff = max_price - execution_price
        mean_diff = execution_price - mean_price

        # Calculate percentile of execution
        if len(window_gpu) > 1:
            sorted_prices_gpu = cp.sort(window_gpu)
            #
            # Convert to numpy for searchsorted which may not be optimized in
            # CuPy
            sorted_prices = cp.asnumpy(sorted_prices_gpu)
            percentile = (
                np.searchsorted(sorted_prices, execution_price)
                / len(sorted_prices)
                * 100
            )
        else:
            percentile = 50.0

        # Calculate timing score (0-100, higher is better)
        if len(window_gpu) > 1:
            if min_price == max_price:
                timing_score = 100.0  # No price variation
            else:
                # For buys, lower is better; for sells, higher is better
                # We'll use a neutral score based on distance from mean
                timing_score = 100.0 - (
                    abs(execution_price - mean_price) / (max_price - min_price) * 100.0
                )
        else:
            timing_score = 50.0  # Neutral score if not enough data

        return {
            "execution_price": float(execution_price),
            "min_price": float(min_price),
            "max_price": float(max_price),
            "mean_price": float(mean_price),
            "min_diff": float(min_diff),
            "max_diff": float(max_diff),
            "mean_diff": float(mean_diff),
            "percentile": float(percentile),
            "timing_score": float(timing_score),
            "window_size": window_size,
            "computed_on": "gpu",
        }

    def _analyze_liquidity_impact(
        self,
        pre_trade_bid_ask: Dict[str, Any],
        post_trade_bid_ask: Dict[str, Any],
        side: str,
        time_window: int = 5,
    ) -> Dict[str, Any]:
        """
        Analyze impact on market liquidity.

        Args:
            pre_trade_bid_ask: Bid-ask data before trade
            post_trade_bid_ask: Bid-ask data after trade
            side: Trade side ('buy' or 'sell')
            time_window: Time window in minutes

        Returns:
            Liquidity impact analysis results
        """
        # Extract bid-ask data
        pre_bid = pre_trade_bid_ask.get("bid_price", 0)
        pre_ask = pre_trade_bid_ask.get("ask_price", 0)
        pre_bid_size = pre_trade_bid_ask.get("bid_size", 0)
        pre_ask_size = pre_trade_bid_ask.get("ask_size", 0)

        post_bid = post_trade_bid_ask.get("bid_price", 0)
        post_ask = post_trade_bid_ask.get("ask_price", 0)
        post_bid_size = post_trade_bid_ask.get("bid_size", 0)
        post_ask_size = post_trade_bid_ask.get("ask_size", 0)

        # Calculate spread changes
        pre_spread = pre_ask - pre_bid
        post_spread = post_ask - post_bid
        spread_change = post_spread - pre_spread
        spread_change_percent = (
            spread_change / pre_spread * 100 if pre_spread > 0 else 0
        )

        # Calculate size changes
        if side.lower() == "buy":
            # For buys, we're more interested in ask side
            size_change = post_ask_size - pre_ask_size
            size_change_percent = (
                size_change / pre_ask_size * 100 if pre_ask_size > 0 else 0
            )
            price_change = post_ask - pre_ask
        else:  # sell
            # For sells, we're more interested in bid side
            size_change = post_bid_size - pre_bid_size
            size_change_percent = (
                size_change / pre_bid_size * 100 if pre_bid_size > 0 else 0
            )
            price_change = post_bid - pre_bid

        price_change_percent = (
            price_change / (pre_bid if side.lower() == "sell" else pre_ask) * 100
        )

        # Calculate liquidity score (0-100, higher means less impact)
        if pre_spread > 0:
            spread_factor = min(100, max(0, 100 - (spread_change_percent * 2)))
        else:
            spread_factor = 50

        if (pre_ask_size if side.lower() == "buy" else pre_bid_size) > 0:
            size_factor = min(100, max(0, 100 - abs(size_change_percent)))
        else:
            size_factor = 50

        liquidity_score = (spread_factor + size_factor) / 2

        return {
            "pre_spread": float(pre_spread),
            "post_spread": float(post_spread),
            "spread_change": float(spread_change),
            "spread_change_percent": float(spread_change_percent),
            "size_change": float(size_change),
            "size_change_percent": float(size_change_percent),
            "price_change": float(price_change),
            "price_change_percent": float(price_change_percent),
            "liquidity_score": float(liquidity_score),
            "time_window_minutes": time_window,
        }

    # Public API methods

    def calculate_slippage(
        self,
        expected_price: float,
        executed_price: float,
        side: str,
        quantity: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculate execution slippage.

        Args:
            expected_price: Expected execution price
            executed_price: Actual execution price
            side: Trade side ('buy' or 'sell')
            quantity: Trade quantity

        Returns:
            Dictionary with slippage metrics
        """
        params = {
            "expected_price": expected_price,
            "executed_price": executed_price,
            "side": side,
            "quantity": quantity,
        }
        return self.call_endpoint("calculate_slippage", params)

    def analyze_market_impact(
        self,
        pre_trade_prices: List[float],
        post_trade_prices: List[float],
        trade_size: float,
        average_volume: float,
    ) -> Dict[str, Any]:
        """
        Analyze market impact of a trade.

        Args:
            pre_trade_prices: Price series before trade
            post_trade_prices: Price series after trade
            trade_size: Size of the executed trade
            average_volume: Average trading volume

        Returns:
            Dictionary with market impact analysis
        """
        params = {
            "pre_trade_prices": pre_trade_prices,
            "post_trade_prices": post_trade_prices,
            "trade_size": trade_size,
            "average_volume": average_volume,
        }
        return self.call_endpoint("analyze_market_impact", params)

    def evaluate_execution_timing(
        self, execution_price: float, price_series: List[float], execution_index: int
    ) -> Dict[str, Any]:
        """
        Evaluate execution timing optimality.

        Args:
            execution_price: Price at which the trade was executed
            price_series: Full price series around execution
            execution_index: Index in the series where execution occurred

        Returns:
            Dictionary with execution timing evaluation
        """
        params = {
            "execution_price": execution_price,
            "price_series": price_series,
            "execution_index": execution_index,
        }
        return self.call_endpoint("evaluate_execution_timing", params)

    def analyze_liquidity_impact(
        self,
        pre_trade_bid_ask: Dict[str, Any],
        post_trade_bid_ask: Dict[str, Any],
        side: str,
    ) -> Dict[str, Any]:
        """
        Analyze impact on market liquidity.

        Args:
            pre_trade_bid_ask: Bid-ask data before trade
            post_trade_bid_ask: Bid-ask data after trade
            side: Trade side ('buy' or 'sell')

        Returns:
            Dictionary with liquidity impact analysis
        """
        params = {
            "pre_trade_bid_ask": pre_trade_bid_ask,
            "post_trade_bid_ask": post_trade_bid_ask,
            "side": side,
        }
        return self.call_endpoint("analyze_liquidity_impact", params)
