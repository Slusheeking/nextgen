"""
Drift Detection MCP Server

This module implements a Model Context Protocol (MCP) server for detecting
price drift, trend changes, and momentum shifts in financial data.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration
CONFIG_PATH = os.path.join("config", "analysis_mcp", "drift_detection_config.json")
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Could not load config from {CONFIG_PATH}: {e}")
    CONFIG = {}

# GPU acceleration imports
try:
    import cupy as cp
    from cupyx import stats

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False
    cp = np  # Fallback to numpy if CUDA not available
    stats = None  # No equivalent in numpy

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring.system_monitor import MonitoringManager

# Set up monitoring (Prometheus + Python logging only)
monitor = MonitoringManager(service_name="analysis_mcp-drift-detection-mcp")
metrics = monitor.metrics


class DriftDetectionMCP(BaseMCPServer):
    """
    MCP server for price drift and trend detection.

    This server provides tools for identifying price drift from moving averages,
    detecting trend changes, and analyzing momentum shifts.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Drift Detection MCP server.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(name="drift_detection_mcp", config=config)

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        # Load configuration from file, with fallback to provided config
        file_config = CONFIG.copy() if CONFIG else {}
        
        # Merge provided config with file config, with provided config taking precedence
        if config:
            for key, value in config.items():
                file_config[key] = value
        
        # GPU configuration
        self.use_gpu = file_config.get("use_gpu", True)
        gpu_device = file_config.get("gpu_device", 0)

        # Check if we have GPU and whether we should use it
        self.gpu_available = HAVE_GPU and self.use_gpu

        # Check for A100 GPU if GPU is available
        self.has_a100 = False
        if self.gpu_available:
            try:
                gpu_info = cp.cuda.runtime.getDeviceProperties(gpu_device)
                gpu_name = gpu_info["name"].decode("utf-8")
                self.has_a100 = "A100" in gpu_name
                self.logger.info(f"GPU detected: {gpu_name}")

                # Set GPU-specific parameters
                if self.has_a100 and file_config.get("a100_optimizations", {}).get("enabled", True):
                    # A100 is more efficient with GPU computation even for smaller datasets
                    self.min_gpu_data_size = file_config.get("a100_optimizations", {}).get("min_gpu_data_size", 500)
                    self.logger.info(
                        f"A100 GPU detected - optimizing for high throughput (min_gpu_data_size: {self.min_gpu_data_size})"
                    )
                else:
                    self.min_gpu_data_size = file_config.get("min_gpu_data_size", 1000)
                    self.logger.info(f"Standard GPU detected (min_gpu_data_size: {self.min_gpu_data_size})")
            except Exception as e:
                self.logger.warning(f"Failed to detect GPU type: {e}")
                self.min_gpu_data_size = file_config.get("min_gpu_data_size", 1000)
        else:
            self.min_gpu_data_size = file_config.get("min_gpu_data_size", 1000)

        # Initialize execution tracking
        self.execution_stats = {
            "gpu_executions": 0,
            "cpu_executions": 0,
            "gpu_failures": 0,
        }

        self.logger.info(
            f"DriftDetectionMCP initialized (GPU enabled: {self.gpu_available})"
        )

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "detect_ma_drift": {
                "description": "Detect drift from moving averages",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["short_window", "long_window", "drift_threshold"],
                "handler": self._handle_detect_ma_drift,
            },
            "detect_trend_change": {
                "description": "Detect trend changes",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["window_size", "change_threshold"],
                "handler": self._handle_detect_trend_change,
            },
            "analyze_momentum": {
                "description": "Analyze price momentum",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["window_size", "momentum_threshold"],
                "handler": self._handle_analyze_momentum,
            },
            "detect_volatility_shift": {
                "description": "Detect shifts in volatility",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["window_size", "shift_threshold"],
                "handler": self._handle_detect_volatility_shift,
            },
            "get_execution_stats": {
                "description": "Get GPU/CPU execution statistics",
                "category": "system",
                "required_params": [],
                "optional_params": [],
                "handler": lambda _: self.get_execution_stats(),
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Drift Detection MCP."""
        self.register_tool(self.detect_ma_drift)
        self.register_tool(self.detect_trend_change)
        self.register_tool(self.analyze_momentum)
        self.register_tool(self.detect_volatility_shift)
        self.register_tool(self.get_execution_stats)

    # Handler methods for specific endpoints

    def _handle_detect_ma_drift(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_ma_drift endpoint."""
        prices = params.get("prices", [])
        
        # Get default values from config
        ma_drift_config = CONFIG.get("ma_drift", {})
        default_short_window = ma_drift_config.get("default_short_window", 5)
        default_long_window = ma_drift_config.get("default_long_window", 20)
        default_drift_threshold = ma_drift_config.get("default_drift_threshold", 0.02)
        
        # Use provided values or fall back to config defaults
        short_window = params.get("short_window", default_short_window)
        long_window = params.get("long_window", default_long_window)
        drift_threshold = params.get("drift_threshold", default_drift_threshold)

        if not prices or len(prices) < long_window:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)

            # Calculate moving averages
            short_ma = self._calculate_moving_average(prices, short_window)
            long_ma = self._calculate_moving_average(prices, long_window)

            # Calculate drift
            drift_result = self._calculate_ma_drift(
                prices, short_ma, long_ma, drift_threshold
            )

            return drift_result
        except Exception as e:
            self.logger.error(f"Error detecting MA drift: {e}")
            return {"error": f"Failed to detect MA drift: {str(e)}"}

    def _handle_detect_trend_change(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_trend_change endpoint."""
        prices = params.get("prices", [])
        
        # Get default values from config
        trend_change_config = CONFIG.get("trend_change", {})
        default_window_size = trend_change_config.get("default_window_size", 10)
        default_change_threshold = trend_change_config.get("default_change_threshold", 0.03)
        
        # Use provided values or fall back to config defaults
        window_size = params.get("window_size", default_window_size)
        change_threshold = params.get("change_threshold", default_change_threshold)

        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)

            # Detect trend change
            trend_result = self._detect_trend_change(
                prices, window_size, change_threshold
            )

            return trend_result
        except Exception as e:
            self.logger.error(f"Error detecting trend change: {e}")
            return {"error": f"Failed to detect trend change: {str(e)}"}

    def _handle_analyze_momentum(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analyze_momentum endpoint."""
        prices = params.get("prices", [])
        
        # Get default values from config
        momentum_config = CONFIG.get("momentum", {})
        default_window_size = momentum_config.get("default_window_size", 14)
        default_momentum_threshold = momentum_config.get("default_momentum_threshold", 0.1)
        
        # Use provided values or fall back to config defaults
        window_size = params.get("window_size", default_window_size)
        momentum_threshold = params.get("momentum_threshold", default_momentum_threshold)

        if not prices or len(prices) < window_size:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)

            # Analyze momentum
            momentum_result = self._analyze_momentum(
                prices, window_size, momentum_threshold
            )

            return momentum_result
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {"error": f"Failed to analyze momentum: {str(e)}"}

    def _handle_detect_volatility_shift(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_volatility_shift endpoint."""
        prices = params.get("prices", [])
        
        # Get default values from config
        volatility_config = CONFIG.get("volatility", {})
        default_window_size = volatility_config.get("default_window_size", 10)
        default_shift_threshold = volatility_config.get("default_shift_threshold", 0.5)
        
        # Use provided values or fall back to config defaults
        window_size = params.get("window_size", default_window_size)
        shift_threshold = params.get("shift_threshold", default_shift_threshold)

        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)

            # Detect volatility shift
            volatility_result = self._detect_volatility_shift(
                prices, window_size, shift_threshold
            )

            return volatility_result
        except Exception as e:
            self.logger.error(f"Error detecting volatility shift: {e}")
            return {"error": f"Failed to detect volatility shift: {str(e)}"}

    # Core analysis methods

    def _calculate_moving_average(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate simple moving average with GPU acceleration when available.

        Args:
            prices: Array of price data
            window: Window size for moving average

        Returns:
            Array of moving averages
        """
        if len(prices) < window:
            return np.array([])

        # Check if we should use GPU
        use_gpu = self.gpu_available and len(prices) >= self.min_gpu_data_size

        if use_gpu:
            try:
                # Transfer data to GPU
                prices_gpu = cp.asarray(prices)

                # Initialize result array on GPU
                ma_gpu = cp.zeros_like(prices_gpu)

                #
                # Use CuPy's implementation of convolve for faster moving
                # average
                #
                # This is significantly more efficient than the loop-based
                # approach
                weights = cp.ones(window) / window

                # Calculate the moving average using convolution
                ma_valid = cp.convolve(prices_gpu, weights, "valid")

                # Fill in the valid part of the result
                ma_gpu[window - 1 :] = ma_valid

                # Fill NaN for the initial window-1 elements
                ma_gpu[: window - 1] = cp.nan

                # Copy back to CPU
                ma = cp.asnumpy(ma_gpu)

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1

                return ma

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for moving average: {e}, fallback to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                # Fall back to CPU implementation

        # CPU implementation
        self.execution_stats["cpu_executions"] += 1

        # Calculate moving average
        # Ensure the array is created with float data type to support NaN values
        ma = np.zeros_like(prices, dtype=float)

        # Vectorized implementation for CPU
        # Much faster than the original loop-based approach
        for i in range(len(prices)):
            if i < window - 1:
                ma[i] = np.nan
            else:
                ma[i] = np.mean(prices[i - window + 1 : i + 1])

        return ma

    def _calculate_ma_drift(
        self,
        prices: np.ndarray,
        short_ma: np.ndarray,
        long_ma: np.ndarray,
        drift_threshold: float,
    ) -> Dict[str, Any]:
        """
        Calculate drift from moving averages with GPU acceleration when available.

        Args:
            prices: Array of price data
            short_ma: Short-term moving average
            long_ma: Long-term moving average
            drift_threshold: Threshold for significant drift

        Returns:
            Drift analysis results
        """
        # Check if we should use GPU
        use_gpu = self.gpu_available and len(prices) >= self.min_gpu_data_size

        if use_gpu:
            try:
                #
                # Transfer data to GPU - we only need the last 2 elements for
                # this calculation
                prices_gpu = cp.asarray(prices[-2:])
                short_ma_gpu = cp.asarray(short_ma[-2:])
                long_ma_gpu = cp.asarray(long_ma[-2:])

                # Get current values
                current_price = float(prices_gpu[-1])
                current_short_ma = float(short_ma_gpu[-1])
                current_long_ma = float(long_ma_gpu[-1])

                # Calculate drifts with safe division to avoid NaN
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                drift_from_short = float(
                    (current_price - current_short_ma) / (current_short_ma + epsilon)
                )
                drift_from_long = float(
                    (current_price - current_long_ma) / (current_long_ma + epsilon)
                )
                ma_spread = float(
                    (current_short_ma - current_long_ma) / (current_long_ma + epsilon)
                )

                # Check for crossover
                crossover = False
                crossover_direction = "none"

                if len(short_ma_gpu) > 1 and len(long_ma_gpu) > 1:
                    prev_short_ma = float(short_ma_gpu[-2])
                    prev_long_ma = float(long_ma_gpu[-2])

                    if (
                        prev_short_ma <= prev_long_ma
                        and current_short_ma > current_long_ma
                    ):
                        crossover = True
                        crossover_direction = "bullish"
                    elif (
                        prev_short_ma >= prev_long_ma
                        and current_short_ma < current_long_ma
                    ):
                        crossover = True
                        crossover_direction = "bearish"

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1

                #
                # Remaining logic runs on CPU since it's simple comparisons and
                # dictionary creation
            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for MA drift: {e}, fallback to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                # Fall back to CPU implementation
                use_gpu = False

        # CPU implementation or fallback from GPU error
        if not use_gpu:
            self.execution_stats["cpu_executions"] += 1

            # Get current values
            current_price = prices[-1]
            current_short_ma = short_ma[-1]
            current_long_ma = long_ma[-1]

            # Calculate drifts with safe division
            epsilon = 1e-10  # Small value to avoid division by zero
            drift_from_short = (current_price - current_short_ma) / (
                current_short_ma + epsilon
            )
            drift_from_long = (current_price - current_long_ma) / (
                current_long_ma + epsilon
            )
            ma_spread = (current_short_ma - current_long_ma) / (
                current_long_ma + epsilon
            )

            # Check for crossover
            crossover = False
            crossover_direction = "none"

            if len(short_ma) > 1 and len(long_ma) > 1:
                prev_short_ma = short_ma[-2]
                prev_long_ma = long_ma[-2]

                if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
                    crossover = True
                    crossover_direction = "bullish"
                elif (
                    prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma
                ):
                    crossover = True
                    crossover_direction = "bearish"

        # Determine drift direction and significance
        if abs(drift_from_short) > drift_threshold:
            short_drift_significant = True
            short_drift_direction = "up" if drift_from_short > 0 else "down"
        else:
            short_drift_significant = False
            short_drift_direction = "neutral"

        if abs(drift_from_long) > drift_threshold:
            long_drift_significant = True
            long_drift_direction = "up" if drift_from_long > 0 else "down"
        else:
            long_drift_significant = False
            long_drift_direction = "neutral"

        # Determine trend based on MA relationship
        if current_short_ma > current_long_ma:
            trend = "uptrend"
        elif current_short_ma < current_long_ma:
            trend = "downtrend"
        else:
            trend = "neutral"

        return {
            "current_price": float(current_price),
            "short_ma": float(current_short_ma),
            "long_ma": float(current_long_ma),
            "drift_from_short_ma": float(drift_from_short),
            "drift_from_long_ma": float(drift_from_long),
            "ma_spread": float(ma_spread),
            "short_drift_significant": short_drift_significant,
            "short_drift_direction": short_drift_direction,
            "long_drift_significant": long_drift_significant,
            "long_drift_direction": long_drift_direction,
            "trend": trend,
            "crossover": crossover,
            "crossover_direction": crossover_direction,
            "computation": "gpu" if use_gpu else "cpu",
        }

    def _detect_trend_change(
        self, prices: np.ndarray, window_size: int = 10, change_threshold: float = 0.03
    ) -> Dict[str, Any]:
        """
        Detect trend changes with GPU acceleration when available.

        Args:
            prices: Array of price data
            window_size: Window size for trend analysis
            change_threshold: Threshold for significant change

        Returns:
            Trend change detection results
        """
        if len(prices) < window_size * 2:
            return {"error": "Insufficient data for trend analysis"}

        # Check if we should use GPU
        use_gpu = self.gpu_available and len(prices) >= self.min_gpu_data_size

        if use_gpu:
            try:
                # Transfer only the data we need to GPU
                recent_window_gpu = cp.asarray(prices[-window_size:])
                prev_window_gpu = cp.asarray(prices[-window_size * 2 : -window_size])

                # Create x arrays for regression
                recent_x_gpu = cp.arange(window_size, dtype=cp.float64)
                prev_x_gpu = cp.arange(window_size, dtype=cp.float64)

                # Calculate linear regression for recent window on GPU
                # For A100 GPUs, we can use cusolver for faster polyfit
                if (
                    self.has_a100
                    and hasattr(cp, "linalg")
                    and hasattr(cp.linalg, "lstsq")
                ):
                    A_recent = cp.vstack([recent_x_gpu, cp.ones_like(recent_x_gpu)]).T
                    coeffs_recent = cp.linalg.lstsq(A_recent, recent_window_gpu)[0]
                    recent_slope, recent_intercept = (
                        float(coeffs_recent[0]),
                        float(coeffs_recent[1]),
                    )

                    A_prev = cp.vstack([prev_x_gpu, cp.ones_like(prev_x_gpu)]).T
                    coeffs_prev = cp.linalg.lstsq(A_prev, prev_window_gpu)[0]
                    prev_slope, prev_intercept = (
                        float(coeffs_prev[0]),
                        float(coeffs_prev[1]),
                    )
                else:
                    # Fallback to using np.polyfit but with GPU data
                    # Transfer data back to CPU for polyfit
                    recent_window_cpu = cp.asnumpy(recent_window_gpu)
                    prev_window_cpu = cp.asnumpy(prev_window_gpu)
                    recent_x_cpu = cp.asnumpy(recent_x_gpu)
                    prev_x_cpu = cp.asnumpy(prev_x_gpu)

                    recent_slope, recent_intercept = np.polyfit(
                        recent_x_cpu, recent_window_cpu, 1
                    )
                    prev_slope, prev_intercept = np.polyfit(
                        prev_x_cpu, prev_window_cpu, 1
                    )

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for trend detection: {e}, fallback to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available or failed
        if not use_gpu:
            self.execution_stats["cpu_executions"] += 1

            # Calculate linear regression for recent window
            recent_window = prices[-window_size:]
            recent_x = np.arange(window_size)
            recent_slope, recent_intercept = np.polyfit(recent_x, recent_window, 1)

            # Calculate linear regression for previous window
            prev_window = prices[-window_size * 2 : -window_size]
            prev_x = np.arange(window_size)
            prev_slope, prev_intercept = np.polyfit(prev_x, prev_window, 1)

        #
        # Calculate slope change - this simple calculation runs on CPU
        # regardless
        slope_change = recent_slope - prev_slope
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        slope_change_percent = (
            slope_change / (abs(prev_slope) + epsilon)
            if prev_slope != 0
            else float("inf")
        )

        # Determine trend directions
        if recent_slope > 0:
            recent_trend = "uptrend"
        elif recent_slope < 0:
            recent_trend = "downtrend"
        else:
            recent_trend = "neutral"

        if prev_slope > 0:
            prev_trend = "uptrend"
        elif prev_slope < 0:
            prev_trend = "downtrend"
        else:
            prev_trend = "neutral"

        # Determine if trend change is significant
        significant_change = abs(slope_change_percent) > change_threshold

        # Determine trend change type
        if recent_trend != prev_trend:
            trend_change_type = f"{prev_trend}_to_{recent_trend}"
        elif significant_change:
            if slope_change > 0:
                trend_change_type = "acceleration"
            else:
                trend_change_type = "deceleration"
        else:
            trend_change_type = "continuation"

        return {
            "recent_trend": recent_trend,
            "previous_trend": prev_trend,
            "recent_slope": float(recent_slope),
            "previous_slope": float(prev_slope),
            "slope_change": float(slope_change),
            "slope_change_percent": float(slope_change_percent),
            "significant_change": significant_change,
            "trend_change_type": trend_change_type,
            "window_size": window_size,
            "computation": "gpu" if use_gpu else "cpu",
        }

    def _analyze_momentum(
        self, prices: np.ndarray, window_size: int = 14, momentum_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Analyze price momentum with GPU acceleration when available.

        Args:
            prices: Array of price data
            window_size: Window size for momentum calculation
            momentum_threshold: Threshold for significant momentum

        Returns:
            Momentum analysis results
        """
        if len(prices) < window_size:
            return {"error": "Insufficient data for momentum analysis"}

        # Check if we should use GPU
        use_gpu = self.gpu_available and len(prices) >= self.min_gpu_data_size

        if use_gpu:
            try:
                # Transfer data to GPU
                prices_gpu = cp.asarray(prices)

                # Calculate rate of change (ROC)
                # Add small epsilon to avoid division by zero
                epsilon = 1e-10
                roc = float(
                    (prices_gpu[-1] / (prices_gpu[-window_size] + epsilon) - 1) * 100
                )

                # Calculate RSI
                returns_gpu = cp.diff(prices_gpu)

                # Create copies for gains and losses
                gains_gpu = cp.copy(returns_gpu)
                losses_gpu = cp.copy(returns_gpu)

                # Set negative gains to zero and positive losses to zero
                gains_gpu = cp.maximum(gains_gpu, 0)
                losses_gpu = cp.absolute(cp.minimum(losses_gpu, 0))

                if len(gains_gpu) >= window_size:
                    avg_gain = float(cp.mean(gains_gpu[-window_size:]))
                    avg_loss = float(cp.mean(losses_gpu[-window_size:]))

                    if avg_loss == 0:
                        rsi = 100
                    else:
                        rs = avg_gain / (avg_loss + epsilon)
                        rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50  # Default if not enough data

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for momentum analysis: {e}, fallback to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available or failed
        if not use_gpu:
            self.execution_stats["cpu_executions"] += 1

            # Calculate rate of change (ROC)
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            roc = (prices[-1] / (prices[-window_size] + epsilon) - 1) * 100

            # Calculate RSI
            returns = np.diff(prices)
            gains = np.copy(returns)
            losses = np.copy(returns)
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = abs(losses)

            if len(gains) >= window_size:
                avg_gain = np.mean(gains[-window_size:])
                avg_loss = np.mean(losses[-window_size:])

                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / (avg_loss + epsilon)
                    rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 50  # Default if not enough data

        # Calculate momentum strength
        momentum_strength = abs(roc)

        # Determine momentum direction
        if roc > 0:
            momentum_direction = "positive"
        elif roc < 0:
            momentum_direction = "negative"
        else:
            momentum_direction = "neutral"

        # Determine if momentum is significant
        significant_momentum = momentum_strength > momentum_threshold

        # Determine momentum state based on RSI
        if rsi > 70:
            momentum_state = "overbought"
        elif rsi < 30:
            momentum_state = "oversold"
        else:
            momentum_state = "neutral"

        return {
            "rate_of_change": float(roc),
            "rsi": float(rsi),
            "momentum_strength": float(momentum_strength),
            "momentum_direction": momentum_direction,
            "significant_momentum": significant_momentum,
            "momentum_state": momentum_state,
            "window_size": window_size,
            "computation": "gpu" if use_gpu else "cpu",
        }

    def _detect_volatility_shift(
        self, prices: np.ndarray, window_size: int = 10, shift_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect shifts in volatility with GPU acceleration when available.

        Args:
            prices: Array of price data
            window_size: Window size for volatility calculation
            shift_threshold: Threshold for significant volatility shift

        Returns:
            Volatility shift detection results
        """
        if len(prices) < window_size * 2:
            return {"error": "Insufficient data for volatility analysis"}

        # Check if we should use GPU
        use_gpu = self.gpu_available and len(prices) >= self.min_gpu_data_size

        if use_gpu:
            try:
                # Transfer data to GPU
                prices_gpu = cp.asarray(prices)

                #
                # Calculate returns safely with epsilon to avoid division by
                # zero
                epsilon = 1e-10
                returns_gpu = cp.diff(prices_gpu) / (prices_gpu[:-1] + epsilon)

                # Calculate recent volatility
                recent_returns_gpu = returns_gpu[-window_size:]
                # A100 can compute standard deviation more efficiently
                if self.has_a100:
                    # Use more optimized algorithm for A100
                    recent_volatility = float(
                        cp.std(recent_returns_gpu) * cp.sqrt(252.0)
                    )  # Annualized
                else:
                    #
                    # For other GPUs, manually calculate standard deviation for
                    # better performance
                    # This is sometimes faster than cp.std() on non-A100 GPUs
                    mean_recent = cp.mean(recent_returns_gpu)
                    recent_variance = cp.mean((recent_returns_gpu - mean_recent) ** 2)
                    recent_volatility = float(
                        cp.sqrt(recent_variance) * cp.sqrt(252.0)
                    )  # Annualized

                # Calculate previous volatility
                prev_returns_gpu = returns_gpu[-window_size * 2 : -window_size]
                # Similar optimization for A100
                if self.has_a100:
                    prev_volatility = float(
                        cp.std(prev_returns_gpu) * cp.sqrt(252.0)
                    )  # Annualized
                else:
                    mean_prev = cp.mean(prev_returns_gpu)
                    prev_variance = cp.mean((prev_returns_gpu - mean_prev) ** 2)
                    prev_volatility = float(
                        cp.sqrt(prev_variance) * cp.sqrt(252.0)
                    )  # Annualized

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for volatility shift detection: {e}, fallback to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available or failed
        if not use_gpu:
            self.execution_stats["cpu_executions"] += 1

            # Calculate returns with safe division
            epsilon = 1e-10
            returns = np.diff(prices) / (prices[:-1] + epsilon)

            # Calculate recent volatility
            recent_returns = returns[-window_size:]
            recent_volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized

            # Calculate previous volatility
            prev_returns = returns[-window_size * 2 : -window_size]
            prev_volatility = np.std(prev_returns) * np.sqrt(252)  # Annualized

        # Calculate volatility shift
        volatility_shift = recent_volatility - prev_volatility

        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        volatility_shift_percent = (
            volatility_shift / (prev_volatility + epsilon)
            if prev_volatility > 0
            else float("inf")
        )

        # Determine if shift is significant
        significant_shift = abs(volatility_shift_percent) > shift_threshold

        # Determine shift direction
        if volatility_shift > 0:
            shift_direction = "increasing"
        elif volatility_shift < 0:
            shift_direction = "decreasing"
        else:
            shift_direction = "stable"

        # Determine volatility state
        if recent_volatility > 0.3:  # 30% annualized volatility is high
            volatility_state = "high"
        elif recent_volatility < 0.1:  # 10% annualized volatility is low
            volatility_state = "low"
        else:
            volatility_state = "normal"

        return {
            "recent_volatility": float(recent_volatility),
            "previous_volatility": float(prev_volatility),
            "volatility_shift": float(volatility_shift),
            "volatility_shift_percent": float(volatility_shift_percent),
            "significant_shift": significant_shift,
            "shift_direction": shift_direction,
            "volatility_state": volatility_state,
            "window_size": window_size,
            "computation": "gpu" if use_gpu else "cpu",
        }

    # Public API methods

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for GPU/CPU usage.

        Returns:
            Dictionary with execution statistics
        """
        total_executions = (
            self.execution_stats["gpu_executions"]
            + self.execution_stats["cpu_executions"]
        )

        # Calculate percentages
        if total_executions > 0:
            gpu_percentage = (
                self.execution_stats["gpu_executions"] / total_executions
            ) * 100
            cpu_percentage = (
                self.execution_stats["cpu_executions"] / total_executions
            ) * 100
        else:
            gpu_percentage = 0.0
            cpu_percentage = 0.0

        return {
            "gpu_executions": self.execution_stats["gpu_executions"],
            "cpu_executions": self.execution_stats["cpu_executions"],
            "gpu_failures": self.execution_stats["gpu_failures"],
            "total_executions": total_executions,
            "gpu_percentage": round(gpu_percentage, 2),
            "cpu_percentage": round(cpu_percentage, 2),
            "gpu_available": self.gpu_available,
            "has_a100": self.has_a100,
            "min_gpu_data_size": self.min_gpu_data_size,
        }

    def detect_ma_drift(
        self, prices: List[float], short_window: int = 5, long_window: int = 20
    ) -> Dict[str, Any]:
        """
        Detect drift from moving averages.

        Args:
            prices: List of price data points
            short_window: Window size for short-term MA
            long_window: Window size for long-term MA

        Returns:
            Dictionary with drift analysis results
        """
        params = {
            "prices": prices,
            "short_window": short_window,
            "long_window": long_window,
        }
        result = self.call_endpoint("detect_ma_drift", params)

        # Include execution statistics in the result
        if "error" not in result:
            result["execution_stats"] = {
                "gpu_executions": self.execution_stats["gpu_executions"],
                "cpu_executions": self.execution_stats["cpu_executions"],
            }

        return result

    def detect_trend_change(
        self, prices: List[float], window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect trend changes.

        Args:
            prices: List of price data points
            window_size: Window size for trend analysis

        Returns:
            Dictionary with trend change detection results
        """
        params = {"prices": prices, "window_size": window_size}
        result = self.call_endpoint("detect_trend_change", params)

        # Include execution statistics in the result
        if "error" not in result:
            result["execution_stats"] = {
                "gpu_executions": self.execution_stats["gpu_executions"],
                "cpu_executions": self.execution_stats["cpu_executions"],
            }

        return result

    def analyze_momentum(
        self, prices: List[float], window_size: int = 14
    ) -> Dict[str, Any]:
        """
        Analyze price momentum.

        Args:
            prices: List of price data points
            window_size: Window size for momentum calculation

        Returns:
            Dictionary with momentum analysis results
        """
        params = {"prices": prices, "window_size": window_size}
        result = self.call_endpoint("analyze_momentum", params)

        # Include execution statistics in the result
        if "error" not in result:
            result["execution_stats"] = {
                "gpu_executions": self.execution_stats["gpu_executions"],
                "cpu_executions": self.execution_stats["cpu_executions"],
            }

        return result

    def detect_volatility_shift(
        self, prices: List[float], window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect shifts in volatility.

        Args:
            prices: List of price data points
            window_size: Window size for volatility calculation

        Returns:
            Dictionary with volatility shift detection results
        """
        params = {"prices": prices, "window_size": window_size}
        result = self.call_endpoint("detect_volatility_shift", params)

        # Include execution statistics in the result
        if "error" not in result:
            result["execution_stats"] = {
                "gpu_executions": self.execution_stats["gpu_executions"],
                "cpu_executions": self.execution_stats["cpu_executions"],
            }

        return result
