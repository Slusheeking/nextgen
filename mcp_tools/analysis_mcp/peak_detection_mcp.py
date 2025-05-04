"""
Peak Detection MCP Server

This module implements a Model Context Protocol (MCP) server for detecting price peaks,
valleys, support/resistance levels, and breakout patterns in financial data.
"""

import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from scipy import signal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load configuration
CONFIG_PATH = os.path.join("config", "analysis_mcp", "peak_detection_config.json")
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"Warning: Could not load config from {CONFIG_PATH}: {e}")
    CONFIG = {}

# GPU acceleration imports
try:
    import cupy as cp
    from cupyx.scipy import signal as cusignal

    HAVE_GPU = True
except ImportError:
    HAVE_GPU = False
    cp = np  # Fallback to numpy if CUDA not available
    cusignal = signal  # Fallback to scipy.signal if CUDA not available

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring.system_monitor import MonitoringManager

# Set up monitoring (Prometheus + Python logging only)
monitor = MonitoringManager(service_name="analysis_mcp-peak-detection-mcp")
metrics = monitor.metrics


class PeakDetectionMCP(BaseMCPServer):
    """
    MCP server for price peak/valley detection and pattern recognition.

    This server provides tools for identifying significant price levels,
    breakout patterns, and support/resistance zones.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Peak Detection MCP server.

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(name="peak_detection_mcp", config=config)

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
        
        # Configure GPU usage
        self.use_gpu = file_config.get("use_gpu", HAVE_GPU)
        self.min_data_size_for_gpu = file_config.get("min_data_size_for_gpu", 1000)

        if self.use_gpu and HAVE_GPU:
            gpu_device = file_config.get("gpu_device", 0)
            try:
                # Set the active CUDA device
                cp.cuda.Device(gpu_device).use()
                self.logger.info(f"Using GPU device {gpu_device} for peak detection")

                # Check if device is A100 and set optimizations
                device_attributes = cp.cuda.runtime.getDeviceProperties(gpu_device)
                if "A100" in device_attributes["name"]:
                    self.logger.info(f"Detected A100 GPU: {device_attributes['name']}")
                    # A100-specific optimizations
                    self.min_data_size_for_gpu = self.config.get(
                        "min_data_size_for_gpu", 500
                    )  # Lower threshold for A100
                    # Could enable more specific A100 optimizations here
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
                self.logger.info("Using CPU for peak detection")
            self.use_gpu = False

        self.logger.info("PeakDetectionMCP initialized")

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "detect_peaks": {
                "description": "Detect price peaks and valleys",
                "category": "pattern_recognition",
                "required_params": ["prices"],
                "optional_params": ["window_size", "prominence", "width", "distance"],
                "handler": self._handle_detect_peaks,
            },
            "detect_support_resistance": {
                "description": "Identify support and resistance levels",
                "category": "pattern_recognition",
                "required_params": ["prices", "volumes"],
                "optional_params": ["window_size", "num_levels", "price_threshold"],
                "handler": self._handle_detect_support_resistance,
            },
            "detect_breakout": {
                "description": "Detect breakout patterns",
                "category": "pattern_recognition",
                "required_params": ["prices", "volumes"],
                "optional_params": [
                    "lookback_period",
                    "volume_factor",
                    "price_threshold",
                ],
                "handler": self._handle_detect_breakout,
            },
            "detect_consolidation": {
                "description": "Detect price consolidation patterns",
                "category": "pattern_recognition",
                "required_params": ["prices"],
                "optional_params": ["window_size", "volatility_threshold"],
                "handler": self._handle_detect_consolidation,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Peak Detection MCP."""
        self.register_tool(self.detect_peaks)
        self.register_tool(self.detect_support_resistance)
        self.register_tool(self.detect_breakout)
        self.register_tool(self.detect_consolidation)

    # Handler methods for specific endpoints

    def _handle_detect_peaks(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_peaks endpoint."""
        prices = params.get("prices", [])
        
        # Get default values from config
        peak_detection_config = CONFIG.get("peak_detection", {})
        default_window_size = peak_detection_config.get("default_window_size", 5)
        default_prominence = peak_detection_config.get("default_prominence", 0.5)
        default_width = peak_detection_config.get("default_width", 1)
        default_distance = peak_detection_config.get("default_distance", 3)
        
        # Use provided values or fall back to config defaults
        window_size = params.get("window_size", default_window_size)
        prominence = params.get("prominence", default_prominence)
        width = params.get("width", default_width)
        distance = params.get("distance", default_distance)

        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)

            # Find peaks
            peaks = self._find_peaks(prices, window_size, prominence, width, distance)

            # Find valleys (peaks in negative prices)
            valleys = self._find_peaks(
                -prices, window_size, prominence, width, distance
            )

            return {
                "peaks": peaks,
                "valleys": valleys,
                "count_peaks": len(peaks),
                "count_valleys": len(valleys),
            }
        except Exception as e:
            self.logger.error(f"Error detecting peaks: {e}")
            return {"error": f"Failed to detect peaks: {str(e)}"}

    def _handle_detect_support_resistance(
        self, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle detect_support_resistance endpoint."""
        prices = params.get("prices", [])
        volumes = params.get("volumes", [])
        
        # Get default values from config
        support_resistance_config = CONFIG.get("support_resistance", {})
        default_window_size = support_resistance_config.get("default_window_size", 10)
        default_num_levels = support_resistance_config.get("default_num_levels", 3)
        default_price_threshold = support_resistance_config.get("default_price_threshold", 0.01)
        
        # Use provided values or fall back to config defaults
        window_size = params.get("window_size", default_window_size)
        num_levels = params.get("num_levels", default_num_levels)
        price_threshold = params.get("price_threshold", default_price_threshold)

        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy arrays if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            if not isinstance(volumes, np.ndarray) and volumes:
                volumes = np.array(volumes)

            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(
                prices, volumes, window_size, num_levels, price_threshold
            )

            return {
                "support_levels": support_levels,
                "resistance_levels": resistance_levels,
                "current_price": prices[-1] if len(prices) > 0 else None,
            }
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {e}")
            return {"error": f"Failed to detect support/resistance: {str(e)}"}

    def _handle_detect_breakout(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_breakout endpoint."""
        prices = params.get("prices", [])
        volumes = params.get("volumes", [])
        
        # Get default values from config
        breakout_config = CONFIG.get("breakout", {})
        default_lookback_period = breakout_config.get("default_lookback_period", 20)
        default_volume_factor = breakout_config.get("default_volume_factor", 1.5)
        default_price_threshold = breakout_config.get("default_price_threshold", 0.02)
        
        # Use provided values or fall back to config defaults
        lookback_period = params.get("lookback_period", default_lookback_period)
        volume_factor = params.get("volume_factor", default_volume_factor)
        price_threshold = params.get("price_threshold", default_price_threshold)

        if not prices or len(prices) < lookback_period:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy arrays if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            if not isinstance(volumes, np.ndarray) and volumes:
                volumes = np.array(volumes)

            # Detect breakout
            breakout_result = self._detect_breakout_pattern(
                prices, volumes, lookback_period, volume_factor, price_threshold
            )

            return breakout_result
        except Exception as e:
            self.logger.error(f"Error detecting breakout: {e}")
            return {"error": f"Failed to detect breakout: {str(e)}"}

    def _handle_detect_consolidation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_consolidation endpoint."""
        prices = params.get("prices", [])
        
        # Get default values from config
        consolidation_config = CONFIG.get("consolidation", {})
        default_window_size = consolidation_config.get("default_window_size", 10)
        default_volatility_threshold = consolidation_config.get("default_volatility_threshold", 0.01)
        
        # Use provided values or fall back to config defaults
        window_size = params.get("window_size", default_window_size)
        volatility_threshold = params.get("volatility_threshold", default_volatility_threshold)

        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}

        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)

            # Detect consolidation
            consolidation_result = self._detect_consolidation_pattern(
                prices, window_size, volatility_threshold
            )

            return consolidation_result
        except Exception as e:
            self.logger.error(f"Error detecting consolidation: {e}")
            return {"error": f"Failed to detect consolidation: {str(e)}"}

    # Core analysis methods

    def _find_peaks(
        self,
        prices: np.ndarray,
        window_size: int = 5,
        prominence: float = 0.5,
        width: int = 1,
        distance: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find peaks in price data.

        Args:
            prices: Array of price data
            window_size: Window size for peak detection
            prominence: Minimum prominence of peaks
            width: Minimum width of peaks
            distance: Minimum distance between peaks

        Returns:
            List of peak dictionaries with index and price
        """
        # Use GPU implementation if available and data is large enough
        if self.use_gpu and HAVE_GPU and len(prices) >= self.min_data_size_for_gpu:
            try:
                return self._find_peaks_gpu(
                    prices, window_size, prominence, width, distance
                )
            except Exception as e:
                self.logger.warning(
                    f"GPU peak detection failed: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        #
        # Use scipy.signal.find_peaks for larger datasets for better
        # performance
        if len(prices) > window_size * 10:
            try:
                # Calculate average peak height for proper prominence scaling
                avg_price = np.mean(prices)
                scaled_prominence = prominence * avg_price

                # Use scipy's find_peaks for better performance
                peak_indices, properties = signal.find_peaks(
                    prices, prominence=scaled_prominence, width=width, distance=distance
                )

                peaks = []
                for i, idx in enumerate(peak_indices):
                    # Skip peaks too close to edges
                    if idx < window_size or idx >= len(prices) - window_size:
                        continue

                    # Extract prominence from properties
                    peak_prominence = properties["prominences"][i]

                    peaks.append(
                        {
                            "index": int(idx),
                            "price": float(prices[idx]),
                            "prominence": float(peak_prominence),
                        }
                    )

                return peaks

            except Exception as e:
                self.logger.warning(
                    f"SciPy peak detection failed: {e}, falling back to simple algorithm"
                )
                # Fall back to simple algorithm

        # Simple peak detection algorithm (traditional implementation)
        peaks = []
        for i in range(window_size, len(prices) - window_size):
            is_peak = True
            for j in range(1, window_size + 1):
                if prices[i] <= prices[i - j] or prices[i] <= prices[i + j]:
                    is_peak = False
                    break

            if is_peak:
                # Check if it's far enough from previous peak
                if not peaks or i - peaks[-1]["index"] >= distance:
                    # Check prominence
                    left_min = min(prices[i - window_size : i])
                    right_min = min(prices[i + 1 : i + window_size + 1])
                    base = max(left_min, right_min)
                    peak_prominence = prices[i] - base

                    if peak_prominence >= prominence:
                        peaks.append(
                            {
                                "index": i,
                                "price": float(prices[i]),
                                "prominence": float(peak_prominence),
                            }
                        )

        return peaks

    def _find_peaks_gpu(
        self,
        prices: np.ndarray,
        window_size: int = 5,
        prominence: float = 0.5,
        width: int = 1,
        distance: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        GPU-accelerated peak detection using CuPy and cuSignal.

        Args:
            prices: Array of price data
            window_size: Window size for peak detection
            prominence: Minimum prominence of peaks
            width: Minimum width of peaks
            distance: Minimum distance between peaks

        Returns:
            List of peak dictionaries with index and price
        """
        # Transfer data to GPU
        prices_gpu = cp.asarray(prices)

        # Calculate average peak height for proper prominence scaling
        avg_price = float(cp.mean(prices_gpu))
        scaled_prominence = prominence * avg_price

        # Use cusignal's find_peaks (similar to scipy)
        peak_indices, properties = cusignal.find_peaks(
            prices_gpu, prominence=scaled_prominence, width=width, distance=distance
        )

        # Transfer results back to CPU if needed
        if isinstance(peak_indices, cp.ndarray):
            peak_indices = cp.asnumpy(peak_indices)

        # Process prominences
        prominences = properties.get("prominences", None)
        if prominences is not None and isinstance(prominences, cp.ndarray):
            prominences = cp.asnumpy(prominences)

        # Create result list
        peaks = []
        for i, idx in enumerate(peak_indices):
            # Skip peaks too close to edges
            if idx < window_size or idx >= len(prices) - window_size:
                continue

            # Extract prominence
            peak_prominence = float(prominences[i]) if prominences is not None else 0.0

            peaks.append(
                {
                    "index": int(idx),
                    "price": float(prices[idx]),
                    "prominence": peak_prominence,
                    "computed_on": "gpu",
                }
            )

        return peaks

    def _find_support_resistance(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None,
        window_size: int = 10,
        num_levels: int = 3,
        price_threshold: float = 0.01,
    ) -> Tuple[List[float], List[float]]:
        """
        Find support and resistance levels.

        Args:
            prices: Array of price data
            volumes: Array of volume data (optional)
            window_size: Window size for level detection
            num_levels: Number of levels to return
            price_threshold: Minimum price difference between levels

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        # Find peaks and valleys
        peaks = self._find_peaks(prices, window_size)
        valleys = self._find_peaks(-prices, window_size)

        # Extract prices
        peak_prices = [p["price"] for p in peaks]
        valley_prices = [prices[v["index"]] for v in valleys]

        # Group similar price levels
        resistance_levels = self._cluster_price_levels(peak_prices, price_threshold)
        support_levels = self._cluster_price_levels(valley_prices, price_threshold)

        # Sort by strength (frequency) and limit to num_levels
        resistance_levels = sorted(
            resistance_levels, key=lambda x: x["strength"], reverse=True
        )[:num_levels]
        support_levels = sorted(
            support_levels, key=lambda x: x["strength"], reverse=True
        )[:num_levels]

        # Extract just the price levels
        resistance_prices = [level["price"] for level in resistance_levels]
        support_prices = [level["price"] for level in support_levels]

        return support_prices, resistance_prices

    def _cluster_price_levels(
        self, prices: List[float], threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Cluster similar price levels.

        Args:
            prices: List of price points
            threshold: Relative threshold for clustering

        Returns:
            List of clustered price levels with strength
        """
        if not prices:
            return []

        # Sort prices
        sorted_prices = sorted(prices)

        # Initialize clusters
        clusters = []
        current_cluster = [sorted_prices[0]]

        # Cluster similar prices
        for i in range(1, len(sorted_prices)):
            if (sorted_prices[i] - current_cluster[0]) / current_cluster[
                0
            ] <= threshold:
                current_cluster.append(sorted_prices[i])
            else:
                # Save current cluster and start a new one
                avg_price = sum(current_cluster) / len(current_cluster)
                clusters.append({"price": avg_price, "strength": len(current_cluster)})
                current_cluster = [sorted_prices[i]]

        # Add the last cluster
        if current_cluster:
            avg_price = sum(current_cluster) / len(current_cluster)
            clusters.append({"price": avg_price, "strength": len(current_cluster)})

        return clusters

    def _detect_breakout_pattern(
        self,
        prices: np.ndarray,
        volumes: np.ndarray = None,
        lookback_period: int = 20,
        volume_factor: float = 1.5,
        price_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Detect breakout patterns.

        Args:
            prices: Array of price data
            volumes: Array of volume data (optional)
            lookback_period: Period to look back for range
            volume_factor: Volume increase factor for confirmation
            price_threshold: Price movement threshold

        Returns:
            Breakout detection results
        """
        if len(prices) < lookback_period:
            return {"breakout_detected": False, "reason": "Insufficient data"}

        # Use GPU implementation if available, data is large enough,
        # and volumes data is available in the correct format
        if (
            self.use_gpu
            and HAVE_GPU
            and len(prices) >= self.min_data_size_for_gpu
            and volumes is not None
            and len(volumes) >= lookback_period
        ):
            try:
                return self._detect_breakout_pattern_gpu(
                    prices, volumes, lookback_period, volume_factor, price_threshold
                )
            except Exception as e:
                self.logger.warning(
                    f"GPU breakout detection failed: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # Get recent price range
        recent_prices = prices[-lookback_period:-1]
        recent_high = np.max(recent_prices)
        recent_low = np.min(recent_prices)
        recent_range = recent_high - recent_low

        # Current price
        current_price = prices[-1]

        # Check volume if available
        volume_confirmation = False
        if volumes is not None and len(volumes) >= lookback_period:
            recent_avg_volume = np.mean(volumes[-lookback_period:-1])
            current_volume = volumes[-1]
            volume_confirmation = current_volume > (recent_avg_volume * volume_factor)

        # Check for breakout
        breakout_up = (
            current_price > recent_high
            and (current_price - recent_high) / recent_high > price_threshold
        )
        breakout_down = (
            current_price < recent_low
            and (recent_low - current_price) / recent_low > price_threshold
        )

        if breakout_up:
            return {
                "breakout_detected": True,
                "direction": "up",
                "price": float(current_price),
                "breakout_level": float(recent_high),
                "breakout_percentage": float(
                    (current_price - recent_high) / recent_high * 100
                ),
                "volume_confirmation": volume_confirmation,
                "computed_on": "cpu",
            }
        elif breakout_down:
            return {
                "breakout_detected": True,
                "direction": "down",
                "price": float(current_price),
                "breakout_level": float(recent_low),
                "breakout_percentage": float(
                    (recent_low - current_price) / recent_low * 100
                ),
                "volume_confirmation": volume_confirmation,
                "computed_on": "cpu",
            }
        else:
            return {
                "breakout_detected": False,
                "price": float(current_price),
                "recent_high": float(recent_high),
                "recent_low": float(recent_low),
                "computed_on": "cpu",
            }

    def _detect_breakout_pattern_gpu(
        self,
        prices: np.ndarray,
        volumes: np.ndarray,
        lookback_period: int = 20,
        volume_factor: float = 1.5,
        price_threshold: float = 0.02,
    ) -> Dict[str, Any]:
        """
        GPU-accelerated breakout pattern detection.

        Args:
            prices: Array of price data
            volumes: Array of volume data
            lookback_period: Period to look back for range
            volume_factor: Volume increase factor for confirmation
            price_threshold: Price movement threshold

        Returns:
            Breakout detection results
        """
        # Transfer data to GPU
        prices_gpu = cp.asarray(prices)
        volumes_gpu = cp.asarray(volumes)

        # Get recent price range
        recent_prices_gpu = prices_gpu[-lookback_period:-1]
        recent_high = float(cp.max(recent_prices_gpu).get())
        recent_low = float(cp.min(recent_prices_gpu).get())

        # Current price
        current_price = float(prices_gpu[-1].get())

        # Check volume
        recent_volume_gpu = volumes_gpu[-lookback_period:-1]
        recent_avg_volume = float(cp.mean(recent_volume_gpu).get())
        current_volume = float(volumes_gpu[-1].get())
        volume_confirmation = current_volume > (recent_avg_volume * volume_factor)

        # Check for breakout (CPU calculations for simple comparisons)
        breakout_up = (
            current_price > recent_high
            and (current_price - recent_high) / recent_high > price_threshold
        )
        breakout_down = (
            current_price < recent_low
            and (recent_low - current_price) / recent_low > price_threshold
        )

        if breakout_up:
            return {
                "breakout_detected": True,
                "direction": "up",
                "price": current_price,
                "breakout_level": recent_high,
                "breakout_percentage": float(
                    (current_price - recent_high) / recent_high * 100
                ),
                "volume_confirmation": volume_confirmation,
                "computed_on": "gpu",
            }
        elif breakout_down:
            return {
                "breakout_detected": True,
                "direction": "down",
                "price": current_price,
                "breakout_level": recent_low,
                "breakout_percentage": float(
                    (recent_low - current_price) / recent_low * 100
                ),
                "volume_confirmation": volume_confirmation,
                "computed_on": "gpu",
            }
        else:
            return {
                "breakout_detected": False,
                "price": current_price,
                "recent_high": recent_high,
                "recent_low": recent_low,
                "computed_on": "gpu",
            }

    def _detect_consolidation_pattern(
        self,
        prices: np.ndarray,
        window_size: int = 10,
        volatility_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Detect price consolidation patterns.

        Args:
            prices: Array of price data
            window_size: Window size for volatility calculation
            volatility_threshold: Maximum volatility for consolidation

        Returns:
            Consolidation detection results
        """
        if len(prices) < window_size * 2:
            return {"consolidation_detected": False, "reason": "Insufficient data"}

        # Use GPU implementation if available and data is large enough
        if self.use_gpu and HAVE_GPU and len(prices) >= self.min_data_size_for_gpu:
            try:
                return self._detect_consolidation_pattern_gpu(
                    prices, window_size, volatility_threshold
                )
            except Exception as e:
                self.logger.warning(
                    f"GPU consolidation detection failed: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # Calculate recent volatility
        recent_prices = prices[-window_size:]
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        recent_volatility = np.std(recent_returns)

        # Calculate previous volatility
        previous_prices = prices[-window_size * 2 : -window_size]
        previous_returns = np.diff(previous_prices) / previous_prices[:-1]
        previous_volatility = np.std(previous_returns)

        # Check for consolidation
        consolidation_detected = recent_volatility < volatility_threshold
        volatility_reduction = (
            previous_volatility > 0 and recent_volatility < previous_volatility
        )

        return {
            "consolidation_detected": consolidation_detected,
            "recent_volatility": float(recent_volatility),
            "previous_volatility": float(previous_volatility),
            "volatility_reduction": volatility_reduction,
            "price_range": {
                "min": float(np.min(recent_prices)),
                "max": float(np.max(recent_prices)),
                "current": float(prices[-1]),
            },
            "computed_on": "cpu",
        }

    def _detect_consolidation_pattern_gpu(
        self,
        prices: np.ndarray,
        window_size: int = 10,
        volatility_threshold: float = 0.01,
    ) -> Dict[str, Any]:
        """
        GPU-accelerated consolidation pattern detection.

        Args:
            prices: Array of price data
            window_size: Window size for volatility calculation
            volatility_threshold: Maximum volatility for consolidation

        Returns:
            Consolidation detection results
        """
        # Transfer data to GPU
        prices_gpu = cp.asarray(prices)

        # Calculate recent volatility
        recent_prices_gpu = prices_gpu[-window_size:]
        # Use safe division (avoiding division by zero)
        recent_returns_gpu = cp.diff(recent_prices_gpu) / cp.maximum(
            recent_prices_gpu[:-1], 1e-8
        )
        recent_volatility = float(cp.std(recent_returns_gpu).get())

        # Calculate previous volatility
        previous_prices_gpu = prices_gpu[-window_size * 2 : -window_size]
        # Use safe division
        previous_returns_gpu = cp.diff(previous_prices_gpu) / cp.maximum(
            previous_prices_gpu[:-1], 1e-8
        )
        previous_volatility = float(cp.std(previous_returns_gpu).get())

        # CPU calculations for the rest (simpler operations)
        # Check for consolidation
        consolidation_detected = recent_volatility < volatility_threshold
        volatility_reduction = (
            previous_volatility > 0 and recent_volatility < previous_volatility
        )

        # Get min, max, and current prices
        min_price = float(cp.min(recent_prices_gpu).get())
        max_price = float(cp.max(recent_prices_gpu).get())
        current_price = float(prices_gpu[-1].get())

        return {
            "consolidation_detected": consolidation_detected,
            "recent_volatility": recent_volatility,
            "previous_volatility": previous_volatility,
            "volatility_reduction": volatility_reduction,
            "price_range": {
                "min": min_price,
                "max": max_price,
                "current": current_price,
            },
            "computed_on": "gpu",
        }

    # Public API methods
    
    def call_endpoint(self, endpoint_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an endpoint by name with the given parameters.
        
        Args:
            endpoint_name: Name of the endpoint to call
            params: Parameters to pass to the endpoint
            
        Returns:
            Result from the endpoint handler
        """
        if endpoint_name not in self.endpoints:
            return {"error": f"Unknown endpoint: {endpoint_name}"}
            
        endpoint = self.endpoints[endpoint_name]
        handler = endpoint["handler"]
        
        # Validate required parameters
        required_params = endpoint.get("required_params", [])
        for param in required_params:
            if param not in params:
                return {"error": f"Missing required parameter: {param}"}
                
        # Call the handler
        return handler(params)

    def detect_peaks(
        self, prices: List[float], window_size: int = 5, prominence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect price peaks and valleys.

        Args:
            prices: List of price data points
            window_size: Window size for peak detection
            prominence: Minimum prominence of peaks

        Returns:
            Dictionary with peaks and valleys
        """
        params = {
            "prices": prices,
            "window_size": window_size,
            "prominence": prominence,
        }
        return self.call_endpoint("detect_peaks", params)

    def detect_support_resistance(
        self, prices: List[float], volumes: List[float] = None, num_levels: int = 3
    ) -> Dict[str, Any]:
        """
        Identify support and resistance levels.

        Args:
            prices: List of price data points
            volumes: List of volume data points (optional)
            num_levels: Number of levels to return

        Returns:
            Dictionary with support and resistance levels
        """
        params = {
            "prices": prices,
            "volumes": volumes if volumes else [],
            "num_levels": num_levels,
        }
        return self.call_endpoint("detect_support_resistance", params)

    def detect_breakout(
        self,
        prices: List[float],
        volumes: List[float] = None,
        lookback_period: int = 20,
    ) -> Dict[str, Any]:
        """
        Detect breakout patterns.

        Args:
            prices: List of price data points
            volumes: List of volume data points (optional)
            lookback_period: Period to look back for range

        Returns:
            Dictionary with breakout detection results
        """
        params = {
            "prices": prices,
            "volumes": volumes if volumes else [],
            "lookback_period": lookback_period,
        }
        return self.call_endpoint("detect_breakout", params)

    def detect_consolidation(
        self, prices: List[float], window_size: int = 10
    ) -> Dict[str, Any]:
        """
        Detect price consolidation patterns.

        Args:
            prices: List of price data points
            window_size: Window size for volatility calculation

        Returns:
            Dictionary with consolidation detection results
        """
        params = {"prices": prices, "window_size": window_size}
        return self.call_endpoint("detect_consolidation", params)
