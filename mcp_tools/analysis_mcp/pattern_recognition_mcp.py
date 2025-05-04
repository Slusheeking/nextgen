"""
Pattern Recognition MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
chart pattern recognition capabilities for financial market data.
"""

import os
import json
import numpy as np
import pandas as pd
import cudf
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# For computer vision and machine learning capabilities
try:
    HAVE_ML_LIBS = True
except ImportError:
    HAVE_ML_LIBS = False

# For GPU acceleration if available
try:
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False

from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring utilities
from monitoring import setup_monitoring

# Set up monitoring
monitor, metrics = setup_monitoring(
    service_name="analysis-mcp-pattern-recognition",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class PatternRecognitionMCP(BaseMCPServer):
    """
    MCP server for recognizing chart patterns in financial market data.

    This tool identifies technical chart patterns such as head and shoulders,
    double tops/bottoms, triangles, flags, wedges, etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Pattern Recognition MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - model_path: Path to pattern recognition models
                - use_ml: Whether to use ML-based pattern recognition
                - use_gpu: Whether to use GPU acceleration if available
                - confidence_threshold: Minimum confidence for pattern detection
                - custom_patterns_path: Path to custom pattern definitions
                - cache_dir: Directory for caching intermediate results
        """
        super().__init__(name="pattern_recognition_mcp", config=config)

        # Set default configurations
        self.model_path = self.config.get(
            "model_path", os.path.join(os.path.dirname(__file__), "models")
        )
        self.use_ml = self.config.get("use_ml", HAVE_ML_LIBS)
        self.use_gpu = self.config.get("use_gpu", HAVE_CUDA)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.65)
        self.custom_patterns_path = self.config.get(
            "custom_patterns_path",
            os.path.join(os.path.dirname(__file__), "data/custom_patterns.json"),
        )
        self.cache_dir = self.config.get("cache_dir", "./pattern_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize models and pattern definitions
        self._load_pattern_definitions()
        self._load_ml_models()

        # Register tools
        self._register_tools()

    def _load_pattern_definitions(self):
        """Load pattern definitions from file."""
        # Define default patterns with their algorithmic definitions
        self.patterns = {
            "double_top": {
                "description": "Two distinct peaks at approximately the same price level",
                "min_points": 20,  # Minimum number of points needed to detect
                "lookback": 60,  # Days to look back for the pattern
                "parameters": {
                    "height_threshold": 0.03,  # Max difference between peaks (%)
                    "valley_depth": 0.05,  # Min depth of valley between peaks (%)
                    "min_gap": 5,  # Min bars between peaks
                },
            },
            "double_bottom": {
                "description": "Two distinct bottoms at approximately the same price level",
                "min_points": 20,
                "lookback": 60,
                "parameters": {
                    "height_threshold": 0.03,
                    "peak_height": 0.05,
                    "min_gap": 5,
                },
            },
            "head_and_shoulders": {
                "description": "Three peaks with the middle one higher, signaling potential reversal",
                "min_points": 30,
                "lookback": 100,
                "parameters": {
                    "shoulder_height_diff": 0.05,  # Max difference between shoulders
                    "head_height_min": 0.03,  # Min height of head above shoulders
                    "neckline_slope_max": 0.2,  # Max slope of neckline
                },
            },
            "ascending_triangle": {
                "description": "Flat upper resistance with rising lower support line",
                "min_points": 15,
                "lookback": 60,
                "parameters": {
                    "resistance_variation": 0.02,  # Max variation in resistance line
                    "support_min_slope": 0.002,  # Min slope of support line
                    "min_touches": 3,  # Min number of touches on resistance/support
                },
            },
            "descending_triangle": {
                "description": "Flat lower support with falling upper resistance line",
                "min_points": 15,
                "lookback": 60,
                "parameters": {
                    "support_variation": 0.02,
                    "resistance_min_slope": -0.002,
                    "min_touches": 3,
                },
            },
            "symmetrical_triangle": {
                "description": "Converging trend lines with similar slopes",
                "min_points": 15,
                "lookback": 60,
                "parameters": {
                    "support_min_slope": 0.002,
                    "resistance_max_slope": -0.002,
                    "min_touches": 2,
                    "convergence_min": 0.02,
                },
            },
            "flag": {
                "description": "Small channel counter to the prevailing trend",
                "min_points": 10,
                "lookback": 30,
                "parameters": {
                    "trend_min_slope": 0.01,
                    "flag_max_slope": 0.005,
                    "flag_duration_min": 5,
                    "flag_duration_max": 20,
                },
            },
            "pennant": {
                "description": "Small symmetrical triangle after a strong trend",
                "min_points": 15,
                "lookback": 40,
                "parameters": {
                    "trend_min_slope": 0.01,
                    "convergence_min": 0.02,
                    "pennant_duration_min": 5,
                    "pennant_duration_max": 20,
                },
            },
            "cup_and_handle": {
                "description": "U-shaped pattern followed by a small downward drift",
                "min_points": 40,
                "lookback": 150,
                "parameters": {
                    "cup_depth": 0.1,
                    "cup_duration_min": 20,
                    "cup_duration_max": 100,
                    "handle_max_depth": 0.05,
                    "handle_duration_max": 20,
                },
            },
            "wedge": {
                "description": "Converging trend lines moving in the same direction",
                "min_points": 20,
                "lookback": 60,
                "parameters": {
                    "upper_slope": -0.003,
                    "lower_slope": -0.001,
                    "convergence_min": 0.01,
                    "min_touches": 2,
                },
            },
        }

        # Load custom patterns if available
        if os.path.exists(self.custom_patterns_path):
            try:
                with open(self.custom_patterns_path, "r") as f:
                    custom_patterns = json.load(f)

                #
                # Merge with default patterns, overriding defaults if
                # duplicates
                self.patterns.update(custom_patterns)
                self.logger.info(
                    f"Loaded {len(custom_patterns)} custom pattern definitions"
                )
            except Exception as e:
                self.logger.error(f"Error loading custom pattern definitions: {e}")
        else:
            self.logger.info(
                f"No custom pattern definitions found at {self.custom_patterns_path}"
            )

        self.logger.info(f"Initialized with {len(self.patterns)} pattern definitions")

    def _load_ml_models(self):
        """Load ML models for pattern recognition if available."""
        self.ml_models = {}

        if not self.use_ml or not HAVE_ML_LIBS:
            self.logger.info(
                "ML-based pattern recognition disabled or required libraries not available"
            )
            return

        # In a real implementation, we would load pre-trained models here
        # For this example, we'll just set flags indicating availability

        model_types = [
            "cnn_pattern_detector",
            "candlestick_classifier",
            "timeseries_classifier",
        ]

        for model_type in model_types:
            model_path = os.path.join(self.model_path, f"{model_type}.h5")
            if os.path.exists(model_path):
                # In a real implementation, load the actual model
                self.ml_models[model_type] = {"path": model_path, "loaded": True}
                self.logger.info(f"Loaded {model_type} model from {model_path}")
            else:
                self.ml_models[model_type] = {"path": model_path, "loaded": False}
                self.logger.warning(f"Model {model_type} not found at {model_path}")

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "detect_patterns",
            self.detect_patterns,
            "Detect technical chart patterns in price data",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with 'date', 'open', 'high', 'low', 'close', 'volume' fields",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of patterns to detect (default: all patterns)",
                    "required": False,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence score for pattern detection (0.0-1.0)",
                    "required": False,
                },
                "lookback": {
                    "type": "integer",
                    "description": "Number of bars to look back for pattern detection",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "patterns": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "detect_candlestick_patterns",
            self.detect_candlestick_patterns,
            "Detect candlestick patterns in price data",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with 'date', 'open', 'high', 'low', 'close' fields",
                },
                "patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candlestick patterns to detect (default: all patterns)",
                    "required": False,
                },
                "min_confidence": {
                    "type": "number",
                    "description": "Minimum confidence score for pattern detection (0.0-1.0)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "patterns": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "detect_support_resistance",
            self.detect_support_resistance,
            "Detect support and resistance levels in price data",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with 'date', 'high', 'low', 'close' fields",
                },
                "method": {
                    "type": "string",
                    "description": "Method to use for detection: 'peaks', 'fractals', or 'zones'",
                    "required": False,
                },
                "sensitivity": {
                    "type": "number",
                    "description": "Sensitivity for detection (0.0-1.0)",
                    "required": False,
                },
                "lookback": {
                    "type": "integer",
                    "description": "Number of bars to look back",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "levels": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "analyze_pattern",
            self.analyze_pattern,
            "Analyze a specific pattern occurrence in detail",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries",
                },
                "pattern": {"type": "string", "description": "Pattern to analyze"},
                "start_index": {
                    "type": "integer",
                    "description": "Start index of the pattern in the data",
                },
                "end_index": {
                    "type": "integer",
                    "description": "End index of the pattern in the data",
                },
            },
            {
                "type": "object",
                "properties": {
                    "analysis": {"type": "object"},
                    "statistics": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "get_historical_pattern_performance",
            self.get_historical_pattern_performance,
            "Get historical performance statistics for a pattern",
            {
                "pattern": {
                    "type": "string",
                    "description": "Pattern to get statistics for",
                },
                "market": {
                    "type": "string",
                    "description": "Market to get statistics for (e.g., 'us_equities', 'forex', 'crypto')",
                    "required": False,
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe to get statistics for (e.g., 'daily', '4h', '1h')",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "statistics": {"type": "object"},
                    "examples": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

    def _prepare_dataframe(self, price_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert price data list to pandas DataFrame.

        Args:
            price_data: List of price data dictionaries

        Returns:
            Pandas DataFrame with price data
        """
        try:
            # Create DataFrame
            df = pd.DataFrame(price_data)

            # Convert date to datetime if it's not already
            if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df["date"]
            ):
                df["date"] = pd.to_datetime(df["date"])

            # Sort by date
            if "date" in df.columns:
                df = df.sort_values("date")

            # Check required columns
            required_cols = ["open", "high", "low", "close"]
            for col in required_cols:
                if col not in df.columns:
                    self.logger.warning(f"Missing required column: {col}")
                    df[col] = np.nan

            # Convert price and volume columns to numeric
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            # If using gpu and available, convert to cudf DataFrame
            if self.use_gpu and HAVE_CUDA:
                try:
                    df_gpu = cudf.DataFrame.from_pandas(df)
                    self.logger.info("Using GPU acceleration for calculations")
                    return df_gpu
                except Exception as e:
                    self.logger.warning(
                        f"Error converting to GPU DataFrame: {e}. Falling back to CPU."
                    )

            return df

        except Exception as e:
            self.logger.error(f"Error preparing DataFrame: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=["date", "open", "high", "low", "close", "volume"]
            )

    def detect_patterns(
        self,
        price_data: List[Dict[str, Any]],
        patterns: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        lookback: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect technical chart patterns in price data.

        Args:
            price_data: List of price data dictionaries
            patterns: List of patterns to detect (default: all patterns)
            min_confidence: Minimum confidence score for pattern detection
            lookback: Number of bars to look back for pattern detection

        Returns:
            Dictionary with detected patterns and metadata
        """
        start_time = time.time()

        # Use provided parameters or defaults
        min_confidence = min_confidence or self.confidence_threshold
        patterns_to_detect = patterns or list(self.patterns.keys())

        # Prepare DataFrame
        df = self._prepare_dataframe(price_data)

        if len(df) < 10:
            return {
                "patterns": [],
                "error": "Not enough data points for pattern detection",
                "processing_time": time.time() - start_time,
            }

        # Store detected patterns
        detected_patterns = []

        # Try to detect each pattern
        for pattern_name in patterns_to_detect:
            if pattern_name not in self.patterns:
                self.logger.warning(f"Unknown pattern: {pattern_name}")
                continue

            pattern_def = self.patterns[pattern_name]
            min_points = pattern_def.get("min_points", 20)
            pattern_lookback = lookback or pattern_def.get("lookback", 60)
            pattern_params = pattern_def.get("parameters", {})

            # Make sure we have enough data
            if len(df) < min_points:
                continue

            # Limit lookback to available data
            actual_lookback = min(pattern_lookback, len(df))
            lookback_data = df.iloc[-actual_lookback:]

            # Detect patterns based on type
            try:
                # First try ML-based detection if available
                if (
                    self.use_ml
                    and pattern_name in self.ml_models
                    and self.ml_models[pattern_name]["loaded"]
                ):
                    ml_detections = self._detect_pattern_ml(lookback_data, pattern_name)
                    detected_patterns.extend(ml_detections)
                else:
                    # Fall back to algorithmic detection
                    algo_detections = self._detect_pattern_algorithmic(
                        lookback_data, pattern_name, pattern_params
                    )
                    detected_patterns.extend(algo_detections)
            except Exception as e:
                self.logger.error(f"Error detecting {pattern_name}: {e}")

        # Filter by confidence threshold
        filtered_patterns = [
            p for p in detected_patterns if p["confidence"] >= min_confidence
        ]

        # Sort patterns by start date (most recent first)
        filtered_patterns.sort(key=lambda x: x["end_index"], reverse=True)

        return {
            "patterns": filtered_patterns,
            "total_detected": len(filtered_patterns),
            "processing_time": time.time() - start_time,
        }

    def _detect_pattern_algorithmic(
        self, df: pd.DataFrame, pattern_name: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns using algorithmic methods.

        Args:
            df: DataFrame with price data
            pattern_name: Name of pattern to detect
            params: Parameters for pattern detection

        Returns:
            List of detected pattern instances
        """
        detected = []

        # Get closing prices as numpy array for faster processing
        closes = df["close"].values if not self.use_gpu else df["close"].to_numpy()
        highs = df["high"].values if not self.use_gpu else df["high"].to_numpy()
        lows = df["low"].values if not self.use_gpu else df["low"].to_numpy()

        # Different detection logic based on pattern type
        if pattern_name == "double_top":
            detected = self._detect_double_top(df, closes, highs, lows, params)

        elif pattern_name == "double_bottom":
            detected = self._detect_double_bottom(df, closes, highs, lows, params)

        elif pattern_name == "head_and_shoulders":
            detected = self._detect_head_and_shoulders(df, closes, highs, lows, params)

        elif pattern_name == "ascending_triangle":
            detected = self._detect_ascending_triangle(df, closes, highs, lows, params)

        elif pattern_name == "descending_triangle":
            detected = self._detect_descending_triangle(df, closes, highs, lows, params)

        elif pattern_name == "symmetrical_triangle":
            detected = self._detect_symmetrical_triangle(
                df, closes, highs, lows, params
            )

        # Add pattern name to each detection
        for detection in detected:
            detection["pattern"] = pattern_name
            detection["method"] = "algorithmic"

            # Add date information if available
            if "date" in df.columns:
                start_date = df["date"].iloc[detection["start_index"]]
                end_date = df["date"].iloc[detection["end_index"]]
                detection["start_date"] = (
                    start_date.strftime("%Y-%m-%d")
                    if isinstance(start_date, pd.Timestamp)
                    else str(start_date)
                )
                detection["end_date"] = (
                    end_date.strftime("%Y-%m-%d")
                    if isinstance(end_date, pd.Timestamp)
                    else str(end_date)
                )

        return detected

    def _detect_pattern_ml(
        self, df: pd.DataFrame, pattern_name: str
    ) -> List[Dict[str, Any]]:
        """
        Detect patterns using machine learning models.
        This is a placeholder implementation.

        Args:
            df: DataFrame with price data
            pattern_name: Name of pattern to detect

        Returns:
            List of detected pattern instances
        """
        # In a real implementation, this would use the loaded ML models
        # For this example, we'll return an empty list
        return []

    def _detect_double_top(
        self,
        df: pd.DataFrame,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect double top pattern.

        Args:
            df: DataFrame with price data
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            params: Detection parameters

        Returns:
            List of detected pattern instances
        """
        height_threshold = params.get("height_threshold", 0.03)
        valley_depth = params.get("valley_depth", 0.05)
        min_gap = params.get("min_gap", 5)

        detected = []
        n = len(closes)

        # Find local maxima in closing prices
        peaks = []
        for i in range(1, n - 1):
            if closes[i] > closes[i - 1] and closes[i] > closes[i + 1]:
                peaks.append(i)

        # Look for pairs of peaks that form double tops
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1_idx = peaks[i]
                peak2_idx = peaks[j]

                # Check if peaks are far enough apart
                if peak2_idx - peak1_idx < min_gap:
                    continue

                peak1_val = closes[peak1_idx]
                peak2_val = closes[peak2_idx]

                # Check if peaks are at similar heights
                height_diff = abs(peak1_val - peak2_val) / peak1_val
                if height_diff > height_threshold:
                    continue

                # Check if there's a significant valley between peaks
                min_val = np.min(closes[peak1_idx : peak2_idx + 1])
                valley_depth_pct = (peak1_val - min_val) / peak1_val
                if valley_depth_pct < valley_depth:
                    continue

                # Check if pattern has completed - a close below the valley
                if peak2_idx < n - 1 and closes[peak2_idx + 1] < min_val:
                    # Pattern found
                    detected.append(
                        {
                            "start_index": peak1_idx,
                            "end_index": peak2_idx + 1,  # Include confirmation candle
                            "confidence": 0.7 * (1 - height_diff / height_threshold)
                            + 0.3 * (valley_depth_pct / valley_depth),
                            "details": {
                                "peak1": float(peak1_val),
                                "peak2": float(peak2_val),
                                "valley": float(min_val),
                                "height_diff_pct": float(height_diff * 100),
                                "valley_depth_pct": float(valley_depth_pct * 100),
                            },
                        }
                    )

        return detected

    def _detect_double_bottom(
        self,
        df: pd.DataFrame,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect double bottom pattern.

        Args:
            df: DataFrame with price data
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            params: Detection parameters

        Returns:
            List of detected pattern instances
        """
        height_threshold = params.get("height_threshold", 0.03)
        peak_height = params.get("peak_height", 0.05)
        min_gap = params.get("min_gap", 5)

        detected = []
        n = len(closes)

        # Find local minima in closing prices
        bottoms = []
        for i in range(1, n - 1):
            if closes[i] < closes[i - 1] and closes[i] < closes[i + 1]:
                bottoms.append(i)

        # Look for pairs of bottoms that form double bottoms
        for i in range(len(bottoms) - 1):
            for j in range(i + 1, len(bottoms)):
                bottom1_idx = bottoms[i]
                bottom2_idx = bottoms[j]

                # Check if bottoms are far enough apart
                if bottom2_idx - bottom1_idx < min_gap:
                    continue

                bottom1_val = closes[bottom1_idx]
                bottom2_val = closes[bottom2_idx]

                # Check if bottoms are at similar heights
                height_diff = abs(bottom1_val - bottom2_val) / bottom1_val
                if height_diff > height_threshold:
                    continue

                # Check if there's a significant peak between bottoms
                max_val = np.max(closes[bottom1_idx : bottom2_idx + 1])
                peak_height_pct = (max_val - bottom1_val) / bottom1_val
                if peak_height_pct < peak_height:
                    continue

                # Check if pattern has completed - a close above the peak
                if bottom2_idx < n - 1 and closes[bottom2_idx + 1] > max_val:
                    # Pattern found
                    detected.append(
                        {
                            "start_index": bottom1_idx,
                            "end_index": bottom2_idx + 1,  # Include confirmation candle
                            "confidence": 0.7 * (1 - height_diff / height_threshold)
                            + 0.3 * (peak_height_pct / peak_height),
                            "details": {
                                "bottom1": float(bottom1_val),
                                "bottom2": float(bottom2_val),
                                "peak": float(max_val),
                                "height_diff_pct": float(height_diff * 100),
                                "peak_height_pct": float(peak_height_pct * 100),
                            },
                        }
                    )

        return detected

    def _detect_head_and_shoulders(
        self,
        df: pd.DataFrame,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect head and shoulders pattern.

        Args:
            df: DataFrame with price data
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            params: Detection parameters

        Returns:
            List of detected pattern instances
        """
        shoulder_height_diff = params.get("shoulder_height_diff", 0.05)
        head_height_min = params.get("head_height_min", 0.03)
        neckline_slope_max = params.get("neckline_slope_max", 0.2)

        # This is a placeholder implementation
        #
        # A full implementation would look for three peaks with specific
        # characteristics
        return []

    def _detect_ascending_triangle(
        self,
        df: pd.DataFrame,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect ascending triangle pattern.

        Args:
            df: DataFrame with price data
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            params: Detection parameters

        Returns:
            List of detected pattern instances
        """
        # This is a placeholder implementation
        return []

    def _detect_descending_triangle(
        self,
        df: pd.DataFrame,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect descending triangle pattern.

        Args:
            df: DataFrame with price data
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            params: Detection parameters

        Returns:
            List of detected pattern instances
        """
        # This is a placeholder implementation
        return []

    def _detect_symmetrical_triangle(
        self,
        df: pd.DataFrame,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        params: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Detect symmetrical triangle pattern.

        Args:
            df: DataFrame with price data
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            params: Detection parameters

        Returns:
            List of detected pattern instances
        """
        # This is a placeholder implementation
        return []

    def detect_candlestick_patterns(
        self,
        price_data: List[Dict[str, Any]],
        patterns: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect candlestick patterns in price data.

        Args:
            price_data: List of price data dictionaries
            patterns: List of candlestick patterns to detect
            min_confidence: Minimum confidence score for pattern detection

        Returns:
            Dictionary with detected patterns and metadata
        """
        start_time = time.time()

        # Prepare DataFrame
        df = self._prepare_dataframe(price_data)

        # Use provided parameters or defaults
        min_confidence = min_confidence or self.confidence_threshold

        # Define available candlestick patterns
        available_patterns = {
            "doji": "A candlestick with a very small body",
            "hammer": "Small body at the high with a long lower shadow",
            "shooting_star": "Small body at the low with a long upper shadow",
            "engulfing_bullish": "Bullish candle engulfs the previous bearish candle",
            "engulfing_bearish": "Bearish candle engulfs the previous bullish candle",
            "morning_star": "Three-candle bullish reversal pattern",
            "evening_star": "Three-candle bearish reversal pattern",
            "three_white_soldiers": "Three consecutive bullish candles with higher closes",
            "three_black_crows": "Three consecutive bearish candles with lower closes",
            "harami_bullish": "Small bullish candle contained within previous bearish candle",
            "harami_bearish": "Small bearish candle contained within previous bullish candle",
        }

        # Filter patterns if specified
        patterns_to_detect = patterns or list(available_patterns.keys())

        # Detected patterns
        detected_patterns = []

        # Calculate candlestick properties
        df["body_size"] = abs(df["close"] - df["open"])
        df["upper_shadow"] = df.apply(
            lambda x: x["high"] - max(x["open"], x["close"]), axis=1
        )
        df["lower_shadow"] = df.apply(
            lambda x: min(x["open"], x["close"]) - x["low"], axis=1
        )
        df["range"] = df["high"] - df["low"]
        df["is_bullish"] = df["close"] > df["open"]

        # Detect each pattern
        for i in range(1, len(df)):
            # Can't detect some patterns at the beginning of the dataset
            if i < 2:
                continue

            patterns_at_index = []

            if "doji" in patterns_to_detect:
                # Doji: body size is very small compared to range
                body_to_range = (
                    df.iloc[i]["body_size"] / df.iloc[i]["range"]
                    if df.iloc[i]["range"] > 0
                    else 0
                )
                if body_to_range < 0.1:
                    confidence = 1.0 - body_to_range / 0.1
                    patterns_at_index.append(
                        {
                            "pattern": "doji",
                            "index": i,
                            "confidence": float(confidence),
                            "methods": "algorithmic",
                        }
                    )

            if "hammer" in patterns_to_detect:
                # Hammer: small body at the top, long lower shadow
                body_to_range = (
                    df.iloc[i]["body_size"] / df.iloc[i]["range"]
                    if df.iloc[i]["range"] > 0
                    else 0
                )
                lower_to_range = (
                    df.iloc[i]["lower_shadow"] / df.iloc[i]["range"]
                    if df.iloc[i]["range"] > 0
                    else 0
                )
                upper_to_range = (
                    df.iloc[i]["upper_shadow"] / df.iloc[i]["range"]
                    if df.iloc[i]["range"] > 0
                    else 0
                )

                if (
                    body_to_range < 0.3
                    and lower_to_range > 0.6
                    and upper_to_range < 0.1
                ):
                    confidence = 0.6 * lower_to_range + 0.4 * (1 - body_to_range)
                    patterns_at_index.append(
                        {
                            "pattern": "hammer",
                            "index": i,
                            "confidence": float(confidence),
                            "methods": "algorithmic",
                        }
                    )

            if "engulfing_bullish" in patterns_to_detect:
                #
                # Bullish Engulfing: current bullish candle engulfs previous
                # bearish candle
                prev_bearish = df.iloc[i - 1]["close"] < df.iloc[i - 1]["open"]
                curr_bullish = df.iloc[i]["close"] > df.iloc[i]["open"]
                curr_open_lower = df.iloc[i]["open"] < df.iloc[i - 1]["close"]
                curr_close_higher = df.iloc[i]["close"] > df.iloc[i - 1]["open"]

                if (
                    prev_bearish
                    and curr_bullish
                    and curr_open_lower
                    and curr_close_higher
                ):
                    # Calculate how much bigger the current candle is
                    size_ratio = (
                        df.iloc[i]["body_size"] / df.iloc[i - 1]["body_size"]
                        if df.iloc[i - 1]["body_size"] > 0
                        else 2
                    )
                    confidence = min(1.0, 0.7 + 0.3 * min(1, (size_ratio - 1) / 2))
                    patterns_at_index.append(
                        {
                            "pattern": "engulfing_bullish",
                            "index": i,
                            "confidence": float(confidence),
                            "methods": "algorithmic",
                        }
                    )

            # Add detected patterns to the overall list
            for pattern in patterns_at_index:
                if pattern["confidence"] >= min_confidence:
                    if "date" in df.columns:
                        pattern_date = df["date"].iloc[pattern["index"]]
                        pattern["date"] = (
                            pattern_date.strftime("%Y-%m-%d")
                            if isinstance(pattern_date, pd.Timestamp)
                            else str(pattern_date)
                        )
                    detected_patterns.append(pattern)

        return {
            "patterns": detected_patterns,
            "total_detected": len(detected_patterns),
            "processing_time": time.time() - start_time,
        }

    def detect_support_resistance(
        self,
        price_data: List[Dict[str, Any]],
        method: Optional[str] = "peaks",
        sensitivity: Optional[float] = 0.5,
        lookback: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Detect support and resistance levels in price data.

        Args:
            price_data: List of price data dictionaries
            method: Method to use for detection
            sensitivity: Sensitivity parameter (0.0-1.0)
            lookback: Number of bars to look back

        Returns:
            Dictionary with detected support/resistance levels
        """
        start_time = time.time()

        # Prepare DataFrame
        df = self._prepare_dataframe(price_data)

        # Default lookback if not provided
        lookback = lookback or min(200, len(df))

        # Limit data to lookback period
        if len(df) > lookback:
            df = df.iloc[-lookback:]

        # Use sensitivity to adjust detection parameters
        if sensitivity is None:
            sensitivity = 0.5

        window_size = int(max(5, lookback * (0.03 + sensitivity * 0.07)))
        min_touches = max(2, int(3 - sensitivity))

        # Detect levels using specified method
        levels = {}

        if method == "peaks":
            levels = self._detect_sr_peaks(df, window_size, min_touches)
        elif method == "fractals":
            levels = self._detect_sr_fractals(df, window_size, min_touches)
        elif method == "zones":
            levels = self._detect_sr_zones(df, window_size, min_touches, sensitivity)
        else:
            # Default to peaks method
            levels = self._detect_sr_peaks(df, window_size, min_touches)

        return {
            "levels": levels,
            "method": method,
            "sensitivity": sensitivity,
            "lookback": lookback,
            "processing_time": time.time() - start_time,
        }

    def _detect_sr_peaks(
        self, df: pd.DataFrame, window_size: int, min_touches: int
    ) -> Dict[str, Any]:
        """
        Detect support and resistance levels using peaks method.

        Args:
            df: DataFrame with price data
            window_size: Window size for peak detection
            min_touches: Minimum number of touches to confirm a level

        Returns:
            Dictionary with support and resistance levels
        """
        # Find local maxima and minima
        max_idx = []
        min_idx = []

        for i in range(window_size, len(df) - window_size):
            if (
                df["high"].iloc[i]
                == df["high"].iloc[i - window_size : i + window_size + 1].max()
            ):
                max_idx.append(i)
            if (
                df["low"].iloc[i]
                == df["low"].iloc[i - window_size : i + window_size + 1].min()
            ):
                min_idx.append(i)

        # Convert to price levels
        resistance_levels = [float(df["high"].iloc[i]) for i in max_idx]
        support_levels = [float(df["low"].iloc[i]) for i in min_idx]

        # Cluster similar levels
        clustered_resistance = self._cluster_levels(resistance_levels)
        clustered_support = self._cluster_levels(support_levels)

        # Count touches for each level
        resistance_touches = self._count_touches(df, clustered_resistance, "resistance")
        support_touches = self._count_touches(df, clustered_support, "support")

        # Filter by minimum touches
        resistance_levels = [
            level
            for level, touches in resistance_touches.items()
            if touches >= min_touches
        ]
        support_levels = [
            level
            for level, touches in support_touches.items()
            if touches >= min_touches
        ]

        # Sort levels
        resistance_levels.sort()
        support_levels.sort()

        # Calculate strength of each level (0.0-1.0)
        max_touches = max(
            max(resistance_touches.values(), default=min_touches),
            max(support_touches.values(), default=min_touches),
        )

        resistance_strength = {
            level: min(1.0, touches / max_touches)
            for level, touches in resistance_touches.items()
            if touches >= min_touches
        }

        support_strength = {
            level: min(1.0, touches / max_touches)
            for level, touches in support_touches.items()
            if touches >= min_touches
        }

        return {
            "resistance": {
                "levels": resistance_levels,
                "strength": resistance_strength,
                "touches": {
                    str(k): v for k, v in resistance_touches.items() if v >= min_touches
                },
            },
            "support": {
                "levels": support_levels,
                "strength": support_strength,
                "touches": {
                    str(k): v for k, v in support_touches.items() if v >= min_touches
                },
            },
            "details": {"window_size": window_size, "min_touches": min_touches},
        }

    def _detect_sr_fractals(
        self, df: pd.DataFrame, window_size: int, min_touches: int
    ) -> Dict[str, Any]:
        """
        Detect support and resistance levels using fractals method.
        This is a placeholder implementation.

        Args:
            df: DataFrame with price data
            window_size: Window size for fractal detection
            min_touches: Minimum number of touches to confirm a level

        Returns:
            Dictionary with support and resistance levels
        """
        # This would implement Bill Williams' fractal indicator
        # For this example, we'll return empty results
        return {
            "resistance": {"levels": [], "strength": {}, "touches": {}},
            "support": {"levels": [], "strength": {}, "touches": {}},
            "details": {"window_size": window_size, "min_touches": min_touches},
        }

    def _detect_sr_zones(
        self, df: pd.DataFrame, window_size: int, min_touches: int, sensitivity: float
    ) -> Dict[str, Any]:
        """
        Detect support and resistance zones.
        This is a placeholder implementation.

        Args:
            df: DataFrame with price data
            window_size: Window size for zone detection
            min_touches: Minimum number of touches to confirm a zone
            sensitivity: Sensitivity parameter

        Returns:
            Dictionary with support and resistance zones
        """
        # This would implement zone-based detection
        # For this example, we'll return empty results
        return {
            "resistance": {"zones": [], "strength": {}, "touches": {}},
            "support": {"zones": [], "strength": {}, "touches": {}},
            "details": {
                "window_size": window_size,
                "min_touches": min_touches,
                "sensitivity": sensitivity,
            },
        }

    def _cluster_levels(
        self, levels: List[float], tolerance: float = 0.01
    ) -> List[float]:
        """
        Cluster similar price levels.

        Args:
            levels: List of price levels
            tolerance: Tolerance for clustering as percentage

        Returns:
            List of clustered levels
        """
        if not levels:
            return []

        # Sort levels
        sorted_levels = sorted(levels)

        # Calculate relative tolerance (percentage of price range)
        price_range = max(sorted_levels) - min(sorted_levels)
        rel_tolerance = price_range * tolerance if price_range > 0 else 0.001

        # Cluster levels
        clustered = []
        current_cluster = [sorted_levels[0]]

        for level in sorted_levels[1:]:
            # Check if level belongs to current cluster
            if level - current_cluster[0] <= rel_tolerance:
                current_cluster.append(level)
            else:
                # Add average of current cluster and start a new one
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]

        # Add last cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))

        return clustered

    def _count_touches(
        self, df: pd.DataFrame, levels: List[float], level_type: str
    ) -> Dict[float, int]:
        """
        Count touches for each level.

        Args:
            df: DataFrame with price data
            levels: List of price levels
            level_type: 'support' or 'resistance'

        Returns:
            Dictionary with level as key and touch count as value
        """
        touches = {level: 0 for level in levels}

        # Calculate price range for tolerance
        price_range = df["high"].max() - df["low"].min()
        tolerance = price_range * 0.005  # 0.5% of price range

        for level in levels:
            if level_type == "resistance":
                # Count candles that touch the resistance level
                for i in range(len(df)):
                    if abs(df["high"].iloc[i] - level) <= tolerance:
                        touches[level] += 1
            else:  # support
                # Count candles that touch the support level
                for i in range(len(df)):
                    if abs(df["low"].iloc[i] - level) <= tolerance:
                        touches[level] += 1

        return touches

    def analyze_pattern(
        self,
        price_data: List[Dict[str, Any]],
        pattern: str,
        start_index: int,
        end_index: int,
    ) -> Dict[str, Any]:
        """
        Analyze a specific pattern occurrence in detail.

        Args:
            price_data: List of price data dictionaries
            pattern: Pattern to analyze
            start_index: Start index of the pattern
            end_index: End index of the pattern

        Returns:
            Dictionary with pattern analysis
        """
        start_time = time.time()

        # Prepare DataFrame
        df = self._prepare_dataframe(price_data)

        # Check indices
        if start_index < 0 or end_index >= len(df) or start_index >= end_index:
            return {
                "error": "Invalid indices",
                "processing_time": time.time() - start_time,
            }

        # Extract pattern data
        pattern_df = df.iloc[start_index : end_index + 1]

        # Basic pattern statistics
        stats = {
            "duration": end_index - start_index + 1,
            "price_range": float(pattern_df["high"].max() - pattern_df["low"].min()),
            "price_range_pct": float(
                (pattern_df["high"].max() - pattern_df["low"].min())
                / pattern_df["low"].min()
                * 100
            ),
            "volume_avg": float(pattern_df["volume"].mean())
            if "volume" in pattern_df.columns
            else None,
            "start_price": float(pattern_df["close"].iloc[0]),
            "end_price": float(pattern_df["close"].iloc[-1]),
            "price_change_pct": float(
                (pattern_df["close"].iloc[-1] - pattern_df["close"].iloc[0])
                / pattern_df["close"].iloc[0]
                * 100
            ),
        }

        # Pattern-specific analysis
        pattern_analysis = {}

        if pattern == "double_top":
            pattern_analysis = self._analyze_double_top(pattern_df)
        elif pattern == "double_bottom":
            pattern_analysis = self._analyze_double_bottom(pattern_df)
        elif pattern == "head_and_shoulders":
            pattern_analysis = self._analyze_head_and_shoulders(pattern_df)
        # Additional pattern analyses would be added here

        # Historical pattern performance
        historical_performance = self.get_historical_pattern_performance(
            pattern, "us_equities", "daily"
        )

        return {
            "pattern": pattern,
            "start_index": start_index,
            "end_index": end_index,
            "start_date": pattern_df["date"].iloc[0].strftime("%Y-%m-%d")
            if "date" in pattern_df.columns
            else None,
            "end_date": pattern_df["date"].iloc[-1].strftime("%Y-%m-%d")
            if "date" in pattern_df.columns
            else None,
            "statistics": stats,
            "analysis": pattern_analysis,
            "historical_performance": historical_performance.get("statistics", {}),
            "processing_time": time.time() - start_time,
        }

    def _analyze_double_top(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a double top pattern.
        This is a placeholder implementation.

        Args:
            df: DataFrame with pattern price data

        Returns:
            Dictionary with pattern analysis
        """
        # This would provide detailed analysis of a double top pattern
        return {
            "neckline": float(df["close"].min()),
            "distance_between_peaks": 0,  # Would calculate the distance
            "peak1_height": 0,  # Would calculate height of first peak
            "peak2_height": 0,  # Would calculate height of second peak
            "target": 0,  # Would calculate price target
        }

    def _analyze_double_bottom(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a double bottom pattern.
        This is a placeholder implementation.

        Args:
            df: DataFrame with pattern price data

        Returns:
            Dictionary with pattern analysis
        """
        return {}

    def _analyze_head_and_shoulders(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a head and shoulders pattern.
        This is a placeholder implementation.

        Args:
            df: DataFrame with pattern price data

        Returns:
            Dictionary with pattern analysis
        """
        return {}

    def get_historical_pattern_performance(
        self,
        pattern: str,
        market: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get historical performance statistics for a pattern.

        Args:
            pattern: Pattern to get statistics for
            market: Market to get statistics for
            timeframe: Timeframe to get statistics for

        Returns:
            Dictionary with historical performance statistics
        """
        start_time = time.time()

        # This would fetch historical statistics from a database
        # For this example, we'll return placeholder data

        pattern_statistics = {
            "double_top": {
                "accuracy": 0.68,
                "avg_decline": 12.5,
                "avg_timeframe": 23,
                "false_signals": 0.22,
                "samples": 2480,
            },
            "double_bottom": {
                "accuracy": 0.72,
                "avg_rally": 14.2,
                "avg_timeframe": 18,
                "false_signals": 0.19,
                "samples": 2120,
            },
            "head_and_shoulders": {
                "accuracy": 0.61,
                "avg_decline": 15.8,
                "avg_timeframe": 35,
                "false_signals": 0.25,
                "samples": 1850,
            },
            "inverse_head_and_shoulders": {
                "accuracy": 0.63,
                "avg_rally": 16.2,
                "avg_timeframe": 32,
                "false_signals": 0.24,
                "samples": 1790,
            },
            "ascending_triangle": {
                "accuracy": 0.75,
                "avg_rally": 10.8,
                "avg_timeframe": 28,
                "false_signals": 0.17,
                "samples": 2310,
            },
            "descending_triangle": {
                "accuracy": 0.71,
                "avg_decline": 11.2,
                "avg_timeframe": 26,
                "false_signals": 0.18,
                "samples": 2250,
            },
            "symmetrical_triangle": {
                "accuracy": 0.59,
                "avg_move": 9.5,
                "avg_timeframe": 22,
                "false_signals": 0.28,
                "samples": 3100,
            },
        }

        # Get statistics for the requested pattern
        statistics = pattern_statistics.get(pattern, {})

        return {
            "pattern": pattern,
            "market": market,
            "timeframe": timeframe,
            "statistics": statistics,
            "examples": [],  # Would include examples of the pattern
            "processing_time": time.time() - start_time,
        }


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {"use_ml": True, "use_gpu": HAVE_CUDA, "confidence_threshold": 0.65}

    # Create and start the server
    server = PatternRecognitionMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("PatternRecognitionMCP server started")

    # Example usage
    import datetime

    # Generate some sample price data with a double top pattern
    price_data = []
    base_price = 100.0
    pattern_start = 30

    for i in range(100):
        day = datetime.datetime.now() - datetime.timedelta(days=100 - i)

        # Create a double top pattern
        if i >= pattern_start and i < pattern_start + 10:
            # First peak
            price = base_price + 10 * (1 - abs(i - pattern_start - 5) / 5)
        elif i >= pattern_start + 15 and i < pattern_start + 25:
            # Second peak
            price = base_price + 10 * (1 - abs(i - pattern_start - 20) / 5)
        elif i >= pattern_start + 25:
            # Decline after pattern
            price = base_price - (i - pattern_start - 25)
        else:
            # Before pattern
            price = base_price + i * 0.1

        # Add some noise
        price += np.random.normal(0, 0.5)

        price_data.append(
            {
                "date": day.strftime("%Y-%m-%d"),
                "open": price - 0.5,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price,
                "volume": 1000000 + np.random.randint(-200000, 200000),
            }
        )

        base_price = price

    # Detect patterns
    result = server.detect_patterns(
        price_data=price_data, patterns=["double_top", "double_bottom"]
    )

    print(f"Detected {len(result['patterns'])} patterns")
