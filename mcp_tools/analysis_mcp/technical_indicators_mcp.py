"""
Technical Indicators MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
technical analysis and indicator calculations for financial market data.
"""

import os
import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any, Optional

# For GPU acceleration if available
try:
    import cudf
    import cupy as cp

    HAVE_CUDA = True
except ImportError:
    HAVE_CUDA = False
    cp = np  # Use NumPy as fallback

# Import monitoring
from monitoring import setup_monitoring

from mcp_tools.base_mcp_server import BaseMCPServer


class TechnicalIndicatorsMCP(BaseMCPServer):
    """
    MCP server for calculating technical indicators and patterns.

    This tool calculates common technical indicators like moving averages,
    RSI, MACD, Bollinger Bands, etc., and can use GPU acceleration when available.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Technical Indicators MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - use_gpu: Whether to use GPU acceleration if available (default: True)
                - custom_indicators_path: Path to custom indicator definitions
                - cache_dir: Directory for caching intermediate results
                - enable_monitoring: Whether to enable monitoring (default: True)
                - enable_prometheus: Whether to enable Prometheus metrics (default: True)
                - enable_loki: Whether to enable Loki logging (default: True)
                - metrics_port: Port for Prometheus metrics (default: auto-assigned)
        """
        super().__init__(name="technical_indicators_mcp", config=config)

        # Set default configurations
        self.use_gpu = self.config.get("use_gpu", HAVE_CUDA)
        self.custom_indicators_path = self.config.get(
            "custom_indicators_path",
            os.path.join(os.path.dirname(__file__), "data/custom_indicators.json"),
        )
        self.cache_dir = self.config.get("cache_dir", "./cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # Setup monitoring
        self._setup_monitoring()

        # Initialize
        self._load_custom_indicators()

        # Register tools
        self._register_tools()

        # Log initialization
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "TechnicalIndicatorsMCP initialized",
                component="technical_indicators_mcp",
                gpu_available=HAVE_CUDA,
                gpu_enabled=self.use_gpu,
            )

    def _setup_monitoring(self):
        """Set up monitoring with Prometheus and Loki."""
        # Check if monitoring is enabled in config
        enable_monitoring = self.config.get("enable_monitoring", True)

        if not enable_monitoring:
            return

        # Set up monitoring with configuration from config
        try:
            enable_prometheus = self.config.get("enable_prometheus", True)
            enable_loki = self.config.get("enable_loki", True)
            metrics_port = self.config.get("metrics_port", None)

            self.monitor, self.metrics = setup_monitoring(
                service_name="technical-indicators-mcp",
                enable_prometheus=enable_prometheus,
                enable_loki=enable_loki,
                metrics_port=metrics_port,
                default_labels={"component": "technical_indicators_mcp"},
            )

            # Create specific metrics for technical indicators
            if hasattr(self, "metrics") and self.metrics and enable_prometheus:
                #
                # Processing time histogram with buckets appropriate for
                # indicator calculations
                self.metrics["indicator_processing_time"] = (
                    self.monitor.create_histogram(
                        "indicator_processing_time_seconds",
                        "Time to calculate technical indicators in seconds",
                        ["indicator_type", "data_size", "gpu_used"],
                        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
                    )
                )

                # Counter for indicator calculations
                self.metrics["indicators_calculated"] = self.monitor.create_counter(
                    "indicators_calculated_total",
                    "Number of technical indicators calculated",
                    ["indicator_type", "gpu_used"],
                )

                # Counter for signals generated
                self.metrics["signals_generated"] = self.monitor.create_counter(
                    "signals_generated_total",
                    "Number of trading signals generated from indicators",
                    ["signal_type", "indicator_type"],
                )

                # GPU utilization metrics if GPU is available
                if HAVE_CUDA:
                    self.metrics["gpu_memory_used"] = self.monitor.create_gauge(
                        "gpu_memory_used_bytes",
                        "GPU memory used for calculations in bytes",
                        ["operation"],
                    )

        except Exception as e:
            self.logger.warning(f"Failed to initialize monitoring: {e}")
            # Continue without monitoring

    def _load_custom_indicators(self):
        """Load custom indicator definitions if available."""
        self.custom_indicators = {}

        if os.path.exists(self.custom_indicators_path):
            try:
                with open(self.custom_indicators_path, "r") as f:
                    self.custom_indicators = json.load(f)
                self.logger.info(
                    f"Loaded {len(self.custom_indicators)} custom indicators"
                )
            except Exception as e:
                self.logger.error(f"Error loading custom indicators: {e}")
        else:
            self.logger.info(
                f"No custom indicators file found at {self.custom_indicators_path}"
            )

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "calculate_indicators",
            self.calculate_indicators,
            "Calculate multiple technical indicators for price data",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with 'date', 'open', 'high', 'low', 'close', 'volume' fields",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of indicators to calculate",
                },
                "periods": {
                    "type": "object",
                    "description": "Dictionary of periods for each indicator (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "indicators": {"type": "object"},
                    "metadata": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "moving_averages",
            self.moving_averages,
            "Calculate various types of moving averages",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with 'date' and 'close' fields at minimum",
                },
                "ma_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Types of moving averages to calculate (e.g., 'sma', 'ema', 'wma')",
                },
                "periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Periods for the moving averages",
                    "default": [20, 50, 200],
                },
            },
            {
                "type": "object",
                "properties": {
                    "moving_averages": {"type": "object"},
                    "crossovers": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "momentum_oscillators",
            self.momentum_oscillators,
            "Calculate momentum oscillators like RSI, Stochastic, MACD",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with required fields",
                },
                "oscillators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of oscillators to calculate",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for oscillators (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "oscillators": {"type": "object"},
                    "signals": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "volatility_indicators",
            self.volatility_indicators,
            "Calculate volatility indicators like Bollinger Bands, ATR",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of volatility indicators to calculate",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for indicators (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "indicators": {"type": "object"},
                    "signals": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "volume_indicators",
            self.volume_indicators,
            "Calculate volume-based indicators like OBV, Money Flow",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries with 'close' and 'volume' fields",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of volume indicators to calculate",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for indicators (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "indicators": {"type": "object"},
                    "signals": {"type": "array"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "trend_indicators",
            self.trend_indicators,
            "Calculate trend indicators like ADX, Ichimoku Cloud",
            {
                "price_data": {
                    "type": "array",
                    "description": "Array of price data dictionaries",
                },
                "indicators": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of trend indicators to calculate",
                },
                "parameters": {
                    "type": "object",
                    "description": "Parameters for indicators (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "indicators": {"type": "object"},
                    "signals": {"type": "array"},
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
            required_cols = ["close"]
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

    def calculate_indicators(
        self,
        price_data: List[Dict[str, Any]],
        indicators: List[str],
        periods: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate multiple technical indicators for price data.

        Args:
            price_data: List of price data dictionaries
            indicators: List of indicators to calculate
            periods: Dictionary of periods for each indicator

        Returns:
            Dictionary of calculated indicators, signals, and metadata
        """
        start_time = time.time()
        stats = {
            "indicators_requested": len(indicators),
            "data_points": len(price_data),
            "successful_indicators": 0,
            "signals_generated": 0,
            "errors": 0,
        }

        # Log the request
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Calculating technical indicators",
                indicator_count=len(indicators),
                data_points=len(price_data),
                indicators=",".join(indicators),
            )

        # Default periods if not specified
        default_periods = {
            "sma": [20, 50, 200],
            "ema": [12, 26, 50],
            "rsi": [14],
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bbands": {"period": 20, "stddev": 2},
            "atr": [14],
            "adx": [14],
            "stoch": {"k": 14, "d": 3},
            "obv": [],
            "vwap": [],
        }

        # Use provided periods or defaults
        periods = periods or {}
        for indicator in indicators:
            if indicator not in periods and indicator in default_periods:
                periods[indicator] = default_periods[indicator]

        # Track GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:

                initial_mem = cp.cuda.memory_info()[0]  # Get initial memory usage
                self.metrics["gpu_memory_used"].labels(operation="initial").set(
                    initial_mem
                )
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track initial GPU memory: {e}", error=str(e)
                    )

        # Prepare DataFrame
        df_preparation_start = time.time()
        try:
            df = self._prepare_dataframe(price_data)

            # Set a flag for error handling
            if df.empty:
                raise ValueError("Empty DataFrame created from input data")

            df_preparation_time = time.time() - df_preparation_start
            if (
                hasattr(self, "monitor")
                and hasattr(self, "metrics")
                and "indicator_processing_time" in self.metrics
            ):
                self.monitor.observe_histogram(
                    "indicator_processing_time",
                    df_preparation_time,
                    indicator_type="data_preparation",
                    data_size=str(len(price_data)),
                    gpu_used=str(self.use_gpu and HAVE_CUDA),
                )

        except Exception as e:
            error_msg = f"Failed to prepare data for indicator calculation: {str(e)}"
            stats["errors"] += 1

            if hasattr(self, "monitor"):
                self.monitor.log_error(
                    error_msg,
                    error_type="data_preparation",
                    traceback=str(e),
                    data_points=len(price_data),
                )

                if hasattr(self, "metrics") and "errors_total" in self.metrics:
                    self.monitor.increment_counter(
                        "errors_total",
                        1,
                        type="data_preparation",
                        operation="calculate_indicators",
                    )

            return {
                "error": error_msg,
                "indicators": {},
                "signals": [],
                "metadata": {
                    "success": False,
                    "error": error_msg,
                    "indicators_requested": stats["indicators_requested"],
                    "data_points": stats["data_points"],
                },
                "processing_time": time.time() - start_time,
            }

        # Calculate requested indicators
        results = {}
        signals = []

        # Track which computations used GPU
        gpu_used = self.use_gpu and HAVE_CUDA

        for indicator in indicators:
            indicator_start_time = time.time()

            try:
                if indicator == "sma":
                    sma_periods = periods.get("sma", [20, 50, 200])
                    sma_results = self._calculate_sma(df, sma_periods)
                    results["sma"] = sma_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated SMA indicator",
                            indicator="sma",
                            periods=str(sma_periods),
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="sma",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="sma",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for crossovers
                    if len(sma_periods) > 1:
                        crossover_signals = []
                        for i in range(len(sma_periods)):
                            for j in range(i + 1, len(sma_periods)):
                                p1, p2 = sma_periods[i], sma_periods[j]
                                crossovers = self._detect_crossovers(
                                    df, f"sma_{p1}", f"sma_{p2}"
                                )
                                for crossover in crossovers:
                                    signal = {
                                        "type": "crossover",
                                        "indicator1": f"SMA({p1})",
                                        "indicator2": f"SMA({p2})",
                                        "direction": crossover["direction"],
                                        "date": crossover["date"],
                                        "index": crossover["index"],
                                    }
                                    signals.append(signal)
                                    crossover_signals.append(signal)

                        # Log signals generated
                        if hasattr(self, "monitor") and crossover_signals:
                            stats["signals_generated"] += len(crossover_signals)
                            self.monitor.log_info(
                                f"Generated {len(crossover_signals)} SMA crossover signals",
                                indicator="sma",
                                signal_count=len(crossover_signals),
                            )

                            if (
                                hasattr(self, "metrics")
                                and "signals_generated" in self.metrics
                            ):
                                self.monitor.increment_counter(
                                    "signals_generated",
                                    len(crossover_signals),
                                    signal_type="crossover",
                                    indicator_type="sma",
                                )

                elif indicator == "ema":
                    ema_periods = periods.get("ema", [12, 26, 50])
                    ema_results = self._calculate_ema(df, ema_periods)
                    results["ema"] = ema_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated EMA indicator",
                            indicator="ema",
                            periods=str(ema_periods),
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="ema",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="ema",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for crossovers
                    if len(ema_periods) > 1:
                        crossover_signals = []
                        for i in range(len(ema_periods)):
                            for j in range(i + 1, len(ema_periods)):
                                p1, p2 = ema_periods[i], ema_periods[j]
                                crossovers = self._detect_crossovers(
                                    df, f"ema_{p1}", f"ema_{p2}"
                                )
                                for crossover in crossovers:
                                    signal = {
                                        "type": "crossover",
                                        "indicator1": f"EMA({p1})",
                                        "indicator2": f"EMA({p2})",
                                        "direction": crossover["direction"],
                                        "date": crossover["date"],
                                        "index": crossover["index"],
                                    }
                                    signals.append(signal)
                                    crossover_signals.append(signal)

                        # Log signals generated
                        if hasattr(self, "monitor") and crossover_signals:
                            stats["signals_generated"] += len(crossover_signals)
                            self.monitor.log_info(
                                f"Generated {len(crossover_signals)} EMA crossover signals",
                                indicator="ema",
                                signal_count=len(crossover_signals),
                            )

                            if (
                                hasattr(self, "metrics")
                                and "signals_generated" in self.metrics
                            ):
                                self.monitor.increment_counter(
                                    "signals_generated",
                                    len(crossover_signals),
                                    signal_type="crossover",
                                    indicator_type="ema",
                                )

                elif indicator == "rsi":
                    rsi_period = periods.get("rsi", [14])[0]
                    rsi_results = self._calculate_rsi(df, rsi_period)
                    results["rsi"] = rsi_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated RSI indicator",
                            indicator="rsi",
                            period=rsi_period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="rsi",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="rsi",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for overbought/oversold conditions
                    rsi_signals = []
                    for i in range(len(df) - 1):
                        if i < rsi_period:
                            continue

                        rsi_value = rsi_results["values"][i]
                        prev_rsi = rsi_results["values"][i - 1]

                        if prev_rsi > 70 and rsi_value <= 70:
                            signal = {
                                "type": "overbought_exit",
                                "indicator": "RSI",
                                "value": rsi_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            rsi_signals.append(signal)
                        elif prev_rsi < 30 and rsi_value >= 30:
                            signal = {
                                "type": "oversold_exit",
                                "indicator": "RSI",
                                "value": rsi_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            rsi_signals.append(signal)
                        elif prev_rsi <= 70 and rsi_value > 70:
                            signal = {
                                "type": "overbought_enter",
                                "indicator": "RSI",
                                "value": rsi_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            rsi_signals.append(signal)
                        elif prev_rsi >= 30 and rsi_value < 30:
                            signal = {
                                "type": "oversold_enter",
                                "indicator": "RSI",
                                "value": rsi_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            rsi_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and rsi_signals:
                        stats["signals_generated"] += len(rsi_signals)
                        self.monitor.log_info(
                            f"Generated {len(rsi_signals)} RSI signals",
                            indicator="rsi",
                            signal_count=len(rsi_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(rsi_signals),
                                signal_type="overbought_oversold",
                                indicator_type="rsi",
                            )

                elif indicator == "macd":
                    macd_params = periods.get(
                        "macd", {"fast": 12, "slow": 26, "signal": 9}
                    )
                    macd_results = self._calculate_macd(
                        df,
                        macd_params["fast"],
                        macd_params["slow"],
                        macd_params["signal"],
                    )
                    results["macd"] = macd_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated MACD indicator",
                            indicator="macd",
                            fast_period=macd_params["fast"],
                            slow_period=macd_params["slow"],
                            signal_period=macd_params["signal"],
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="macd",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="macd",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for signal line crossovers
                    macd_signals = []
                    for i in range(len(df) - 1):
                        if (
                            i
                            < max(macd_params["slow"], macd_params["fast"])
                            + macd_params["signal"]
                        ):
                            continue

                        macd_line = macd_results["macd_line"][i]
                        signal_line = macd_results["signal_line"][i]
                        prev_macd = macd_results["macd_line"][i - 1]
                        prev_signal = macd_results["signal_line"][i - 1]

                        if prev_macd < prev_signal and macd_line > signal_line:
                            signal = {
                                "type": "bullish_crossover",
                                "indicator": "MACD",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            macd_signals.append(signal)
                        elif prev_macd > prev_signal and macd_line < signal_line:
                            signal = {
                                "type": "bearish_crossover",
                                "indicator": "MACD",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            macd_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and macd_signals:
                        stats["signals_generated"] += len(macd_signals)
                        self.monitor.log_info(
                            f"Generated {len(macd_signals)} MACD signals",
                            indicator="macd",
                            signal_count=len(macd_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(macd_signals),
                                signal_type="crossover",
                                indicator_type="macd",
                            )

                elif indicator == "bbands":
                    bb_params = periods.get("bbands", {"period": 20, "stddev": 2})
                    bb_results = self._calculate_bollinger_bands(
                        df, bb_params["period"], bb_params["stddev"]
                    )
                    results["bbands"] = bb_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Bollinger Bands indicator",
                            indicator="bbands",
                            period=bb_params["period"],
                            stddev=bb_params["stddev"],
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="bbands",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="bbands",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for price touching bands
                    bb_signals = []
                    for i in range(len(df) - 1):
                        if i < bb_params["period"]:
                            continue

                        price = df["close"].iloc[i]
                        upper_band = bb_results["upper_band"][i]
                        lower_band = bb_results["lower_band"][i]

                        if price >= upper_band:
                            signal = {
                                "type": "upper_band_touch",
                                "indicator": "Bollinger Bands",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            bb_signals.append(signal)
                        elif price <= lower_band:
                            signal = {
                                "type": "lower_band_touch",
                                "indicator": "Bollinger Bands",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            bb_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and bb_signals:
                        stats["signals_generated"] += len(bb_signals)
                        self.monitor.log_info(
                            f"Generated {len(bb_signals)} Bollinger Bands signals",
                            indicator="bbands",
                            signal_count=len(bb_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(bb_signals),
                                signal_type="band_touch",
                                indicator_type="bbands",
                            )

                elif indicator == "atr":
                    atr_period = periods.get("atr", [14])[0]
                    atr_results = self._calculate_atr(df, atr_period)
                    results["atr"] = atr_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated ATR indicator",
                            indicator="atr",
                            period=atr_period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="atr",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="atr",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                elif indicator == "adx":
                    adx_period = periods.get("adx", [14])[0]
                    adx_results = self._calculate_adx(df, adx_period)
                    results["adx"] = adx_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated ADX indicator",
                            indicator="adx",
                            period=adx_period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="adx",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="adx",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for strong trend
                    adx_signals = []
                    for i in range(len(df) - 1):
                        if i < adx_period * 2:  # ADX needs more data points
                            continue

                        adx_value = adx_results["adx"][i]
                        prev_adx = adx_results["adx"][i - 1]

                        if prev_adx < 25 and adx_value >= 25:
                            signal = {
                                "type": "trend_strengthening",
                                "indicator": "ADX",
                                "value": adx_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            adx_signals.append(signal)
                        elif prev_adx > 25 and adx_value <= 25:
                            signal = {
                                "type": "trend_weakening",
                                "indicator": "ADX",
                                "value": adx_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            adx_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and adx_signals:
                        stats["signals_generated"] += len(adx_signals)
                        self.monitor.log_info(
                            f"Generated {len(adx_signals)} ADX signals",
                            indicator="adx",
                            signal_count=len(adx_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(adx_signals),
                                signal_type="trend",
                                indicator_type="adx",
                            )

                elif indicator == "stoch":
                    stoch_params = periods.get("stoch", {"k": 14, "d": 3})
                    stoch_results = self._calculate_stochastic(
                        df, stoch_params["k"], stoch_params["d"]
                    )
                    results["stoch"] = stoch_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Stochastic indicator",
                            indicator="stoch",
                            k_period=stoch_params["k"],
                            d_period=stoch_params["d"],
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="stoch",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="stoch",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                elif indicator == "obv":
                    obv_results = self._calculate_obv(df)
                    results["obv"] = obv_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated OBV indicator",
                            indicator="obv",
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="obv",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="obv",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                elif indicator == "vwap":
                    vwap_results = self._calculate_vwap(df)
                    results["vwap"] = vwap_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated VWAP indicator",
                            indicator="vwap",
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="vwap",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="vwap",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                else:
                    warning_msg = f"Unknown indicator: {indicator}"
                    self.logger.warning(warning_msg)
                    if hasattr(self, "monitor"):
                        self.monitor.log_warning(warning_msg, indicator=indicator)

            except Exception as e:
                error_msg = f"Error calculating {indicator}: {str(e)}"
                self.logger.error(error_msg)
                stats["errors"] += 1
                results[indicator] = {"error": str(e)}

                if hasattr(self, "monitor"):
                    self.monitor.log_error(
                        error_msg,
                        indicator=indicator,
                        error_type="calculation_error",
                        traceback=str(e),
                    )

                    if hasattr(self, "metrics") and "errors_total" in self.metrics:
                        self.monitor.increment_counter(
                            "errors_total",
                            1,
                            type="indicator_calculation",
                            indicator=indicator,
                        )

        # Track final GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                final_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(operation="final").set(final_mem)
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track final GPU memory: {e}", error=str(e)
                    )

        # Convert numpy arrays to lists for JSON serialization
        for indicator, data in results.items():
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series)):
                    if hasattr(value, "tolist"):
                        results[indicator][key] = value.tolist()
                    else:
                        results[indicator][key] = list(value)

        # Create metadata
        metadata = {
            "indicators_calculated": list(results.keys()),
            "data_points": len(df),
            "signals_found": len(signals),
            "start_date": df["date"].iloc[0].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "end_date": df["date"].iloc[-1].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "calculation_stats": stats,
            "gpu_used": gpu_used,
        }

        total_time = time.time() - start_time

        # Log overall performance
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Completed technical indicator calculations",
                duration=total_time,
                indicators_requested=stats["indicators_requested"],
                indicators_calculated=stats["successful_indicators"],
                signals_generated=stats["signals_generated"],
                errors=stats["errors"],
                gpu_used=gpu_used,
            )

        return {
            "indicators": results,
            "signals": signals,
            "metadata": metadata,
            "processing_time": total_time,
        }

    def moving_averages(
        self,
        price_data: List[Dict[str, Any]],
        ma_types: List[str],
        periods: List[int] = [20, 50, 200],
    ) -> Dict[str, Any]:
        """
        Calculate various types of moving averages.

        Args:
            price_data: List of price data dictionaries
            ma_types: Types of moving averages ('sma', 'ema', 'wma', etc.)
            periods: Periods for the moving averages

        Returns:
            Dictionary of moving averages and crossover signals
        """
        start_time = time.time()

        # Prepare DataFrame
        df = self._prepare_dataframe(price_data)

        results = {}
        crossovers = []

        for ma_type in ma_types:
            ma_results = {}

            try:
                if ma_type.lower() == "sma":
                    ma_results = self._calculate_sma(df, periods)

                elif ma_type.lower() == "ema":
                    ma_results = self._calculate_ema(df, periods)

                elif ma_type.lower() == "wma":
                    ma_results = self._calculate_wma(df, periods)

                elif ma_type.lower() == "vwma":
                    ma_results = self._calculate_vwma(df, periods)

                elif ma_type.lower() == "hull":
                    ma_results = self._calculate_hull_ma(df, periods)

                else:
                    self.logger.warning(f"Unknown moving average type: {ma_type}")
                    continue

                # Convert numpy arrays to lists
                for key, value in ma_results.items():
                    if isinstance(value, (np.ndarray, pd.Series)):
                        if hasattr(value, "tolist"):
                            ma_results[key] = value.tolist()
                        else:
                            ma_results[key] = list(value)

                results[ma_type] = ma_results

                # Check for crossovers within this MA type
                if len(periods) > 1:
                    for i in range(len(periods)):
                        for j in range(i + 1, len(periods)):
                            p1, p2 = periods[i], periods[j]
                            col1 = f"{ma_type.lower()}_{p1}"
                            col2 = f"{ma_type.lower()}_{p2}"

                            # Detect crossovers
                            ma_crossovers = self._detect_crossovers(df, col1, col2)

                            # Add to results
                            for cross in ma_crossovers:
                                crossovers.append(
                                    {
                                        "type": "crossover",
                                        "ma_type": ma_type,
                                        "fast_period": p1,
                                        "slow_period": p2,
                                        "direction": cross["direction"],
                                        "date": cross["date"],
                                        "index": cross["index"],
                                    }
                                )

            except Exception as e:
                self.logger.error(f"Error calculating {ma_type}: {e}")
                results[ma_type] = {"error": str(e)}

        # Check for crossovers between different MA types
        if len(ma_types) > 1:
            for i in range(len(ma_types)):
                for j in range(i + 1, len(ma_types)):
                    ma_type1, ma_type2 = ma_types[i], ma_types[j]

                    # Compare same periods across different MA types
                    for period in periods:
                        col1 = f"{ma_type1.lower()}_{period}"
                        col2 = f"{ma_type2.lower()}_{period}"

                        # Detect crossovers
                        type_crossovers = self._detect_crossovers(df, col1, col2)

                        # Add to results
                        for cross in type_crossovers:
                            crossovers.append(
                                {
                                    "type": "ma_type_crossover",
                                    "ma_type1": ma_type1,
                                    "ma_type2": ma_type2,
                                    "period": period,
                                    "direction": cross["direction"],
                                    "date": cross["date"],
                                    "index": cross["index"],
                                }
                            )

        return {
            "moving_averages": results,
            "crossovers": crossovers,
            "processing_time": time.time() - start_time,
        }

    def momentum_oscillators(
        self,
        price_data: List[Dict[str, Any]],
        oscillators: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate momentum oscillators like RSI, Stochastic, MACD.

        Args:
            price_data: List of price data dictionaries
            oscillators: List of oscillators to calculate
            parameters: Parameters for oscillators

        Returns:
            Dictionary of oscillator results and signals
        """
        start_time = time.time()

        # Default parameters
        default_params = {
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "stoch": {"k_period": 14, "d_period": 3},
            "cci": {"period": 20},
            "mfi": {"period": 14},
            "roc": {"period": 12},
            "williams_r": {"period": 14},
        }

        # Use provided parameters or defaults
        parameters = parameters or {}
        for osc in oscillators:
            if osc not in parameters and osc in default_params:
                parameters[osc] = default_params[osc]

        # Prepare DataFrame
        df = self._prepare_dataframe(price_data)

        results = {}
        signals = []

        for osc in oscillators:
            try:
                if osc == "rsi":
                    period = parameters.get("rsi", {}).get("period", 14)
                    rsi_results = self._calculate_rsi(df, period)
                    results["rsi"] = rsi_results

                    # Check for overbought/oversold conditions
                    for i in range(len(df) - 1):
                        if i < period:
                            continue

                        rsi_value = rsi_results["values"][i]
                        prev_rsi = rsi_results["values"][i - 1]

                        if prev_rsi > 70 and rsi_value <= 70:
                            signals.append(
                                {
                                    "type": "overbought_exit",
                                    "indicator": "RSI",
                                    "value": rsi_value,
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )
                        elif prev_rsi < 30 and rsi_value >= 30:
                            signals.append(
                                {
                                    "type": "oversold_exit",
                                    "indicator": "RSI",
                                    "value": rsi_value,
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )

                elif osc == "macd":
                    params = parameters.get("macd", {})
                    fast = params.get("fast", 12)
                    slow = params.get("slow", 26)
                    signal = params.get("signal", 9)

                    macd_results = self._calculate_macd(df, fast, slow, signal)
                    results["macd"] = macd_results

                    # Check for signal line crossovers
                    for i in range(len(df) - 1):
                        if i < max(slow, fast) + signal:
                            continue

                        macd_line = macd_results["macd_line"][i]
                        signal_line = macd_results["signal_line"][i]
                        prev_macd = macd_results["macd_line"][i - 1]
                        prev_signal = macd_results["signal_line"][i - 1]

                        if prev_macd < prev_signal and macd_line > signal_line:
                            signals.append(
                                {
                                    "type": "bullish_crossover",
                                    "indicator": "MACD",
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )
                        elif prev_macd > prev_signal and macd_line < signal_line:
                            signals.append(
                                {
                                    "type": "bearish_crossover",
                                    "indicator": "MACD",
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )

                elif osc == "stoch":
                    params = parameters.get("stoch", {})
                    k_period = params.get("k_period", 14)
                    d_period = params.get("d_period", 3)

                    stoch_results = self._calculate_stochastic(df, k_period, d_period)
                    results["stoch"] = stoch_results

                    # Check for overbought/oversold conditions
                    for i in range(len(df) - 1):
                        if i < k_period + d_period:
                            continue

                        k_val = stoch_results["k"][i]
                        d_val = stoch_results["d"][i]
                        prev_k = stoch_results["k"][i - 1]
                        prev_d = stoch_results["d"][i - 1]

                        if prev_k > 80 and k_val <= 80:
                            signals.append(
                                {
                                    "type": "stoch_overbought_exit",
                                    "indicator": "Stochastic",
                                    "k_value": k_val,
                                    "d_value": d_val,
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )
                        elif prev_k < 20 and k_val >= 20:
                            signals.append(
                                {
                                    "type": "stoch_oversold_exit",
                                    "indicator": "Stochastic",
                                    "k_value": k_val,
                                    "d_value": d_val,
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )

                        # Check for K-D crossovers
                        if prev_k < prev_d and k_val > d_val:
                            signals.append(
                                {
                                    "type": "bullish_stoch_crossover",
                                    "indicator": "Stochastic",
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )
                        elif prev_k > prev_d and k_val < d_val:
                            signals.append(
                                {
                                    "type": "bearish_stoch_crossover",
                                    "indicator": "Stochastic",
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                            )

                elif osc == "cci":
                    period = parameters.get("cci", {}).get("period", 20)
                    cci_results = self._calculate_cci(df, period)
                    results["cci"] = cci_results

                elif osc == "mfi":
                    period = parameters.get("mfi", {}).get("period", 14)
                    mfi_results = self._calculate_mfi(df, period)
                    results["mfi"] = mfi_results

                elif osc == "roc":
                    period = parameters.get("roc", {}).get("period", 12)
                    roc_results = self._calculate_roc(df, period)
                    results["roc"] = roc_results

                elif osc == "williams_r":
                    period = parameters.get("williams_r", {}).get("period", 14)
                    williams_results = self._calculate_williams_r(df, period)
                    results["williams_r"] = williams_results

                else:
                    self.logger.warning(f"Unknown oscillator: {osc}")

            except Exception as e:
                self.logger.error(f"Error calculating {osc}: {e}")
                results[osc] = {"error": str(e)}

        # Convert numpy arrays to lists
        for osc, data in results.items():
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series)):
                    if hasattr(value, "tolist"):
                        results[osc][key] = value.tolist()
                    else:
                        results[osc][key] = list(value)

        return {
            "oscillators": results,
            "signals": signals,
            "processing_time": time.time() - start_time,
        }

    # Implementations for other registered tools
    def volatility_indicators(
        self,
        price_data: List[Dict[str, Any]],
        indicators: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate volatility indicators like Bollinger Bands, ATR, etc.

        Args:
            price_data: List of price data dictionaries with required fields
            indicators: List of volatility indicators to calculate
            parameters: Parameters for the indicators (optional)

        Returns:
            Dictionary containing calculated indicators, signals, and metadata
        """
        start_time = time.time()

        # Log the request
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Calculating volatility indicators",
                indicator_count=len(indicators),
                data_points=len(price_data),
                indicators=",".join(indicators),
            )

        # Default parameters
        default_params = {
            "bbands": {"period": 20, "stddev": 2},
            "atr": {"period": 14},
            "keltner": {"ema_period": 20, "atr_period": 10, "atr_multiplier": 2},
            "donchian": {"period": 20},
            "historical_vol": {"period": 20, "annualization": 252},
        }

        # Use provided parameters or defaults
        parameters = parameters or {}
        for ind in indicators:
            if ind not in parameters and ind in default_params:
                parameters[ind] = default_params[ind]

        # Track GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                initial_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(
                    operation="volatility_initial"
                ).set(initial_mem)
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track initial GPU memory: {e}", error=str(e)
                    )

        # Prepare DataFrame
        try:
            df = self._prepare_dataframe(price_data)

            # Check if DataFrame is valid
            if df.empty:
                raise ValueError("Empty DataFrame created from input data")

        except Exception as e:
            error_msg = (
                f"Failed to prepare data for volatility indicator calculation: {str(e)}"
            )

            if hasattr(self, "monitor"):
                self.monitor.log_error(
                    error_msg,
                    error_type="data_preparation",
                    traceback=str(e),
                    data_points=len(price_data),
                )

                if hasattr(self, "metrics") and "errors_total" in self.metrics:
                    self.monitor.increment_counter(
                        "errors_total",
                        1,
                        type="data_preparation",
                        operation="volatility_indicators",
                    )

            return {
                "error": error_msg,
                "indicators": {},
                "signals": [],
                "metadata": {
                    "success": False,
                    "error": error_msg,
                    "indicators_requested": len(indicators),
                    "data_points": len(price_data),
                },
                "processing_time": time.time() - start_time,
            }

        # Setup result storage
        results = {}
        signals = []
        stats = {
            "indicators_requested": len(indicators),
            "data_points": len(price_data),
            "successful_indicators": 0,
            "signals_generated": 0,
            "errors": 0,
        }

        # Track which computations used GPU
        gpu_used = self.use_gpu and HAVE_CUDA

        # Calculate each requested indicator
        for indicator in indicators:
            indicator_start_time = time.time()

            try:
                # Bollinger Bands
                if indicator == "bbands":
                    bb_params = parameters.get("bbands", default_params["bbands"])
                    bb_results = self._calculate_bollinger_bands(
                        df, bb_params.get("period", 20), bb_params.get("stddev", 2)
                    )
                    results["bbands"] = bb_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Bollinger Bands indicator",
                            indicator="bbands",
                            period=bb_params.get("period", 20),
                            stddev=bb_params.get("stddev", 2),
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="bbands",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="bbands",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Check for price touching bands
                    bb_signals = []
                    for i in range(len(df) - 1):
                        if i < bb_params.get("period", 20):
                            continue

                        price = df["close"].iloc[i]
                        upper_band = bb_results["upper_band"][i]
                        lower_band = bb_results["lower_band"][i]

                        if price >= upper_band:
                            signal = {
                                "type": "upper_band_touch",
                                "indicator": "Bollinger Bands",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            bb_signals.append(signal)
                        elif price <= lower_band:
                            signal = {
                                "type": "lower_band_touch",
                                "indicator": "Bollinger Bands",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            bb_signals.append(signal)

                    if hasattr(self, "monitor") and bb_signals:
                        stats["signals_generated"] += len(bb_signals)
                        self.monitor.log_info(
                            f"Generated {len(bb_signals)} Bollinger Bands signals",
                            indicator="bbands",
                            signal_count=len(bb_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(bb_signals),
                                signal_type="band_touch",
                                indicator_type="bbands",
                            )

                # Average True Range
                elif indicator == "atr":
                    atr_params = parameters.get("atr", default_params["atr"])
                    atr_period = atr_params.get("period", 14)
                    atr_results = self._calculate_atr(df, atr_period)
                    results["atr"] = atr_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated ATR indicator",
                            indicator="atr",
                            period=atr_period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="atr",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="atr",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Keltner Channels
                elif indicator == "keltner":
                    keltner_params = parameters.get(
                        "keltner", default_params["keltner"]
                    )
                    ema_period = keltner_params.get("ema_period", 20)
                    atr_period = keltner_params.get("atr_period", 10)
                    atr_multiplier = keltner_params.get("atr_multiplier", 2)

                    # Calculate EMA for middle line
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        # For cuDF, manually implement EMA
                        alpha = 2 / (ema_period + 1)
                        ema = df["close"].rolling(window=ema_period).mean()
                        for i in range(ema_period, len(df)):
                            ema[i] = alpha * df["close"][i] + (1 - alpha) * ema[i - 1]
                    else:
                        ema = df["close"].ewm(span=ema_period, adjust=False).mean()

                    # Calculate ATR for channel width
                    atr_vals = self._calculate_atr(df, atr_period)["values"]

                    # Calculate upper and lower bands
                    upper_band = ema + (atr_multiplier * atr_vals)
                    lower_band = ema - (atr_multiplier * atr_vals)

                    keltner_results = {
                        "middle_line": ema,
                        "upper_band": upper_band,
                        "lower_band": lower_band,
                        "ema_period": ema_period,
                        "atr_period": atr_period,
                        "atr_multiplier": atr_multiplier,
                    }

                    results["keltner"] = keltner_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Keltner Channels indicator",
                            indicator="keltner",
                            ema_period=ema_period,
                            atr_period=atr_period,
                            atr_multiplier=atr_multiplier,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="keltner",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="keltner",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Donchian Channels
                elif indicator == "donchian":
                    donchian_params = parameters.get(
                        "donchian", default_params["donchian"]
                    )
                    period = donchian_params.get("period", 20)

                    if "high" not in df.columns or "low" not in df.columns:
                        raise ValueError(
                            "High and Low data required for Donchian Channels calculation"
                        )

                    # Calculate upper and lower bands
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        upper_band = df["high"].rolling(period).max()
                        lower_band = df["low"].rolling(period).min()
                    else:
                        upper_band = df["high"].rolling(window=period).max()
                        lower_band = df["low"].rolling(window=period).min()

                    # Calculate middle line
                    middle_line = (upper_band + lower_band) / 2

                    donchian_results = {
                        "upper_band": upper_band,
                        "middle_line": middle_line,
                        "lower_band": lower_band,
                        "period": period,
                    }

                    results["donchian"] = donchian_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Donchian Channels indicator",
                            indicator="donchian",
                            period=period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="donchian",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="donchian",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Historical Volatility
                elif indicator == "historical_vol":
                    hist_vol_params = parameters.get(
                        "historical_vol", default_params["historical_vol"]
                    )
                    period = hist_vol_params.get("period", 20)
                    annualization = hist_vol_params.get(
                        "annualization", 252
                    )  # Trading days in a year

                    # Calculate log returns
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        log_returns = cp.log(df["close"] / df["close"].shift(1))
                        rolling_std = log_returns.rolling(period).std()
                        hist_vol = rolling_std * cp.sqrt(annualization)
                    else:
                        log_returns = np.log(df["close"] / df["close"].shift(1))
                        rolling_std = log_returns.rolling(window=period).std()
                        hist_vol = rolling_std * np.sqrt(annualization)

                    historical_vol_results = {
                        "values": hist_vol,
                        "period": period,
                        "annualization": annualization,
                    }

                    results["historical_vol"] = historical_vol_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Historical Volatility indicator",
                            indicator="historical_vol",
                            period=period,
                            annualization=annualization,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="historical_vol",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="historical_vol",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                else:
                    warning_msg = f"Unknown volatility indicator: {indicator}"
                    self.logger.warning(warning_msg)
                    if hasattr(self, "monitor"):
                        self.monitor.log_warning(warning_msg, indicator=indicator)

            except Exception as e:
                error_msg = f"Error calculating {indicator}: {str(e)}"
                self.logger.error(error_msg)
                stats["errors"] += 1
                results[indicator] = {"error": str(e)}

                if hasattr(self, "monitor"):
                    self.monitor.log_error(
                        error_msg,
                        indicator=indicator,
                        error_type="calculation_error",
                        traceback=str(e),
                    )

                    if hasattr(self, "metrics") and "errors_total" in self.metrics:
                        self.monitor.increment_counter(
                            "errors_total",
                            1,
                            type="indicator_calculation",
                            indicator=indicator,
                        )

        # Track final GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                final_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(
                    operation="volatility_final"
                ).set(final_mem)
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track final GPU memory: {e}", error=str(e)
                    )

        # Convert numpy arrays to lists for JSON serialization
        for indicator, data in results.items():
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series)):
                    if hasattr(value, "tolist"):
                        results[indicator][key] = value.tolist()
                    else:
                        results[indicator][key] = list(value)

        # Create metadata
        metadata = {
            "indicators_calculated": list(results.keys()),
            "data_points": len(df),
            "signals_found": len(signals),
            "start_date": df["date"].iloc[0].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "end_date": df["date"].iloc[-1].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "calculation_stats": stats,
            "gpu_used": gpu_used,
        }

        total_time = time.time() - start_time

        # Log overall performance
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Completed volatility indicator calculations",
                duration=total_time,
                indicators_requested=stats["indicators_requested"],
                indicators_calculated=stats["successful_indicators"],
                signals_generated=stats["signals_generated"],
                errors=stats["errors"],
                gpu_used=gpu_used,
            )

        return {
            "indicators": results,
            "signals": signals,
            "metadata": metadata,
            "processing_time": total_time,
        }

    def volume_indicators(
        self,
        price_data: List[Dict[str, Any]],
        indicators: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate volume-based indicators like OBV, Money Flow, VWAP, etc.

        Args:
            price_data: List of price data dictionaries with 'close' and 'volume' fields
            indicators: List of volume indicators to calculate
            parameters: Parameters for the indicators (optional)

        Returns:
            Dictionary containing calculated indicators, signals, and metadata
        """
        start_time = time.time()

        # Log the request
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Calculating volume indicators",
                indicator_count=len(indicators),
                data_points=len(price_data),
                indicators=",".join(indicators),
            )

        # Default parameters
        default_params = {
            "obv": {},  # On-Balance Volume doesn't need parameters
            "vwap": {},  # Volume Weighted Average Price doesn't need parameters
            "mfi": {"period": 14},  # Money Flow Index
            "cmf": {"period": 20},  # Chaikin Money Flow
            "ad": {},  # Accumulation/Distribution Line
            "pvt": {},  # Price Volume Trend
        }

        # Use provided parameters or defaults
        parameters = parameters or {}
        for ind in indicators:
            if ind not in parameters and ind in default_params:
                parameters[ind] = default_params[ind]

        # Track GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                initial_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(operation="volume_initial").set(
                    initial_mem
                )
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track initial GPU memory: {e}", error=str(e)
                    )

        # Prepare DataFrame
        try:
            df = self._prepare_dataframe(price_data)

            # Check if DataFrame is valid
            if df.empty:
                raise ValueError("Empty DataFrame created from input data")

            # Check if required columns exist
            if "volume" not in df.columns:
                raise ValueError("Volume data is required for volume indicators")

        except Exception as e:
            error_msg = (
                f"Failed to prepare data for volume indicator calculation: {str(e)}"
            )

            if hasattr(self, "monitor"):
                self.monitor.log_error(
                    error_msg,
                    error_type="data_preparation",
                    traceback=str(e),
                    data_points=len(price_data),
                )

                if hasattr(self, "metrics") and "errors_total" in self.metrics:
                    self.monitor.increment_counter(
                        "errors_total",
                        1,
                        type="data_preparation",
                        operation="volume_indicators",
                    )

            return {
                "error": error_msg,
                "indicators": {},
                "signals": [],
                "metadata": {
                    "success": False,
                    "error": error_msg,
                    "indicators_requested": len(indicators),
                    "data_points": len(price_data),
                },
                "processing_time": time.time() - start_time,
            }

        # Setup result storage
        results = {}
        signals = []
        stats = {
            "indicators_requested": len(indicators),
            "data_points": len(price_data),
            "successful_indicators": 0,
            "signals_generated": 0,
            "errors": 0,
        }

        # Track which computations used GPU
        gpu_used = self.use_gpu and HAVE_CUDA

        # Calculate each requested indicator
        for indicator in indicators:
            indicator_start_time = time.time()

            try:
                # On-Balance Volume (OBV)
                if indicator == "obv":
                    obv_results = self._calculate_obv(df)
                    results["obv"] = obv_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated OBV indicator",
                            indicator="obv",
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="obv",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="obv",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Volume Weighted Average Price (VWAP)
                elif indicator == "vwap":
                    vwap_results = self._calculate_vwap(df)
                    results["vwap"] = vwap_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated VWAP indicator",
                            indicator="vwap",
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="vwap",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="vwap",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Money Flow Index (MFI)
                elif indicator == "mfi":
                    mfi_params = parameters.get("mfi", default_params["mfi"])
                    period = mfi_params.get("period", 14)

                    if "high" not in df.columns or "low" not in df.columns:
                        raise ValueError(
                            "High and Low price data required for MFI calculation"
                        )

                    # Calculate typical price
                    typical_price = (df["high"] + df["low"] + df["close"]) / 3

                    # Calculate money flow
                    money_flow = typical_price * df["volume"]

                    # Get positives and negatives
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        positive_flow = cudf.Series(
                            cp.where(
                                typical_price > typical_price.shift(1), money_flow, 0
                            )
                        )
                        negative_flow = cudf.Series(
                            cp.where(
                                typical_price < typical_price.shift(1), money_flow, 0
                            )
                        )
                    else:
                        positive_flow = pd.Series(
                            np.where(
                                typical_price > typical_price.shift(1), money_flow, 0
                            )
                        )
                        negative_flow = pd.Series(
                            np.where(
                                typical_price < typical_price.shift(1), money_flow, 0
                            )
                        )

                    # Calculate money ratio and MFI
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        positive_sum = positive_flow.rolling(period).sum()
                        negative_sum = negative_flow.rolling(period).sum()
                        money_ratio = positive_sum / negative_sum.replace(
                            0, 1e-10
                        )  # Avoid division by zero
                        mfi = 100 - (100 / (1 + money_ratio))
                    else:
                        positive_sum = positive_flow.rolling(window=period).sum()
                        negative_sum = negative_flow.rolling(window=period).sum()
                        money_ratio = positive_sum / negative_sum.replace(
                            0, 1e-10
                        )  # Avoid division by zero
                        mfi = 100 - (100 / (1 + money_ratio))

                    mfi_results = {"values": mfi, "period": period}

                    results["mfi"] = mfi_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated MFI indicator",
                            indicator="mfi",
                            period=period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="mfi",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="mfi",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Generate signals for MFI (overbought/oversold)
                    mfi_signals = []
                    for i in range(len(df) - 1):
                        if i < period:
                            continue

                        mfi_value = mfi.iloc[i]
                        prev_mfi = mfi.iloc[i - 1]

                        if prev_mfi > 80 and mfi_value <= 80:
                            signal = {
                                "type": "overbought_exit",
                                "indicator": "MFI",
                                "value": float(mfi_value),
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            mfi_signals.append(signal)
                        elif prev_mfi < 20 and mfi_value >= 20:
                            signal = {
                                "type": "oversold_exit",
                                "indicator": "MFI",
                                "value": float(mfi_value),
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            mfi_signals.append(signal)
                        elif prev_mfi <= 80 and mfi_value > 80:
                            signal = {
                                "type": "overbought_enter",
                                "indicator": "MFI",
                                "value": float(mfi_value),
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            mfi_signals.append(signal)
                        elif prev_mfi >= 20 and mfi_value < 20:
                            signal = {
                                "type": "oversold_enter",
                                "indicator": "MFI",
                                "value": float(mfi_value),
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            mfi_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and mfi_signals:
                        stats["signals_generated"] += len(mfi_signals)
                        self.monitor.log_info(
                            f"Generated {len(mfi_signals)} MFI signals",
                            indicator="mfi",
                            signal_count=len(mfi_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(mfi_signals),
                                signal_type="overbought_oversold",
                                indicator_type="mfi",
                            )

                # Chaikin Money Flow (CMF)
                elif indicator == "cmf":
                    cmf_params = parameters.get("cmf", default_params["cmf"])
                    period = cmf_params.get("period", 20)

                    if "high" not in df.columns or "low" not in df.columns:
                        raise ValueError(
                            "High and Low price data required for CMF calculation"
                        )

                    # Calculate Money Flow Multiplier
                    high_low_range = df["high"] - df["low"]
                    close_low = df["close"] - df["low"]
                    high_close = df["high"] - df["close"]

                    # Avoid division by zero
                    money_flow_multiplier = (
                        (close_low - high_close) / (high_low_range + 1e-10)
                    ).fillna(0)
                    money_flow_volume = money_flow_multiplier * df["volume"]

                    # Calculate CMF
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        cmf = money_flow_volume.rolling(period).sum() / df[
                            "volume"
                        ].rolling(period).sum().replace(0, 1e-10)
                    else:
                        cmf = money_flow_volume.rolling(window=period).sum() / df[
                            "volume"
                        ].rolling(window=period).sum().replace(0, 1e-10)

                    cmf_results = {"values": cmf, "period": period}

                    results["cmf"] = cmf_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated CMF indicator",
                            indicator="cmf",
                            period=period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="cmf",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="cmf",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Accumulation/Distribution Line (AD)
                elif indicator == "ad":
                    if "high" not in df.columns or "low" not in df.columns:
                        raise ValueError(
                            "High and Low price data required for AD calculation"
                        )

                    # Calculate Money Flow Multiplier
                    high_low_range = df["high"] - df["low"]
                    close_low = df["close"] - df["low"]
                    high_close = df["high"] - df["close"]

                    # Avoid division by zero
                    money_flow_multiplier = (
                        (close_low - high_close) / (high_low_range + 1e-10)
                    ).fillna(0)
                    money_flow_volume = money_flow_multiplier * df["volume"]

                    # Calculate AD Line (cumulative sum of money flow volume)
                    ad_line = money_flow_volume.cumsum()

                    ad_results = {"values": ad_line}

                    results["ad"] = ad_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated AD Line indicator",
                            indicator="ad",
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="ad",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="ad",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                # Price Volume Trend (PVT)
                elif indicator == "pvt":
                    # Calculate percentage price change
                    price_change_pct = (df["close"] - df["close"].shift(1)) / df[
                        "close"
                    ].shift(1).replace(0, 1e-10)

                    # Calculate PVT values
                    pvt_values = price_change_pct * df["volume"]
                    pvt_line = pvt_values.cumsum()

                    pvt_results = {"values": pvt_line}

                    results["pvt"] = pvt_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated PVT indicator",
                            indicator="pvt",
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="pvt",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="pvt",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                else:
                    warning_msg = f"Unknown volume indicator: {indicator}"
                    self.logger.warning(warning_msg)
                    if hasattr(self, "monitor"):
                        self.monitor.log_warning(warning_msg, indicator=indicator)

            except Exception as e:
                error_msg = f"Error calculating {indicator}: {str(e)}"
                self.logger.error(error_msg)
                stats["errors"] += 1
                results[indicator] = {"error": str(e)}

                if hasattr(self, "monitor"):
                    self.monitor.log_error(
                        error_msg,
                        indicator=indicator,
                        error_type="calculation_error",
                        traceback=str(e),
                    )

                    if hasattr(self, "metrics") and "errors_total" in self.metrics:
                        self.monitor.increment_counter(
                            "errors_total",
                            1,
                            type="indicator_calculation",
                            indicator=indicator,
                        )

        # Track final GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                final_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(operation="volume_final").set(
                    final_mem
                )
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track final GPU memory: {e}", error=str(e)
                    )

        # Convert numpy arrays to lists for JSON serialization
        for indicator, data in results.items():
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series)):
                    if hasattr(value, "tolist"):
                        results[indicator][key] = value.tolist()
                    else:
                        results[indicator][key] = list(value)

        # Create metadata
        metadata = {
            "indicators_calculated": list(results.keys()),
            "data_points": len(df),
            "signals_found": len(signals),
            "start_date": df["date"].iloc[0].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "end_date": df["date"].iloc[-1].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "calculation_stats": stats,
            "gpu_used": gpu_used,
        }

        total_time = time.time() - start_time

        # Log overall performance
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Completed volume indicator calculations",
                duration=total_time,
                indicators_requested=stats["indicators_requested"],
                indicators_calculated=stats["successful_indicators"],
                signals_generated=stats["signals_generated"],
                errors=stats["errors"],
                gpu_used=gpu_used,
            )

        return {
            "indicators": results,
            "signals": signals,
            "metadata": metadata,
            "processing_time": total_time,
        }

    def trend_indicators(
        self,
        price_data: List[Dict[str, Any]],
        indicators: List[str],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate trend indicators like ADX, Ichimoku Cloud, SuperTrend, etc.

        Args:
            price_data: List of price data dictionaries with required fields
            indicators: List of trend indicators to calculate
            parameters: Parameters for the indicators (optional)

        Returns:
            Dictionary containing calculated indicators, signals, and metadata
        """
        start_time = time.time()

        # Log the request
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Calculating trend indicators",
                indicator_count=len(indicators),
                data_points=len(price_data),
                indicators=",".join(indicators),
            )

        # Default parameters
        default_params = {
            "adx": {"period": 14},  # Average Directional Index
            "ichimoku": {  # Ichimoku Cloud parameters
                "tenkan_period": 9,
                "kijun_period": 26,
                "senkou_b_period": 52,
                "displacement": 26,
            },
            "supertrend": {  # SuperTrend parameters
                "period": 10,
                "multiplier": 3,
            },
            "parabolic_sar": {  # Parabolic SAR parameters
                "initial_af": 0.02,
                "max_af": 0.2,
            },
            "zigzag": {"pct_change": 5},  # ZigZag percent change
            "aroon": {"period": 25},  # Aroon indicator period
        }

        # Use provided parameters or defaults
        parameters = parameters or {}
        for ind in indicators:
            if ind not in parameters and ind in default_params:
                parameters[ind] = default_params[ind]

        # Track GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                initial_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(operation="trend_initial").set(
                    initial_mem
                )
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track initial GPU memory: {e}", error=str(e)
                    )

        # Prepare DataFrame
        try:
            df = self._prepare_dataframe(price_data)

            # Check if DataFrame is valid
            if df.empty:
                raise ValueError("Empty DataFrame created from input data")

            # Check if required columns exist for certain indicators
            has_ohlc = all(
                col in df.columns for col in ["open", "high", "low", "close"]
            )
            if not has_ohlc and any(
                ind in indicators for ind in ["ichimoku", "supertrend", "parabolic_sar"]
            ):
                raise ValueError("OHLC data required for selected trend indicators")

        except Exception as e:
            error_msg = (
                f"Failed to prepare data for trend indicator calculation: {str(e)}"
            )

            if hasattr(self, "monitor"):
                self.monitor.log_error(
                    error_msg,
                    error_type="data_preparation",
                    traceback=str(e),
                    data_points=len(price_data),
                )

                if hasattr(self, "metrics") and "errors_total" in self.metrics:
                    self.monitor.increment_counter(
                        "errors_total",
                        1,
                        type="data_preparation",
                        operation="trend_indicators",
                    )

            return {
                "error": error_msg,
                "indicators": {},
                "signals": [],
                "metadata": {
                    "success": False,
                    "error": error_msg,
                    "indicators_requested": len(indicators),
                    "data_points": len(price_data),
                },
                "processing_time": time.time() - start_time,
            }

        # Setup result storage
        results = {}
        signals = []
        stats = {
            "indicators_requested": len(indicators),
            "data_points": len(price_data),
            "successful_indicators": 0,
            "signals_generated": 0,
            "errors": 0,
        }

        # Track which computations used GPU
        gpu_used = self.use_gpu and HAVE_CUDA

        # Calculate each requested indicator
        for indicator in indicators:
            indicator_start_time = time.time()

            try:
                # Average Directional Index (ADX)
                if indicator == "adx":
                    adx_params = parameters.get("adx", default_params["adx"])
                    period = adx_params.get("period", 14)

                    adx_results = self._calculate_adx(df, period)
                    results["adx"] = adx_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated ADX indicator",
                            indicator="adx",
                            period=period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="adx",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="adx",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Generate ADX signals for trend strength changes
                    adx_signals = []
                    for i in range(len(df) - 1):
                        if i < period * 2:  # ADX needs more data points
                            continue

                        adx_value = adx_results["adx"][i]
                        prev_adx = adx_results["adx"][i - 1]

                        if prev_adx < 25 and adx_value >= 25:
                            signal = {
                                "type": "trend_strengthening",
                                "indicator": "ADX",
                                "value": adx_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            adx_signals.append(signal)
                        elif prev_adx > 25 and adx_value <= 25:
                            signal = {
                                "type": "trend_weakening",
                                "indicator": "ADX",
                                "value": adx_value,
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            adx_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and adx_signals:
                        stats["signals_generated"] += len(adx_signals)
                        self.monitor.log_info(
                            f"Generated {len(adx_signals)} ADX signals",
                            indicator="adx",
                            signal_count=len(adx_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(adx_signals),
                                signal_type="trend",
                                indicator_type="adx",
                            )

                # Ichimoku Cloud
                elif indicator == "ichimoku":
                    if not has_ohlc:
                        raise ValueError(
                            "OHLC data required for Ichimoku Cloud calculation"
                        )

                    ichimoku_params = parameters.get(
                        "ichimoku", default_params["ichimoku"]
                    )
                    tenkan_period = ichimoku_params.get("tenkan_period", 9)
                    kijun_period = ichimoku_params.get("kijun_period", 26)
                    senkou_b_period = ichimoku_params.get("senkou_b_period", 52)
                    displacement = ichimoku_params.get("displacement", 26)

                    # Function to get average of highest high and lowest low
                    def donchian_midpoint(high_series, low_series, period):
                        if (
                            self.use_gpu
                            and HAVE_CUDA
                            and (
                                isinstance(high_series, cudf.Series)
                                and isinstance(low_series, cudf.Series)
                            )
                        ):
                            highest_high = high_series.rolling(period).max()
                            lowest_low = low_series.rolling(period).min()
                        else:
                            highest_high = high_series.rolling(window=period).max()
                            lowest_low = low_series.rolling(window=period).min()
                        return (highest_high + lowest_low) / 2

                    # Calculate Tenkan-sen (Conversion Line)
                    tenkan_sen = donchian_midpoint(df["high"], df["low"], tenkan_period)

                    # Calculate Kijun-sen (Base Line)
                    kijun_sen = donchian_midpoint(df["high"], df["low"], kijun_period)

                    # Calculate Senkou Span A (Leading Span A)
                    senkou_span_a = (tenkan_sen + kijun_sen) / 2

                    # Calculate Senkou Span B (Leading Span B)
                    senkou_span_b = donchian_midpoint(
                        df["high"], df["low"], senkou_b_period
                    )

                    #
                    # Calculate Chikou Span (Lagging Span) - Closing price
                    # shifted back
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        chikou_span = df["close"].shift(-displacement)
                    else:
                        chikou_span = df["close"].shift(-displacement)

                    # Store results
                    ichimoku_results = {
                        "tenkan_sen": tenkan_sen,
                        "kijun_sen": kijun_sen,
                        "senkou_span_a": senkou_span_a,
                        "senkou_span_b": senkou_span_b,
                        "chikou_span": chikou_span,
                        "parameters": {
                            "tenkan_period": tenkan_period,
                            "kijun_period": kijun_period,
                            "senkou_b_period": senkou_b_period,
                            "displacement": displacement,
                        },
                    }

                    results["ichimoku"] = ichimoku_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Ichimoku Cloud indicator",
                            indicator="ichimoku",
                            tenkan_period=tenkan_period,
                            kijun_period=kijun_period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="ichimoku",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="ichimoku",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Generate Ichimoku signals
                    ichimoku_signals = []
                    for i in range(len(df) - 1):
                        if i < max(tenkan_period, kijun_period, senkou_b_period):
                            continue

                        # Tenkan/Kijun Cross
                        if (
                            tenkan_sen.iloc[i - 1] < kijun_sen.iloc[i - 1]
                            and tenkan_sen.iloc[i] >= kijun_sen.iloc[i]
                        ):
                            signal = {
                                "type": "bullish_tk_cross",
                                "indicator": "Ichimoku",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            ichimoku_signals.append(signal)
                        elif (
                            tenkan_sen.iloc[i - 1] > kijun_sen.iloc[i - 1]
                            and tenkan_sen.iloc[i] <= kijun_sen.iloc[i]
                        ):
                            signal = {
                                "type": "bearish_tk_cross",
                                "indicator": "Ichimoku",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            ichimoku_signals.append(signal)

                        # Price crossing Kumo (Cloud)
                        if i + displacement < len(
                            senkou_span_a
                        ) and i + displacement < len(senkou_span_b):
                            cloud_top = max(
                                senkou_span_a.iloc[i + displacement],
                                senkou_span_b.iloc[i + displacement],
                            )
                            cloud_bottom = min(
                                senkou_span_a.iloc[i + displacement],
                                senkou_span_b.iloc[i + displacement],
                            )

                            if (
                                df["close"].iloc[i - 1] <= cloud_bottom
                                and df["close"].iloc[i] > cloud_bottom
                            ):
                                signal = {
                                    "type": "price_above_kumo",
                                    "indicator": "Ichimoku",
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                                signals.append(signal)
                                ichimoku_signals.append(signal)
                            elif (
                                df["close"].iloc[i - 1] >= cloud_top
                                and df["close"].iloc[i] < cloud_top
                            ):
                                signal = {
                                    "type": "price_below_kumo",
                                    "indicator": "Ichimoku",
                                    "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                    if "date" in df.columns
                                    else i,
                                    "index": i,
                                }
                                signals.append(signal)
                                ichimoku_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and ichimoku_signals:
                        stats["signals_generated"] += len(ichimoku_signals)
                        self.monitor.log_info(
                            f"Generated {len(ichimoku_signals)} Ichimoku signals",
                            indicator="ichimoku",
                            signal_count=len(ichimoku_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(ichimoku_signals),
                                signal_type="trend",
                                indicator_type="ichimoku",
                            )

                # SuperTrend
                elif indicator == "supertrend":
                    if not has_ohlc:
                        raise ValueError(
                            "OHLC data required for SuperTrend calculation"
                        )

                    supertrend_params = parameters.get(
                        "supertrend", default_params["supertrend"]
                    )
                    period = supertrend_params.get("period", 10)
                    multiplier = supertrend_params.get("multiplier", 3)

                    # Step 1: Calculate ATR
                    atr_values = self._calculate_atr(df, period)["values"]

                    # Step 2: Calculate Basic Upper and Lower Bands
                    hl2 = (df["high"] + df["low"]) / 2
                    basic_upper_band = hl2 + (multiplier * atr_values)
                    basic_lower_band = hl2 - (multiplier * atr_values)

                    # Step 3: Calculate SuperTrend
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        #
                        # For GPU, implement SuperTrend calculation with custom
                        # cuDF/cuPy operations
                        #
                        # This is simplified and should be expanded for
                        # production
                        final_upper_band = cp.zeros(len(df))
                        final_lower_band = cp.zeros(len(df))
                        supertrend = cp.zeros(len(df))
                        direction = cp.zeros(len(df))

                        # Manual calculation (simplified for illustration)
                        for i in range(1, len(df)):
                            if i < period:
                                continue

                            final_upper_band[i] = (
                                min(basic_upper_band[i], final_upper_band[i - 1])
                                if df["close"][i - 1] > final_upper_band[i - 1]
                                else basic_upper_band[i]
                            )
                            final_lower_band[i] = (
                                max(basic_lower_band[i], final_lower_band[i - 1])
                                if df["close"][i - 1] < final_lower_band[i - 1]
                                else basic_lower_band[i]
                            )

                            # Determine trend direction (1 for up, -1 for down)
                            if df["close"][i] > final_upper_band[i - 1]:
                                direction[i] = 1
                            elif df["close"][i] < final_lower_band[i - 1]:
                                direction[i] = -1
                            else:
                                direction[i] = direction[i - 1]

                            # Set SuperTrend value based on direction
                            supertrend[i] = (
                                final_lower_band[i]
                                if direction[i] == 1
                                else final_upper_band[i]
                            )

                    else:
                        # For CPU, implement SuperTrend with pandas
                        final_upper_band = pd.Series(index=df.index)
                        final_lower_band = pd.Series(index=df.index)
                        supertrend = pd.Series(index=df.index)
                        direction = pd.Series(index=df.index)

                        # Initialize first values
                        final_upper_band.iloc[0] = basic_upper_band.iloc[0]
                        final_lower_band.iloc[0] = basic_lower_band.iloc[0]
                        supertrend.iloc[0] = (
                            final_upper_band.iloc[0] + final_lower_band.iloc[0]
                        ) / 2
                        direction.iloc[0] = 0

                        # Calculate SuperTrend
                        for i in range(1, len(df)):
                            # Final Upper Band
                            if (
                                basic_upper_band.iloc[i] < final_upper_band.iloc[i - 1]
                                or df["close"].iloc[i - 1]
                                > final_upper_band.iloc[i - 1]
                            ):
                                final_upper_band.iloc[i] = basic_upper_band.iloc[i]
                            else:
                                final_upper_band.iloc[i] = final_upper_band.iloc[i - 1]

                            # Final Lower Band
                            if (
                                basic_lower_band.iloc[i] > final_lower_band.iloc[i - 1]
                                or df["close"].iloc[i - 1]
                                < final_lower_band.iloc[i - 1]
                            ):
                                final_lower_band.iloc[i] = basic_lower_band.iloc[i]
                            else:
                                final_lower_band.iloc[i] = final_lower_band.iloc[i - 1]

                            # SuperTrend Direction
                            if (
                                supertrend.iloc[i - 1] == final_upper_band.iloc[i - 1]
                                and df["close"].iloc[i] > final_upper_band.iloc[i]
                            ):
                                direction.iloc[i] = 1  # Uptrend
                            elif (
                                supertrend.iloc[i - 1] == final_lower_band.iloc[i - 1]
                                and df["close"].iloc[i] < final_lower_band.iloc[i]
                            ):
                                direction.iloc[i] = -1  # Downtrend
                            else:
                                direction.iloc[i] = direction.iloc[
                                    i - 1
                                ]  # Continue previous trend

                            # Set SuperTrend Value
                            if direction.iloc[i] == 1:
                                supertrend.iloc[i] = final_lower_band.iloc[i]
                            else:
                                supertrend.iloc[i] = final_upper_band.iloc[i]

                    supertrend_results = {
                        "supertrend": supertrend,
                        "upper_band": final_upper_band,
                        "lower_band": final_lower_band,
                        "direction": direction,
                        "period": period,
                        "multiplier": multiplier,
                    }

                    results["supertrend"] = supertrend_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated SuperTrend indicator",
                            indicator="supertrend",
                            period=period,
                            multiplier=multiplier,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="supertrend",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="supertrend",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Generate SuperTrend signals (trend changes)
                    supertrend_signals = []
                    for i in range(1, len(df)):
                        if i < period:
                            continue

                        if direction[i] == 1 and direction[i - 1] == -1:
                            signal = {
                                "type": "bullish_trend_change",
                                "indicator": "SuperTrend",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            supertrend_signals.append(signal)
                        elif direction[i] == -1 and direction[i - 1] == 1:
                            signal = {
                                "type": "bearish_trend_change",
                                "indicator": "SuperTrend",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            supertrend_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and supertrend_signals:
                        stats["signals_generated"] += len(supertrend_signals)
                        self.monitor.log_info(
                            f"Generated {len(supertrend_signals)} SuperTrend signals",
                            indicator="supertrend",
                            signal_count=len(supertrend_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(supertrend_signals),
                                signal_type="trend",
                                indicator_type="supertrend",
                            )

                # Aroon Indicator
                elif indicator == "aroon":
                    aroon_params = parameters.get("aroon", default_params["aroon"])
                    period = aroon_params.get("period", 25)

                    # Calculate Aroon Up and Down
                    if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                        # For GPU
                        high_max = df["high"].rolling(period).max()
                        low_min = df["low"].rolling(period).min()

                        # Get indices of max and min values
                        high_days = cp.zeros(len(df))
                        low_days = cp.zeros(len(df))

                        for i in range(period, len(df)):
                            window_high = df["high"].iloc[i - period + 1 : i + 1].values
                            window_low = df["low"].iloc[i - period + 1 : i + 1].values

                            high_pos = cp.argmax(window_high)
                            low_pos = cp.argmin(window_low)

                            high_days[i] = high_pos
                            low_days[i] = low_pos

                        # Calculate Aroon indicators
                        aroon_up = 100 * (period - high_days) / period
                        aroon_down = 100 * (period - low_days) / period
                        aroon_oscillator = aroon_up - aroon_down
                    else:
                        # For CPU
                        high_max = df["high"].rolling(window=period).max()
                        low_min = df["low"].rolling(window=period).min()

                        # Get days since high and low
                        high_days = np.zeros(len(df))
                        low_days = np.zeros(len(df))

                        for i in range(period, len(df)):
                            # Get last 'period' values
                            high_val = high_max.iloc[i]
                            low_val = low_min.iloc[i]

                            # Find the most recent high/low positions
                            for j in range(period):
                                pos = i - j
                                if pos >= 0:
                                    if df["high"].iloc[pos] == high_val:
                                        high_days[i] = j
                                        break

                            for j in range(period):
                                pos = i - j
                                if pos >= 0:
                                    if df["low"].iloc[pos] == low_val:
                                        low_days[i] = j
                                        break

                        # Calculate Aroon indicators
                        aroon_up = 100 * (period - high_days) / period
                        aroon_down = 100 * (period - low_days) / period
                        aroon_oscillator = aroon_up - aroon_down

                    aroon_results = {
                        "aroon_up": aroon_up,
                        "aroon_down": aroon_down,
                        "aroon_oscillator": aroon_oscillator,
                        "period": period,
                    }

                    results["aroon"] = aroon_results
                    stats["successful_indicators"] += 1

                    # Log successful calculation
                    if hasattr(self, "monitor"):
                        self.monitor.log_info(
                            "Calculated Aroon indicator",
                            indicator="aroon",
                            period=period,
                            duration=time.time() - indicator_start_time,
                        )

                        if (
                            hasattr(self, "metrics")
                            and "indicators_calculated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "indicators_calculated",
                                1,
                                indicator_type="aroon",
                                gpu_used=str(gpu_used),
                            )

                    # Record processing time
                    if (
                        hasattr(self, "monitor")
                        and hasattr(self, "metrics")
                        and "indicator_processing_time" in self.metrics
                    ):
                        self.monitor.observe_histogram(
                            "indicator_processing_time",
                            time.time() - indicator_start_time,
                            indicator_type="aroon",
                            data_size=str(len(price_data)),
                            gpu_used=str(gpu_used),
                        )

                    # Generate Aroon signals
                    aroon_signals = []
                    for i in range(period + 1, len(df)):
                        # Bullish crossover (Up crosses above Down)
                        if (
                            aroon_up[i - 1] < aroon_down[i - 1]
                            and aroon_up[i] > aroon_down[i]
                        ):
                            signal = {
                                "type": "bullish_crossover",
                                "indicator": "Aroon",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            aroon_signals.append(signal)
                        # Bearish crossover (Down crosses above Up)
                        elif (
                            aroon_up[i - 1] > aroon_down[i - 1]
                            and aroon_up[i] < aroon_down[i]
                        ):
                            signal = {
                                "type": "bearish_crossover",
                                "indicator": "Aroon",
                                "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                                if "date" in df.columns
                                else i,
                                "index": i,
                            }
                            signals.append(signal)
                            aroon_signals.append(signal)

                    # Log signals generated
                    if hasattr(self, "monitor") and aroon_signals:
                        stats["signals_generated"] += len(aroon_signals)
                        self.monitor.log_info(
                            f"Generated {len(aroon_signals)} Aroon signals",
                            indicator="aroon",
                            signal_count=len(aroon_signals),
                        )

                        if (
                            hasattr(self, "metrics")
                            and "signals_generated" in self.metrics
                        ):
                            self.monitor.increment_counter(
                                "signals_generated",
                                len(aroon_signals),
                                signal_type="crossover",
                                indicator_type="aroon",
                            )

                else:
                    warning_msg = f"Unknown trend indicator: {indicator}"
                    self.logger.warning(warning_msg)
                    if hasattr(self, "monitor"):
                        self.monitor.log_warning(warning_msg, indicator=indicator)

            except Exception as e:
                error_msg = f"Error calculating {indicator}: {str(e)}"
                self.logger.error(error_msg)
                stats["errors"] += 1
                results[indicator] = {"error": str(e)}

                if hasattr(self, "monitor"):
                    self.monitor.log_error(
                        error_msg,
                        indicator=indicator,
                        error_type="calculation_error",
                        traceback=str(e),
                    )

                    if hasattr(self, "metrics") and "errors_total" in self.metrics:
                        self.monitor.increment_counter(
                            "errors_total",
                            1,
                            type="indicator_calculation",
                            indicator=indicator,
                        )

        # Track final GPU memory usage if available
        if HAVE_CUDA and hasattr(self, "metrics") and "gpu_memory_used" in self.metrics:
            try:
                final_mem = cp.cuda.memory_info()[0]
                self.metrics["gpu_memory_used"].labels(operation="trend_final").set(
                    final_mem
                )
            except Exception as e:
                if hasattr(self, "monitor"):
                    self.monitor.log_warning(
                        f"Failed to track final GPU memory: {e}", error=str(e)
                    )

        # Convert numpy arrays to lists for JSON serialization
        for indicator, data in results.items():
            for key, value in data.items():
                if isinstance(value, (np.ndarray, pd.Series)):
                    if hasattr(value, "tolist"):
                        results[indicator][key] = value.tolist()
                    else:
                        results[indicator][key] = list(value)

        # Create metadata
        metadata = {
            "indicators_calculated": list(results.keys()),
            "data_points": len(df),
            "signals_found": len(signals),
            "start_date": df["date"].iloc[0].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "end_date": df["date"].iloc[-1].strftime("%Y-%m-%d")
            if "date" in df.columns and len(df) > 0
            else None,
            "calculation_stats": stats,
            "gpu_used": gpu_used,
        }

        total_time = time.time() - start_time

        # Log overall performance
        if hasattr(self, "monitor"):
            self.monitor.log_info(
                "Completed trend indicator calculations",
                duration=total_time,
                indicators_requested=stats["indicators_requested"],
                indicators_calculated=stats["successful_indicators"],
                signals_generated=stats["signals_generated"],
                errors=stats["errors"],
                gpu_used=gpu_used,
            )

        return {
            "indicators": results,
            "signals": signals,
            "metadata": metadata,
            "processing_time": total_time,
        }

    # Indicator calculation methods
    def _calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, Any]:
        """Calculate Simple Moving Average for multiple periods."""
        result = {}

        for period in periods:
            if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                sma = df["close"].rolling(period).mean()
            else:
                sma = df["close"].rolling(window=period).mean()

            col_name = f"sma_{period}"
            df[col_name] = sma
            result[col_name] = sma

        return result

    def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, Any]:
        """Calculate Exponential Moving Average for multiple periods."""
        result = {}

        for period in periods:
            if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                # For cuDF, manually implement EMA since it might not have ewm
                alpha = 2 / (period + 1)
                ema = df["close"].rolling(window=period).mean()
                for i in range(period, len(df)):
                    ema[i] = alpha * df["close"][i] + (1 - alpha) * ema[i - 1]
            else:
                ema = df["close"].ewm(span=period, adjust=False).mean()

            col_name = f"ema_{period}"
            df[col_name] = ema
            result[col_name] = ema

        return result

    def _calculate_wma(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, Any]:
        """Calculate Weighted Moving Average for multiple periods."""
        result = {}

        for period in periods:
            weights = np.arange(1, period + 1)
            if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                # For cuDF, convert to numpy for weighted calculation
                closes = df["close"].to_numpy()
                wma = np.zeros_like(closes)
                for i in range(period - 1, len(closes)):
                    wma[i] = (
                        np.sum(closes[i - period + 1 : i + 1] * weights) / weights.sum()
                    )
                wma = cudf.Series(wma)
            else:
                wma = (
                    df["close"]
                    .rolling(window=period)
                    .apply(lambda x: np.sum(x * weights) / weights.sum(), raw=True)
                )

            col_name = f"wma_{period}"
            df[col_name] = wma
            result[col_name] = wma

        return result

    def _calculate_vwma(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, Any]:
        """Calculate Volume Weighted Moving Average for multiple periods."""
        result = {}

        if "volume" not in df.columns:
            self.logger.warning("Volume data not available for VWMA calculation")
            return result

        for period in periods:
            if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                # For cuDF
                vp = df["close"] * df["volume"]
                vwma = vp.rolling(period).sum() / df["volume"].rolling(period).sum()
            else:
                vp = df["close"] * df["volume"]
                vwma = (
                    vp.rolling(window=period).sum()
                    / df["volume"].rolling(window=period).sum()
                )

            col_name = f"vwma_{period}"
            df[col_name] = vwma
            result[col_name] = vwma

        return result

    def _calculate_hull_ma(
        self, df: pd.DataFrame, periods: List[int]
    ) -> Dict[str, Any]:
        """Calculate Hull Moving Average for multiple periods."""
        result = {}

        for period in periods:
            half_period = int(period / 2)
            sqrt_period = int(np.sqrt(period))

            if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
                # For cuDF, hull calculation using cuDF operations
                wma1 = (
                    df["close"].rolling(half_period).mean() * 2
                )  # Simplified for brevity
                wma2 = df["close"].rolling(period).mean()
                raw_hma = wma1 - wma2
                hma = raw_hma.rolling(sqrt_period).mean()
            else:
                # Calculate WMA with period/2
                weights1 = np.arange(1, half_period + 1)
                wma1 = (
                    df["close"]
                    .rolling(window=half_period)
                    .apply(lambda x: np.sum(x * weights1) / weights1.sum(), raw=True)
                )

                # Calculate WMA with period
                weights2 = np.arange(1, period + 1)
                wma2 = (
                    df["close"]
                    .rolling(window=period)
                    .apply(lambda x: np.sum(x * weights2) / weights2.sum(), raw=True)
                )

                # Calculate raw HMA
                raw_hma = 2 * wma1 - wma2

                # Calculate WMA with sqrt(period) on the raw HMA
                weights3 = np.arange(1, sqrt_period + 1)
                hma = raw_hma.rolling(window=sqrt_period).apply(
                    lambda x: np.sum(x * weights3) / weights3.sum(), raw=True
                )

            col_name = f"hull_{period}"
            df[col_name] = hma
            result[col_name] = hma

        return result

    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Relative Strength Index."""
        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            # For cuDF, convert to numpy for RSI calculation
            close = df["close"].to_numpy()

            # Calculate price changes
            delta = np.zeros_like(close)
            delta[1:] = close[1:] - close[:-1]

            # Create arrays for gains and losses
            gain = np.zeros_like(delta)
            loss = np.zeros_like(delta)

            gain[delta > 0] = delta[delta > 0]
            loss[delta < 0] = -delta[delta < 0]

            # Calculate average gains and losses
            avg_gain = np.zeros_like(close)
            avg_loss = np.zeros_like(close)

            # First average is simple average
            avg_gain[period] = np.mean(gain[1 : period + 1])
            avg_loss[period] = np.mean(loss[1 : period + 1])

            # Calculate subsequent averages
            for i in range(period + 1, len(close)):
                avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

            # Calculate RS and RSI
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return {"values": rsi, "period": period}
        else:
            # Calculate price changes
            delta = df["close"].diff()

            # Create Series for gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)

            # Calculate average gains and losses
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()

            # Calculate RS and RSI
            rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
            rsi = 100.0 - (100.0 / (1.0 + rs))

            return {"values": rsi, "period": period}

    def _calculate_macd(
        self, df: pd.DataFrame, fast_period: int, slow_period: int, signal_period: int
    ) -> Dict[str, Any]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            # For cuDF, similar to EMA calculation
            close = df["close"]

            # Calculate fast EMA
            alpha_fast = 2 / (fast_period + 1)
            ema_fast = close.rolling(window=fast_period).mean()
            for i in range(fast_period, len(close)):
                ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i - 1]

            # Calculate slow EMA
            alpha_slow = 2 / (slow_period + 1)
            ema_slow = close.rolling(window=slow_period).mean()
            for i in range(slow_period, len(close)):
                ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i - 1]

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate Signal line
            alpha_signal = 2 / (signal_period + 1)
            signal_line = macd_line.rolling(window=signal_period).mean()
            for i in range(slow_period + signal_period, len(close)):
                signal_line[i] = (
                    alpha_signal * macd_line[i]
                    + (1 - alpha_signal) * signal_line[i - 1]
                )

            # Calculate histogram
            histogram = macd_line - signal_line

            return {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            }
        else:
            # Calculate EMAs
            ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()
            ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()

            # Calculate MACD line
            macd_line = ema_fast - ema_slow

            # Calculate Signal line (EMA of MACD line)
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

            # Calculate histogram
            histogram = macd_line - signal_line

            return {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "fast_period": fast_period,
                "slow_period": slow_period,
                "signal_period": signal_period,
            }

    def _calculate_bollinger_bands(
        self, df: pd.DataFrame, period: int, num_std: float
    ) -> Dict[str, Any]:
        """Calculate Bollinger Bands."""
        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            # For cuDF
            rolling_mean = df["close"].rolling(period).mean()
            rolling_std = df["close"].rolling(period).std()

            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)

            return {
                "middle_band": rolling_mean,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "period": period,
                "num_std": num_std,
            }
        else:
            rolling_mean = df["close"].rolling(window=period).mean()
            rolling_std = df["close"].rolling(window=period).std()

            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)

            return {
                "middle_band": rolling_mean,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "period": period,
                "num_std": num_std,
            }

    def _calculate_atr(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Average True Range."""
        if "high" not in df.columns or "low" not in df.columns:
            self.logger.warning("High and Low data required for ATR calculation")
            return {"values": np.nan, "period": period}

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            # For cuDF, calculate the true ranges
            high = df["high"].to_array()
            low = df["low"].to_array()
            close = df["close"].to_array()

            prev_close = np.zeros_like(close)
            prev_close[1:] = close[:-1]

            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)

            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            tr[0] = tr1[0]  # First TR is just high - low

            # Calculate ATR
            atr = np.zeros_like(tr)
            atr[:period] = np.nan
            atr[period] = np.mean(tr[1 : period + 1])

            for i in range(period + 1, len(tr)):
                atr[i] = ((period - 1) * atr[i - 1] + tr[i]) / period

            return {"values": atr, "period": period}
        else:
            # Calculate True Range
            high = df["high"]
            low = df["low"]
            close = df["close"]

            prev_close = close.shift(1)

            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()

            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calculate ATR
            atr = tr.rolling(window=period).mean()

            return {"values": atr, "period": period}

    def _calculate_adx(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Average Directional Index (ADX)."""
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            self.logger.warning(
                "High, Low, and Close data required for ADX calculation"
            )
            return {
                "adx": np.zeros(len(df)),
                "plus_di": np.zeros(len(df)),
                "minus_di": np.zeros(len(df)),
                "period": period,
            }

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                high = df["high"].to_array()
                low = df["low"].to_array()
                close = df["close"].to_array()

                # Create arrays for calculations
                plus_dm = np.zeros(len(df))
                minus_dm = np.zeros(len(df))
                tr = np.zeros(len(df))

                # Calculate True Range and Directional Movement
                for i in range(1, len(df)):
                    # True Range
                    tr1 = high[i] - low[i]
                    tr2 = abs(high[i] - close[i - 1])
                    tr3 = abs(low[i] - close[i - 1])
                    tr[i] = max(tr1, tr2, tr3)

                    # Directional Movement
                    up_move = high[i] - high[i - 1]
                    down_move = low[i - 1] - low[i]

                    if up_move > down_move and up_move > 0:
                        plus_dm[i] = up_move
                    else:
                        plus_dm[i] = 0

                    if down_move > up_move and down_move > 0:
                        minus_dm[i] = down_move
                    else:
                        minus_dm[i] = 0

                # Smooth values using Wilder's smoothing technique
                tr_smooth = np.zeros(len(df))
                plus_dm_smooth = np.zeros(len(df))
                minus_dm_smooth = np.zeros(len(df))

                # Initialize first values
                tr_smooth[period] = np.sum(tr[1 : period + 1])
                plus_dm_smooth[period] = np.sum(plus_dm[1 : period + 1])
                minus_dm_smooth[period] = np.sum(minus_dm[1 : period + 1])

                # Calculate smoothed values
                for i in range(period + 1, len(df)):
                    tr_smooth[i] = (
                        tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
                    )
                    plus_dm_smooth[i] = (
                        plus_dm_smooth[i - 1]
                        - (plus_dm_smooth[i - 1] / period)
                        + plus_dm[i]
                    )
                    minus_dm_smooth[i] = (
                        minus_dm_smooth[i - 1]
                        - (minus_dm_smooth[i - 1] / period)
                        + minus_dm[i]
                    )

                # Calculate +DI and -DI
                plus_di = np.zeros(len(df))
                minus_di = np.zeros(len(df))

                for i in range(period, len(df)):
                    plus_di[i] = (
                        100 * plus_dm_smooth[i] / tr_smooth[i]
                        if tr_smooth[i] != 0
                        else 0
                    )
                    minus_di[i] = (
                        100 * minus_dm_smooth[i] / tr_smooth[i]
                        if tr_smooth[i] != 0
                        else 0
                    )

                # Calculate DX
                dx = np.zeros(len(df))
                for i in range(period, len(df)):
                    if plus_di[i] + minus_di[i] != 0:
                        dx[i] = (
                            100
                            * abs(plus_di[i] - minus_di[i])
                            / (plus_di[i] + minus_di[i])
                        )
                    else:
                        dx[i] = 0

                # Calculate ADX
                adx = np.zeros(len(df))
                adx[2 * period - 1] = np.mean(dx[period : 2 * period])

                for i in range(2 * period, len(df)):
                    adx[i] = ((period - 1) * adx[i - 1] + dx[i]) / period

                # Transfer back to GPU if required for other calculations
                adx_gpu = cp.array(adx)
                plus_di_gpu = cp.array(plus_di)
                minus_di_gpu = cp.array(minus_di)

                return {
                    "adx": adx,
                    "plus_di": plus_di,
                    "minus_di": minus_di,
                    "period": period,
                }

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for ADX: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Calculate +DM, -DM, and TR
            high = df["high"]
            low = df["low"]
            close = df["close"]

            # True Range calculation
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement calculation
            high_diff = high.diff()
            low_diff = low.diff() * -1

            plus_dm = pd.Series(
                np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0),
                index=df.index,
            )
            minus_dm = pd.Series(
                np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0),
                index=df.index,
            )

            #
            # Smoothed True Range and Directional Movement using Wilder's
            # smoothing
            tr_period = tr.rolling(window=period).sum()
            plus_dm_period = plus_dm.rolling(window=period).sum()
            minus_dm_period = minus_dm.rolling(window=period).sum()

            # Calculate +DI and -DI
            plus_di = 100 * plus_dm_period / tr_period
            minus_di = 100 * minus_dm_period / tr_period

            # Calculate the Directional Index (DX)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).abs()

            # Calculate ADX using Wilder's smoothing
            adx = dx.rolling(window=period).mean()

            return {
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "period": period,
            }

        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return {
                "adx": np.zeros(len(df)),
                "plus_di": np.zeros(len(df)),
                "minus_di": np.zeros(len(df)),
                "period": period,
            }

    def _calculate_stochastic(
        self, df: pd.DataFrame, k_period: int, d_period: int
    ) -> Dict[str, Any]:
        """Calculate Stochastic Oscillator."""
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            self.logger.warning(
                "High, Low, and Close data required for Stochastic Oscillator calculation"
            )
            return {
                "k": np.zeros(len(df)),
                "d": np.zeros(len(df)),
                "k_period": k_period,
                "d_period": d_period,
            }

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                high = df["high"].to_array()
                low = df["low"].to_array()
                close = df["close"].to_array()

                # Calculate %K values
                k_values = np.zeros(len(df))

                for i in range(k_period - 1, len(df)):
                    high_window = high[i - k_period + 1 : i + 1]
                    low_window = low[i - k_period + 1 : i + 1]
                    highest_high = np.max(high_window)
                    lowest_low = np.min(low_window)

                    if highest_high - lowest_low != 0:
                        k_values[i] = (
                            100 * (close[i] - lowest_low) / (highest_high - lowest_low)
                        )
                    else:
                        k_values[i] = 50  # Default value when range is zero

                # Calculate %D values (simple moving average of %K)
                d_values = np.zeros(len(df))
                for i in range(k_period + d_period - 2, len(df)):
                    d_values[i] = np.mean(k_values[i - d_period + 1 : i + 1])

                # Transfer back to GPU if required for other calculations
                k_gpu = cp.array(k_values)
                d_gpu = cp.array(d_values)

                return {
                    "k": k_values,
                    "d": d_values,
                    "k_period": k_period,
                    "d_period": d_period,
                }

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for Stochastic Oscillator: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Calculate %K
            lowest_low = df["low"].rolling(window=k_period).min()
            highest_high = df["high"].rolling(window=k_period).max()

            # Safely handle division (avoid division by zero)
            high_low_range = highest_high - lowest_low
            high_low_range = high_low_range.replace(0, np.nan)  # Replace zeros with NaN

            # Calculate %K
            k = 100 * (df["close"] - lowest_low) / high_low_range
            k = k.fillna(50)  # Fill NaN values with 50 (neutral)

            # Calculate %D (simple moving average of %K)
            d = k.rolling(window=d_period).mean()

            return {"k": k, "d": d, "k_period": k_period, "d_period": d_period}

        except Exception as e:
            self.logger.error(f"Error calculating Stochastic Oscillator: {e}")
            return {
                "k": np.zeros(len(df)),
                "d": np.zeros(len(df)),
                "k_period": k_period,
                "d_period": d_period,
            }

    def _calculate_obv(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate On-Balance Volume (OBV)."""
        if "close" not in df.columns or "volume" not in df.columns:
            self.logger.warning("Close and Volume data required for OBV calculation")
            return {"values": np.zeros(len(df))}

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                close = df["close"].to_array()
                volume = df["volume"].to_array()

                # Calculate OBV values
                obv = np.zeros(len(df))

                # First value is just the volume
                obv[0] = volume[0]

                # Calculate subsequent values
                for i in range(1, len(df)):
                    if close[i] > close[i - 1]:
                        # Price up, add volume
                        obv[i] = obv[i - 1] + volume[i]
                    elif close[i] < close[i - 1]:
                        # Price down, subtract volume
                        obv[i] = obv[i - 1] - volume[i]
                    else:
                        # Price unchanged, OBV unchanged
                        obv[i] = obv[i - 1]

                # Transfer back to GPU if required for other calculations
                obv_gpu = cp.array(obv)

                return {"values": obv}

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for OBV: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Get price and volume data
            close = df["close"]
            volume = df["volume"]

            # Calculate daily price change direction
            price_change = close.diff()

            # Create a Series with -1, 0, or 1 based on price change direction
            direction = pd.Series(
                np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)),
                index=df.index,
            )

            # Calculate OBV directly with vectorized operations
            direction_volume = direction * volume

            # Shift the first value to deal with NaN from diff()
            direction_volume.iloc[0] = volume.iloc[0]  # Initialize with first volume

            # Cumulative sum gives us OBV
            obv = direction_volume.cumsum()

            return {"values": obv}

        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return {"values": np.zeros(len(df))}

    def _calculate_vwap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Volume Weighted Average Price (VWAP)."""
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
            or "volume" not in df.columns
        ):
            self.logger.warning(
                "High, Low, Close, and Volume data required for VWAP calculation"
            )
            return {"values": np.zeros(len(df))}

        # Check if 'date' column is available to reset VWAP for each day
        has_dates = "date" in df.columns

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                high = df["high"].to_array()
                low = df["low"].to_array()
                close = df["close"].to_array()
                volume = df["volume"].to_array()

                # Calculate typical price: (high + low + close) / 3
                tp = (high + low + close) / 3

                # Calculate cumulative TP * Volume and cumulative Volume
                cumulative_tpv = np.zeros(len(df))
                cumulative_vol = np.zeros(len(df))

                # Reset VWAP at the start of each day
                if has_dates:
                    dates = df["date"].to_array()
                    current_date = None

                    for i in range(len(df)):
                        date_val = pd.Timestamp(dates[i]).date()

                        # Reset on new day
                        if current_date != date_val:
                            current_date = date_val
                            cumulative_tpv[i] = tp[i] * volume[i]
                            cumulative_vol[i] = volume[i]
                        else:
                            cumulative_tpv[i] = cumulative_tpv[i - 1] + (
                                tp[i] * volume[i]
                            )
                            cumulative_vol[i] = cumulative_vol[i - 1] + volume[i]
                else:
                    # If no date column, calculate continuous VWAP
                    cumulative_tpv[0] = tp[0] * volume[0]
                    cumulative_vol[0] = volume[0]

                    for i in range(1, len(df)):
                        cumulative_tpv[i] = cumulative_tpv[i - 1] + (tp[i] * volume[i])
                        cumulative_vol[i] = cumulative_vol[i - 1] + volume[i]

                # Calculate VWAP with safe division
                vwap = np.zeros(len(df))
                for i in range(len(df)):
                    if cumulative_vol[i] > 0:
                        vwap[i] = cumulative_tpv[i] / cumulative_vol[i]
                    else:
                        vwap[i] = tp[i]  # If no volume, use typical price

                # Transfer back to GPU if required for other calculations
                vwap_gpu = cp.array(vwap)

                return {"values": vwap}

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for VWAP: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Calculate typical price: (high + low + close) / 3
            df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

            # Calculate VWAP
            if has_dates:
                # Convert date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(df["date"]):
                    df["date"] = pd.to_datetime(df["date"])

                # Extract date part for grouping
                df["date_only"] = df["date"].dt.date

                # Group by date and calculate VWAP for each day
                df["tp_volume"] = df["tp"] * df["volume"]
                df["cum_tp_volume"] = df.groupby("date_only")["tp_volume"].cumsum()
                df["cum_volume"] = df.groupby("date_only")["volume"].cumsum()

                # Calculate VWAP with safe division
                vwap = df["cum_tp_volume"] / df["cum_volume"].replace(0, np.nan)
                vwap = vwap.fillna(df["tp"])  # If no volume, use typical price
            else:
                # Continuous VWAP
                df["tp_volume"] = df["tp"] * df["volume"]
                df["cum_tp_volume"] = df["tp_volume"].cumsum()
                df["cum_volume"] = df["volume"].cumsum()

                # Calculate VWAP with safe division
                vwap = df["cum_tp_volume"] / df["cum_volume"].replace(0, np.nan)
                vwap = vwap.fillna(df["tp"])  # If no volume, use typical price

            return {"values": vwap}

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return {"values": np.zeros(len(df))}

    def _calculate_cci(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Commodity Channel Index (CCI)."""
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            self.logger.warning(
                "High, Low, and Close data required for CCI calculation"
            )
            return {"values": np.zeros(len(df)), "period": period}

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                high = df["high"].to_array()
                low = df["low"].to_array()
                close = df["close"].to_array()

                # Calculate typical price: (high + low + close) / 3
                tp = (high + low + close) / 3

                # Calculate SMA of typical price
                tp_sma = np.zeros(len(df))
                # Mean deviation
                md = np.zeros(len(df))

                # Calculate SMA and Mean Deviation
                for i in range(period - 1, len(df)):
                    # SMA of typical price
                    tp_window = tp[i - period + 1 : i + 1]
                    tp_sma[i] = np.mean(tp_window)

                    # Mean deviation
                    md[i] = np.mean(np.abs(tp_window - tp_sma[i]))

                # Calculate CCI with safe division
                cci = np.zeros(len(df))
                for i in range(period - 1, len(df)):
                    if md[i] != 0:
                        cci[i] = (tp[i] - tp_sma[i]) / (0.015 * md[i])
                    else:
                        cci[i] = 0  # Avoid division by zero

                # Transfer back to GPU if required for other calculations
                cci_gpu = cp.array(cci)

                return {"values": cci, "period": period}

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for CCI: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Calculate typical price
            df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

            # Calculate SMA of typical price
            tp_sma = df["tp"].rolling(window=period).mean()

            # Calculate Mean Deviation
            def mean_dev(x):
                return np.mean(np.abs(x - x.mean()))

            # Calculate Mean Deviation using the rolling window
            mean_dev_tp = df["tp"].rolling(window=period).apply(mean_dev, raw=True)

            # Calculate CCI with safe division (avoid division by zero)
            # The constant 0.015 is traditionally used in CCI calculation
            cci = (df["tp"] - tp_sma) / (0.015 * mean_dev_tp.replace(0, np.nan))
            cci = cci.fillna(0)  # Replace NaN values with 0

            return {"values": cci, "period": period}

        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return {"values": np.zeros(len(df)), "period": period}

    def _calculate_mfi(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Money Flow Index (MFI)."""
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
            or "volume" not in df.columns
        ):
            self.logger.warning(
                "High, Low, Close, and Volume data required for MFI calculation"
            )
            return {"values": np.zeros(len(df)), "period": period}

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                high = df["high"].to_array()
                low = df["low"].to_array()
                close = df["close"].to_array()
                volume = df["volume"].to_array()

                # Calculate typical price: (high + low + close) / 3
                tp = (high + low + close) / 3

                # Calculate money flow
                money_flow = tp * volume

                # Determine positive and negative money flow
                pos_flow = np.zeros(len(df))
                neg_flow = np.zeros(len(df))

                # First point has no price change
                for i in range(1, len(df)):
                    if tp[i] > tp[i - 1]:
                        pos_flow[i] = money_flow[i]
                        neg_flow[i] = 0
                    elif tp[i] < tp[i - 1]:
                        pos_flow[i] = 0
                        neg_flow[i] = money_flow[i]
                    else:
                        # No change
                        pos_flow[i] = 0
                        neg_flow[i] = 0

                # Calculate positive and negative money flow ratio over period
                pos_flow_sum = np.zeros(len(df))
                neg_flow_sum = np.zeros(len(df))

                for i in range(period, len(df)):
                    pos_flow_sum[i] = np.sum(pos_flow[i - period + 1 : i + 1])
                    neg_flow_sum[i] = np.sum(neg_flow[i - period + 1 : i + 1])

                # Calculate MFI with safe division
                mfi = np.zeros(len(df))
                for i in range(period, len(df)):
                    if neg_flow_sum[i] != 0:
                        money_ratio = pos_flow_sum[i] / neg_flow_sum[i]
                        mfi[i] = 100 - (100 / (1 + money_ratio))
                    elif pos_flow_sum[i] != 0:
                        # All positive money flow
                        mfi[i] = 100
                    else:
                        # No money flow (no volume or all unchanged prices)
                        mfi[i] = 50  # Neutral value

                # Transfer back to GPU if required for other calculations
                mfi_gpu = cp.array(mfi)

                return {"values": mfi, "period": period}

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for MFI: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Calculate typical price
            df["tp"] = (df["high"] + df["low"] + df["close"]) / 3

            # Calculate raw money flow
            df["money_flow"] = df["tp"] * df["volume"]

            # Get price changes to determine positive/negative money flow
            df["tp_change"] = df["tp"].diff()

            # Create positive and negative money flow series
            df["positive_flow"] = np.where(df["tp_change"] > 0, df["money_flow"], 0)
            df["negative_flow"] = np.where(df["tp_change"] < 0, df["money_flow"], 0)

            # Calculate the sums for the period
            pos_flow_sum = df["positive_flow"].rolling(window=period).sum()
            neg_flow_sum = df["negative_flow"].rolling(window=period).sum()

            # Calculate the ratio with safe division
            money_ratio = pos_flow_sum / neg_flow_sum.replace(0, np.nan)

            # Calculate MFI
            mfi = 100 - (100 / (1 + money_ratio))

            # Handle special cases
            #
            # Where there was no negative money flow, MFI is 100 (completely
            # bullish)
            mfi = mfi.fillna(100)
            #
            # Where there was neither positive nor negative flow, MFI is 50
            # (neutral)
            mfi = np.where((pos_flow_sum == 0) & (neg_flow_sum == 0), 50, mfi)

            return {"values": mfi, "period": period}

        except Exception as e:
            self.logger.error(f"Error calculating MFI: {e}")
            return {"values": np.zeros(len(df)), "period": period}

    def _calculate_roc(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Rate of Change (ROC)."""
        if "close" not in df.columns:
            self.logger.warning("Close price data required for ROC calculation")
            return {"values": np.zeros(len(df)), "period": period}

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                close = df["close"].to_array()

                #
                # Calculate ROC: (current_price - price_n_periods_ago) /
                # price_n_periods_ago * 100
                roc = np.zeros(len(df))

                for i in range(period, len(df)):
                    price_past = close[i - period]
                    if price_past != 0:  # Avoid division by zero
                        roc[i] = ((close[i] - price_past) / price_past) * 100
                    else:
                        roc[i] = 0  # Default value when price_past is zero

                # Transfer back to GPU if required for other calculations
                roc_gpu = cp.array(roc)

                return {"values": roc, "period": period}

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for ROC: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            #
            # Calculate ROC: (current_price - price_n_periods_ago) /
            # price_n_periods_ago * 100
            price_past = df["close"].shift(period)

            # Use vectorized operations with safe division
            roc = ((df["close"] - price_past) / price_past.replace(0, np.nan)) * 100
            roc = roc.fillna(0)  # Replace NaN values with 0

            return {"values": roc, "period": period}

        except Exception as e:
            self.logger.error(f"Error calculating ROC: {e}")
            return {"values": np.zeros(len(df)), "period": period}

    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> Dict[str, Any]:
        """Calculate Williams %R."""
        if (
            "high" not in df.columns
            or "low" not in df.columns
            or "close" not in df.columns
        ):
            self.logger.warning(
                "High, Low, and Close data required for Williams %R calculation"
            )
            return {"values": np.zeros(len(df)), "period": period}

        if self.use_gpu and HAVE_CUDA and isinstance(df, cudf.DataFrame):
            try:
                # Convert to NumPy arrays for calculation
                high = df["high"].to_array()
                low = df["low"].to_array()
                close = df["close"].to_array()

                #
                # Calculate Williams %R: (Highest High - Close) / (Highest High
                # - Lowest Low) * -100
                williams_r = np.zeros(len(df))

                for i in range(period - 1, len(df)):
                    highest_high = np.max(high[i - period + 1 : i + 1])
                    lowest_low = np.min(low[i - period + 1 : i + 1])

                    # Safely calculate Williams %R (avoid division by zero)
                    hl_range = highest_high - lowest_low
                    if hl_range != 0:
                        williams_r[i] = ((highest_high - close[i]) / hl_range) * -100
                    else:
                        williams_r[i] = -50  # Default neutral value when range is zero

                # Transfer back to GPU if required for other calculations
                williams_r_gpu = cp.array(williams_r)

                return {"values": williams_r, "period": period}

            except Exception as e:
                self.logger.error(
                    f"GPU calculation failed for Williams %R: {e}, falling back to CPU"
                )
                # Fall back to CPU implementation

        # CPU implementation
        try:
            # Get highest high and lowest low over the period
            highest_high = df["high"].rolling(window=period).max()
            lowest_low = df["low"].rolling(window=period).min()

            # Safely calculate the range (avoid division by zero)
            hl_range = highest_high - lowest_low
            hl_range = hl_range.replace(0, np.nan)  # Replace zeros with NaN

            # Calculate Williams %R
            williams_r = ((highest_high - df["close"]) / hl_range) * -100

            # Handle special cases (no range)
            williams_r = williams_r.fillna(-50)  # Fill NaN with neutral value

            return {"values": williams_r, "period": period}

        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return {"values": np.zeros(len(df)), "period": period}

    def _detect_crossovers(
        self, df: pd.DataFrame, col1: str, col2: str
    ) -> List[Dict[str, Any]]:
        """
        Detect crossovers between two indicators.

        Args:
            df: DataFrame containing the indicators
            col1: Column name for first indicator
            col2: Column name for second indicator

        Returns:
            List of crossover events
        """
        if col1 not in df.columns or col2 not in df.columns:
            return []

        crossovers = []

        # Get the indicator values
        ind1 = df[col1]
        ind2 = df[col2]

        # Check for crossovers
        for i in range(1, len(df)):
            # Skip NaN values
            if (
                pd.isna(ind1.iloc[i - 1])
                or pd.isna(ind1.iloc[i])
                or pd.isna(ind2.iloc[i - 1])
                or pd.isna(ind2.iloc[i])
            ):
                continue

            # Bullish crossover: ind1 crosses above ind2
            if ind1.iloc[i - 1] <= ind2.iloc[i - 1] and ind1.iloc[i] > ind2.iloc[i]:
                crossovers.append(
                    {
                        "direction": "bullish",
                        "index": i,
                        "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                        if "date" in df.columns
                        else i,
                        "value1": float(ind1.iloc[i]),
                        "value2": float(ind2.iloc[i]),
                    }
                )

            # Bearish crossover: ind1 crosses below ind2
            elif ind1.iloc[i - 1] >= ind2.iloc[i - 1] and ind1.iloc[i] < ind2.iloc[i]:
                crossovers.append(
                    {
                        "direction": "bearish",
                        "index": i,
                        "date": df["date"].iloc[i].strftime("%Y-%m-%d")
                        if "date" in df.columns
                        else i,
                        "value1": float(ind1.iloc[i]),
                        "value2": float(ind2.iloc[i]),
                    }
                )

        return crossovers


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration with GPU support if available
    config = {"use_gpu": HAVE_CUDA, "cache_dir": "./model_cache/technical"}

    # Create and start the server
    server = TechnicalIndicatorsMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("TechnicalIndicatorsMCP server started")

    # Example usage
    import datetime

    # Generate some sample price data
    price_data = []
    base_price = 100.0
    for i in range(100):
        day = datetime.datetime.now() - datetime.timedelta(days=100 - i)
        price = base_price + i * 0.5 + np.sin(i / 10) * 5
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

    # Calculate some indicators
    result = server.calculate_indicators(
        price_data=price_data, indicators=["sma", "ema", "rsi", "macd"]
    )

    print(
        f"Calculated {len(result['indicators'])} indicators with {len(result['signals'])} signals"
    )
