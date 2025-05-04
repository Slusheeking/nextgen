"""
Growth Analysis MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
growth analysis capabilities for financial data, focusing on revenue growth,
earnings growth, and other growth-related metrics.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

# GPU acceleration imports with fallbacks to CPU (NumPy)
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
    service_name="analysis-mcp-growth-analysis",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class GrowthAnalysisMCP(BaseMCPServer):
    """
    MCP server for analyzing growth metrics and trends.

    This tool analyzes historical financial data to calculate growth rates,
    identify growth trends, and evaluate growth sustainability. It supports
    GPU acceleration for faster processing of large datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Growth Analysis MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - growth_benchmarks_path: Path to growth benchmark data
                - growth_weights_path: Path to growth model weights
                - cache_dir: Directory for caching intermediate results
                - use_gpu: Whether to use GPU acceleration (default: True)
                - min_company_batch: Minimum number of companies to process with GPU (default: 10)
                - min_periods: Minimum number of periods to process with GPU (default: 20)
                - enable_monitoring: Whether to enable monitoring (default: True)
        """
        super().__init__(name="growth_analysis_mcp", config=config)

        # Set default configurations
        self.growth_benchmarks_path = self.config.get(
            "growth_benchmarks_path",
            os.path.join(os.path.dirname(__file__), "data/growth_benchmarks.json"),
        )
        self.growth_weights_path = self.config.get(
            "growth_weights_path",
            os.path.join(os.path.dirname(__file__), "data/growth_weights.json"),
        )
        self.cache_dir = self.config.get("cache_dir", "./growth_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # GPU configuration
        self.use_gpu = self.config.get("use_gpu", True)

        # Check if we have GPU and whether we should use it
        self.gpu_available = HAVE_GPU and self.use_gpu

        # Set batch processing thresholds
        self.min_company_batch = self.config.get("min_company_batch", 10)
        self.min_periods = self.config.get("min_periods", 20)

        # Initialize execution tracking
        self.execution_stats = {
            "gpu_executions": 0,
            "cpu_executions": 0,
            "gpu_failures": 0,
            "total_processing_time": 0.0,
            "gpu_processing_time": 0.0,
            "cpu_processing_time": 0.0,
        }

        # Load benchmarks and weights
        self._load_growth_benchmarks()
        self._load_growth_weights()

        # Register tools
        self._register_tools()

        self.logger.info(
            f"GrowthAnalysisMCP initialized (GPU enabled: {self.gpu_available})"
        )

    def _load_growth_benchmarks(self):
        """Load growth benchmark data for comparison."""
        self.growth_benchmarks = {}

        if os.path.exists(self.growth_benchmarks_path):
            try:
                with open(self.growth_benchmarks_path, "r") as f:
                    self.growth_benchmarks = json.load(f)
                self.logger.info(
                    f"Loaded growth benchmarks from {self.growth_benchmarks_path}"
                )
            except Exception as e:
                self.logger.error(f"Error loading growth benchmarks: {e}")
                self._create_default_benchmarks()
        else:
            self.logger.warning(
                f"Growth benchmarks file not found: {self.growth_benchmarks_path}"
            )
            self._create_default_benchmarks()

    def _create_default_benchmarks(self):
        """Create default growth benchmarks."""
        self.growth_benchmarks = {
            "Technology": {
                "revenue_growth": 15.0,
                "earnings_growth": 20.0,
                "fcf_growth": 18.0,
                "dividend_growth": 5.0,
                "r_and_d_growth": 12.0,
            },
            "Financial Services": {
                "revenue_growth": 8.0,
                "earnings_growth": 10.0,
                "fcf_growth": 9.0,
                "dividend_growth": 7.0,
                "r_and_d_growth": 3.0,
            },
            "Healthcare": {
                "revenue_growth": 10.0,
                "earnings_growth": 12.0,
                "fcf_growth": 11.0,
                "dividend_growth": 6.0,
                "r_and_d_growth": 15.0,
            },
            "Consumer Cyclical": {
                "revenue_growth": 7.0,
                "earnings_growth": 9.0,
                "fcf_growth": 8.0,
                "dividend_growth": 5.0,
                "r_and_d_growth": 5.0,
            },
            "Consumer Defensive": {
                "revenue_growth": 5.0,
                "earnings_growth": 7.0,
                "fcf_growth": 6.0,
                "dividend_growth": 8.0,
                "r_and_d_growth": 4.0,
            },
            "Industrials": {
                "revenue_growth": 6.0,
                "earnings_growth": 8.0,
                "fcf_growth": 7.0,
                "dividend_growth": 6.0,
                "r_and_d_growth": 7.0,
            },
            "Energy": {
                "revenue_growth": 4.0,
                "earnings_growth": 6.0,
                "fcf_growth": 5.0,
                "dividend_growth": 9.0,
                "r_and_d_growth": 5.0,
            },
            "Utilities": {
                "revenue_growth": 3.0,
                "earnings_growth": 4.0,
                "fcf_growth": 3.0,
                "dividend_growth": 5.0,
                "r_and_d_growth": 2.0,
            },
            "Real Estate": {
                "revenue_growth": 5.0,
                "earnings_growth": 6.0,
                "fcf_growth": 4.0,
                "dividend_growth": 7.0,
                "r_and_d_growth": 1.0,
            },
            "Basic Materials": {
                "revenue_growth": 4.0,
                "earnings_growth": 5.0,
                "fcf_growth": 4.0,
                "dividend_growth": 6.0,
                "r_and_d_growth": 4.0,
            },
            "Communication Services": {
                "revenue_growth": 9.0,
                "earnings_growth": 11.0,
                "fcf_growth": 10.0,
                "dividend_growth": 5.0,
                "r_and_d_growth": 8.0,
            },
        }
        self.logger.info("Created default growth benchmarks")

    def _load_growth_weights(self):
        """Load weights for the growth scoring model."""
        self.growth_weights = {}

        if os.path.exists(self.growth_weights_path):
            try:
                with open(self.growth_weights_path, "r") as f:
                    self.growth_weights = json.load(f)
                self.logger.info(
                    f"Loaded growth weights from {self.growth_weights_path}"
                )
            except Exception as e:
                self.logger.error(f"Error loading growth weights: {e}")
                self._create_default_weights()
        else:
            self.logger.warning(
                f"Growth weights file not found: {self.growth_weights_path}"
            )
            self._create_default_weights()

    def _create_default_weights(self):
        """Create default weights for the growth scoring model."""
        self.growth_weights = {
            "revenue_growth": 0.30,
            "earnings_growth": 0.25,
            "fcf_growth": 0.20,
            "dividend_growth": 0.10,
            "r_and_d_growth": 0.15,
        }
        self.logger.info("Created default growth weights")

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "calculate_growth_rates",
            self.calculate_growth_rates,
            "Calculate growth rates from historical financial data",
            {
                "financial_data": {
                    "type": "object",
                    "description": "Dictionary containing historical financial data with time periods as keys",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of metrics to calculate growth rates for (default: all common metrics)",
                    "required": False,
                },
                "periods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of time periods to analyze (e.g., 'annual', 'quarterly')",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "growth_rates": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "analyze_growth_trends",
            self.analyze_growth_trends,
            "Analyze growth trends and patterns from historical data",
            {
                "growth_rates": {
                    "type": "object",
                    "description": "Dictionary of calculated growth rates",
                },
                "min_periods": {
                    "type": "integer",
                    "description": "Minimum number of periods required for trend analysis",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "trends": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "score_growth_quality",
            self.score_growth_quality,
            "Score the quality and sustainability of growth",
            {
                "growth_rates": {
                    "type": "object",
                    "description": "Dictionary of calculated growth rates",
                },
                "trends": {
                    "type": "object",
                    "description": "Dictionary of growth trends analysis",
                    "required": False,
                },
                "sector": {
                    "type": "string",
                    "description": "Industry sector for benchmark comparison",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "metric_scores": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "compare_to_sector_growth",
            self.compare_to_sector_growth,
            "Compare company's growth metrics to sector benchmarks",
            {
                "growth_rates": {
                    "type": "object",
                    "description": "Dictionary of company's growth rates",
                },
                "sector": {
                    "type": "string",
                    "description": "Industry sector of the company",
                },
            },
            {
                "type": "object",
                "properties": {
                    "comparison": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "analyze_earnings_report",
            self.analyze_earnings_report,
            "Analyze an earnings report for growth implications",
            {
                "report_data": {
                    "type": "object",
                    "description": "Dictionary containing earnings report data",
                },
                "historical_data": {
                    "type": "object",
                    "description": "Dictionary containing historical financial data for comparison",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "analysis": {"type": "object"},
                    "growth_implications": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "forecast_future_growth",
            self.forecast_future_growth,
            "Forecast future growth based on historical patterns",
            {
                "growth_rates": {
                    "type": "object",
                    "description": "Dictionary of historical growth rates",
                },
                "periods_ahead": {
                    "type": "integer",
                    "description": "Number of periods to forecast ahead",
                    "default": 4,
                },
                "confidence_interval": {
                    "type": "number",
                    "description": "Confidence interval for forecasts (0.0-1.0)",
                    "default": 0.95,
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "forecasts": {"type": "object"},
                    "confidence_intervals": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "get_execution_stats",
            self.get_execution_stats,
            "Get GPU/CPU execution statistics for this MCP server",
            {},
            {
                "type": "object",
                "properties": {
                    "gpu_executions": {"type": "number"},
                    "cpu_executions": {"type": "number"},
                    "gpu_failures": {"type": "number"},
                    "total_executions": {"type": "number"},
                    "gpu_percentage": {"type": "number"},
                    "cpu_percentage": {"type": "number"},
                    "gpu_available": {"type": "boolean"},
                    "total_processing_time": {"type": "number"},
                    "gpu_speedup": {"type": "number"},
                },
            },
        )

    def calculate_growth_rates(
        self,
        financial_data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
        periods: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate growth rates from historical financial data.

        Args:
            financial_data: Dictionary containing historical financial data with time periods as keys
            metrics: List of metrics to calculate growth rates for
            periods: List of time periods to analyze (e.g., 'annual', 'quarterly')

        Returns:
            Dictionary of calculated growth rates
        """
        start_time = time.time()

        # Default metrics if not specified
        default_metrics = [
            "revenue",
            "net_income",
            "operating_income",
            "gross_profit",
            "free_cash_flow",
            "dividends",
            "r_and_d",
            "capex",
        ]
        metrics_to_calculate = metrics or default_metrics

        # Default periods if not specified
        default_periods = ["annual", "quarterly"]
        periods_to_analyze = periods or default_periods

        # Determine if we should use GPU
        use_gpu = False

        # For batch processing or large datasets
        if isinstance(financial_data.get("annual", {}), list) or isinstance(
            financial_data.get("quarterly", {}), list
        ):
            batch_size = len(financial_data.get("annual", [])) or len(
                financial_data.get("quarterly", [])
            )
            use_gpu = self.gpu_available and batch_size >= self.min_company_batch
        else:
            # For single company with many periods
            period_count = 0
            for period_type in periods_to_analyze:
                if period_type in financial_data:
                    period_count += len(financial_data[period_type])
            use_gpu = self.gpu_available and period_count >= self.min_periods

        growth_rates = {}

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for growth rate calculation")

                for period_type in periods_to_analyze:
                    if period_type not in financial_data:
                        continue

                    period_data = financial_data[period_type]

                    # Sort periods chronologically
                    if isinstance(period_data, dict):
                        # Convert to list of (period, data) tuples and sort
                        period_items = sorted(period_data.items())
                        periods_list = [item[0] for item in period_items]

                        # Process each metric
                        for metric in metrics_to_calculate:
                            # Extract values for this metric across periods
                            values = []
                            for _, data in period_items:
                                values.append(data.get(metric, np.nan))

                            # Convert to GPU array
                            values_gpu = cp.array(values, dtype=cp.float64)

                            # Calculate YoY growth rates
                            growth_gpu = cp.zeros_like(values_gpu)
                            growth_gpu[1:] = (
                                values_gpu[1:] / (values_gpu[:-1] + 1e-10)
                            ) - 1.0
                            growth_gpu[0] = cp.nan  # First period has no growth rate

                            # Calculate CAGR for different timeframes
                            cagr_1yr = growth_gpu[-1] if len(growth_gpu) > 0 else cp.nan

                            # 3-year CAGR
                            if len(values_gpu) >= 4:
                                cagr_3yr = (
                                    cp.power(
                                        (values_gpu[-1] / (values_gpu[-4] + 1e-10)),
                                        1 / 3,
                                    )
                                    - 1.0
                                )
                            else:
                                cagr_3yr = cp.nan

                            # 5-year CAGR
                            if len(values_gpu) >= 6:
                                cagr_5yr = (
                                    cp.power(
                                        (values_gpu[-1] / (values_gpu[-6] + 1e-10)),
                                        1 / 5,
                                    )
                                    - 1.0
                                )
                            else:
                                cagr_5yr = cp.nan

                            # Store results
                            if period_type not in growth_rates:
                                growth_rates[period_type] = {}

                            growth_rates[period_type][metric] = {
                                "periods": periods_list,
                                "values": cp.asnumpy(values_gpu).tolist(),
                                "growth_rates": cp.asnumpy(growth_gpu).tolist(),
                                "cagr_1yr": float(cagr_1yr)
                                if not cp.isnan(cagr_1yr)
                                else None,
                                "cagr_3yr": float(cagr_3yr)
                                if not cp.isnan(cagr_3yr)
                                else None,
                                "cagr_5yr": float(cagr_5yr)
                                if not cp.isnan(cagr_5yr)
                                else None,
                            }

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for growth rates: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            for period_type in periods_to_analyze:
                if period_type not in financial_data:
                    continue

                period_data = financial_data[period_type]

                # Sort periods chronologically
                if isinstance(period_data, dict):
                    # Convert to list of (period, data) tuples and sort
                    period_items = sorted(period_data.items())
                    periods_list = [item[0] for item in period_items]

                    # Process each metric
                    for metric in metrics_to_calculate:
                        # Extract values for this metric across periods
                        values = []
                        for _, data in period_items:
                            values.append(data.get(metric, np.nan))

                        # Convert to numpy array
                        values_np = np.array(values, dtype=np.float64)

                        # Calculate YoY growth rates
                        growth_np = np.zeros_like(values_np)
                        growth_np[1:] = (values_np[1:] / (values_np[:-1] + 1e-10)) - 1.0
                        growth_np[0] = np.nan  # First period has no growth rate

                        # Calculate CAGR for different timeframes
                        cagr_1yr = growth_np[-1] if len(growth_np) > 0 else np.nan

                        # 3-year CAGR
                        if len(values_np) >= 4:
                            cagr_3yr = (
                                np.power(
                                    (values_np[-1] / (values_np[-4] + 1e-10)), 1 / 3
                                )
                                - 1.0
                            )
                        else:
                            cagr_3yr = np.nan

                        # 5-year CAGR
                        if len(values_np) >= 6:
                            cagr_5yr = (
                                np.power(
                                    (values_np[-1] / (values_np[-6] + 1e-10)), 1 / 5
                                )
                                - 1.0
                            )
                        else:
                            cagr_5yr = np.nan

                        # Store results
                        if period_type not in growth_rates:
                            growth_rates[period_type] = {}

                        growth_rates[period_type][metric] = {
                            "periods": periods_list,
                            "values": values_np.tolist(),
                            "growth_rates": growth_np.tolist(),
                            "cagr_1yr": float(cagr_1yr)
                            if not np.isnan(cagr_1yr)
                            else None,
                            "cagr_3yr": float(cagr_3yr)
                            if not np.isnan(cagr_3yr)
                            else None,
                            "cagr_5yr": float(cagr_5yr)
                            if not np.isnan(cagr_5yr)
                            else None,
                        }

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "growth_rates": growth_rates,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def analyze_growth_trends(
        self, growth_rates: Dict[str, Any], min_periods: Optional[int] = 4
    ) -> Dict[str, Any]:
        """
        Analyze growth trends and patterns from historical data.

        Args:
            growth_rates: Dictionary of calculated growth rates
            min_periods: Minimum number of periods required for trend analysis

        Returns:
            Dictionary with growth trend analysis
        """
        start_time = time.time()

        # Determine if we should use GPU
        use_gpu = False

        # Count total number of metrics and periods
        metric_period_count = 0
        for period_type, metrics in growth_rates.items():
            for metric, data in metrics.items():
                if len(data.get("growth_rates", [])) >= min_periods:
                    metric_period_count += 1

        use_gpu = self.gpu_available and metric_period_count >= self.min_periods

        trends = {}

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for growth trend analysis")

                for period_type, metrics in growth_rates.items():
                    trends[period_type] = {}

                    for metric, data in metrics.items():
                        growth_values = data.get("growth_rates", [])

                        # Skip if not enough data points
                        if len(growth_values) < min_periods:
                            continue

                        # Convert to GPU array
                        growth_gpu = cp.array(
                            [v for v in growth_values if not np.isnan(v)],
                            dtype=cp.float64,
                        )

                        if len(growth_gpu) < min_periods:
                            continue

                        # Calculate trend metrics
                        mean_growth = cp.mean(growth_gpu)
                        median_growth = cp.median(growth_gpu)
                        std_dev = cp.std(growth_gpu)
                        volatility = std_dev / (cp.abs(mean_growth) + 1e-10)

                        # Calculate linear regression for trend direction
                        x = cp.arange(len(growth_gpu), dtype=cp.float64)
                        A = cp.vstack([x, cp.ones(len(x))]).T
                        slope, _ = cp.linalg.lstsq(A, growth_gpu, rcond=None)[0]

                        # Determine trend direction
                        if slope > 0.01:
                            trend_direction = "accelerating"
                        elif slope < -0.01:
                            trend_direction = "decelerating"
                        else:
                            trend_direction = "stable"

                        #
                        # Calculate consistency (percentage of periods with
                        # positive growth)
                        positive_periods = cp.sum(growth_gpu > 0)
                        consistency = float(positive_periods / len(growth_gpu))

                        # Store results
                        trends[period_type][metric] = {
                            "mean_growth": float(mean_growth),
                            "median_growth": float(median_growth),
                            "std_dev": float(std_dev),
                            "volatility": float(volatility),
                            "trend_direction": trend_direction,
                            "consistency": consistency,
                            "slope": float(slope),
                        }

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for growth trend analysis: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            for period_type, metrics in growth_rates.items():
                trends[period_type] = {}

                for metric, data in metrics.items():
                    growth_values = data.get("growth_rates", [])

                    # Skip if not enough data points
                    if len(growth_values) < min_periods:
                        continue

                    # Filter out NaN values
                    growth_np = np.array(
                        [v for v in growth_values if not np.isnan(v)], dtype=np.float64
                    )

                    if len(growth_np) < min_periods:
                        continue

                    # Calculate trend metrics
                    mean_growth = np.mean(growth_np)
                    median_growth = np.median(growth_np)
                    std_dev = np.std(growth_np)
                    volatility = std_dev / (np.abs(mean_growth) + 1e-10)

                    # Calculate linear regression for trend direction
                    x = np.arange(len(growth_np))
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, _ = np.linalg.lstsq(A, growth_np, rcond=None)[0]

                    # Determine trend direction
                    if slope > 0.01:
                        trend_direction = "accelerating"
                    elif slope < -0.01:
                        trend_direction = "decelerating"
                    else:
                        trend_direction = "stable"

                    #
                    # Calculate consistency (percentage of periods with
                    # positive growth)
                    positive_periods = np.sum(growth_np > 0)
                    consistency = float(positive_periods / len(growth_np))

                    # Store results
                    trends[period_type][metric] = {
                        "mean_growth": float(mean_growth),
                        "median_growth": float(median_growth),
                        "std_dev": float(std_dev),
                        "volatility": float(volatility),
                        "trend_direction": trend_direction,
                        "consistency": consistency,
                        "slope": float(slope),
                    }

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "trends": trends,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def score_growth_quality(
        self,
        growth_rates: Dict[str, Any],
        trends: Optional[Dict[str, Any]] = None,
        sector: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Score the quality and sustainability of growth.

        Args:
            growth_rates: Dictionary of calculated growth rates
            trends: Dictionary of growth trends analysis (optional)
            sector: Industry sector for benchmark comparison (optional)

        Returns:
            Dictionary with overall growth score and metric scores
        """
        start_time = time.time()

        # Determine if we should use GPU
        use_gpu = False
        metric_count = 0
        for period_type, metrics in growth_rates.items():
            metric_count += len(metrics)
        use_gpu = self.gpu_available and metric_count >= self.min_periods

        metric_scores = {}
        overall_score = 0.0
        total_weight = 0.0

        # Get benchmarks for the sector or use defaults
        benchmarks = self.growth_benchmarks.get(sector, {})
        if not benchmarks:
            self.logger.warning(
                f"No growth benchmarks found for sector: {sector}. Using general comparison."
            )
            # Use average of all sectors as a fallback
            all_benchmarks = list(self.growth_benchmarks.values())
            if all_benchmarks:
                benchmarks = {
                    k: np.mean([b.get(k, np.nan) for b in all_benchmarks])
                    for k in all_benchmarks[0].keys()
                }

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for growth quality scoring")

                for period_type, metrics in growth_rates.items():
                    for metric, data in metrics.items():
                        weight = self.growth_weights.get(metric, 0.0)
                        if weight == 0.0:
                            continue

                        # Get latest growth rate and benchmark
                        latest_growth = data.get("cagr_1yr")
                        benchmark_growth = benchmarks.get(metric)

                        if (
                            latest_growth is not None
                            and benchmark_growth is not None
                            and not cp.isnan(cp.array([benchmark_growth]))[0]
                        ):
                            # Transfer to GPU
                            latest_growth_gpu = cp.array([latest_growth])
                            benchmark_growth_gpu = cp.array([benchmark_growth])
                            weight_gpu = cp.array([weight])

                            # Score based on comparison to benchmark
                            if benchmark_growth_gpu != 0:
                                relative_diff = (
                                    latest_growth_gpu - benchmark_growth_gpu
                                ) / cp.abs(benchmark_growth_gpu)
                                score_gpu = 0.5 + 0.5 * cp.clip(relative_diff, -1, 1)
                            else:
                                score_gpu = (
                                    cp.array([0.75])
                                    if latest_growth_gpu > 0
                                    else cp.array([0.25])
                                )

                            # Adjust score based on trend if available
                            if (
                                trends
                                and period_type in trends
                                and metric in trends[period_type]
                            ):
                                trend_info = trends[period_type][metric]
                                trend_direction = trend_info.get("trend_direction")
                                volatility = trend_info.get("volatility", 1.0)
                                consistency = trend_info.get("consistency", 0.5)

                                # Trend adjustment
                                if trend_direction == "accelerating":
                                    score_gpu *= 1.1
                                elif trend_direction == "decelerating":
                                    score_gpu *= 0.9

                                #
                                # Volatility adjustment (lower volatility is
                                # better)
                                score_gpu *= cp.clip(1.0 - volatility * 0.5, 0.5, 1.5)

                                # Consistency adjustment
                                score_gpu *= 0.8 + consistency * 0.4

                            # Clip score to [0, 1]
                            score_gpu = cp.clip(score_gpu, 0, 1)

                            # Store metric score
                            metric_scores[f"{period_type}_{metric}"] = float(
                                score_gpu[0]
                            )

                            # Add to overall score
                            overall_score += float(score_gpu[0] * weight_gpu[0])
                            total_weight += float(weight_gpu[0])

                # Normalize overall score
                if total_weight > 0:
                    final_overall_score = overall_score / total_weight
                else:
                    final_overall_score = 0.5

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for growth quality scoring: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            for period_type, metrics in growth_rates.items():
                for metric, data in metrics.items():
                    weight = self.growth_weights.get(metric, 0.0)
                    if weight == 0.0:
                        continue

                    # Get latest growth rate and benchmark
                    latest_growth = data.get("cagr_1yr")
                    benchmark_growth = benchmarks.get(metric)

                    if (
                        latest_growth is not None
                        and benchmark_growth is not None
                        and not np.isnan(benchmark_growth)
                    ):
                        # Score based on comparison to benchmark
                        if benchmark_growth != 0:
                            relative_diff = (latest_growth - benchmark_growth) / abs(
                                benchmark_growth
                            )
                            score = 0.5 + 0.5 * np.clip(relative_diff, -1, 1)
                        else:
                            score = 0.75 if latest_growth > 0 else 0.25

                        # Adjust score based on trend if available
                        if (
                            trends
                            and period_type in trends
                            and metric in trends[period_type]
                        ):
                            trend_info = trends[period_type][metric]
                            trend_direction = trend_info.get("trend_direction")
                            volatility = trend_info.get("volatility", 1.0)
                            consistency = trend_info.get("consistency", 0.5)

                            # Trend adjustment
                            if trend_direction == "accelerating":
                                score *= 1.1
                            elif trend_direction == "decelerating":
                                score *= 0.9

                            #
                            # Volatility adjustment (lower volatility is
                            # better)
                            score *= np.clip(1.0 - volatility * 0.5, 0.5, 1.5)

                            # Consistency adjustment
                            score *= 0.8 + consistency * 0.4

                        # Clip score to [0, 1]
                        score = np.clip(score, 0, 1)

                        # Store metric score
                        metric_scores[f"{period_type}_{metric}"] = float(score)

                        # Add to overall score
                        overall_score += score * weight
                        total_weight += weight

            # Normalize overall score
            if total_weight > 0:
                final_overall_score = overall_score / total_weight
            else:
                final_overall_score = 0.5

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "overall_score": float(np.clip(final_overall_score, 0, 1)),
            "metric_scores": metric_scores,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def compare_to_sector_growth(
        self, growth_rates: Dict[str, Any], sector: str
    ) -> Dict[str, Any]:
        """
        Compare company's growth metrics to sector benchmarks.

        Args:
            growth_rates: Dictionary of company's growth rates
            sector: Industry sector of the company

        Returns:
            Dictionary comparing each growth metric to its sector benchmark
        """
        start_time = time.time()

        # Get benchmarks for the sector
        benchmarks = self.growth_benchmarks.get(sector)

        if not benchmarks:
            return {
                "comparison": {},
                "error": f"No growth benchmarks found for sector: {sector}",
                "processing_time": time.time() - start_time,
            }

        # Determine if we should use GPU
        use_gpu = False
        metric_count = 0
        for period_type, metrics in growth_rates.items():
            metric_count += len(metrics)
        use_gpu = self.gpu_available and metric_count >= self.min_periods

        comparison = {}

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for sector growth comparison")

                for period_type, metrics in growth_rates.items():
                    comparison[period_type] = {}
                    for metric, data in metrics.items():
                        company_growth = data.get(
                            "cagr_1yr"
                        )  # Use 1-year CAGR for comparison
                        benchmark_growth = benchmarks.get(metric)

                        if (
                            company_growth is not None
                            and benchmark_growth is not None
                            and not cp.isnan(cp.array([benchmark_growth]))[0]
                        ):
                            # Transfer to GPU
                            company_growth_gpu = cp.array([company_growth])
                            benchmark_growth_gpu = cp.array([benchmark_growth])

                            # Calculate difference and relative difference
                            difference = company_growth_gpu - benchmark_growth_gpu
                            epsilon = 1e-10
                            with cp.errstate(divide="ignore", invalid="ignore"):
                                relative_difference = (
                                    difference
                                    / (cp.abs(benchmark_growth_gpu) + epsilon)
                                ) * 100

                            # Extract CPU values
                            difference_val = float(difference[0])
                            relative_difference_val = (
                                float(relative_difference[0])
                                if cp.isfinite(relative_difference[0])
                                else None
                            )

                            # Determine status
                            status = (
                                "better"
                                if difference_val > 0
                                else "worse"
                                if difference_val < 0
                                else "in_line"
                            )

                            comparison[period_type][metric] = {
                                "company_growth": float(company_growth),
                                "benchmark_growth": float(benchmark_growth),
                                "difference": difference_val,
                                "relative_difference_pct": relative_difference_val,
                                "status": status,
                            }
                        else:
                            comparison[period_type][metric] = {
                                "company_growth": float(company_growth)
                                if company_growth is not None
                                else None,
                                "benchmark_growth": float(benchmark_growth)
                                if benchmark_growth is not None
                                else None,
                                "status": "cannot_compare",
                            }

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for sector growth comparison: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            for period_type, metrics in growth_rates.items():
                comparison[period_type] = {}
                for metric, data in metrics.items():
                    company_growth = data.get(
                        "cagr_1yr"
                    )  # Use 1-year CAGR for comparison
                    benchmark_growth = benchmarks.get(metric)

                    if (
                        company_growth is not None
                        and benchmark_growth is not None
                        and not np.isnan(benchmark_growth)
                    ):
                        # Calculate difference and relative difference
                        difference = company_growth - benchmark_growth
                        epsilon = 1e-10
                        relative_difference = (
                            difference / (abs(benchmark_growth) + epsilon)
                        ) * 100

                        # Determine status
                        status = (
                            "better"
                            if difference > 0
                            else "worse"
                            if difference < 0
                            else "in_line"
                        )

                        comparison[period_type][metric] = {
                            "company_growth": float(company_growth),
                            "benchmark_growth": float(benchmark_growth),
                            "difference": float(difference),
                            "relative_difference_pct": float(relative_difference)
                            if np.isfinite(relative_difference)
                            else None,
                            "status": status,
                        }
                    else:
                        comparison[period_type][metric] = {
                            "company_growth": float(company_growth)
                            if company_growth is not None
                            else None,
                            "benchmark_growth": float(benchmark_growth)
                            if benchmark_growth is not None
                            else None,
                            "status": "cannot_compare",
                        }

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "comparison": comparison,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def analyze_earnings_report(
        self,
        report_data: Dict[str, Any],
        historical_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an earnings report for growth implications.

        Args:
            report_data: Dictionary containing earnings report data
            historical_data: Dictionary containing historical financial data for comparison

        Returns:
            Dictionary with earnings report analysis
        """
        start_time = time.time()

        # Validate input
        if not report_data:
            return {
                "error": "No report data provided",
                "processing_time": time.time() - start_time,
                "computation": "cpu",
            }

        # Extract key metrics from report data
        actual_revenue = report_data.get("revenue", 0)
        estimated_revenue = report_data.get("estimated_revenue", 0)
        actual_eps = report_data.get("earnings_per_share", 0)
        estimated_eps = report_data.get("estimated_eps", 0)
        previous_guidance = report_data.get("previous_guidance", {})
        current_guidance = report_data.get("guidance", {})
        quarter = report_data.get("period", "")

        # Calculate surprises
        revenue_surprise = 0
        if estimated_revenue > 0:
            revenue_surprise = (actual_revenue - estimated_revenue) / estimated_revenue

        eps_surprise = 0
        if estimated_eps > 0:
            eps_surprise = (actual_eps - estimated_eps) / estimated_eps

        # Determine guidance change
        guidance_change = "unchanged"
        guidance_metrics = ["revenue", "eps", "operating_margin", "growth_rate"]

        if current_guidance and previous_guidance:
            guidance_diffs = []
            for metric in guidance_metrics:
                if metric in current_guidance and metric in previous_guidance:
                    current = current_guidance[metric]
                    previous = previous_guidance[metric]
                    if isinstance(current, (int, float)) and isinstance(
                        previous, (int, float)
                    ):
                        pct_change = (
                            (current - previous) / previous if previous != 0 else 0
                        )
                        guidance_diffs.append(pct_change)

            # Determine overall guidance direction
            if guidance_diffs:
                avg_change = sum(guidance_diffs) / len(guidance_diffs)
                if avg_change > 0.02:  # 2% threshold for "raised"
                    guidance_change = "raised"
                elif avg_change < -0.02:  # -2% threshold for "lowered"
                    guidance_change = "lowered"

        # Compare to historical data if available
        growth_trend = "stable"
        confidence = 0.5

        if historical_data:
            # Calculate sequential and year-over-year growth
            current_period_data = report_data
            previous_periods = historical_data.get("quarterly", {})

            # Get previous quarter and same quarter last year
            previous_quarter = None
            year_ago_quarter = None

            # Find the right periods for comparison
            for period, data in previous_periods.items():
                if period == quarter.replace("Q", "Q-1"):  # Previous quarter
                    previous_quarter = data
                elif period == quarter.replace(
                    str(datetime.now().year), str(datetime.now().year - 1)
                ):  # Same quarter last year
                    year_ago_quarter = data

            # Calculate sequential growth
            sequential_growth = {}
            if previous_quarter:
                for metric in ["revenue", "net_income", "operating_income"]:
                    if metric in current_period_data and metric in previous_quarter:
                        current_value = current_period_data.get(metric, 0)
                        previous_value = previous_quarter.get(metric, 0)
                        if previous_value != 0:
                            sequential_growth[metric] = (
                                current_value - previous_value
                            ) / previous_value

            # Calculate year-over-year growth
            yoy_growth = {}
            if year_ago_quarter:
                for metric in ["revenue", "net_income", "operating_income"]:
                    if metric in current_period_data and metric in year_ago_quarter:
                        current_value = current_period_data.get(metric, 0)
                        previous_value = year_ago_quarter.get(metric, 0)
                        if previous_value != 0:
                            yoy_growth[metric] = (
                                current_value - previous_value
                            ) / previous_value

            # Determine growth trend
            if yoy_growth:
                avg_yoy_growth = sum(yoy_growth.values()) / len(yoy_growth)
                avg_sequential_growth = (
                    sum(sequential_growth.values()) / len(sequential_growth)
                    if sequential_growth
                    else 0
                )

                if avg_yoy_growth > 0.15:  # 15% YoY growth threshold for "accelerating"
                    growth_trend = (
                        "accelerating" if avg_sequential_growth > 0 else "decelerating"
                    )
                    confidence = 0.8
                elif avg_yoy_growth < 0:
                    growth_trend = "contracting"
                    confidence = 0.7
                else:
                    growth_trend = "stable"
                    confidence = 0.6

        # Generate key takeaways
        key_takeaways = []

        # Revenue takeaway
        if revenue_surprise > 0.05:
            key_takeaways.append(
                f"Strong revenue growth with {revenue_surprise:.1%} beat against estimates."
            )
        elif revenue_surprise > 0:
            key_takeaways.append(
                f"Revenue slightly above expectations with {revenue_surprise:.1%} beat."
            )
        elif revenue_surprise < -0.05:
            key_takeaways.append(
                f"Significant revenue miss with {-revenue_surprise:.1%} below estimates."
            )
        else:
            key_takeaways.append("Revenue in line with expectations.")

        # EPS takeaway
        if eps_surprise > 0.10:
            key_takeaways.append(f"Strong earnings with {eps_surprise:.1%} EPS beat.")
        elif eps_surprise > 0:
            key_takeaways.append(
                f"Earnings slightly above expectations with {eps_surprise:.1%} EPS beat."
            )
        elif eps_surprise < -0.10:
            key_takeaways.append(
                f"Significant earnings miss with {-eps_surprise:.1%} below estimates."
            )
        else:
            key_takeaways.append("Earnings in line with expectations.")

        # Guidance takeaway
        if guidance_change == "raised":
            key_takeaways.append(
                "Management raised future guidance, indicating confidence in continued growth."
            )
        elif guidance_change == "lowered":
            key_takeaways.append(
                "Management lowered future guidance, suggesting caution about future performance."
            )
        else:
            key_takeaways.append("Management maintained previous guidance.")

        # Growth trend takeaway
        if growth_trend == "accelerating":
            key_takeaways.append("Growth is accelerating compared to previous periods.")
        elif growth_trend == "contracting":
            key_takeaways.append("Growth is contracting compared to previous periods.")

        # Create comprehensive analysis
        analysis = {
            "revenue_surprise": float(revenue_surprise),
            "eps_surprise": float(eps_surprise),
            "guidance_change": guidance_change,
            "growth_trend": growth_trend,
            "key_takeaways": key_takeaways,
        }

        # Determine growth implications
        short_term = "neutral"
        if revenue_surprise > 0.03 and eps_surprise > 0.03:
            short_term = "positive"
        elif revenue_surprise < -0.03 and eps_surprise < -0.03:
            short_term = "negative"

        long_term = "neutral"
        if guidance_change == "raised" or growth_trend == "accelerating":
            long_term = "positive"
        elif guidance_change == "lowered" or growth_trend == "contracting":
            long_term = "negative"

        growth_implications = {
            "short_term": short_term,
            "long_term": long_term,
            "confidence": float(confidence),
        }

        computation_type = "cpu"  # Earnings report analysis is CPU-bound

        processing_time = time.time() - start_time
        return {
            "analysis": analysis,
            "growth_implications": growth_implications,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def forecast_future_growth(
        self,
        growth_rates: Dict[str, Any],
        periods_ahead: int = 4,
        confidence_interval: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Forecast future growth based on historical patterns using time series models.

        Args:
            growth_rates: Dictionary of historical growth rates
            periods_ahead: Number of periods to forecast ahead
            confidence_interval: Confidence interval for forecasts (0.0-1.0)

        Returns:
            Dictionary with growth forecasts and confidence intervals
        """
        start_time = time.time()

        try:
            # Check for statsmodels for time series forecasting
            import statsmodels.api as sm
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            from statsmodels.tsa.arima.model import ARIMA

            HAVE_STATSMODELS = True
        except ImportError:
            self.logger.warning("statsmodels not available, using simple forecasting")
            HAVE_STATSMODELS = False

        forecasts = {}
        confidence_intervals = {}
        computation_type = "cpu"  # Time series forecasting is CPU-bound

        # Process each period type and metric
        for period_type, metrics in growth_rates.items():
            forecasts[period_type] = {}
            confidence_intervals[period_type] = {}

            for metric, data in metrics.items():
                growth_values = data.get("growth_rates", [])

                # Filter out NaN values
                valid_growth = [g for g in growth_values if not np.isnan(g)]

                # Need at least 4 data points for meaningful forecasting
                if len(valid_growth) < 4:
                    self.logger.warning(
                        f"Not enough data points for {metric} forecasting"
                    )
                    forecasts[period_type][metric] = [0.0] * periods_ahead
                    confidence_intervals[period_type][metric] = [
                        (0.0, 0.0)
                    ] * periods_ahead
                    continue

                # Convert to numpy array for analysis
                growth_array = np.array(valid_growth)

                if HAVE_STATSMODELS:
                    try:
                        #
                        # Determine the best forecasting method based on data
                        # characteristics
                        # Check for trend and seasonality
                        has_trend = self._check_for_trend(growth_array)

                        if has_trend:
                            # Use ARIMA for data with trend
                            #
                            # Simple auto ARIMA-like process to find p,d,q
                            # parameters
                            best_aic = float("inf")
                            best_model = None
                            best_params = None

                            # Try different ARIMA configurations
                            for p in range(0, 3):
                                for d in range(0, 2):
                                    for q in range(0, 3):
                                        try:
                                            model = ARIMA(growth_array, order=(p, d, q))
                                            model_fit = model.fit()

                                            if model_fit.aic < best_aic:
                                                best_aic = model_fit.aic
                                                best_model = model_fit
                                                best_params = (p, d, q)
                                        except:
                                            continue

                            if best_model is not None:
                                # Generate forecast
                                forecast_result = best_model.forecast(
                                    steps=periods_ahead
                                )
                                forecast_values = forecast_result

                                # Calculate prediction intervals
                                pred_intervals = best_model.get_forecast(
                                    steps=periods_ahead
                                ).conf_int(alpha=1 - confidence_interval)
                                intervals = [
                                    (float(lower), float(upper))
                                    for lower, upper in zip(
                                        pred_intervals[:, 0], pred_intervals[:, 1]
                                    )
                                ]

                                self.logger.info(
                                    f"ARIMA({best_params[0]},{best_params[1]},{best_params[2]}) selected for {metric}"
                                )
                            else:
                                # Fallback to exponential smoothing
                                model = ExponentialSmoothing(
                                    growth_array, trend="add", seasonal=None
                                )
                                model_fit = model.fit()
                                forecast_values = model_fit.forecast(periods_ahead)

                                # Simple approximation of confidence intervals
                                std_dev = np.std(growth_array)
                                z_value = 1.96  # Approximately 95% confidence
                                half_width = z_value * std_dev
                                intervals = [
                                    (float(f - half_width), float(f + half_width))
                                    for f in forecast_values
                                ]
                        else:
                            # Use Exponential Smoothing for stable data
                            model = ExponentialSmoothing(
                                growth_array, trend=None, seasonal=None
                            )
                            model_fit = model.fit()
                            forecast_values = model_fit.forecast(periods_ahead)

                            # Simple approximation of confidence intervals
                            std_dev = np.std(growth_array)
                            z_value = 1.96  # Approximately 95% confidence
                            half_width = z_value * std_dev
                            intervals = [
                                (float(f - half_width), float(f + half_width))
                                for f in forecast_values
                            ]

                        # Store the forecasts and intervals
                        forecasts[period_type][metric] = [
                            float(f) for f in forecast_values
                        ]
                        confidence_intervals[period_type][metric] = intervals

                    except Exception as e:
                        self.logger.error(
                            f"Error in time series forecasting for {metric}: {e}, falling back to simple method"
                        )
                        # Fallback to simple method
                        HAVE_STATSMODELS = False

                # Simple forecasting method (fallback)
                if not HAVE_STATSMODELS:
                    # Use the average of recent values with a trend factor
                    recent_growth = growth_array[-4:]  # Last 4 periods
                    avg_growth = np.mean(recent_growth)

                    # Check for trend in recent values
                    if len(recent_growth) >= 3:
                        # Simple linear regression to detect trend
                        x = np.arange(len(recent_growth))
                        slope, _ = np.polyfit(x, recent_growth, 1)

                        # Apply trend factor to forecast
                        forecast_values = [
                            avg_growth + slope * i for i in range(1, periods_ahead + 1)
                        ]
                    else:
                        forecast_values = [avg_growth] * periods_ahead

                    #
                    # Simple confidence intervals based on historical
                    # volatility
                    std_dev = np.std(growth_array)
                    z_value = 1.96  # Approximately 95% confidence
                    half_width = z_value * std_dev
                    intervals = [
                        (f - half_width, f + half_width) for f in forecast_values
                    ]

                    # Store the forecasts and intervals
                    forecasts[period_type][metric] = forecast_values
                    confidence_intervals[period_type][metric] = intervals

        processing_time = time.time() - start_time

        return {
            "forecasts": forecasts,
            "confidence_intervals": confidence_intervals,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def _check_for_trend(self, data: np.ndarray) -> bool:
        """
        Check if a time series has a significant trend.

        Args:
            data: Numpy array of time series data

        Returns:
            Boolean indicating if a trend is present
        """
        # Use simple linear regression to detect trend
        if len(data) < 4:
            return False

        try:
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)

            # Calculate R-squared to determine trend significance
            y_pred = slope * x + intercept
            ss_total = np.sum((data - np.mean(data)) ** 2)
            ss_residual = np.sum((data - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            #
            # Consider a trend significant if R-squared > 0.3 and absolute
            # slope > 0.01
            return r_squared > 0.3 and abs(slope) > 0.01
        except:
            return False

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for GPU/CPU usage during growth calculations.

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

        # Calculate average processing times
        avg_total_time = 0.0
        if total_executions > 0:
            avg_total_time = (
                self.execution_stats["total_processing_time"] / total_executions
            )

        avg_gpu_time = 0.0
        if self.execution_stats["gpu_executions"] > 0:
            avg_gpu_time = (
                self.execution_stats["gpu_processing_time"]
                / self.execution_stats["gpu_executions"]
            )

        avg_cpu_time = 0.0
        if self.execution_stats["cpu_executions"] > 0:
            avg_cpu_time = (
                self.execution_stats["cpu_processing_time"]
                / self.execution_stats["cpu_executions"]
            )

        # Calculate speedup if both CPU and GPU have been used
        speedup = 0.0
        if avg_cpu_time > 0 and avg_gpu_time > 0:
            speedup = avg_cpu_time / avg_gpu_time

        return {
            "gpu_executions": self.execution_stats["gpu_executions"],
            "cpu_executions": self.execution_stats["cpu_executions"],
            "gpu_failures": self.execution_stats["gpu_failures"],
            "total_executions": total_executions,
            "gpu_percentage": round(gpu_percentage, 2),
            "cpu_percentage": round(cpu_percentage, 2),
            "total_processing_time": round(
                self.execution_stats["total_processing_time"], 4
            ),
            "gpu_processing_time": round(
                self.execution_stats["gpu_processing_time"], 4
            ),
            "cpu_processing_time": round(
                self.execution_stats["cpu_processing_time"], 4
            ),
            "avg_total_time": round(avg_total_time, 4),
            "avg_gpu_time": round(avg_gpu_time, 4),
            "avg_cpu_time": round(avg_cpu_time, 4),
            "gpu_speedup": round(speedup, 2) if speedup > 0 else None,
            "gpu_available": self.gpu_available,
            "min_company_batch": self.min_company_batch,
            "min_periods": self.min_periods,
        }


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {}

    # Create and start the server
    server = GrowthAnalysisMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("GrowthAnalysisMCP server started")

    # Example usage
    sample_historical_data = {
        "annual": {
            "2020": {"revenue": 1000, "net_income": 100, "free_cash_flow": 80},
            "2021": {"revenue": 1200, "net_income": 130, "free_cash_flow": 100},
            "2022": {"revenue": 1500, "net_income": 180, "free_cash_flow": 140},
            "2023": {"revenue": 1800, "net_income": 220, "free_cash_flow": 170},
            "2024": {"revenue": 2100, "net_income": 270, "free_cash_flow": 200},
        }
    }

    # Calculate growth rates
    growth_result = server.calculate_growth_rates(sample_historical_data)
    print(
        f"Calculated Growth Rates: {json.dumps(growth_result['growth_rates'], indent=2)}"
    )
    print(f"Computation: {growth_result.get('computation', 'cpu')}")

    # Analyze trends
    trends_result = server.analyze_growth_trends(growth_result["growth_rates"])
    print(f"Growth Trends: {json.dumps(trends_result['trends'], indent=2)}")
    print(f"Computation: {trends_result.get('computation', 'cpu')}")

    # Score growth quality
    score_result = server.score_growth_quality(
        growth_result["growth_rates"], trends_result["trends"], sector="Technology"
    )
    print(f"Growth Quality Score: {json.dumps(score_result, indent=2)}")
    print(f"Computation: {score_result.get('computation', 'cpu')}")

    # Compare to sector
    comparison_result = server.compare_to_sector_growth(
        growth_result["growth_rates"], sector="Technology"
    )
    print(
        f"Sector Growth Comparison: {json.dumps(comparison_result['comparison'], indent=2)}"
    )
    print(f"Computation: {comparison_result.get('computation', 'cpu')}")

    # Forecast growth
    forecast_result = server.forecast_future_growth(growth_result["growth_rates"])
    print(f"Growth Forecast: {json.dumps(forecast_result['forecasts'], indent=2)}")
    print(f"Computation: {forecast_result.get('computation', 'cpu')}")

    # Get execution stats
    stats = server.get_execution_stats()
    print("\nExecution Statistics:")
    print(f"  GPU Available: {stats['gpu_available']}")
    print(f"  GPU Executions: {stats['gpu_executions']} ({stats['gpu_percentage']}%)")
    print(f"  CPU Executions: {stats['cpu_executions']} ({stats['cpu_percentage']}%)")
    print(f"  GPU Failures: {stats['gpu_failures']}")
    if stats["gpu_speedup"]:
        print(f"  GPU Speedup: {stats['gpu_speedup']}x")
