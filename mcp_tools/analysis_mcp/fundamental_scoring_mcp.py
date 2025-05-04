"""
Fundamental Scoring MCP Tool

This module implements a Model Context Protocol (MCP) server that provides
fundamental analysis scoring capabilities based on financial statements and metrics.
It supports GPU acceleration for faster processing of large datasets.
"""

import os
import json
import numpy as np
import time
from typing import Dict, List, Any, Optional

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
    service_name="analysis-mcp-fundamental-scoring",
    enable_prometheus=True,
    enable_loki=True,
    default_labels={"component": "mcp_tools/analysis_mcp"},
)


class FundamentalScoringMCP(BaseMCPServer):
    """
    MCP server for calculating fundamental scores and metrics.

    This tool analyzes financial statements (income statement, balance sheet, cash flow)
    to calculate key ratios, value metrics, and financial health scores.
    It supports GPU acceleration for faster processing of large datasets.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Fundamental Scoring MCP server.

        Args:
            config: Optional configuration dictionary, may contain:
                - sector_benchmarks_path: Path to sector benchmark data
                - scoring_weights_path: Path to scoring model weights
                - cache_dir: Directory for caching intermediate results
                - use_gpu: Whether to use GPU acceleration (default: True)
                - min_company_batch: Minimum number of companies to process with GPU (default: 10)
                - min_ratio_count: Minimum number of ratios to process with GPU (default: 20)
        """
        super().__init__(name="fundamental_scoring_mcp", config=config)

        # Set default configurations
        self.sector_benchmarks_path = self.config.get(
            "sector_benchmarks_path",
            os.path.join(os.path.dirname(__file__), "data/sector_benchmarks.json"),
        )
        self.scoring_weights_path = self.config.get(
            "scoring_weights_path",
            os.path.join(
                os.path.dirname(__file__), "data/fundamental_scoring_weights.json"
            ),
        )
        self.cache_dir = self.config.get("cache_dir", "./fundamental_cache")

        # Create cache directory if needed
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)

        # GPU configuration
        self.use_gpu = self.config.get("use_gpu", True)

        # Check if we have GPU and whether we should use it
        self.gpu_available = HAVE_GPU and self.use_gpu

        # Check for A100 GPU if GPU is available
        self.has_a100 = False
        if self.gpu_available:
            try:
                gpu_info = cp.cuda.runtime.getDeviceProperties(0)
                gpu_name = gpu_info["name"].decode("utf-8")
                self.has_a100 = "A100" in gpu_name
                self.logger.info(f"GPU detected: {gpu_name}")

                # Set GPU-specific parameters
                if self.has_a100:
                    #
                    # A100 is more efficient with GPU computation even for
                    # smaller datasets
                    self.min_company_batch = self.config.get(
                        "min_company_batch", 5
                    )  # Smaller batch size threshold for A100
                    self.min_ratio_count = self.config.get(
                        "min_ratio_count", 10
                    )  # Smaller ratio count threshold for A100
                    self.logger.info(
                        "A100 GPU detected - optimizing for high throughput financial analysis"
                    )
                else:
                    self.min_company_batch = self.config.get(
                        "min_company_batch", 10
                    )  # Default batch size threshold
                    self.min_ratio_count = self.config.get(
                        "min_ratio_count", 20
                    )  # Default ratio count threshold
                    self.logger.info("Standard GPU detected")
            except Exception as e:
                self.logger.warning(f"Failed to detect GPU type: {e}")
                self.gpu_available = False  # Disable GPU if detection fails
                self.min_company_batch = self.config.get("min_company_batch", 10)
                self.min_ratio_count = self.config.get("min_ratio_count", 20)
        else:
            self.min_company_batch = self.config.get("min_company_batch", 10)
            self.min_ratio_count = self.config.get("min_ratio_count", 20)

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
        self._load_sector_benchmarks()
        self._load_scoring_weights()

        # Register tools
        self._register_tools()

        self.logger.info(
            f"FundamentalScoringMCP initialized (GPU enabled: {self.gpu_available})"
        )

    def _load_sector_benchmarks(self):
        """Load sector benchmark data for comparison."""
        self.sector_benchmarks = {}

        if os.path.exists(self.sector_benchmarks_path):
            try:
                with open(self.sector_benchmarks_path, "r") as f:
                    self.sector_benchmarks = json.load(f)
                self.logger.info(
                    f"Loaded sector benchmarks from {self.sector_benchmarks_path}"
                )
            except Exception as e:
                self.logger.error(f"Error loading sector benchmarks: {e}")
                self._create_default_benchmarks()
        else:
            self.logger.warning(
                f"Sector benchmarks file not found: {self.sector_benchmarks_path}"
            )
            self._create_default_benchmarks()

    def _create_default_benchmarks(self):
        """Create default sector benchmarks."""
        self.sector_benchmarks = {
            "Technology": {
                "pe_ratio": 25.0,
                "pb_ratio": 5.0,
                "roe": 15.0,
                "debt_equity": 0.5,
            },
            "Financial Services": {
                "pe_ratio": 15.0,
                "pb_ratio": 1.5,
                "roe": 10.0,
                "debt_equity": 1.0,
            },
            "Healthcare": {
                "pe_ratio": 20.0,
                "pb_ratio": 4.0,
                "roe": 12.0,
                "debt_equity": 0.6,
            },
            "Consumer Cyclical": {
                "pe_ratio": 18.0,
                "pb_ratio": 3.0,
                "roe": 14.0,
                "debt_equity": 0.7,
            },
            "Consumer Defensive": {
                "pe_ratio": 16.0,
                "pb_ratio": 2.5,
                "roe": 13.0,
                "debt_equity": 0.4,
            },
            "Industrials": {
                "pe_ratio": 17.0,
                "pb_ratio": 2.8,
                "roe": 11.0,
                "debt_equity": 0.8,
            },
            "Energy": {
                "pe_ratio": 12.0,
                "pb_ratio": 1.8,
                "roe": 8.0,
                "debt_equity": 0.9,
            },
            "Utilities": {
                "pe_ratio": 14.0,
                "pb_ratio": 1.6,
                "roe": 9.0,
                "debt_equity": 1.2,
            },
            "Real Estate": {
                "pe_ratio": 19.0,
                "pb_ratio": 2.0,
                "roe": 7.0,
                "debt_equity": 1.5,
            },
            "Basic Materials": {
                "pe_ratio": 15.0,
                "pb_ratio": 2.2,
                "roe": 10.0,
                "debt_equity": 0.7,
            },
            "Communication Services": {
                "pe_ratio": 22.0,
                "pb_ratio": 4.5,
                "roe": 16.0,
                "debt_equity": 0.6,
            },
        }
        self.logger.info("Created default sector benchmarks")

    def _load_scoring_weights(self):
        """Load weights for the fundamental scoring model."""
        self.scoring_weights = {}

        if os.path.exists(self.scoring_weights_path):
            try:
                with open(self.scoring_weights_path, "r") as f:
                    self.scoring_weights = json.load(f)
                self.logger.info(
                    f"Loaded scoring weights from {self.scoring_weights_path}"
                )
            except Exception as e:
                self.logger.error(f"Error loading scoring weights: {e}")
                self._create_default_weights()
        else:
            self.logger.warning(
                f"Scoring weights file not found: {self.scoring_weights_path}"
            )
            self._create_default_weights()

    def _create_default_weights(self):
        """Create default weights for the scoring model."""
        self.scoring_weights = {
            "profitability": {
                "gross_margin": 0.15,
                "operating_margin": 0.20,
                "net_margin": 0.15,
                "roe": 0.25,
                "roa": 0.15,
                "roic": 0.10,
            },
            "valuation": {
                "pe_ratio": 0.25,
                "pb_ratio": 0.20,
                "ps_ratio": 0.15,
                "ev_ebitda": 0.20,
                "dividend_yield": 0.10,
                "fcf_yield": 0.10,
            },
            "solvency": {
                "debt_equity": 0.30,
                "current_ratio": 0.25,
                "quick_ratio": 0.20,
                "interest_coverage": 0.25,
            },
            "efficiency": {
                "asset_turnover": 0.30,
                "inventory_turnover": 0.25,
                "receivables_turnover": 0.25,
                "days_sales_outstanding": 0.20,
            },
            "growth": {
                "revenue_growth": 0.30,
                "earnings_growth": 0.30,
                "fcf_growth": 0.20,
                "dividend_growth": 0.20,
            },
            "overall_score_weights": {
                "profitability": 0.25,
                "valuation": 0.20,
                "solvency": 0.20,
                "efficiency": 0.15,
                "growth": 0.20,
            },
        }
        self.logger.info("Created default fundamental scoring weights")

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        self.register_tool(
            "calculate_ratios",
            self.calculate_ratios,
            "Calculate key financial ratios from financial statement data",
            {
                "financial_data": {
                    "type": "object",
                    "description": "Dictionary containing financial statement data (income, balance sheet, cash flow)",
                },
                "ratios": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ratios to calculate (default: all common ratios)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "ratios": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "score_financial_health",
            self.score_financial_health,
            "Calculate a financial health score based on key ratios",
            {
                "ratios": {
                    "type": "object",
                    "description": "Dictionary of calculated financial ratios",
                },
                "sector": {
                    "type": "string",
                    "description": "Industry sector for benchmark comparison (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "category_scores": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "calculate_value_metrics",
            self.calculate_value_metrics,
            "Calculate common valuation metrics",
            {
                "market_data": {
                    "type": "object",
                    "description": "Dictionary with market data (price, market cap, shares outstanding)",
                },
                "financial_data": {
                    "type": "object",
                    "description": "Dictionary with financial statement data (revenue, earnings, book value, etc.)",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of value metrics to calculate (default: all common metrics)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "metrics": {"type": "object"},
                    "processing_time": {"type": "number"},
                },
            },
        )

        self.register_tool(
            "compare_to_sector",
            self.compare_to_sector,
            "Compare company's financial ratios to sector benchmarks",
            {
                "ratios": {
                    "type": "object",
                    "description": "Dictionary of company's financial ratios",
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
                },
            },
        )

    def calculate_ratios(
        self, financial_data: Dict[str, Any], ratios: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate key financial ratios from financial statement data with GPU acceleration when beneficial.

        Args:
            financial_data: Dictionary containing financial statement data.
                            Expected keys: 'income_statement', 'balance_sheet', 'cash_flow'.
                            Each should contain relevant line items.
            ratios: Optional list of specific ratios to calculate.

        Returns:
            Dictionary of calculated ratios.
        """
        start_time = time.time()

        # Extract data (handle potential missing keys gracefully)
        income = financial_data.get("income_statement", {})
        balance = financial_data.get("balance_sheet", {})
        cash_flow = financial_data.get("cash_flow", {})

        # Helper function to safely get values
        def get_val(data, key, default=np.nan):
            return data.get(key, default)

        # Determine if we should use GPU
        # For single company ratio calculation, GPU is rarely beneficial
        # We'll optimize for batch processing of multiple companies later
        use_gpu = False

        #
        # If this is a batch of companies (detected by income_statement being a
        # list)
        if isinstance(financial_data.get("income_statement", {}), list):
            batch_size = len(financial_data.get("income_statement", []))
            use_gpu = self.gpu_available and batch_size >= self.min_company_batch

        # For heavily nested dictionaries that might contain many ratios
        if isinstance(financial_data.get("income_statement", {}), dict):
            if len(financial_data.get("income_statement", {})) >= self.min_ratio_count:
                use_gpu = self.gpu_available

        calculated_ratios = {}

        if use_gpu:
            try:
                # GPU implementation for ratio calculation
                self.logger.debug("Using GPU for ratio calculation")

                #
                # Convert key financial metrics to GPU arrays for faster
                # computation
                # Extract the necessary values
                revenue = get_val(income, "revenue")
                cogs = get_val(income, "cost_of_goods_sold", 0)
                operating_income = get_val(income, "operating_income")
                net_income = get_val(income, "net_income")
                total_equity = get_val(balance, "total_equity")
                total_assets = get_val(balance, "total_assets")
                total_debt = get_val(balance, "total_debt")
                current_assets = get_val(balance, "current_assets")
                current_liabilities = get_val(balance, "current_liabilities")
                inventory = get_val(balance, "inventory", 0)
                interest_expense = get_val(income, "interest_expense", 0)
                accounts_receivable = get_val(balance, "accounts_receivable")

                #
                # Create a small epsilon value to prevent division by zero
                # errors
                epsilon = 1e-10

                # Transfer values to GPU
                values_dict = {
                    "revenue": revenue,
                    "cogs": cogs,
                    "operating_income": operating_income,
                    "net_income": net_income,
                    "total_equity": total_equity,
                    "total_assets": total_assets,
                    "total_debt": total_debt,
                    "current_assets": current_assets,
                    "current_liabilities": current_liabilities,
                    "inventory": inventory,
                    "interest_expense": interest_expense,
                    "accounts_receivable": accounts_receivable,
                }

                # Transfer to GPU if not batch data, otherwise keep as is
                gpu_values = {}
                for k, v in values_dict.items():
                    if not isinstance(v, (list, tuple, np.ndarray)):
                        gpu_values[k] = (
                            cp.array([v])
                            if not cp.isnan(cp.array([v]))[0]
                            else cp.array([cp.nan])
                        )
                    else:
                        gpu_values[k] = cp.asarray(v)

                # Calculate gross profit
                if "revenue" in gpu_values and "cogs" in gpu_values:
                    gross_profit = gpu_values["revenue"] - gpu_values["cogs"]
                else:
                    gross_profit = cp.array([cp.nan])

                # Calculate quick assets
                if "current_assets" in gpu_values and "inventory" in gpu_values:
                    quick_assets = (
                        gpu_values["current_assets"] - gpu_values["inventory"]
                    )
                else:
                    quick_assets = cp.array([cp.nan])

                # Calculate ratios on GPU
                ratio_funcs = {
                    "gross_margin": lambda: cp.true_divide(
                        gross_profit, gpu_values["revenue"] + epsilon
                    ),
                    "operating_margin": lambda: cp.true_divide(
                        gpu_values["operating_income"], gpu_values["revenue"] + epsilon
                    ),
                    "net_margin": lambda: cp.true_divide(
                        gpu_values["net_income"], gpu_values["revenue"] + epsilon
                    ),
                    "roe": lambda: cp.true_divide(
                        gpu_values["net_income"], gpu_values["total_equity"] + epsilon
                    ),
                    "roa": lambda: cp.true_divide(
                        gpu_values["net_income"], gpu_values["total_assets"] + epsilon
                    ),
                    "asset_turnover": lambda: cp.true_divide(
                        gpu_values["revenue"], gpu_values["total_assets"] + epsilon
                    ),
                    "debt_equity": lambda: cp.true_divide(
                        gpu_values["total_debt"], gpu_values["total_equity"] + epsilon
                    ),
                    "current_ratio": lambda: cp.true_divide(
                        gpu_values["current_assets"],
                        gpu_values["current_liabilities"] + epsilon,
                    ),
                    "quick_ratio": lambda: cp.true_divide(
                        quick_assets, gpu_values["current_liabilities"] + epsilon
                    ),
                    "interest_coverage": lambda: cp.true_divide(
                        gpu_values["operating_income"],
                        gpu_values["interest_expense"] + epsilon,
                    ),
                    "inventory_turnover": lambda: cp.true_divide(
                        gpu_values["cogs"], gpu_values["inventory"] + epsilon
                    ),
                    "receivables_turnover": lambda: cp.true_divide(
                        gpu_values["revenue"],
                        gpu_values["accounts_receivable"] + epsilon,
                    ),
                }

                # Filter for requested ratios if specified
                ratio_keys = ratios if ratios else ratio_funcs.keys()

                # Calculate the requested ratios
                for ratio_name in ratio_keys:
                    if ratio_name in ratio_funcs:
                        try:
                            ratio_value = ratio_funcs[ratio_name]()
                            # For single values, extract from array
                            if len(ratio_value) == 1:
                                calculated_ratios[ratio_name] = float(ratio_value[0])
                            else:
                                calculated_ratios[ratio_name] = cp.asnumpy(ratio_value)
                        except Exception as e:
                            self.logger.debug(
                                f"GPU calculation failed for ratio {ratio_name}: {e}"
                            )
                            calculated_ratios[ratio_name] = np.nan

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for ratios: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            # Profitability Ratios
            revenue = get_val(income, "revenue")
            cogs = get_val(income, "cost_of_goods_sold", 0)
            gross_profit = revenue - cogs
            operating_income = get_val(income, "operating_income")
            net_income = get_val(income, "net_income")
            total_equity = get_val(balance, "total_equity")
            total_assets = get_val(balance, "total_assets")

            epsilon = 1e-10  # Small value to prevent division by zero

            if revenue > 0:
                calculated_ratios["gross_margin"] = (
                    gross_profit / revenue if not np.isnan(gross_profit) else np.nan
                )
                calculated_ratios["operating_margin"] = (
                    operating_income / revenue
                    if not np.isnan(operating_income)
                    else np.nan
                )
                calculated_ratios["net_margin"] = (
                    net_income / revenue if not np.isnan(net_income) else np.nan
                )

            if total_equity > 0:
                calculated_ratios["roe"] = (
                    net_income / total_equity if not np.isnan(net_income) else np.nan
                )

            if total_assets > 0:
                calculated_ratios["roa"] = (
                    net_income / total_assets if not np.isnan(net_income) else np.nan
                )
                calculated_ratios["asset_turnover"] = (
                    revenue / total_assets if not np.isnan(revenue) else np.nan
                )

            # Solvency Ratios
            total_debt = get_val(balance, "total_debt")
            if total_equity > 0:
                calculated_ratios["debt_equity"] = (
                    total_debt / total_equity if not np.isnan(total_debt) else np.nan
                )

            current_assets = get_val(balance, "current_assets")
            current_liabilities = get_val(balance, "current_liabilities")
            if current_liabilities > 0:
                calculated_ratios["current_ratio"] = (
                    current_assets / current_liabilities
                    if not np.isnan(current_assets)
                    else np.nan
                )

            inventory = get_val(balance, "inventory", 0)
            quick_assets = current_assets - inventory
            if current_liabilities > 0:
                calculated_ratios["quick_ratio"] = (
                    quick_assets / current_liabilities
                    if not np.isnan(quick_assets)
                    else np.nan
                )

            interest_expense = get_val(income, "interest_expense", 0)
            ebit = operating_income  # Approximation
            if interest_expense > 0:
                calculated_ratios["interest_coverage"] = (
                    ebit / interest_expense if not np.isnan(ebit) else np.nan
                )

            # Efficiency Ratios
            if cogs > 0:
                calculated_ratios["inventory_turnover"] = (
                    cogs / inventory
                    if not np.isnan(inventory) and inventory > 0
                    else np.nan
                )

            accounts_receivable = get_val(balance, "accounts_receivable")
            if revenue > 0:
                calculated_ratios["receivables_turnover"] = (
                    revenue / accounts_receivable
                    if not np.isnan(accounts_receivable) and accounts_receivable > 0
                    else np.nan
                )

            # Filter for requested ratios if specified
            if ratios:
                filtered_ratios = {r: calculated_ratios.get(r, np.nan) for r in ratios}
                calculated_ratios = filtered_ratios

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Replace NaN with None for JSON compatibility
        final_ratios = {
            k: (float(v) if not np.isnan(v) else None)
            for k, v in calculated_ratios.items()
        }

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "ratios": final_ratios,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def score_financial_health(
        self, ratios: Dict[str, float], sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate a financial health score based on key ratios with GPU acceleration when beneficial.

        Args:
            ratios: Dictionary of calculated financial ratios.
            sector: Optional industry sector for benchmark comparison.

        Returns:
            Dictionary with overall score and category scores.
        """
        start_time = time.time()

        # Determine if we should use GPU
        # This function is most beneficial on GPU when:
        # 1. We have many ratios to evaluate (>= self.min_ratio_count)
        #
        # 2. We're processing multiple companies in batch mode (input is a
        # list/array)
        use_gpu = False

        # Check if ratios is a list (batch processing)
        is_batch = isinstance(ratios, list)
        if is_batch:
            batch_size = len(ratios)
            use_gpu = self.gpu_available and batch_size >= self.min_company_batch
        else:
            # For single company analysis, check number of ratios
            ratio_count = len(ratios)
            use_gpu = self.gpu_available and ratio_count >= self.min_ratio_count

        category_scores = {}

        # Get benchmarks for the sector or use defaults
        benchmarks = self.sector_benchmarks.get(sector, {})
        if not benchmarks:
            self.logger.warning(
                f"No benchmarks found for sector: {sector}. Using general comparison."
            )
            # Use average of all sectors as a fallback
            all_benchmarks = list(self.sector_benchmarks.values())
            if all_benchmarks:
                benchmarks = {
                    k: np.mean([b.get(k, np.nan) for b in all_benchmarks])
                    for k in all_benchmarks[0].keys()
                }

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for financial health scoring")

                # Define which ratios are 'higher is better'
                higher_is_better = [
                    "gross_margin",
                    "operating_margin",
                    "net_margin",
                    "roe",
                    "roa",
                    "asset_turnover",
                    "inventory_turnover",
                    "receivables_turnover",
                    "interest_coverage",
                    "dividend_yield",
                    "fcf_yield",
                    "revenue_growth",
                    "earnings_growth",
                    "fcf_growth",
                    "dividend_growth",
                ]

                # Define which ratios are 'lower is better'
                lower_is_better = [
                    "pe_ratio",
                    "pb_ratio",
                    "ps_ratio",
                    "ev_ebitda",
                    "debt_equity",
                    "days_sales_outstanding",
                ]

                # Calculate score for each category
                for category, weights in self.scoring_weights.items():
                    if category == "overall_score_weights":
                        continue

                    # Transfer weights to GPU
                    gpu_weights = {}
                    for ratio_name, weight in weights.items():
                        gpu_weights[ratio_name] = cp.array([weight])

                    category_score_gpu = cp.array([0.0])
                    total_weight_gpu = cp.array([0.0])

                    #
                    # Process each ratio in parallel on GPU for better
                    # performance
                    for ratio_name, weight in weights.items():
                        ratio_value = ratios.get(ratio_name)
                        benchmark_value = benchmarks.get(ratio_name)

                        if (
                            ratio_value is not None
                            and benchmark_value is not None
                            and not cp.isnan(cp.array([benchmark_value]))[0]
                        ):
                            # Convert to GPU arrays
                            ratio_val_gpu = cp.array([ratio_value])
                            benchmark_val_gpu = cp.array([benchmark_value])
                            weight_gpu = cp.array([weight])

                            # Calculate score based on ratio type
                            if ratio_name in higher_is_better:
                                # Handle division by zero
                                if benchmark_value != 0:
                                    relative_diff = (
                                        ratio_val_gpu - benchmark_val_gpu
                                    ) / cp.abs(benchmark_val_gpu)
                                    score_gpu = 0.5 + 0.5 * cp.clip(
                                        relative_diff, -1, 1
                                    )
                                else:
                                    score_gpu = (
                                        cp.array([0.75])
                                        if ratio_value > 0
                                        else cp.array([0.25])
                                    )

                            elif ratio_name in lower_is_better:
                                if benchmark_value != 0:
                                    relative_diff = (
                                        benchmark_val_gpu - ratio_val_gpu
                                    ) / cp.abs(benchmark_val_gpu)
                                    score_gpu = 0.5 + 0.5 * cp.clip(
                                        relative_diff, -1, 1
                                    )
                                else:
                                    score_gpu = (
                                        cp.array([0.75])
                                        if ratio_value < 0
                                        else cp.array([0.25])
                                    )

                            # Special handling for ratios with ideal ranges
                            elif ratio_name == "current_ratio":
                                score_gpu = cp.where(
                                    (ratio_val_gpu >= 1.5) & (ratio_val_gpu <= 3.0),
                                    cp.array([1.0]),
                                    cp.where(
                                        ratio_val_gpu > 1.0,
                                        cp.array([0.5]),
                                        cp.array([0.2]),
                                    ),
                                )
                            elif ratio_name == "quick_ratio":
                                score_gpu = cp.where(
                                    ratio_val_gpu >= 1.0,
                                    cp.array([1.0]),
                                    cp.where(
                                        ratio_val_gpu > 0.5,
                                        cp.array([0.5]),
                                        cp.array([0.2]),
                                    ),
                                )
                            else:
                                score_gpu = cp.array([0.5])

                            # Add to category score
                            category_score_gpu += score_gpu * weight_gpu
                            total_weight_gpu += weight_gpu

                    # Normalize category score
                    if total_weight_gpu > 0:
                        category_scores[category] = float(
                            category_score_gpu / total_weight_gpu
                        )
                    else:
                        category_scores[category] = (
                            0.5  # Default if no ratios available
                        )

                # Calculate overall score on GPU
                overall_score_gpu = cp.array([0.0])
                total_overall_weight_gpu = cp.array([0.0])
                overall_weights = self.scoring_weights.get("overall_score_weights", {})

                for category, weight in overall_weights.items():
                    score = category_scores.get(category, 0.5)
                    score_gpu = cp.array([score])
                    weight_gpu = cp.array([weight])

                    overall_score_gpu += score_gpu * weight_gpu
                    total_overall_weight_gpu += weight_gpu

                # Normalize overall score
                if total_overall_weight_gpu > 0:
                    final_overall_score = float(
                        overall_score_gpu / total_overall_weight_gpu
                    )
                else:
                    final_overall_score = 0.5  # Default if no weights

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for financial health scoring: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            # Calculate score for each category
            for category, weights in self.scoring_weights.items():
                if category == "overall_score_weights":
                    continue

                category_score = 0.0
                total_weight = 0.0

                for ratio_name, weight in weights.items():
                    ratio_value = ratios.get(ratio_name)
                    benchmark_value = benchmarks.get(ratio_name)

                    if (
                        ratio_value is not None
                        and benchmark_value is not None
                        and not np.isnan(benchmark_value)
                    ):
                        #
                        # Normalize score (0-1) based on comparison to
                        # benchmark
                        # This is a simplified scoring logic
                        #
                        # Higher is better for margins, ROE, ROA, turnover
                        # ratios
                        # Lower is better for P/E, P/B, P/S, Debt/Equity

                        score = 0.5  # Default score

                        # Define which ratios are 'higher is better'
                        higher_is_better = [
                            "gross_margin",
                            "operating_margin",
                            "net_margin",
                            "roe",
                            "roa",
                            "asset_turnover",
                            "inventory_turnover",
                            "receivables_turnover",
                            "interest_coverage",
                            "dividend_yield",
                            "fcf_yield",
                            "revenue_growth",
                            "earnings_growth",
                            "fcf_growth",
                            "dividend_growth",
                        ]

                        # Define which ratios are 'lower is better'
                        lower_is_better = [
                            "pe_ratio",
                            "pb_ratio",
                            "ps_ratio",
                            "ev_ebitda",
                            "debt_equity",
                            "days_sales_outstanding",
                        ]

                        if ratio_name in higher_is_better:
                            # Score based on being above/below benchmark
                            if benchmark_value != 0:
                                relative_diff = (ratio_value - benchmark_value) / abs(
                                    benchmark_value
                                )
                                score = 0.5 + 0.5 * np.clip(
                                    relative_diff, -1, 1
                                )  # Scale difference to [-0.5, 0.5]
                            else:
                                score = (
                                    0.75 if ratio_value > 0 else 0.25
                                )  # Handle zero benchmark

                        elif ratio_name in lower_is_better:
                            # Score based on being below/above benchmark
                            if benchmark_value != 0:
                                relative_diff = (benchmark_value - ratio_value) / abs(
                                    benchmark_value
                                )
                                score = 0.5 + 0.5 * np.clip(
                                    relative_diff, -1, 1
                                )  # Scale difference to [-0.5, 0.5]
                            else:
                                score = (
                                    0.75 if ratio_value < 0 else 0.25
                                )  # Handle zero benchmark

                        #
                        # Special handling for ratios like current_ratio,
                        # quick_ratio (ideal range)
                        elif ratio_name == "current_ratio":
                            score = (
                                1.0
                                if ratio_value >= 1.5 and ratio_value <= 3.0
                                else 0.5
                                if ratio_value > 1.0
                                else 0.2
                            )
                        elif ratio_name == "quick_ratio":
                            score = (
                                1.0
                                if ratio_value >= 1.0
                                else 0.5
                                if ratio_value > 0.5
                                else 0.2
                            )

                        category_score += score * weight
                        total_weight += weight

                # Normalize category score
                if total_weight > 0:
                    category_scores[category] = category_score / total_weight
                else:
                    category_scores[category] = 0.5  # Default if no ratios available

            # Calculate overall score
            overall_score = 0.0
            total_overall_weight = 0.0
            overall_weights = self.scoring_weights.get("overall_score_weights", {})

            for category, weight in overall_weights.items():
                score = category_scores.get(
                    category, 0.5
                )  # Use default score if category not calculated
                overall_score += score * weight
                total_overall_weight += weight

            # Normalize overall score
            if total_overall_weight > 0:
                final_overall_score = overall_score / total_overall_weight
            else:
                final_overall_score = 0.5  # Default if no weights

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "overall_score": float(
                np.clip(final_overall_score, 0, 1)
            ),  # Ensure score is between 0 and 1
            "category_scores": {
                k: float(np.clip(v, 0, 1)) for k, v in category_scores.items()
            },
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def calculate_value_metrics(
        self,
        market_data: Dict[str, Any],
        financial_data: Dict[str, Any],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate common valuation metrics with GPU acceleration when beneficial.

        Args:
            market_data: Dictionary with market data (price, market_cap, shares_outstanding).
            financial_data: Dictionary with financial statement data.
            metrics: Optional list of specific metrics to calculate.

        Returns:
            Dictionary of calculated valuation metrics.
        """
        start_time = time.time()

        # Extract data
        price = market_data.get("price")
        market_cap = market_data.get("market_cap")
        shares_outstanding = market_data.get("shares_outstanding")

        income = financial_data.get("income_statement", {})
        balance = financial_data.get("balance_sheet", {})
        cash_flow = financial_data.get("cash_flow", {})

        # Helper function
        def get_val(data, key, default=np.nan):
            return data.get(key, default)

        # Determine if we should use GPU
        # For batch processing or large metrics calculations
        use_gpu = False

        # Check if we're processing multiple companies (batch mode)
        is_batch = isinstance(market_data, list) or isinstance(
            financial_data.get("income_statement", {}), list
        )

        if is_batch:
            # For batch processing, check batch size
            batch_size = (
                len(market_data)
                if isinstance(market_data, list)
                else len(financial_data.get("income_statement", []))
            )
            use_gpu = self.gpu_available and batch_size >= self.min_company_batch
        else:
            # For single company, check number of metrics
            metric_count = len(metrics) if metrics else 12  # Default set of metrics
            use_gpu = self.gpu_available and metric_count >= self.min_ratio_count

        calculated_metrics = {}

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for value metrics calculation")

                # Extract necessary values for calculation
                data_values = {
                    "price": price,
                    "market_cap": market_cap,
                    "shares_outstanding": shares_outstanding,
                    "net_income": get_val(income, "net_income"),
                    "total_equity": get_val(balance, "total_equity"),
                    "revenue": get_val(income, "revenue"),
                    "total_debt": get_val(balance, "total_debt", 0),
                    "cash": get_val(balance, "cash_and_equivalents", 0),
                    "ebit": get_val(income, "operating_income"),
                    "depreciation_amortization": get_val(
                        cash_flow, "depreciation_amortization", 0
                    ),
                    "dividends_paid": abs(get_val(cash_flow, "dividends_paid", 0)),
                    "operating_cash_flow": get_val(cash_flow, "operating_cash_flow"),
                    "capex": abs(get_val(cash_flow, "capital_expenditures", 0)),
                }

                # Transfer values to GPU
                gpu_values = {}
                for k, v in data_values.items():
                    if v is None:
                        gpu_values[k] = cp.array([cp.nan])
                    elif isinstance(v, (list, tuple, np.ndarray)):
                        gpu_values[k] = cp.asarray(v)
                    else:
                        gpu_values[k] = cp.array([v])

                # Small value to prevent division by zero
                epsilon = 1e-10

                # Define functions to calculate metrics
                #
                # These functions are designed to work with both scalar and
                # array inputs
                metric_funcs = {}

                # EPS and PE ratio
                if "shares_outstanding" in gpu_values and "net_income" in gpu_values:
                    if not cp.isnan(gpu_values["shares_outstanding"]).all() and cp.any(
                        gpu_values["shares_outstanding"] > 0
                    ):
                        eps = cp.true_divide(
                            gpu_values["net_income"],
                            gpu_values["shares_outstanding"] + epsilon,
                        )
                        calculated_metrics["eps"] = eps

                        if (
                            "price" in gpu_values
                            and not cp.isnan(gpu_values["price"]).all()
                        ):
                            pe_ratio = cp.true_divide(
                                gpu_values["price"], eps + epsilon
                            )
                            calculated_metrics["pe_ratio"] = pe_ratio

                # BVPS and PB ratio
                if "shares_outstanding" in gpu_values and "total_equity" in gpu_values:
                    if not cp.isnan(gpu_values["shares_outstanding"]).all() and cp.any(
                        gpu_values["shares_outstanding"] > 0
                    ):
                        bvps = cp.true_divide(
                            gpu_values["total_equity"],
                            gpu_values["shares_outstanding"] + epsilon,
                        )
                        calculated_metrics["bvps"] = bvps

                        if (
                            "price" in gpu_values
                            and not cp.isnan(gpu_values["price"]).all()
                        ):
                            pb_ratio = cp.true_divide(
                                gpu_values["price"], bvps + epsilon
                            )
                            calculated_metrics["pb_ratio"] = pb_ratio

                # SPS and PS ratio
                if "shares_outstanding" in gpu_values and "revenue" in gpu_values:
                    if not cp.isnan(gpu_values["shares_outstanding"]).all() and cp.any(
                        gpu_values["shares_outstanding"] > 0
                    ):
                        sps = cp.true_divide(
                            gpu_values["revenue"],
                            gpu_values["shares_outstanding"] + epsilon,
                        )
                        calculated_metrics["sps"] = sps

                        if (
                            "price" in gpu_values
                            and not cp.isnan(gpu_values["price"]).all()
                        ):
                            ps_ratio = cp.true_divide(
                                gpu_values["price"], sps + epsilon
                            )
                            calculated_metrics["ps_ratio"] = ps_ratio

                # Enterprise Value
                if (
                    "market_cap" in gpu_values
                    and not cp.isnan(gpu_values["market_cap"]).all()
                ):
                    ev = (
                        gpu_values["market_cap"]
                        + gpu_values["total_debt"]
                        - gpu_values["cash"]
                    )
                    calculated_metrics["enterprise_value"] = ev

                    # EBITDA
                    if "ebit" in gpu_values and not cp.isnan(gpu_values["ebit"]).all():
                        ebitda = (
                            gpu_values["ebit"] + gpu_values["depreciation_amortization"]
                        )
                        calculated_metrics["ebitda"] = ebitda

                        # EV/EBITDA
                        ev_ebitda = cp.true_divide(ev, ebitda + epsilon)
                        calculated_metrics["ev_ebitda"] = ev_ebitda

                # Dividend Yield
                if (
                    "market_cap" in gpu_values
                    and "dividends_paid" in gpu_values
                    and not cp.isnan(gpu_values["market_cap"]).all()
                    and cp.any(gpu_values["market_cap"] > 0)
                ):
                    dividend_yield = cp.true_divide(
                        gpu_values["dividends_paid"], gpu_values["market_cap"] + epsilon
                    )
                    calculated_metrics["dividend_yield"] = dividend_yield

                # Free Cash Flow
                if (
                    "operating_cash_flow" in gpu_values
                    and not cp.isnan(gpu_values["operating_cash_flow"]).all()
                ):
                    fcf = gpu_values["operating_cash_flow"] - gpu_values["capex"]
                    calculated_metrics["free_cash_flow"] = fcf

                    # FCF Yield
                    if (
                        "market_cap" in gpu_values
                        and not cp.isnan(gpu_values["market_cap"]).all()
                        and cp.any(gpu_values["market_cap"] > 0)
                    ):
                        fcf_yield = cp.true_divide(
                            fcf, gpu_values["market_cap"] + epsilon
                        )
                        calculated_metrics["fcf_yield"] = fcf_yield

                # Convert GPU arrays back to CPU
                for k, v in calculated_metrics.items():
                    # Only convert from GPU to CPU if values are still on GPU
                    if isinstance(v, cp.ndarray):
                        # For single values, extract from array
                        if v.size == 1:
                            calculated_metrics[k] = float(v[0])
                        else:
                            calculated_metrics[k] = cp.asnumpy(v)

                # Update execution stats
                self.execution_stats["gpu_executions"] += 1
                execution_time = time.time() - start_time
                self.execution_stats["gpu_processing_time"] += execution_time
                computation_type = "gpu"

            except Exception as e:
                self.logger.warning(
                    f"GPU calculation failed for value metrics: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            # Earnings Per Share (EPS)
            net_income = get_val(income, "net_income")
            if shares_outstanding and shares_outstanding > 0:
                eps = (
                    net_income / shares_outstanding
                    if not np.isnan(net_income)
                    else np.nan
                )
                calculated_metrics["eps"] = eps

                # Price-to-Earnings (P/E) Ratio
                if price and eps and eps != 0:
                    calculated_metrics["pe_ratio"] = price / eps

            # Book Value Per Share (BVPS)
            total_equity = get_val(balance, "total_equity")
            if shares_outstanding and shares_outstanding > 0:
                bvps = (
                    total_equity / shares_outstanding
                    if not np.isnan(total_equity)
                    else np.nan
                )
                calculated_metrics["bvps"] = bvps

                # Price-to-Book (P/B) Ratio
                if price and bvps and bvps != 0:
                    calculated_metrics["pb_ratio"] = price / bvps

            # Sales Per Share (SPS)
            revenue = get_val(income, "revenue")
            if shares_outstanding and shares_outstanding > 0:
                sps = revenue / shares_outstanding if not np.isnan(revenue) else np.nan
                calculated_metrics["sps"] = sps

                # Price-to-Sales (P/S) Ratio
                if price and sps and sps != 0:
                    calculated_metrics["ps_ratio"] = price / sps

            # Enterprise Value (EV) - Approximation
            total_debt = get_val(balance, "total_debt", 0)
            cash = get_val(balance, "cash_and_equivalents", 0)
            if market_cap:
                ev = market_cap + total_debt - cash
                calculated_metrics["enterprise_value"] = ev

                # EBITDA - Approximation
                ebit = get_val(
                    income, "operating_income"
                )  # Use operating income as proxy
                depreciation_amortization = get_val(
                    cash_flow, "depreciation_amortization", 0
                )
                ebitda = (
                    ebit + depreciation_amortization if not np.isnan(ebit) else np.nan
                )
                calculated_metrics["ebitda"] = ebitda

                # EV/EBITDA
                if ev and ebitda and ebitda != 0:
                    calculated_metrics["ev_ebitda"] = ev / ebitda

            # Dividend Yield
            dividends_paid = abs(
                get_val(cash_flow, "dividends_paid", 0)
            )  # Dividends paid is usually negative
            if market_cap and market_cap > 0:
                calculated_metrics["dividend_yield"] = (
                    dividends_paid / market_cap
                    if not np.isnan(dividends_paid)
                    else np.nan
                )

            # Free Cash Flow (FCF) - Approximation
            operating_cash_flow = get_val(cash_flow, "operating_cash_flow")
            capex = abs(
                get_val(cash_flow, "capital_expenditures", 0)
            )  # Capex is usually negative
            fcf = (
                operating_cash_flow - capex
                if not np.isnan(operating_cash_flow)
                else np.nan
            )
            calculated_metrics["free_cash_flow"] = fcf

            # Free Cash Flow Yield
            if market_cap and market_cap > 0:
                calculated_metrics["fcf_yield"] = (
                    fcf / market_cap if not np.isnan(fcf) else np.nan
                )

            # Update execution stats
            self.execution_stats["cpu_executions"] += 1
            execution_time = time.time() - cpu_start_time
            self.execution_stats["cpu_processing_time"] += execution_time
            computation_type = "cpu"

        # Filter for requested metrics if specified
        if metrics:
            filtered_metrics = {m: calculated_metrics.get(m, np.nan) for m in metrics}
            calculated_metrics = filtered_metrics

        # Replace NaN with None
        final_metrics = {
            k: (float(v) if not np.isnan(v) else None)
            for k, v in calculated_metrics.items()
        }

        # Update total processing time
        processing_time = time.time() - start_time
        self.execution_stats["total_processing_time"] += processing_time

        return {
            "metrics": final_metrics,
            "processing_time": processing_time,
            "computation": computation_type,
        }

    def compare_to_sector(
        self, ratios: Dict[str, float], sector: str
    ) -> Dict[str, Any]:
        """
        Compare company's financial ratios to sector benchmarks with GPU acceleration when beneficial.

        Args:
            ratios: Dictionary of company's financial ratios.
            sector: Industry sector of the company.

        Returns:
            Dictionary comparing each ratio to its sector benchmark.
        """
        start_time = time.time()

        # Get benchmarks for the sector
        benchmarks = self.sector_benchmarks.get(sector)

        if not benchmarks:
            return {
                "comparison": {},
                "error": f"No benchmarks found for sector: {sector}",
                "processing_time": time.time() - start_time,
            }

        # Determine if we should use GPU
        # For batch comparison or comparing many ratios
        use_gpu = False

        # Check if ratios is a batch (list of dictionaries)
        is_batch = isinstance(ratios, list)
        if is_batch:
            batch_size = len(ratios)
            use_gpu = self.gpu_available and batch_size >= self.min_company_batch
        else:
            # For single company, check number of ratios
            ratio_count = len(ratios)
            use_gpu = self.gpu_available and ratio_count >= self.min_ratio_count

        comparison = {}

        if use_gpu:
            try:
                # GPU implementation
                self.logger.debug("Using GPU for sector comparison")

                # Define ratio classifications for evaluation
                higher_is_better = [
                    "gross_margin",
                    "operating_margin",
                    "net_margin",
                    "roe",
                    "roa",
                    "asset_turnover",
                    "inventory_turnover",
                    "receivables_turnover",
                    "interest_coverage",
                    "dividend_yield",
                    "fcf_yield",
                ]
                lower_is_better = [
                    "pe_ratio",
                    "pb_ratio",
                    "ps_ratio",
                    "ev_ebitda",
                    "debt_equity",
                    "days_sales_outstanding",
                ]

                # Process each ratio using GPU
                for ratio_name, company_value in ratios.items():
                    benchmark_value = benchmarks.get(ratio_name)

                    if (
                        company_value is not None
                        and benchmark_value is not None
                        and not cp.isnan(cp.array([benchmark_value]))[0]
                    ):
                        # Transfer to GPU
                        company_val_gpu = cp.array([company_value])
                        benchmark_val_gpu = cp.array([benchmark_value])

                        # Calculate difference and relative difference
                        difference = company_val_gpu - benchmark_val_gpu

                        # Add small epsilon to prevent division by zero
                        epsilon = 1e-10
                        with cp.errstate(divide="ignore", invalid="ignore"):
                            relative_difference = (
                                difference / (cp.abs(benchmark_val_gpu) + epsilon)
                            ) * 100

                        # Extract CPU values
                        difference_val = float(difference[0])
                        relative_difference_val = (
                            float(relative_difference[0])
                            if cp.isfinite(relative_difference[0])
                            else None
                        )

                        # Determine if better or worse than benchmark
                        status = "neutral"
                        if ratio_name in higher_is_better:
                            status = (
                                "better"
                                if difference_val > 0
                                else "worse"
                                if difference_val < 0
                                else "in_line"
                            )
                        elif ratio_name in lower_is_better:
                            status = (
                                "better"
                                if difference_val < 0
                                else "worse"
                                if difference_val > 0
                                else "in_line"
                            )

                        comparison[ratio_name] = {
                            "company_value": float(company_value),
                            "benchmark_value": float(benchmark_value),
                            "difference": difference_val,
                            "relative_difference_pct": relative_difference_val,
                            "status": status,
                        }
                    else:
                        comparison[ratio_name] = {
                            "company_value": float(company_value)
                            if company_value is not None
                            else None,
                            "benchmark_value": float(benchmark_value)
                            if benchmark_value is not None
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
                    f"GPU calculation failed for sector comparison: {e}, falling back to CPU"
                )
                self.execution_stats["gpu_failures"] += 1
                use_gpu = False

        # CPU implementation if GPU is not available, not beneficial, or failed
        if not use_gpu:
            cpu_start_time = time.time()

            # Define ratio classifications
            higher_is_better = [
                "gross_margin",
                "operating_margin",
                "net_margin",
                "roe",
                "roa",
                "asset_turnover",
                "inventory_turnover",
                "receivables_turnover",
                "interest_coverage",
                "dividend_yield",
                "fcf_yield",
            ]
            lower_is_better = [
                "pe_ratio",
                "pb_ratio",
                "ps_ratio",
                "ev_ebitda",
                "debt_equity",
                "days_sales_outstanding",
            ]

            for ratio_name, company_value in ratios.items():
                benchmark_value = benchmarks.get(ratio_name)

                if (
                    company_value is not None
                    and benchmark_value is not None
                    and not np.isnan(benchmark_value)
                ):
                    # Calculate difference and relative difference
                    difference = company_value - benchmark_value
                    # Add small epsilon to prevent division by zero
                    epsilon = 1e-10
                    relative_difference = (
                        difference / (abs(benchmark_value) + epsilon)
                    ) * 100

                    # Determine if better or worse than benchmark
                    status = "neutral"
                    if ratio_name in higher_is_better:
                        status = (
                            "better"
                            if difference > 0
                            else "worse"
                            if difference < 0
                            else "in_line"
                        )
                    elif ratio_name in lower_is_better:
                        status = (
                            "better"
                            if difference < 0
                            else "worse"
                            if difference > 0
                            else "in_line"
                        )

                    comparison[ratio_name] = {
                        "company_value": float(company_value),
                        "benchmark_value": float(benchmark_value),
                        "difference": float(difference),
                        "relative_difference_pct": float(relative_difference)
                        if np.isfinite(relative_difference)
                        else None,
                        "status": status,
                    }
                else:
                    comparison[ratio_name] = {
                        "company_value": float(company_value)
                        if company_value is not None
                        else None,
                        "benchmark_value": float(benchmark_value)
                        if benchmark_value is not None
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

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for GPU/CPU usage during financial calculations.

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
            "has_a100": self.has_a100,
            "min_company_batch": self.min_company_batch,
            "min_ratio_count": self.min_ratio_count,
        }

    def _register_tools(self):
        """Register all available tools for this MCP server."""
        # Original tools
        self.register_tool(
            "calculate_ratios",
            self.calculate_ratios,
            "Calculate key financial ratios from financial statement data",
            {
                "financial_data": {
                    "type": "object",
                    "description": "Dictionary containing financial statement data (income, balance sheet, cash flow)",
                },
                "ratios": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of ratios to calculate (default: all common ratios)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "ratios": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "score_financial_health",
            self.score_financial_health,
            "Calculate a financial health score based on key ratios",
            {
                "ratios": {
                    "type": "object",
                    "description": "Dictionary of calculated financial ratios",
                },
                "sector": {
                    "type": "string",
                    "description": "Industry sector for benchmark comparison (optional)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "overall_score": {"type": "number"},
                    "category_scores": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "calculate_value_metrics",
            self.calculate_value_metrics,
            "Calculate common valuation metrics",
            {
                "market_data": {
                    "type": "object",
                    "description": "Dictionary with market data (price, market cap, shares outstanding)",
                },
                "financial_data": {
                    "type": "object",
                    "description": "Dictionary with financial statement data (revenue, earnings, book value, etc.)",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of value metrics to calculate (default: all common metrics)",
                    "required": False,
                },
            },
            {
                "type": "object",
                "properties": {
                    "metrics": {"type": "object"},
                    "processing_time": {"type": "number"},
                    "computation": {"type": "string"},
                },
            },
        )

        self.register_tool(
            "compare_to_sector",
            self.compare_to_sector,
            "Compare company's financial ratios to sector benchmarks",
            {
                "ratios": {
                    "type": "object",
                    "description": "Dictionary of company's financial ratios",
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

        # Add a new tool for execution statistics
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
                    "has_a100": {"type": "boolean"},
                    "total_processing_time": {"type": "number"},
                    "gpu_speedup": {"type": "number"},
                },
            },
        )


# For running as a standalone MCP server
if __name__ == "__main__":
    # Set up configuration
    config = {}

    # Create and start the server
    server = FundamentalScoringMCP(config)

    # Run the server (this would be replaced with actual server code)
    print("FundamentalScoringMCP server started")

    # Example usage
    sample_financial_data = {
        "income_statement": {
            "revenue": 1000000,
            "cost_of_goods_sold": 400000,
            "operating_income": 200000,
            "interest_expense": 10000,
            "net_income": 150000,
        },
        "balance_sheet": {
            "total_assets": 1200000,
            "current_assets": 500000,
            "inventory": 100000,
            "accounts_receivable": 150000,
            "cash_and_equivalents": 50000,
            "total_liabilities": 600000,
            "current_liabilities": 300000,
            "total_debt": 400000,
            "total_equity": 600000,
        },
        "cash_flow": {
            "operating_cash_flow": 250000,
            "capital_expenditures": -80000,  # Usually negative
            "dividends_paid": -30000,  # Usually negative
            "depreciation_amortization": 50000,
        },
    }

    sample_market_data = {
        "price": 50.0,
        "market_cap": 30000000,  # 30 Million
        "shares_outstanding": 600000,  # 600k shares
    }

    # Calculate ratios
    ratios_result = server.calculate_ratios(sample_financial_data)
    print(f"Calculated Ratios: {json.dumps(ratios_result['ratios'], indent=2)}")
    print(f"Computation: {ratios_result.get('computation', 'cpu')}")

    # Calculate value metrics
    value_result = server.calculate_value_metrics(
        sample_market_data, sample_financial_data
    )
    print(f"Calculated Value Metrics: {json.dumps(value_result['metrics'], indent=2)}")
    print(f"Computation: {value_result.get('computation', 'cpu')}")

    # Score financial health
    score_result = server.score_financial_health(
        ratios_result["ratios"], sector="Technology"
    )
    print(f"Financial Health Score: {json.dumps(score_result, indent=2)}")
    print(f"Computation: {score_result.get('computation', 'cpu')}")

    # Compare to sector
    comparison_result = server.compare_to_sector(
        ratios_result["ratios"], sector="Technology"
    )
    print(f"Sector Comparison: {json.dumps(comparison_result['comparison'], indent=2)}")
    print(f"Computation: {comparison_result.get('computation', 'cpu')}")

    # Get execution stats
    stats = server.get_execution_stats()
    print("Execution Statistics:")
    print(f"  GPU Available: {stats['gpu_available']}")
    print(f"  A100 GPU: {stats['has_a100']}")
    print(f"  GPU Executions: {stats['gpu_executions']} ({stats['gpu_percentage']}%)")
    print(f"  CPU Executions: {stats['cpu_executions']} ({stats['cpu_percentage']}%)")
    print(f"  GPU Failures: {stats['gpu_failures']}")
    if stats["gpu_speedup"]:
        print(f"  GPU Speedup: {stats['gpu_speedup']}x")
