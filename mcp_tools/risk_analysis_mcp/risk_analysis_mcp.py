#!/usr/bin/env python3
"""
Risk Analysis MCP Tool

This module implements a consolidated Model Context Protocol (MCP) server for
comprehensive risk analysis, including risk metrics calculation, attribution,
portfolio optimization, slippage analysis, and scenario generation.

This production-ready implementation includes:
- Robust error handling and fallbacks
- Performance optimizations with caching
- Comprehensive logging and metrics
- Efficient resource management
- Enhanced monitoring and visualization
"""

import os
import json

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
import time
import logging
import threading
import hashlib
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import core dependencies directly
import numpy as np
import pandas as pd
import dotenv

# Load environment variables

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring components
from monitoring.stock_charts import StockChartGenerator

# Import libraries for risk analysis and optimization
try:
    import pypfopt
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    HAVE_PYPFOPT = True
    print("PyPortfolioOpt successfully integrated.")
except ImportError:
    HAVE_PYPFOPT = False
    print("Warning: PyPortfolioOpt not found or import failed.")

# Using Prophet + XGBoost for scenario generation instead of GluonTS
HAVE_SCENARIO_GEN = False
try:
    from prophet import Prophet
    import xgboost as xgb
    HAVE_SCENARIO_GEN = True
    print("Prophet and XGBoost successfully integrated for advanced scenario generation.")
except ImportError:
    print("Warning: Prophet or XGBoost not found or import failed. Scenario generation will use Monte Carlo.")

# For advanced risk metrics (e.g., VaR, CVaR, Expected Shortfall)
try:
    # Import comprehensive risk libraries
    import scipy.stats
    import statsmodels.api as sm
    
    # Try to import arch for volatility modeling
    try:
        import arch
        HAVE_ARCH = True
        print("ARCH volatility modeling library successfully integrated.")
    except ImportError:
        HAVE_ARCH = False
        print("Warning: ARCH volatility modeling library not available.")
    
    # Try to import PyPortfolioOpt's risk models specifically
    try:
        from pypfopt.risk_models import CovarianceShrinkage
        HAVE_SHRINKAGE = True
        print("Advanced covariance shrinkage methods successfully integrated.")
    except ImportError:
        HAVE_SHRINKAGE = False
        print("Warning: Advanced covariance shrinkage methods not available.")
except ImportError:
    print("Warning: One or more advanced risk libraries (scipy, statsmodels, arch, pypfopt.risk_models) not found.")
    # Set flags accordingly if needed, though HAVE_ARCH etc. handle specifics
    pass # Allow execution to continue without these optional libraries

    # Create a risk library module with custom implementations
    class RiskLib:
        """Custom risk metrics implementation"""
        
        @staticmethod
        def calculate_var(returns, confidence_level=0.95, method='historical'):
            """
            Calculate Value at Risk (VaR)
            
            Args:
                returns: Array-like of returns
                confidence_level: Confidence level (e.g., 0.95 for 95%)
                method: Method to use ('historical', 'parametric', or 'monte_carlo')
                
            Returns:
                Value at Risk estimate
            """
            if method == 'historical':
                # Historical VaR
                return np.percentile(returns, 100 * (1 - confidence_level))
            
            elif method == 'parametric':
                # Parametric VaR (assuming normal distribution)
                mean = np.mean(returns)
                std = np.std(returns)
                return mean + scipy.stats.norm.ppf(1 - confidence_level) * std
            
            elif method == 'monte_carlo':
                # Simple Monte Carlo VaR
                mean = np.mean(returns)
                std = np.std(returns)
                simulations = 10000
                sim_returns = np.random.normal(mean, std, simulations)
                return np.percentile(sim_returns, 100 * (1 - confidence_level))
            
            else:
                raise ValueError(f"Unknown VaR method: {method}")
        
        @staticmethod
        def calculate_cvar(returns, confidence_level=0.95, method='historical'):
            """
            Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
            
            Args:
                returns: Array-like of returns
                confidence_level: Confidence level (e.g., 0.95 for 95%)
                method: Method to use ('historical', 'parametric', or 'monte_carlo')
                
            Returns:
                CVaR estimate
            """
            if method == 'historical':
                # Historical CVaR
                var = RiskLib.calculate_var(returns, confidence_level, 'historical')
                return returns[returns <= var].mean()
            
            elif method == 'parametric':
                # Parametric CVaR (assuming normal distribution)
                mean = np.mean(returns)
                std = np.std(returns)
                var = mean + scipy.stats.norm.ppf(1 - confidence_level) * std
                
                # Expected shortfall for normal distribution
                return mean - std * scipy.stats.norm.pdf(scipy.stats.norm.ppf(1 - confidence_level)) / (1 - confidence_level)
            
            elif method == 'monte_carlo':
                # Monte Carlo CVaR
                var = RiskLib.calculate_var(returns, confidence_level, 'monte_carlo')
                mean = np.mean(returns)
                std = np.std(returns)
                simulations = 10000
                sim_returns = np.random.normal(mean, std, simulations)
                return sim_returns[sim_returns <= var].mean()
            
            else:
                raise ValueError(f"Unknown CVaR method: {method}")
        
        @staticmethod
        def calculate_drawdown(returns):
            """
            Calculate drawdown metrics from returns
            
            Args:
                returns: Array-like of returns
                
            Returns:
                Dictionary with drawdown metrics
            """
            # Calculate cumulative returns
            cum_returns = (1 + returns).cumprod()
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(cum_returns)
            
            # Calculate drawdown
            drawdown = (cum_returns - running_max) / running_max
            
            # Calculate max drawdown and its duration
            max_drawdown = np.min(drawdown)
            max_drawdown_idx = np.argmin(drawdown)
            
            # Find the peak before the max drawdown
            peak_idx = np.argmax(cum_returns[:max_drawdown_idx+1])
            
            # Find the recovery after the max drawdown
            recovery_idx = None
            for i in range(max_drawdown_idx, len(cum_returns)):
                if cum_returns[i] >= cum_returns[peak_idx]:
                    recovery_idx = i
                    break
            
            # Calculate duration
            if recovery_idx is not None:
                duration = recovery_idx - peak_idx
            else:
                duration = len(cum_returns) - peak_idx
            
            return {
                "drawdown_series": drawdown,
                "max_drawdown": max_drawdown,
                "max_drawdown_idx": max_drawdown_idx,
                "peak_idx": peak_idx,
                "recovery_idx": recovery_idx,
                "duration": duration
            }
        
        @staticmethod
        def calculate_tail_risk_metrics(returns, confidence_level=0.95):
            """
            Calculate various tail risk metrics
            
            Args:
                returns: Array-like of returns
                confidence_level: Confidence level
                
            Returns:
                Dictionary with tail risk metrics
            """
            # Calculate VaR and CVaR using different methods
            historical_var = RiskLib.calculate_var(returns, confidence_level, 'historical')
            parametric_var = RiskLib.calculate_var(returns, confidence_level, 'parametric')
            
            historical_cvar = RiskLib.calculate_cvar(returns, confidence_level, 'historical')
            parametric_cvar = RiskLib.calculate_cvar(returns, confidence_level, 'parametric')
            
            # Calculate skewness and kurtosis
            skewness = scipy.stats.skew(returns)
            kurtosis = scipy.stats.kurtosis(returns)
            
            # Calculate downside deviation
            mean = np.mean(returns)
            downside_returns = returns[returns < 0]
            downside_deviation = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
            
            return {
                "historical_var": historical_var,
                "parametric_var": parametric_var,
                "historical_cvar": historical_cvar,
                "parametric_cvar": parametric_cvar,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "downside_deviation": downside_deviation
            }
    
    # Create risk library instance
    risk_lib = RiskLib()
    HAVE_RISK_LIB = True
    print("Comprehensive risk metrics library successfully integrated.")
    
class RiskAnalysisMCP(BaseMCPServer):
    """
    MCP Server implementation for comprehensive risk analysis.
    Inherits from BaseMCPServer and integrates various risk tools.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Risk Analysis MCP server.

        Args:
            config: Optional configuration dictionary.
        """
        # Set default config if none provided
        if config is None:
            # Define a default config path or load defaults
            config_path = os.path.join("config", "mcp_tools", "risk_analysis_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Error loading config from {config_path}: {e}")
                    config = {}
            else:
                print(f"Warning: Config file not found at {config_path}. Using default settings.")
                config = {}

        # Call the parent class constructor with the required name
        super().__init__(name="risk_analysis_mcp", config=config)

        # Initialize attributes specific to RiskAnalysisMCP
        self._cache_lock = threading.RLock()
        self._model_lock = threading.RLock()
        self._cache = {}
        self._last_health_check = 0
        self._health_status = {"status": "initializing"}
        self._last_activity = time.time() # Initialize last activity time

        # Initialize performance counters
        self.risk_metrics_count = 0
        self.risk_attribution_count = 0
        self.portfolio_optimization_count = 0
        self.slippage_analysis_count = 0
        self.scenario_generation_count = 0
        self.total_processing_time_ms = 0
        self.execution_errors = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0

        # Configure from config
        self._configure_from_config()

        # Initialize chart generator if monitoring is enabled
        if self.config.get("enable_monitoring", True):
             # Ensure logger is initialized before passing to chart generator
            if not hasattr(self, 'logger'):
                 # Initialize logger if not already done by superclass (should be)
                 self.logger = logging.getLogger(self.name) # Basic logger as fallback
                 self.logger.warning("Logger not initialized by superclass, using basic logger.")
            # Pass a specific output directory for risk analysis charts
            chart_output_dir = os.path.join("monitoring", "dashboard", "charts", "risk_analysis")
            self.chart_generator = StockChartGenerator(chart_output_dir)
        else:
            self.chart_generator = None # Explicitly set to None if disabled

        # Initialize models
        self._initialize_models()

        # Register tools
        self._register_tools()

        # Log initialization completion
        self.logger.info("RiskAnalysisMCP initialized successfully")

    def _configure_from_config(self):
        """Extract configuration values."""
        self.default_metrics = self.config.get("default_metrics", ["volatility", "sharpe"])
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        self.optimization_objective = self.config.get("optimization_objective", "max_sharpe")
        self.optimization_bounds = tuple(self.config.get("optimization_bounds", (0, 1)))
        self.scenario_count = self.config.get("scenario_count", 100)
        self.scenario_horizon = self.config.get("scenario_horizon", 30) # e.g., 30 days
        self.enable_cache = self.config.get("enable_cache", True)
        self.cache_ttl = self.config.get("cache_ttl", 600) # Default 10 minutes
        self.max_cache_size = self.config.get("max_cache_size", 1000) # Max cache items

        self.logger.info("RiskAnalysisMCP configuration loaded",
                       default_metrics=self.default_metrics,
                       risk_free_rate=self.risk_free_rate,
                       optimization_objective=self.optimization_objective,
                       scenario_count=self.scenario_count,
                       cache_enabled=self.enable_cache,
                       cache_ttl=self.cache_ttl)

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report with detailed metrics.
        
        Returns:
            Dict containing performance metrics and statistics
        """
        # Calculate cache hit rate
        total_cache_requests = self.cache_hit_count + self.cache_miss_count
        cache_hit_rate = (self.cache_hit_count / max(1, total_cache_requests)) * 100
        
        # Calculate average processing time
        total_requests = (self.risk_metrics_count + self.risk_attribution_count + 
                         self.portfolio_optimization_count + self.slippage_analysis_count +
                         self.scenario_generation_count)
        avg_processing_time = self.total_processing_time_ms / max(1, total_requests)
        
        # Calculate error rate
        error_rate = (self.execution_errors / max(1, total_requests)) * 100
        
        # Record metrics
        self.logger.gauge("risk_analysis_mcp.cache_hit_rate", cache_hit_rate)
        self.logger.gauge("risk_analysis_mcp.avg_processing_time_ms", avg_processing_time)
        self.logger.gauge("risk_analysis_mcp.error_rate", error_rate)
        
        # Calculate uptime
        uptime = time.time() - self._last_activity
        hours, remainder = divmod(uptime, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Generate the performance report
        report = {
            "timestamp": datetime.now().isoformat(),
            "uptime": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "request_metrics": {
                "total_requests": total_requests,
                "risk_metrics_requests": self.risk_metrics_count,
                "risk_attribution_requests": self.risk_attribution_count,
                "portfolio_optimization_requests": self.portfolio_optimization_count,
                "slippage_analysis_requests": self.slippage_analysis_count,
                "scenario_generation_requests": self.scenario_generation_count
            },
            "performance_metrics": {
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "execution_errors": self.execution_errors,
                "error_rate": f"{error_rate:.2f}%"
            },
            "cache_metrics": {
                "cache_size": len(self._cache),
                "cache_hit_count": self.cache_hit_count,
                "cache_miss_count": self.cache_miss_count,
                "cache_hit_rate": f"{cache_hit_rate:.2f}%"
            },
            "system_status": {
                "health_status": self._health_status["status"],
                "last_health_check": datetime.fromtimestamp(self._last_health_check).isoformat()
            }
        }
        
        # Log the report generation
        self.logger.info("Generated performance report", 
                       total_requests=total_requests,
                       error_rate=f"{error_rate:.2f}%",
                       cache_hit_rate=f"{cache_hit_rate:.2f}%")
        
        return report
        
    def generate_risk_dashboard(self, asset_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a visual dashboard of risk metrics and analysis results.
        
        Args:
            asset_data: Optional asset data to include in the dashboard. If None,
                      uses cached results if available.
        
        Returns:
            Dict containing dashboard information and chart paths
        """
        dashboard_info = {
            "generated_at": datetime.now().isoformat(),
            "charts": []
        }
        
        try:
            # Example: Generate a system metrics chart
            metrics_data = {
                "Risk Metrics Requests": self.risk_metrics_count,
                "Portfolio Opt Requests": self.portfolio_optimization_count,
                "Scenario Gen Requests": self.scenario_generation_count,
                "Cache Hit Rate (%)": (self.cache_hit_count / max(1, self.cache_hit_count + self.cache_miss_count)) * 100
            }
            
            # Use the chart generator to create a performance chart
            try:
                perf_chart = self.chart_generator.create_performance_chart(
                    metrics_data,
                    title="Risk Analysis MCP Performance Metrics",
                    include_timestamps=True
                )
                dashboard_info["charts"].append({
                    "name": "performance_metrics",
                    "path": perf_chart,
                    "description": "Performance metrics for the Risk Analysis MCP"
                })
            except Exception as e:
                self.logger.warning(f"Failed to generate performance chart: {e}", exc_info=True)
            
            # If we have asset data, generate some risk visualization charts
            if asset_data:
                # Example: Generate a risk vs return scatter plot
                try:
                    risk_return_data = {}
                    for asset, metrics in asset_data.items():
                        if "volatility" in metrics and "sharpe_ratio" in metrics:
                            risk_return_data[asset] = {
                                "Risk (Volatility)": metrics["volatility"],
                                "Return (Sharpe)": metrics["sharpe_ratio"]
                            }
                    
                    if risk_return_data:
                        risk_chart = self.chart_generator.create_scatter_chart(
                            risk_return_data,
                            x_axis="Risk (Volatility)",
                            y_axis="Return (Sharpe)",
                            title="Risk vs Return Analysis"
                        )
                        dashboard_info["charts"].append({
                            "name": "risk_return_analysis",
                            "path": risk_chart,
                            "description": "Risk vs Return Analysis for assets"
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to generate risk vs return chart: {e}", exc_info=True)
            
            self.logger.info(f"Generated risk dashboard with {len(dashboard_info['charts'])} charts")
            return dashboard_info
            
        except Exception as e:
            self.logger.error(f"Error generating risk dashboard: {e}", exc_info=True)
            self.execution_errors += 1
            return {
                "error": f"Failed to generate dashboard: {str(e)}",
                "generated_at": datetime.now().isoformat()
            }
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get the current health status of the MCP server."""
        # Update health status if it's been a while
        if time.time() - self._last_health_check > 60:  # Update every minute
            self._update_health_status()
        
        # Calculate uptime
        uptime_seconds = time.time() - self._last_activity
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_uptime = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        # Calculate cache hit rate
        total_cache_requests = self.cache_hit_count + self.cache_miss_count
        cache_hit_rate = (self.cache_hit_count / max(1, total_cache_requests)) * 100 if total_cache_requests > 0 else 0
        
        # Calculate request stats
        total_requests = (self.risk_metrics_count + self.risk_attribution_count + 
                         self.portfolio_optimization_count + self.slippage_analysis_count +
                         self.scenario_generation_count)
        
        # Capture memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        except ImportError:
            memory_usage_mb = None
            
        # Enhanced health report with detailed metrics
        health_report = {
            "status": self._health_status["status"],
            "uptime": {
                "seconds": uptime_seconds,
                "formatted": formatted_uptime
            },
            "cache": {
                "size": len(self._cache),
                "hit_rate": f"{cache_hit_rate:.2f}%",
                "hits": self.cache_hit_count,
                "misses": self.cache_miss_count
            },
            "requests": {
                "total": total_requests,
                "risk_metrics_count": self.risk_metrics_count,
                "portfolio_optimization_count": self.portfolio_optimization_count,
                "risk_attribution_count": self.risk_attribution_count,
                "slippage_analysis_count": self.slippage_analysis_count,
                "scenario_generation_count": self.scenario_generation_count
            },
            "errors": {
                "count": self.execution_errors,
                "rate": f"{(self.execution_errors / max(1, total_requests) * 100):.2f}%"
            },
            "system": {
                "memory_usage_mb": memory_usage_mb,
                "scenario_model_loaded": HAVE_SCENARIO_GEN and self.scenario_model is not None
            },
            "last_updated": self._health_status.get("last_updated", datetime.now().isoformat())
        }
        
        # Log health check
        self.logger.info("Health status checked", 
                       status=self._health_status["status"],
                       cache_size=len(self._cache),
                       total_requests=total_requests)
        
        return health_report
        
    def _update_health_status(self) -> None:
        """Update the health status of the MCP server."""
        self._last_health_check = time.time()
        
        # Check if models are loaded (for scenario generation)
        scenario_model_loaded = HAVE_SCENARIO_GEN and self.scenario_model is not None
        
        # Calculate metrics for health assessment
        total_requests = (self.risk_metrics_count + self.risk_attribution_count + 
                         self.portfolio_optimization_count + self.slippage_analysis_count +
                         self.scenario_generation_count)
        
        error_rate = (self.execution_errors / max(1, total_requests)) * 100 if total_requests > 0 else 0
        
        total_cache_requests = self.cache_hit_count + self.cache_miss_count
        cache_hit_rate = (self.cache_hit_count / max(1, total_cache_requests)) * 100 if total_cache_requests > 0 else 0
        
        avg_processing_time = self.total_processing_time_ms / max(1, total_requests) if total_requests > 0 else 0
        
        # Determine overall status
        status = "healthy"
        health_issues = []
        
        # Check for potential issues
        if error_rate > 10:  # More than 10% error rate
            status = "degraded"
            health_issues.append(f"High error rate: {error_rate:.2f}%")
            
        if avg_processing_time > 1000:  # Response time > 1 second
            status = "degraded"
            health_issues.append(f"Slow average response time: {avg_processing_time:.2f}ms")
            
        if len(self._cache) > self.max_cache_size * 0.9:  # Cache approaching full
            health_issues.append(f"Cache nearly full: {len(self._cache)}/{self.max_cache_size}")
        
        # Update health status
        self._health_status = {
            "status": status,
            "scenario_model_loaded": scenario_model_loaded,
            "last_updated": datetime.now().isoformat(),
            "metrics": {
                "risk_metrics_count": self.risk_metrics_count,
                "portfolio_optimization_count": self.portfolio_optimization_count,
                "cache_hit_rate": cache_hit_rate,
                "avg_response_time_ms": avg_processing_time,
                "error_rate": error_rate
            },
            "issues": health_issues
        }
        
        # Update gauges
        self.logger.gauge("risk_analysis_mcp.health_status", 1 if status == "healthy" else 0)
        self.logger.gauge("risk_analysis_mcp.error_rate", error_rate)
        self.logger.gauge("risk_analysis_mcp.cache_hit_rate", cache_hit_rate)
        self.logger.gauge("risk_analysis_mcp.avg_response_time_ms", avg_processing_time)
        
        # Log health status update
        self.logger.info(f"Health status updated to: {status}", 
                       error_rate=f"{error_rate:.2f}%",
                       cache_hit_rate=f"{cache_hit_rate:.2f}%",
                       avg_processing_time=f"{avg_processing_time:.2f}ms",
                       issues=health_issues if health_issues else "None")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache usage."""
        with self._cache_lock:
            current_time = time.time()
            cache_size = len(self._cache)
            
            # Count items by type and age
            method_counts = {}
            age_buckets = {"<1min": 0, "1-10min": 0, "10-60min": 0, ">60min": 0}
            
            for key, (timestamp, _) in self._cache.items():
                # Extract method name from key (first part before the hash)
                method = key.split(":")[0] if ":" in key else "unknown"
                method_counts[method] = method_counts.get(method, 0) + 1
                
                # Categorize by age
                age_seconds = current_time - timestamp
                if age_seconds < 60:
                    age_buckets["<1min"] += 1
                elif age_seconds < 600:
                    age_buckets["1-10min"] += 1
                elif age_seconds < 3600:
                    age_buckets["10-60min"] += 1
                else:
                    age_buckets[">60min"] += 1
            
            return {
                "cache_size": cache_size,
                "cache_enabled": self.enable_cache,
                "cache_ttl_seconds": self.cache_ttl,
                "method_counts": method_counts,
                "age_distribution": age_buckets
            }
    
    def clear_cache(self, method_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the cache to free up memory.
        
        Args:
            method_name: Optional method name to clear cache for. If None, clears all cache.
            
        Returns:
            Dict with information about the operation
        """
        with self._cache_lock:
            if method_name:
                # Count items to be removed
                count_before = len(self._cache)
                # Remove items for the specified method
                keys_to_remove = [
                    key for key in list(self._cache.keys())
                    if key.startswith(method_name)
                ]
                for key in keys_to_remove:
                    del self._cache[key]
                count_after = len(self._cache)
                removed = count_before - count_after
                
                self.logger.info(f"Cleared {removed} items from cache for method {method_name}")
                return {
                    "success": True,
                    "items_removed": removed,
                    "remaining_items": count_after,
                    "method": method_name
                }
            else:
                # Clear all cache
                count = len(self._cache)
                self._cache.clear()
                
                self.logger.info(f"Cleared all {count} items from cache")
                return {
                    "success": True,
                    "items_removed": count,
                    "remaining_items": 0
                }
                
    def _get_cache_key(self, method_name: str, **kwargs) -> str:
        """Generate a cache key for the given method and arguments."""
        # Create a string representation of the kwargs, sorted by key
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        # Create a hash of the method name and kwargs
        key = hashlib.md5(f"{method_name}:{kwargs_str}".encode()).hexdigest()
        return f"{method_name}:{key}"

    def _get_from_cache(self, method_name: str, **kwargs) -> Optional[Any]:
        """Get a cached result if available and not expired."""
        if not self.enable_cache:
            return None
            
        with self._cache_lock:
            cache_key = self._get_cache_key(method_name, **kwargs)
            cached_item = self._cache.get(cache_key)
            
            if cached_item is None:
                # Record cache miss
                self.cache_miss_count += 1
                self.logger.counter("risk_analysis_mcp.cache_miss_count")
                
                # Update cache miss rate
                total_cache_requests = self.cache_hit_count + self.cache_miss_count
                cache_hit_rate = (self.cache_hit_count / total_cache_requests) * 100
                self.logger.gauge("risk_analysis_mcp.cache_hit_rate", cache_hit_rate)
                
                self.logger.debug(f"Cache miss for {method_name}", method=method_name)
                return None
                
            timestamp, result = cached_item
            current_time = time.time()
            
            # Check if cached entry has expired
            if current_time - timestamp > self.cache_ttl:
                # Cache entry has expired
                del self._cache[cache_key]
                
                # Record cache miss due to expiration
                self.cache_miss_count += 1
                self.logger.counter("risk_analysis_mcp.cache_miss_count")
                self.logger.counter("risk_analysis_mcp.cache_expiration_count")
                
                # Update cache miss rate
                total_cache_requests = self.cache_hit_count + self.cache_miss_count
                cache_hit_rate = (self.cache_hit_count / total_cache_requests) * 100
                self.logger.gauge("risk_analysis_mcp.cache_hit_rate", cache_hit_rate)
                
                self.logger.debug(f"Cache expired for {method_name}", method=method_name, 
                               age_seconds=current_time - timestamp, ttl=self.cache_ttl)
                return None
            
            # Record cache hit
            self.cache_hit_count += 1
            self.logger.counter("risk_analysis_mcp.cache_hit_count")
            
            # Update cache hit rate
            total_cache_requests = self.cache_hit_count + self.cache_miss_count
            cache_hit_rate = (self.cache_hit_count / total_cache_requests) * 100
            self.logger.gauge("risk_analysis_mcp.cache_hit_rate", cache_hit_rate)
            
            # Record cache age metrics
            cache_age_ms = (current_time - timestamp) * 1000
            self.logger.timing("risk_analysis_mcp.cache_hit_age_ms", cache_age_ms)
            
            self.logger.debug(f"Cache hit for {method_name}", 
                           method=method_name, 
                           cache_key=cache_key, 
                           age_ms=cache_age_ms)
            
            return result

    def _add_to_cache(self, method_name: str, result: Any, **kwargs) -> None:
        """Add a result to the cache."""
        if not self.enable_cache:
            return
            
        with self._cache_lock:
            cache_key = self._get_cache_key(method_name, **kwargs)
            timestamp = time.time()
            self._cache[cache_key] = (timestamp, result)
            
            # Track cache size
            current_cache_size = len(self._cache)
            self.logger.gauge("risk_analysis_mcp.cache_size", current_cache_size)
            self.logger.counter("risk_analysis_mcp.cache_add_count")
            
            # Log detailed information
            self.logger.debug(f"Added item to cache: {method_name}", 
                           method=method_name,
                           cache_key=cache_key,
                           cache_size=current_cache_size,
                           max_size=self.max_cache_size)
            
            # Clean up old cache entries if cache is getting too large
            if current_cache_size > self.max_cache_size:
                self._clean_cache()
                
            # Record cache utilization percentage
            cache_utilization = (current_cache_size / self.max_cache_size) * 100
            self.logger.gauge("risk_analysis_mcp.cache_utilization_pct", cache_utilization)

    def _clean_cache(self) -> None:
        """Remove expired items from the cache."""
        clean_start = time.time()
        removed_count = 0
        
        with self._cache_lock:
            current_time = time.time()
            
            # Find expired items
            expired_keys = [
                key for key, (timestamp, _) in self._cache.items()
                if current_time - timestamp > self.cache_ttl
            ]
            
            # If we still need to remove more items to stay under max_cache_size,
            # remove the oldest items first
            if len(self._cache) - len(expired_keys) > self.max_cache_size:
                # Get all items sorted by age (oldest first)
                all_items = sorted(
                    [(key, timestamp) for key, (timestamp, _) in self._cache.items()],
                    key=lambda x: x[1]
                )
                
                # Calculate how many more items we need to remove
                additional_removals_needed = len(self._cache) - len(expired_keys) - self.max_cache_size
                
                # Add oldest items to our expired_keys list
                for i in range(min(additional_removals_needed, len(all_items))):
                    if all_items[i][0] not in expired_keys:  # Avoid duplicates
                        expired_keys.append(all_items[i][0])
            
            # Remove all expired and oldest items
            for key in expired_keys:
                del self._cache[key]
                removed_count += 1
            
            # Calculate cleaning metrics
            cleaning_time_ms = (time.time() - clean_start) * 1000
            current_cache_size = len(self._cache)
            
            # Update metrics
            self.logger.gauge("risk_analysis_mcp.cache_size", current_cache_size)
            self.logger.timing("risk_analysis_mcp.cache_cleaning_time_ms", cleaning_time_ms)
            self.logger.counter("risk_analysis_mcp.cache_items_removed", count=removed_count)
            
            # Log cache cleaning results
            self.logger.info("Cleaned cache", 
                           expired_items_removed=len(expired_keys),
                           total_removed=removed_count,
                           current_cache_size=current_cache_size,
                           cleaning_time_ms=f"{cleaning_time_ms:.2f}ms")
            
    def shutdown(self) -> None:
        """
        Clean up resources when shutting down the MCP server.
        This method ensures proper cleanup of models and other resources.
        """
        shutdown_start = time.time()
        self.logger.info("Shutting down RiskAnalysisMCP server")
        
        # Generate and log final performance report
        try:
            # Calculate cache hit rate
            total_cache_requests = self.cache_hit_count + self.cache_miss_count
            cache_hit_rate = (self.cache_hit_count / max(1, total_cache_requests)) * 100
            
            # Calculate request stats
            total_requests = (self.risk_metrics_count + self.risk_attribution_count + 
                             self.portfolio_optimization_count + self.slippage_analysis_count +
                             self.scenario_generation_count)
            
            # Calculate error rate
            error_rate = (self.execution_errors / max(1, total_requests)) * 100 if total_requests > 0 else 0
            
            # Calculate uptime
            uptime = time.time() - self._last_activity
            hours, remainder = divmod(uptime, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Log final metrics
            self.logger.info("Risk Analysis MCP Final Metrics", 
                          total_requests=total_requests,
                          risk_metrics_count=self.risk_metrics_count,
                          portfolio_optimization_count=self.portfolio_optimization_count,
                          risk_attribution_count=self.risk_attribution_count,
                          slippage_analysis_count=self.slippage_analysis_count,
                          scenario_generation_count=self.scenario_generation_count,
                          error_count=self.execution_errors,
                          error_rate=f"{error_rate:.2f}%",
                          cache_hits=self.cache_hit_count,
                          cache_misses=self.cache_miss_count,
                          cache_hit_rate=f"{cache_hit_rate:.2f}%",
                          uptime=f"{int(hours)}h {int(minutes)}m {int(seconds)}s")
        except Exception as e:
            self.logger.error(f"Error generating final metrics: {e}", exc_info=True)
        
        # Clean up cache
        with self._cache_lock:
            cache_size = len(self._cache)
            self._cache.clear()
            self.logger.info(f"Cleared {cache_size} items from cache")
        
        # Clean up models and visualization resources
        try:
            # Free chart generator resources
            if hasattr(self, 'chart_generator'):
                # No explicit cleanup needed, but log it
                self.logger.info("Chart generator resources released")
            
            # Clean up models
            with self._model_lock:
                # Free any resources used by scenario model
                self.scenario_model = None
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                self.logger.info("Models unloaded and memory reclaimed")
                
            # Record the shutdown success
            self.logger.gauge("risk_analysis_mcp.shutdown_success", 1)
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}", exc_info=True)
            self.logger.gauge("risk_analysis_mcp.shutdown_success", 0)
        
        # Calculate shutdown time
        shutdown_duration = (time.time() - shutdown_start) * 1000  # milliseconds
        self.logger.timing("risk_analysis_mcp.shutdown_time_ms", shutdown_duration)
        self.logger.info(f"Risk Analysis MCP shutdown completed in {shutdown_duration:.2f}ms")
        
        # Call parent shutdown method
        super().shutdown()

    def _initialize_models(self):
        """Initialize advanced risk or optimization models if configured."""
        # Initialize scenario generator with Prophet and XGBoost models if available
        self.scenario_model = None
        self.xgb_model = None
        
        if HAVE_SCENARIO_GEN:
            try:
                model_config = self.config.get("scenario_model", {})
                model_path = model_config.get("model_path")
                
                # Check for pre-trained model path for XGBoost
                if model_path and os.path.exists(model_path) and model_path.endswith(('.json', '.model')):
                    self.logger.info(f"Loading pre-trained XGBoost model from {model_path}")
                    try:
                        # Load pre-trained XGBoost model
                        self.xgb_model = xgb.Booster()
                        self.xgb_model.load_model(model_path)
                        self.logger.info("XGBoost model loaded successfully")
                    except Exception as e:
                        self.logger.error(f"Failed to load pre-trained XGBoost model: {e}", exc_info=True)
                else:
                    # Initialize a new Prophet model
                    self.logger.info("Initializing new Prophet model for scenario generation")
                    try:
                        # Configure Prophet parameters
                        prophet_params = model_config.get("prophet", {})
                        
                        # Initialize Prophet with configurable parameters
                        self.scenario_model = Prophet(
                            seasonality_mode=prophet_params.get("seasonality_mode", "multiplicative"),
                            yearly_seasonality=prophet_params.get("yearly_seasonality", "auto"),
                            weekly_seasonality=prophet_params.get("weekly_seasonality", "auto"),
                            daily_seasonality=prophet_params.get("daily_seasonality", False)
                        )
                        
                        # Initialize XGBoost model for residuals/features
                        xgb_params = model_config.get("xgboost", {})
                        self.xgb_model = xgb.XGBRegressor(
                            n_estimators=xgb_params.get("n_estimators", 100),
                            learning_rate=xgb_params.get("learning_rate", 0.1),
                            max_depth=xgb_params.get("max_depth", 3),
                            objective="reg:squarederror",
                            random_state=42
                        )
                        
                        self.logger.info("Prophet and XGBoost models initialized with configuration",
                                       seasonality_mode=prophet_params.get("seasonality_mode", "multiplicative"),
                                       xgb_estimators=xgb_params.get("n_estimators", 100))
                    except Exception as e:
                        self.logger.error(f"Failed to initialize Prophet+XGBoost models: {e}", exc_info=True)
                        self.scenario_model = None
                        self.xgb_model = None
            except Exception as e:
                self.logger.error(f"Error in scenario model initialization: {e}", exc_info=True)
                self.scenario_model = None
                self.xgb_model = None
        else:
            self.logger.info("Prophet or XGBoost not available. Using basic Monte Carlo simulation for scenarios.")
        
        # Log the status of model initialization
        self.logger.info("Risk models initialization complete",
                       scenario_model=self.scenario_model is not None,
                       scenario_model_type="Prophet+XGBoost" if self.scenario_model is not None else "None",
                       optimization_available=HAVE_PYPFOPT)

    def _register_tools(self):
        """Register unified risk analysis tools."""
        self.register_tool(
            self.calculate_risk_metrics,
            "calculate_risk_metrics",
            "Calculate risk metrics (e.g., Volatility, Sharpe, Beta, VaR) for assets or portfolios."
        )
        self.register_tool(
            self.attribute_risk,
            "attribute_risk",
            "Perform risk attribution to identify sources of portfolio risk."
        )
        self.register_tool(
            self.optimize_portfolio,
            "optimize_portfolio",
            "Optimize portfolio weights based on risk/return objectives."
        )
        self.register_tool(
            self.analyze_slippage,
            "analyze_slippage",
            "Analyze execution slippage based on order and trade data."
        )
        self.register_tool(
            self.generate_scenarios,
            "generate_scenarios",
            "Generate market scenarios for stress testing or simulation."
        )
        self.register_tool(
            self.generate_risk_dashboard,
            "generate_risk_dashboard",
            "Generate a visual dashboard of risk metrics and analysis results"
        )

    def _validate_returns_data(self, data: Any, required_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Validate and convert input data to DataFrame suitable for returns analysis."""
        if isinstance(data, dict) and data: # Expect dict of lists/arrays for multiple assets
             try:
                 df = pd.DataFrame(data)
                 # Assume index is datetime or convert if possible
                 if not isinstance(df.index, pd.DatetimeIndex):
                      # Try converting index if it looks like dates
                      try:
                           df.index = pd.to_datetime(df.index)
                      except Exception:
                           self.logger.warning("Index is not datetime. Assuming sequential periods.")
                 # Ensure numeric data
                 df = df.apply(pd.to_numeric, errors='coerce').dropna()
                 if required_cols and not all(col in df.columns for col in required_cols):
                      self.logger.error(f"Input data missing required columns: {required_cols}. Found: {list(df.columns)}")
                      return None
                 return df
             except Exception as e:
                  self.logger.error(f"Failed to convert input dict to DataFrame: {e}", exc_info=True)
                  return None
        elif isinstance(data, pd.DataFrame):
             df = data.copy()
             if not isinstance(df.index, pd.DatetimeIndex):
                  self.logger.warning("Input DataFrame index is not DatetimeIndex.")
             if required_cols and not all(col in df.columns for col in required_cols):
                  self.logger.error(f"Input DataFrame missing required columns: {required_cols}. Found: {list(df.columns)}")
                  return None
             return df
        else:
             self.logger.error(f"Invalid input data type: {type(data)}. Expected dict or DataFrame.")
             return None

    # --- Tool Implementations ---

    def calculate_risk_metrics(self, returns_data: Any, metrics: Optional[List[str]] = None, benchmark_returns: Optional[Any] = None) -> Dict[str, Any]:
        """Calculate risk metrics for one or more return series."""
        # Record start time for performance measurement
        operation_start = time.time()
        
        # Update last activity timestamp
        self._last_activity = time.time()
        
        # Increment request counter
        self.risk_metrics_count += 1
        self.logger.counter("risk_analysis_mcp.risk_metrics_count")
        
        # Check cache first
        cache_key_params = {
            "returns_data": returns_data,
            "metrics": metrics,
            "benchmark_returns": benchmark_returns
        }
        cached_result = self._get_from_cache("calculate_risk_metrics", **cache_key_params)
        if cached_result is not None:
            self.logger.info("Using cached risk metrics result", 
                           from_cache=True,
                           metrics_requested=metrics or self.default_metrics)
            cached_result["from_cache"] = True
            return cached_result
        
        # Log the start of calculation with detailed context
        try:
            # Explicitly check input type before validation
            if not isinstance(returns_data, (dict, pd.DataFrame)):
                 self.logger.error(f"Invalid input type for returns_data: {type(returns_data)}. Expected dict or DataFrame.")
                 self.execution_errors += 1
                 self.logger.counter("risk_analysis_mcp.execution_errors")
                 return {"error": f"Invalid input type for returns_data: {type(returns_data)}. Expected dict or DataFrame."}

            # Parse and validate input data
            validation_start = time.time()
            df_returns = self._validate_returns_data(returns_data)
            validation_time = (time.time() - validation_start) * 1000  # ms
            self.logger.timing("risk_analysis_mcp.data_validation_time_ms", validation_time)
            
            if df_returns is None:
                self.logger.error("Invalid returns data format for risk metrics calculation after validation")
                self.execution_errors += 1
                self.logger.counter("risk_analysis_mcp.execution_errors")
                return {"error": "Invalid returns data format after validation."}
            
            # Determine which metrics to calculate
            metrics_to_calc = metrics or self.default_metrics
            # Ensure metrics_to_calc is a list if it's not None
            if metrics_to_calc is not None and not isinstance(metrics_to_calc, list):
                 self.logger.error("Invalid type for metrics parameter. Expected list or None.")
                 self.execution_errors += 1
                 self.logger.counter("risk_analysis_mcp.execution_errors")
                 return {"error": "Invalid type for metrics parameter. Expected list or None."}
            
            asset_count = len(df_returns.columns)
            data_points = len(df_returns)
            
            # Log detailed information about the calculation
            self.logger.info("Starting risk metrics calculation", 
                           metrics=metrics_to_calc, 
                           asset_count=asset_count,
                           data_points=data_points)
            
            # Initialize results
            results = {}
            periods_per_year = 252  # Assuming daily returns for annualization
            
            # Handle benchmark data for beta/alpha calculations
            benchmark_start = time.time()
            df_benchmark = None
            if "beta" in metrics_to_calc or "alpha" in metrics_to_calc:
                if benchmark_returns is None:
                    self.logger.warning("Benchmark returns required for Beta/Alpha calculation, but not provided.")
                else:
                    df_benchmark = self._validate_returns_data(benchmark_returns)
                    if df_benchmark is None:
                        self.logger.warning("Invalid benchmark returns data format.")
                    elif len(df_benchmark.columns) > 1:
                        self.logger.warning("Benchmark data should contain only one series. Using the first column.")
                        df_benchmark = df_benchmark.iloc[:, [0]]
                    # Align indices
                    df_returns, df_benchmark = df_returns.align(df_benchmark, join='inner', axis=0)
            benchmark_time = (time.time() - benchmark_start) * 1000  # ms
            self.logger.timing("risk_analysis_mcp.benchmark_processing_time_ms", benchmark_time)
            
            # Calculate metrics for each asset
            calc_start = time.time()
            for asset in df_returns.columns:
                asset_start = time.time()
                asset_results = {}
                asset_returns = df_returns[asset]
                
                # Check for sufficient data
                if len(asset_returns) < 2:
                    self.logger.warning(f"Insufficient data points for asset {asset}", data_points=len(asset_returns))
                    results[asset] = {"error": "Not enough data points"}
                    continue
                
                # Calculate volatility if requested
                if "volatility" in metrics_to_calc:
                    vol_start = time.time()
                    # Annualized standard deviation
                    vol = asset_returns.std() * np.sqrt(periods_per_year)
                    asset_results["volatility"] = round(vol, 4)
                    vol_time = (time.time() - vol_start) * 1000  # ms
                    self.logger.timing("risk_analysis_mcp.volatility_calculation_time_ms", vol_time)
                
                # Calculate Sharpe ratio if requested
                if "sharpe" in metrics_to_calc:
                    sharpe_start = time.time()
                    # Annualized Sharpe Ratio
                    excess_returns = asset_returns - (self.risk_free_rate / periods_per_year)
                    if excess_returns.std() == 0:
                        sharpe = 0.0
                    else:
                        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
                    asset_results["sharpe_ratio"] = round(sharpe, 4)
                    sharpe_time = (time.time() - sharpe_start) * 1000  # ms
                    self.logger.timing("risk_analysis_mcp.sharpe_calculation_time_ms", sharpe_time)
                
                # Calculate Beta if requested and benchmark data is available
                if "beta" in metrics_to_calc and df_benchmark is not None:
                    beta_start = time.time()
                    if asset in df_returns.columns and df_benchmark.columns[0] in df_benchmark.columns:
                        # Calculate Beta relative to the benchmark
                        cov_matrix = pd.concat([asset_returns, df_benchmark.iloc[:, 0]], axis=1).cov()
                        covariance = cov_matrix.iloc[0, 1]
                        benchmark_variance = df_benchmark.iloc[:, 0].var()
                        if benchmark_variance == 0:
                            beta = np.nan
                        else:
                            beta = covariance / benchmark_variance
                        asset_results["beta"] = round(beta, 4)
                    beta_time = (time.time() - beta_start) * 1000  # ms
                    self.logger.timing("risk_analysis_mcp.beta_calculation_time_ms", beta_time)
                
                # Calculate VaR if requested
                if "VaR" in metrics_to_calc:
                    var_start = time.time()
                    # Example: Historical VaR at 95% confidence
                    var_95 = np.percentile(asset_returns, 5)
                    asset_results["VaR_95_historical"] = round(var_95, 4)
                    var_time = (time.time() - var_start) * 1000  # ms
                    self.logger.timing("risk_analysis_mcp.var_calculation_time_ms", var_time)
                
                # Calculate CVaR if requested
                if "CVaR" in metrics_to_calc:
                    cvar_start = time.time()
                    # Example: Historical CVaR (Expected Shortfall) at 95%
                    var_95 = np.percentile(asset_returns, 5)
                    cvar_95 = asset_returns[asset_returns <= var_95].mean()
                    asset_results["CVaR_95_historical"] = round(cvar_95, 4)
                    cvar_time = (time.time() - cvar_start) * 1000  # ms
                    self.logger.timing("risk_analysis_mcp.cvar_calculation_time_ms", cvar_time)
                
                # Record per-asset calculation time
                asset_time = (time.time() - asset_start) * 1000  # ms
                self.logger.timing("risk_analysis_mcp.per_asset_calculation_time_ms", asset_time)
                
                # Store results for this asset
                results[asset] = asset_results
            
            # Calculate total calculation time
            calc_time = (time.time() - calc_start) * 1000  # ms
            self.logger.timing("risk_analysis_mcp.metrics_calculation_time_ms", calc_time)
            
            # Generate overall stats
            processing_time_ms = (time.time() - operation_start) * 1000
            self.total_processing_time_ms += processing_time_ms
            
            # Log successful completion with timing details
            metrics_count = sum(len(asset_metrics) for asset_metrics in results.values())
            self.logger.info("Risk metrics calculation completed", 
                           asset_count=asset_count,
                           metrics_calculated=metrics_count,
                           processing_time_ms=processing_time_ms)
            
            # Record timing metrics
            self.logger.timing("risk_analysis_mcp.risk_metrics_processing_time_ms", processing_time_ms)
            
            # Package results with detailed metrics
            result = {
                "metrics": results, 
                "processing_time_ms": processing_time_ms,
                "asset_count": asset_count,
                "metrics_count": metrics_count,
                "data_points": data_points
            }
            
            # Cache the result
            self._add_to_cache("calculate_risk_metrics", result, **cache_key_params)
            
            # Use chart generator to visualize results if there are interesting metrics
            try:
                # Only create visualization for significant calculations
                if asset_count >= 3 and "volatility" in metrics_to_calc and "sharpe" in metrics_to_calc:
                    # Extract data for visualization
                    risk_return_data = {}
                    for asset, metrics in results.items():
                        if "volatility" in metrics and "sharpe_ratio" in metrics:
                            risk_return_data[asset] = {
                                "Risk": metrics["volatility"],
                                "Return": metrics["sharpe_ratio"]
                            }
                    
                    # Generate risk-return chart
                    if risk_return_data and len(risk_return_data) >= 3:
                        chart_file = self.chart_generator.create_scatter_chart(
                            risk_return_data,
                            x_axis="Risk",
                            y_axis="Return",
                            title="Risk-Return Analysis"
                        )
                        # Add chart to result
                        result["visualization"] = {
                            "chart_file": chart_file,
                            "chart_type": "risk_return_scatter"
                        }
                        self.logger.info("Generated risk-return visualization chart", chart_path=chart_file)
            except Exception as viz_e:
                self.logger.warning(f"Failed to generate visualization: {viz_e}", exc_info=True)
            
            return result
            
        except Exception as e:
            # Record failure
            self.execution_errors += 1
            self.logger.counter("risk_analysis_mcp.execution_errors")
            
            # Calculate error processing time for metrics
            error_time_ms = (time.time() - operation_start) * 1000
            self.logger.timing("risk_analysis_mcp.error_processing_time_ms", error_time_ms)
            
            # Log detailed error
            self.logger.error(f"Error calculating risk metrics: {e}", 
                           exc_info=True, 
                           metrics_requested=metrics or self.default_metrics,
                           error_time_ms=error_time_ms)
            
            # Return error information
            return {
                "error": str(e),
                "error_details": traceback.format_exc(),
                "error_time_ms": error_time_ms
            }


    def attribute_risk(self, portfolio_returns: Any, factor_returns: Any) -> Dict[str, Any]:
        """
        Perform risk attribution using factor regression model.
        
        Args:
            portfolio_returns: Returns for the portfolio (single series)
            factor_returns: Returns for the risk factors
            
        Returns:
            Dictionary containing risk attribution results
        """
        start_time = time.time()
        self.risk_attribution_count += 1
        self.logger.counter("risk_analysis_mcp.risk_attribution_count")
        
        # Validate inputs
        df_portfolio = self._validate_returns_data(portfolio_returns)
        df_factors = self._validate_returns_data(factor_returns)

        if df_portfolio is None or df_factors is None:
            self.execution_errors += 1
            return {"error": "Invalid portfolio or factor returns data format."}
            
        if len(df_portfolio.columns) > 1:
            self.logger.warning("Portfolio returns should be a single series. Using the first column.")
            df_portfolio = df_portfolio.iloc[:, [0]]

        portfolio_col = df_portfolio.columns[0]
        self.logger.info(f"Performing risk attribution for portfolio '{portfolio_col}'", 
                       factor_count=len(df_factors.columns),
                       portfolio_data_points=len(df_portfolio),
                       factor_data_points=len(df_factors))

        try:
            # Align data first - ensure dates match
            data_aligned = pd.concat([df_portfolio, df_factors], axis=1).dropna()
            if len(data_aligned) < max(5, len(df_factors.columns) + 2):  # Need enough data points
                self.execution_errors += 1
                return {"error": f"Not enough overlapping data for attribution. Need at least {max(5, len(df_factors.columns) + 2)} points, got {len(data_aligned)}."}

            # Calculate covariance matrix for simplified attribution
            cov_matrix = data_aligned.cov()
            portfolio_variance = data_aligned[portfolio_col].var()
            
            # Get the aligned data
            y = data_aligned[portfolio_col]
            X = data_aligned[df_factors.columns]
            
            # Try to import statsmodels for factor regression
            factor_model_results = {}
            try:
                import statsmodels.api as sm
                self.logger.info("Using statsmodels for factor regression")
                
                # Add constant (intercept) to the model
                X_with_const = sm.add_constant(X)
                
                # Fit the model
                model = sm.OLS(y, X_with_const).fit()
                
                # Extract results
                factor_betas = model.params[1:]  # Skip the constant
                factor_pvalues = model.pvalues[1:]  # Skip the constant
                r_squared = model.rsquared
                alpha = model.params[0]  # The constant term
                alpha_pvalue = model.pvalues[0]
                
                # Calculate marginal contribution to risk
                # MCR_i = beta_i * Cov(F_i, P) / Var(P)
                marginal_contributions = {}
                for factor in df_factors.columns:
                    beta = factor_betas[factor]
                    covariance = cov_matrix.loc[portfolio_col, factor]
                    mcr = beta * covariance / portfolio_variance if portfolio_variance > 0 else 0
                    marginal_contributions[factor] = round(mcr, 6)
                
                # Calculate percentage contribution to risk
                total_explained = sum(marginal_contributions.values())
                percentage_contributions = {
                    factor: round(contrib / max(total_explained, 1e-10) * 100, 2)
                    for factor, contrib in marginal_contributions.items()
                }
                
                # Full factor model results
                factor_model_results = {
                    "alpha": round(alpha, 6),
                    "alpha_pvalue": round(alpha_pvalue, 6),
                    "factor_betas": {f: round(factor_betas[f], 6) for f in df_factors.columns},
                    "factor_pvalues": {f: round(factor_pvalues[f], 6) for f in df_factors.columns},
                    "r_squared": round(r_squared, 6),
                    "marginal_contributions": marginal_contributions,
                    "percentage_contributions": percentage_contributions
                }
            except ImportError:
                factor_model_results = {"warning": "statsmodels not available for detailed factor regression"}
            except Exception as model_e:
                self.logger.warning(f"Failed to perform statsmodels regression: {model_e}", exc_info=True)
                factor_model_results = {"error": f"Factor regression failed: {str(model_e)}"}
            
            # Also calculate simplified covariance-based contribution (as a fallback)
            simple_factor_contributions = {}
            if portfolio_variance > 0:
                for factor in df_factors.columns:
                    # Simple covariance contribution
                    covariance = cov_matrix.loc[portfolio_col, factor]
                    simple_factor_contributions[factor] = round(covariance / portfolio_variance, 6)
            
            # Compile full results
            results = {
                "factor_model": factor_model_results,
                "factor_contributions_simplified": simple_factor_contributions,
                "data_points": len(data_aligned),
                "factor_count": len(df_factors.columns)
            }

            # Calculate processing time
            processing_time = time.time() - start_time
            self.total_processing_time_ms += processing_time * 1000
            
            # Log completion
            self.logger.info(f"Risk attribution completed in {processing_time:.3f}s", 
                           r_squared=factor_model_results.get("r_squared", "N/A"),
                           factors=len(df_factors.columns))
            self.logger.timing("risk_analysis_mcp.risk_attribution_time_s", processing_time)
            
            return {
                "attribution": results, 
                "processing_time": round(processing_time, 3)
            }

        except Exception as e:
            # Log error
            self.execution_errors += 1
            self.logger.error(f"Error performing risk attribution: {e}", exc_info=True)
            
            # Return error information
            return {
                "error": str(e),
                "error_details": traceback.format_exc()
            }


    def optimize_portfolio(self, expected_returns_data: Any, covariance_matrix_data: Any, objective: Optional[str] = None) -> Dict[str, Any]:
        """Optimize portfolio weights."""
        start_time = time.time()
        objective = objective or self.optimization_objective

        # Validate inputs (expected returns could be series/dict, cov matrix could be df/dict)
        mu = None
        S = None
        try:
            if isinstance(expected_returns_data, dict):
                mu = pd.Series(expected_returns_data)
            elif isinstance(expected_returns_data, pd.Series):
                mu = expected_returns_data
            else:
                return {"error": "Invalid format for expected_returns_data. Expected dict or pd.Series."}

            if isinstance(covariance_matrix_data, dict):
                S = pd.DataFrame(covariance_matrix_data)
            elif isinstance(covariance_matrix_data, pd.DataFrame):
                S = covariance_matrix_data
            else:
                 return {"error": "Invalid format for covariance_matrix_data. Expected dict or pd.DataFrame."}

            # Ensure assets match
            if not mu.index.equals(S.index) or not mu.index.equals(S.columns):
                 return {"error": "Assets in expected returns and covariance matrix do not match."}

        except Exception as e:
             self.logger.error(f"Error processing optimization inputs: {e}", exc_info=True)
             return {"error": f"Invalid input data format: {e}"}


        self.logger.info(f"Optimizing portfolio for objective: {objective}", asset_count=len(mu))

        if not HAVE_PYPFOPT:
            return {"error": "PyPortfolioOpt not installed. Cannot optimize portfolio."}

        try:
            # Use PyPortfolioOpt for optimization
            ef = EfficientFrontier(mu, S, weight_bounds=self.optimization_bounds)

            if objective == "max_sharpe":
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            elif objective == "min_volatility":
                weights = ef.min_volatility()
            # Add other objectives like efficient_risk, efficient_return
            else:
                self.logger.warning(f"Unsupported optimization objective: {objective}. Defaulting to max_sharpe.")
                weights = ef.max_sharpe(risk_free_rate=self.risk_free_rate)

            cleaned_weights = ef.clean_weights()
            performance = ef.portfolio_performance(verbose=False, risk_free_rate=self.risk_free_rate)

            results = {
                "weights": {k: round(v, 4) for k, v in cleaned_weights.items()},
                "expected_return": round(performance[0], 4),
                "annual_volatility": round(performance[1], 4),
                "sharpe_ratio": round(performance[2], 4)
            }

            processing_time = time.time() - start_time
            self.logger.info(f"Portfolio optimization completed in {processing_time:.3f}s with {len(mu)} assets")
            return {"optimization": results, "processing_time": processing_time}

        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}", exc_info=True)
            return {"error": str(e)}


    def analyze_slippage(self, orders: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze execution slippage."""
        start_time = time.time()
        if not orders or not trades:
            return {"error": "Orders and trades data required for slippage analysis."}

        self.logger.info(f"Analyzing slippage for {len(orders)} orders and {len(trades)} trades")

        try:
            # Basic slippage calculation: Compare average trade price to order price (e.g., limit price or arrival price)
            # This requires matching trades to orders (e.g., by order ID)
            slippage_results = {}
            trades_df = pd.DataFrame(trades)
            orders_df = pd.DataFrame(orders)

            # Example: Calculate slippage per order_id
            for order_id, group in trades_df.groupby('order_id'):
                 order_info = orders_df[orders_df['order_id'] == order_id].iloc[0] # Assuming unique order IDs
                 avg_fill_price = (group['price'] * group['quantity']).sum() / group['quantity'].sum()
                 arrival_price = order_info.get('arrival_price') # Need arrival price (e.g., mid-quote at order time)
                 limit_price = order_info.get('limit_price')
                 side = order_info.get('side', 'buy').lower()

                 slippage = np.nan
                 if arrival_price is not None:
                      if side == 'buy':
                           slippage = avg_fill_price - arrival_price # Positive slippage is bad for buys
                      elif side == 'sell':
                           slippage = arrival_price - avg_fill_price # Positive slippage is bad for sells
                 elif limit_price is not None:
                      # Slippage vs limit price (less common definition)
                      if side == 'buy':
                           slippage = avg_fill_price - limit_price
                      elif side == 'sell':
                           slippage = limit_price - avg_fill_price

                 slippage_results[order_id] = {
                     "avg_fill_price": round(avg_fill_price, 4),
                     "arrival_price": arrival_price,
                     "limit_price": limit_price,
                     "slippage": round(slippage, 4) if not np.isnan(slippage) else None,
                     "total_quantity": group['quantity'].sum()
                 }

            # Calculate overall average slippage (needs weighting or careful interpretation)
            valid_slippages = [s['slippage'] for s in slippage_results.values() if s['slippage'] is not None]
            avg_slippage = np.mean(valid_slippages) if valid_slippages else None

            results = {
                "per_order_slippage": slippage_results,
                "average_slippage": round(avg_slippage, 4) if avg_slippage is not None else None
            }

            processing_time = time.time() - start_time
            self.logger.info(f"Slippage analysis completed in {processing_time:.3f}s for {len(orders)} orders")
            return {"slippage_analysis": results, "processing_time": processing_time}

        except Exception as e:
            self.logger.error(f"Error analyzing slippage: {e}", exc_info=True)
            return {"error": str(e)}


    def generate_scenarios(self, historical_data: Any, count: Optional[int] = None, horizon: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate market scenarios using either DeepAR forecasting model or Monte Carlo simulation.
        
        Args:
            historical_data: Historical time series data (returns or prices)
            count: Number of scenarios to generate (default: configured value)
            horizon: Length of each scenario (default: configured value)
            
        Returns:
            Dictionary containing generated scenarios and metadata
        """
        start_time = time.time()
        self.scenario_generation_count += 1
        self.logger.counter("risk_analysis_mcp.scenario_generation_count")
        
        # Validate and prepare data
        df_hist = self._validate_returns_data(historical_data)
        if df_hist is None:
            self.execution_errors += 1
            return {"error": "Invalid historical data format for scenario generation."}

        scenario_count = count or self.scenario_count
        scenario_horizon = horizon or self.scenario_horizon
        
        self.logger.info(f"Generating {scenario_count} scenarios for {scenario_horizon} steps", 
                       assets=list(df_hist.columns),
                       data_points=len(df_hist),
                       use_deepar=HAVE_SCENARIO_GEN and self.scenario_model is not None)
        
        try:
            # Check if we can use Prophet + XGBoost model
            if HAVE_SCENARIO_GEN and self.scenario_model is not None:
                self.logger.info("Using Prophet + XGBoost models for advanced scenario generation")
                
                try:
                    # Prepare for time-based modeling
                    model_start_time = time.time()
                    
                    # Container for scenarios
                    all_scenarios = []
                    asset_names = list(df_hist.columns)
                    
                    # For each asset, generate scenarios using Prophet + XGBoost
                    for asset_idx, asset_name in enumerate(asset_names):
                        asset_series = df_hist[asset_name]
                        
                        # Create Prophet dataset with proper formatting
                        prophet_df = pd.DataFrame({
                            'ds': df_hist.index if isinstance(df_hist.index, pd.DatetimeIndex)
                                  else pd.date_range(start='2000-01-01', periods=len(asset_series), freq='D'),
                            'y': asset_series.values
                        })
                        
                        # Fit Prophet model for this asset
                        self.scenario_model.fit(prophet_df)
                        
                        # Create future dataframe for prediction
                        future_df = self.scenario_model.make_future_dataframe(periods=scenario_horizon)
                        
                        # Get Prophet prediction
                        forecast = self.scenario_model.predict(future_df)
                        
                        # Extract components for XGBoost feature engineering
                        components = forecast[['ds', 'trend', 'yhat']]
                        
                        # Prepare training data for XGBoost (use historical data + Prophet components)
                        train_df = components[:-scenario_horizon].copy()
                        
                        # Add time features for XGBoost
                        for df in [train_df, components]:
                            df['day_of_week'] = df['ds'].dt.dayofweek
                            df['month'] = df['ds'].dt.month
                            df['day'] = df['ds'].dt.day
                            
                        # Get actual values for training
                        train_df['y'] = prophet_df['y']
                        
                        # Select features for XGBoost
                        feature_cols = ['trend', 'day_of_week', 'month', 'day']
                        
                        # Train XGBoost on historical data + Prophet components
                        if self.xgb_model is not None:
                            # Train XGBoost model
                            self.xgb_model.fit(
                                train_df[feature_cols],
                                train_df['y']
                            )
                            
                            # Future component data for scenarios
                            future_components = components[-scenario_horizon:].copy()
                            
                            # Generate multiple scenarios using XGBoost with randomness
                            asset_scenarios = []
                            for i in range(scenario_count):
                                scenario_pred = self.xgb_model.predict(future_components[feature_cols])
                                
                                # Add randomness based on historical residuals
                                if len(train_df) > 1:
                                    # Calculate historical prediction and residuals
                                    hist_pred = self.xgb_model.predict(train_df[feature_cols])
                                    residuals = train_df['y'] - hist_pred
                                    
                                    # Sample from residuals distribution
                                    noise = np.random.choice(residuals, size=len(scenario_pred))
                                    scenario_pred += noise
                                
                                asset_scenarios.append(scenario_pred)
                        else:
                            # Fallback to Prophet-only scenarios with sampling
                            future_yhat = forecast['yhat'].values[-scenario_horizon:]
                            future_yhat_lower = forecast['yhat_lower'].values[-scenario_horizon:]
                            future_yhat_upper = forecast['yhat_upper'].values[-scenario_horizon:]
                            
                            asset_scenarios = []
                            for i in range(scenario_count):
                                # Sample between lower and upper bounds
                                scenario_pred = np.random.uniform(
                                    future_yhat_lower,
                                    future_yhat_upper,
                                    size=scenario_horizon
                                )
                                asset_scenarios.append(scenario_pred)
                        
                        # Store scenarios for this asset
                        for i in range(scenario_count):
                            if i >= len(all_scenarios):
                                all_scenarios.append({})
                            all_scenarios[i][asset_name] = asset_scenarios[i].tolist()
                    
                    # Convert to the expected format
                    formatted_scenarios = []
                    for scenario in all_scenarios:
                        # Convert to dataframe for consistent structure
                        scenario_df = pd.DataFrame(scenario)
                        formatted_scenarios.append(scenario_df.to_dict('records'))
                    
                    model_time = (time.time() - model_start_time) * 1000  # ms
                    self.logger.timing("risk_analysis_mcp.prophet_xgboost_prediction_time_ms", model_time)
                    
                    results = {
                        "scenarios": formatted_scenarios,
                        "method": "prophet_xgboost",
                        "asset_count": len(df_hist.columns),
                        "scenario_count": len(formatted_scenarios),
                        "horizon": scenario_horizon,
                        "model_time_ms": round(model_time, 2)
                    }
                    
                    self.logger.info(f"Prophet+XGBoost scenario generation completed successfully in {model_time:.2f}ms")
                    
                except Exception as model_error:
                    # Log error and fall back to Monte Carlo
                    self.logger.error(f"Error using Prophet+XGBoost models for scenario generation: {model_error}", exc_info=True)
                    self.execution_errors += 1
                    
                    # Fall back to Monte Carlo simulation
                    self.logger.info("Falling back to Monte Carlo simulation due to Prophet+XGBoost error")
                    
                    # Generate Monte Carlo scenarios using historical statistics
                    scenarios = self._generate_monte_carlo_scenarios(df_hist, scenario_count, scenario_horizon)
                    
                    results = {
                        "scenarios": scenarios,
                        "method": "monte_carlo_fallback",
                        "asset_count": len(df_hist.columns),
                        "scenario_count": len(scenarios),
                        "horizon": scenario_horizon,
                        "fallback_reason": str(model_error)
                    }
            else:
                # Use Monte Carlo simulation
                self.logger.info("Using Monte Carlo simulation for scenario generation (Prophet+XGBoost not available)")
                
                # Generate Monte Carlo scenarios
                scenarios = self._generate_monte_carlo_scenarios(df_hist, scenario_count, scenario_horizon)
                
                results = {
                    "scenarios": scenarios,
                    "method": "monte_carlo",
                    "asset_count": len(df_hist.columns),
                    "scenario_count": len(scenarios),
                    "horizon": scenario_horizon
                }
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            self.total_processing_time_ms += processing_time * 1000
            
            # Log completion and timing
            self.logger.info(f"Scenario generation completed in {processing_time:.3f}s", 
                           scenario_count=scenario_count,
                           method=results["method"])
            self.logger.timing("risk_analysis_mcp.scenario_generation_time_s", processing_time)
            
            # Add processing time to results
            results["processing_time"] = round(processing_time, 3)
            
            return {"scenario_generation": results}
            
        except Exception as e:
            # Log the error
            self.execution_errors += 1
            self.logger.error(f"Error generating scenarios: {e}", exc_info=True)
            
            # Return error information
            return {
                "error": str(e),
                "error_details": traceback.format_exc()
            }
    
    def _generate_monte_carlo_scenarios(self, df_hist: pd.DataFrame, scenario_count: int, scenario_horizon: int) -> List[Any]:
        """Generate scenarios using Monte Carlo simulation based on historical returns statistics."""
        monte_carlo_start = time.time()
        self.logger.info("Generating Monte Carlo scenarios from historical statistics")
        
        # Calculate return statistics
        mean_returns = df_hist.mean()
        cov_matrix = df_hist.cov()
        
        # Generate scenarios
        scenarios = []
        for i in range(scenario_count):
            # Generate random multivariate normal returns
            random_returns = np.random.multivariate_normal(
                mean_returns, 
                cov_matrix, 
                size=scenario_horizon
            )
            
            # Convert to dataframe with consistent column names
            scenario_df = pd.DataFrame(random_returns, columns=df_hist.columns)
            
            # Convert to records format for consistent output
            scenarios.append(scenario_df.to_dict('records'))
        
        # Log timing statistics
        monte_carlo_time = (time.time() - monte_carlo_start) * 1000  # ms
        self.logger.timing("risk_analysis_mcp.monte_carlo_generation_time_ms", monte_carlo_time)
        self.logger.info(f"Monte Carlo simulation completed in {monte_carlo_time:.2f}ms for {scenario_count} scenarios")
        
        return scenarios
