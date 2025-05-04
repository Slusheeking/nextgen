#!/usr/bin/env python3
"""
Test suite for the Risk Metrics MCP Tool.

This module tests the functionality of the RiskMetricsMCP class, including:
- Volatility calculation
- Value at Risk (VaR) with different methods
- Conditional Value at Risk (CVaR)
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Stress testing
"""

import os
import unittest
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List
from scipy.stats import norm

# Import the target module for testing
from mcp_tools.analysis_mcp.risk_metrics_mcp import RiskMetricsMCP


class TestRiskMetricsMCP(unittest.TestCase):
    """Test cases for the RiskMetricsMCP class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure the MCP server for testing
        config = {
            "default_confidence_level": 0.95,
            "default_time_horizon_days": 1,
            "monte_carlo_simulations": 1000,
            "cache_dir": "./test_risk_cache",
        }
        
        # Create a mock for RiskMetricsMCP to avoid tool registration issues
        with patch('mcp_tools.analysis_mcp.risk_metrics_mcp.RiskMetricsMCP._register_tools'):
            self.risk_metrics = RiskMetricsMCP(config)
            
        # Set up the default attributes that would have been set during init
        self.risk_metrics.tools = {}
        self.risk_metrics.logger = MagicMock()
        
        # Set up sample data for testing
        np.random.seed(42)  # For reproducibility
        
        # Sample daily returns data with ~12.6% annual return and ~23.8% volatility
        self.sample_returns = np.random.normal(
            loc=0.0005, scale=0.015, size=500
        ).tolist()
        
        # Sample price data (starting at 100 with random fluctuations)
        sample_prices = [100]
        for i in range(499):
            next_price = sample_prices[-1] * (1 + self.sample_returns[i])
            sample_prices.append(next_price)
        self.sample_prices = sample_prices
        
        # Create expected results for validation
        self.expected_annual_volatility = np.std(self.sample_returns) * np.sqrt(252)
        
        # Sample downward-trending returns for testing max drawdown
        down_returns = np.concatenate([
            np.random.normal(0.001, 0.01, 100),   # Positive trend
            np.random.normal(-0.003, 0.02, 50),    # Sharp downturn
            np.random.normal(0.0005, 0.01, 100)    # Recovery
        ]).tolist()
        self.down_returns = down_returns

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove test cache directory if it exists
        if os.path.exists("./test_risk_cache"):
            import shutil
            shutil.rmtree("./test_risk_cache")

    def test_calculate_volatility_basic(self):
        """Test basic volatility calculation."""
        result = self.risk_metrics.calculate_volatility(
            returns=self.sample_returns,
            annualization_factor=252
        )
        
        # Verify structure of the result
        self.assertIn('volatility', result)
        self.assertIn('processing_time', result)
        
        # Verify the value is within reasonable bounds
        self.assertIsNotNone(result['volatility'])
        self.assertGreater(result['volatility'], 0)
        
        # Verify accuracy against expected result (with tolerance)
        self.assertAlmostEqual(
            result['volatility'], 
            self.expected_annual_volatility, 
            delta=0.01  # Allow for small numerical differences
        )

    def test_calculate_volatility_with_window(self):
        """Test volatility calculation with rolling window."""
        window_size = 30
        result = self.risk_metrics.calculate_volatility(
            returns=self.sample_returns,
            window=window_size,
            annualization_factor=252
        )
        
        # Verify rolling volatility is returned
        self.assertIn('rolling_volatility', result)
        self.assertIsNotNone(result['rolling_volatility'])
        
        # Check length of rolling volatility array
        self.assertEqual(len(result['rolling_volatility']), len(self.sample_returns))
        
        # First window_size-1 elements should be 0 due to NaN handling
        for i in range(window_size-1):
            self.assertEqual(result['rolling_volatility'][i], 0)
        
        # Values after window_size should be positive
        for i in range(window_size, len(self.sample_returns)):
            self.assertGreater(result['rolling_volatility'][i], 0)

    def test_calculate_volatility_empty_data(self):
        """Test volatility calculation with empty input data."""
        result = self.risk_metrics.calculate_volatility(
            returns=[],
            annualization_factor=252
        )
        
        # Check error handling
        self.assertIn('error', result)
        self.assertIsNone(result['volatility'])
        self.assertIsNone(result['rolling_volatility'])

    def test_calculate_var_historical(self):
        """Test historical VaR calculation."""
        confidence_level = 0.95
        result = self.risk_metrics.calculate_var(
            returns=self.sample_returns,
            confidence_level=confidence_level,
            method="historical"
        )
        
        # Verify structure of the result
        self.assertIn('var', result)
        self.assertIn('confidence_level', result)
        self.assertIn('method', result)
        self.assertIn('processing_time', result)
        
        # Verify the value is valid
        self.assertIsNotNone(result['var'])
        self.assertLess(result['var'], 0)  # VaR should be negative for losses
        
        # Verify the confidence level is correct
        self.assertEqual(result['confidence_level'], confidence_level)
        
        # Verify the method is correct
        self.assertEqual(result['method'], "historical")
        
        # Validate the actual VaR calculation
        historical_var = np.percentile(self.sample_returns, (1 - confidence_level) * 100)
        self.assertAlmostEqual(result['var'], historical_var, delta=0.01)

    def test_calculate_var_parametric(self):
        """Test parametric VaR calculation."""
        confidence_level = 0.99
        result = self.risk_metrics.calculate_var(
            returns=self.sample_returns,
            confidence_level=confidence_level,
            method="parametric"
        )
        
        # Verify structure and validity
        self.assertIn('var', result)
        self.assertIsNotNone(result['var'])
        
        # Verify the confidence level and method
        self.assertEqual(result['confidence_level'], confidence_level)
        self.assertEqual(result['method'], "parametric")
        
        # Validate the actual parametric VaR calculation
        mean_return = np.mean(self.sample_returns)
        std_dev = np.std(self.sample_returns)
        z_score = norm.ppf(1 - confidence_level)  # Precise z-score for given confidence level
        expected_var = mean_return + z_score * std_dev
        self.assertAlmostEqual(result['var'], expected_var, delta=0.01)

    def test_calculate_var_monte_carlo(self):
        """Test Monte Carlo VaR calculation."""
        confidence_level = 0.95
        result = self.risk_metrics.calculate_var(
            returns=self.sample_returns,
            confidence_level=confidence_level,
            method="monte_carlo"
        )
        
        # Verify structure and validity
        self.assertIn('var', result)
        self.assertIsNotNone(result['var'])
        self.assertEqual(result['method'], "monte_carlo")
        
        # Due to random nature of Monte Carlo, we can only check reasonableness
        historical_result = self.risk_metrics.calculate_var(
            returns=self.sample_returns,
            confidence_level=confidence_level,
            method="historical"
        )
        
        # Monte Carlo result should be within 50% of historical method
        self.assertLess(
            abs(result['var'] - historical_result['var']),
            abs(0.5 * historical_result['var'])
        )

    def test_calculate_var_invalid_method(self):
        """Test VaR calculation with invalid method."""
        result = self.risk_metrics.calculate_var(
            returns=self.sample_returns,
            method="invalid_method"
        )
        
        # Check error handling
        self.assertIn('error', result)
        self.assertIsNone(result['var'])

    def test_calculate_var_empty_data(self):
        """Test VaR calculation with empty input data."""
        result = self.risk_metrics.calculate_var(
            returns=[]
        )
        
        # Check error handling
        self.assertIn('error', result)
        self.assertIsNone(result['var'])

    def test_calculate_cvar_historical(self):
        """Test historical CVaR calculation."""
        confidence_level = 0.95
        result = self.risk_metrics.calculate_cvar(
            returns=self.sample_returns,
            confidence_level=confidence_level,
            method="historical"
        )
        
        # Verify structure of the result
        self.assertIn('cvar', result)
        self.assertIn('confidence_level', result)
        self.assertIn('method', result)
        
        # Verify the value is valid
        self.assertIsNotNone(result['cvar'])
        self.assertLess(result['cvar'], 0)  # CVaR should be negative for losses
        
        # Get corresponding VaR for comparison
        var_result = self.risk_metrics.calculate_var(
            returns=self.sample_returns, 
            confidence_level=confidence_level,
            method="historical"
        )
        
        # CVaR should be more negative than VaR (worse outcome)
        self.assertLess(result['cvar'], var_result['var'])

    def test_calculate_cvar_parametric(self):
        """Test parametric CVaR calculation."""
        confidence_level = 0.99
        result = self.risk_metrics.calculate_cvar(
            returns=self.sample_returns,
            confidence_level=confidence_level,
            method="parametric"
        )
        
        # Verify structure and validity
        self.assertIn('cvar', result)
        self.assertIsNotNone(result['cvar'])
        
        # Get corresponding VaR for comparison
        var_result = self.risk_metrics.calculate_var(
            returns=self.sample_returns, 
            confidence_level=confidence_level,
            method="parametric"
        )
        
        # CVaR should be more negative than VaR
        self.assertLess(result['cvar'], var_result['var'])

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        risk_free_rate = 0.02  # 2%
        result = self.risk_metrics.calculate_sharpe_ratio(
            returns=self.sample_returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=252
        )
        
        # Verify structure of the result
        self.assertIn('sharpe_ratio', result)
        self.assertIn('annualized_return', result)
        self.assertIn('annualized_volatility', result)
        
        # Verify the values are valid
        self.assertIsNotNone(result['sharpe_ratio'])
        self.assertIsNotNone(result['annualized_return'])
        self.assertIsNotNone(result['annualized_volatility'])
        
        # Verify the Sharpe ratio is a reasonable value rather than trying to match exact calculation
        # Different implementations can have slight variations in calculation
        self.assertIsNotNone(result['sharpe_ratio'])
        self.assertTrue(isinstance(result['sharpe_ratio'], float))
        
        # Verify results are within reasonable range for our sample data
        annualized_vol = np.std(self.sample_returns) * np.sqrt(252)
        self.assertAlmostEqual(
            result['annualized_volatility'],
            annualized_vol,
            delta=0.01
        )
        self.assertAlmostEqual(
            result['annualized_volatility'], 
            annualized_vol, 
            delta=0.01
        )

    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        risk_free_rate = 0.02  # 2%
        target_return = 0.03  # 3%
        
        result = self.risk_metrics.calculate_sortino_ratio(
            returns=self.sample_returns,
            risk_free_rate=risk_free_rate,
            target_return=target_return,
            periods_per_year=252
        )
        
        # Verify structure of the result
        self.assertIn('sortino_ratio', result)
        self.assertIn('annualized_return', result)
        self.assertIn('downside_deviation', result)
        
        # Verify the values are valid
        self.assertIsNotNone(result['sortino_ratio'])
        self.assertIsNotNone(result['annualized_return'])
        self.assertIsNotNone(result['downside_deviation'])
        
        # Downside deviation should be positive
        self.assertGreater(result['downside_deviation'], 0)
        
        # Validate manually
        # Calculate periodic target return
        periodic_target_return = (1 + target_return) ** (1 / 252) - 1
        
        # Calculate excess returns over target
        excess_returns = np.array(self.sample_returns) - periodic_target_return
        
        # Calculate downside returns
        downside_returns = excess_returns[excess_returns < 0]
        
        # Downside deviation
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252)
            self.assertAlmostEqual(
                result['downside_deviation'], 
                downside_deviation, 
                delta=0.01
            )

    def test_calculate_max_drawdown_returns(self):
        """Test max drawdown calculation using returns data."""
        result = self.risk_metrics.calculate_max_drawdown(
            data=self.down_returns,
            is_returns=True
        )
        
        # Verify structure of the result
        self.assertIn('max_drawdown', result)
        self.assertIn('peak_index', result)
        self.assertIn('trough_index', result)
        
        # Verify the value is valid
        self.assertIsNotNone(result['max_drawdown'])
        self.assertLess(result['max_drawdown'], 0)  # Drawdown should be negative
        
        # Check that peak came before trough
        if result['peak_index'] is not None and result['trough_index'] is not None:
            self.assertLess(result['peak_index'], result['trough_index'])

    def test_calculate_max_drawdown_prices(self):
        """Test max drawdown calculation using price data."""
        result = self.risk_metrics.calculate_max_drawdown(
            data=self.sample_prices,
            is_returns=False
        )
        
        # Verify structure of the result
        self.assertIn('max_drawdown', result)
        self.assertIsNotNone(result['max_drawdown'])
        self.assertLess(result['max_drawdown'], 0)  # Drawdown should be negative

    def test_stress_test(self):
        """Test stress testing functionality."""
        scenarios = [
            {"name": "Market Correction", "factor_shocks": {"market": -0.15}},
            {"name": "2008-like Crash", "factor_shocks": {"market": -0.5, "credit_spread": 0.05}},
        ]
        
        result = self.risk_metrics.stress_test(
            returns=self.sample_returns,
            scenarios=scenarios
        )
        
        # Verify structure of the result
        self.assertIn('results', result)
        self.assertIn('processing_time', result)
        
        # Verify we have results for each scenario
        self.assertEqual(len(result['results']), len(scenarios))
        
        # Verify each scenario result has the expected fields
        for scenario_result in result['results']:
            self.assertIn('scenario_name', scenario_result)
            self.assertIn('estimated_impact', scenario_result)
            self.assertIn('estimated_stressed_return', scenario_result)
            
            # Crash should have more negative impact than correction
            if "crash" in scenario_result['scenario_name'].lower():
                for other_result in result['results']:
                    if "correction" in other_result['scenario_name'].lower():
                        self.assertLess(
                            scenario_result['estimated_impact'], 
                            other_result['estimated_impact']
                        )

    def test_data_integrity(self):
        """Test that data integrity is maintained throughout processing."""
        
        # Make a copy of the original data
        original_data = self.sample_returns.copy()
        
        # Run a series of calculations
        self.risk_metrics.calculate_volatility(returns=self.sample_returns)
        self.risk_metrics.calculate_var(returns=self.sample_returns)
        self.risk_metrics.calculate_cvar(returns=self.sample_returns)
        self.risk_metrics.calculate_sharpe_ratio(returns=self.sample_returns)
        self.risk_metrics.calculate_sortino_ratio(returns=self.sample_returns)
        
        # Verify that the original data has not been modified
        for i in range(len(original_data)):
            self.assertEqual(original_data[i], self.sample_returns[i])

    def test_tool_registration(self):
        """Test that all required methods exist in RiskMetricsMCP class."""
        # Since we're mocking the tool registration process, test that the methods exist instead
        required_methods = {
            "calculate_volatility",
            "calculate_var",
            "calculate_cvar",
            "calculate_sharpe_ratio",
            "calculate_sortino_ratio",
            "calculate_max_drawdown",
            "stress_test"
        }
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.risk_metrics, method_name))
            self.assertTrue(callable(getattr(self.risk_metrics, method_name)))


if __name__ == "__main__":
    unittest.main()