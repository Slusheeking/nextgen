#!/usr/bin/env python3
"""
Test suite for the Slippage Analysis MCP Tool.

This module tests the functionality of the SlippageAnalysisMCP class, including:
- Execution slippage calculation
- Market impact analysis
- Execution timing evaluation
- Liquidity impact analysis
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the target module for testing
from monitoring.system_monitor import MonitoringManager

# Create a mock for MonitoringManager to avoid Prometheus conflicts
mock_monitor = MagicMock(spec=MonitoringManager)
mock_monitor.metrics = {}  # Empty metrics dictionary
mock_monitor.log_info = MagicMock()
mock_monitor.log_warning = MagicMock()
mock_monitor.log_error = MagicMock()

# Patch the global MonitoringManager before importing SlippageAnalysisMCP
with patch('monitoring.system_monitor.MonitoringManager', return_value=mock_monitor):
    from mcp_tools.analysis_mcp.slippage_analysis_mcp import SlippageAnalysisMCP


class TestSlippageAnalysisMCP(unittest.TestCase):
    """Test cases for the SlippageAnalysisMCP class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure the MCP server for testing
        config = {
            "use_gpu": False,  # Force CPU usage for testing
        }
        
        # Create a mock for MonitoringManager to avoid Prometheus conflicts
        mock_monitor = MagicMock(spec=MonitoringManager)
        mock_monitor.metrics = {}  # Empty metrics dictionary
        mock_monitor.log_info = MagicMock()
        mock_monitor.log_warning = MagicMock()
        mock_monitor.log_error = MagicMock()
        
        # Patch both the MonitoringManager and _register_specific_tools
        with patch('monitoring.system_monitor.MonitoringManager', return_value=mock_monitor), \
             patch('mcp_tools.analysis_mcp.slippage_analysis_mcp.SlippageAnalysisMCP._register_specific_tools'):
            self.slippage_analysis = SlippageAnalysisMCP(config)
            
        # Set up the default attributes that would have been set during init
        self.slippage_analysis.tools = {}
        self.slippage_analysis.logger = MagicMock()
        
        # Set up sample data for testing
        np.random.seed(42)  # For reproducibility
        
        # Sample price data
        self.sample_prices = 100 + np.cumsum(np.random.normal(0, 1, 100)) * 0.1
        
        # Sample bid-ask data
        self.pre_trade_bid_ask = {
            "bid_price": 99.5,
            "ask_price": 100.5,
            "bid_size": 1000,
            "ask_size": 800
        }
        
        self.post_trade_bid_ask = {
            "bid_price": 99.4,
            "ask_price": 100.6,
            "bid_size": 900,
            "ask_size": 700
        }

    def test_initialization(self):
        """Test initialization of SlippageAnalysisMCP."""
        # Verify that the server was initialized correctly
        self.assertEqual(self.slippage_analysis.name, "slippage_analysis_mcp")
        self.assertFalse(self.slippage_analysis.use_gpu)
        
        # Verify that endpoints were initialized
        self.assertIn("calculate_slippage", self.slippage_analysis.endpoints)
        self.assertIn("analyze_market_impact", self.slippage_analysis.endpoints)
        self.assertIn("evaluate_execution_timing", self.slippage_analysis.endpoints)
        self.assertIn("analyze_liquidity_impact", self.slippage_analysis.endpoints)

    def test_calculate_slippage(self):
        """Test slippage calculation functionality."""
        # Test buy side slippage
        buy_result = self.slippage_analysis._handle_calculate_slippage({
            "expected_price": 100.0,
            "executed_price": 100.5,
            "side": "buy",
            "quantity": 100
        })
        
        # Verify structure of the result
        self.assertIn('expected_price', buy_result)
        self.assertIn('executed_price', buy_result)
        self.assertIn('side', buy_result)
        self.assertIn('slippage_amount', buy_result)
        self.assertIn('slippage_percent', buy_result)
        self.assertIn('cost_impact', buy_result)
        
        # Verify calculations for buy side
        self.assertEqual(buy_result['expected_price'], 100.0)
        self.assertEqual(buy_result['executed_price'], 100.5)
        self.assertEqual(buy_result['side'], 'buy')
        self.assertEqual(buy_result['slippage_amount'], 0.5)  # Executed - Expected for buy
        self.assertEqual(buy_result['slippage_percent'], 0.5)  # (Executed - Expected) / Expected * 100
        self.assertEqual(buy_result['cost_impact'], 50.0)  # slippage_amount * quantity
        
        # Test sell side slippage
        sell_result = self.slippage_analysis._handle_calculate_slippage({
            "expected_price": 100.0,
            "executed_price": 99.5,
            "side": "sell",
            "quantity": 100
        })
        
        # Verify calculations for sell side
        self.assertEqual(sell_result['expected_price'], 100.0)
        self.assertEqual(sell_result['executed_price'], 99.5)
        self.assertEqual(sell_result['side'], 'sell')
        self.assertEqual(sell_result['slippage_amount'], 0.5)  # Expected - Executed for sell
        self.assertEqual(sell_result['slippage_percent'], 0.5)  # (Expected - Executed) / Expected * 100
        self.assertEqual(sell_result['cost_impact'], 50.0)  # slippage_amount * quantity

    def test_calculate_slippage_missing_params(self):
        """Test slippage calculation with missing parameters."""
        result = self.slippage_analysis._handle_calculate_slippage({
            "executed_price": 100.5,
            "side": "buy"
        })
        
        # Verify error handling
        self.assertIn('error', result)

    def test_analyze_market_impact(self):
        """Test market impact analysis functionality."""
        # Create pre and post trade price series
        pre_trade_prices = self.sample_prices[:50]
        post_trade_prices = self.sample_prices[50:] + 0.2  # Slight price increase after trade
        
        result = self.slippage_analysis._handle_analyze_market_impact({
            "pre_trade_prices": pre_trade_prices.tolist(),
            "post_trade_prices": post_trade_prices.tolist(),
            "trade_size": 1000,
            "average_volume": 10000,
            "time_window": 5
        })
        
        # Verify structure of the result
        self.assertIn('price_change', result)
        self.assertIn('price_change_percent', result)
        self.assertIn('volatility_change', result)
        self.assertIn('volatility_change_percent', result)
        self.assertIn('relative_trade_size', result)
        self.assertIn('estimated_impact', result)
        self.assertIn('time_window_minutes', result)
        self.assertIn('computed_on', result)
        
        # Verify calculations
        # Note: We're adding 0.2 to post_trade_prices, but due to random data
        # the actual mean might still be lower, so we don't assert direction
        self.assertIsInstance(result['price_change'], float)
        self.assertIsInstance(result['price_change_percent'], float)
        self.assertEqual(result['relative_trade_size'], 0.1)  # 1000 / 10000
        self.assertEqual(result['time_window_minutes'], 5)
        self.assertEqual(result['computed_on'], 'cpu')

    def test_analyze_market_impact_missing_params(self):
        """Test market impact analysis with missing parameters."""
        result = self.slippage_analysis._handle_analyze_market_impact({
            "pre_trade_prices": self.sample_prices[:50].tolist(),
            "post_trade_prices": self.sample_prices[50:].tolist()
        })
        
        # Verify error handling
        self.assertIn('error', result)

    def test_evaluate_execution_timing(self):
        """Test execution timing evaluation functionality."""
        # Create price series with a known optimal point
        price_series = np.array([100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 98, 99, 100])
        execution_index = 6  # Execute at price 100 (middle of the series)
        
        result = self.slippage_analysis._handle_evaluate_execution_timing({
            "execution_price": price_series[execution_index],
            "price_series": price_series.tolist(),
            "execution_index": execution_index,
            "window_size": 5
        })
        
        # Verify structure of the result
        self.assertIn('execution_price', result)
        self.assertIn('min_price', result)
        self.assertIn('max_price', result)
        self.assertIn('mean_price', result)
        self.assertIn('min_diff', result)
        self.assertIn('max_diff', result)
        self.assertIn('mean_diff', result)
        self.assertIn('percentile', result)
        self.assertIn('timing_score', result)
        self.assertIn('window_size', result)
        self.assertIn('computed_on', result)
        
        # Verify calculations
        self.assertEqual(result['execution_price'], 100.0)
        self.assertEqual(result['min_price'], 97.0)
        self.assertEqual(result['max_price'], 103.0)
        self.assertEqual(result['window_size'], 5)
        self.assertEqual(result['computed_on'], 'cpu')

    def test_evaluate_execution_timing_missing_params(self):
        """Test execution timing evaluation with missing parameters."""
        result = self.slippage_analysis._handle_evaluate_execution_timing({
            "execution_price": 100.0,
            "price_series": self.sample_prices.tolist()
        })
        
        # Verify error handling
        self.assertIn('error', result)

    def test_analyze_liquidity_impact(self):
        """Test liquidity impact analysis functionality."""
        # Test buy side liquidity impact
        buy_result = self.slippage_analysis._handle_analyze_liquidity_impact({
            "pre_trade_bid_ask": self.pre_trade_bid_ask,
            "post_trade_bid_ask": self.post_trade_bid_ask,
            "side": "buy",
            "time_window": 5
        })
        
        # Verify structure of the result
        self.assertIn('pre_spread', buy_result)
        self.assertIn('post_spread', buy_result)
        self.assertIn('spread_change', buy_result)
        self.assertIn('spread_change_percent', buy_result)
        self.assertIn('size_change', buy_result)
        self.assertIn('size_change_percent', buy_result)
        self.assertIn('price_change', buy_result)
        self.assertIn('price_change_percent', buy_result)
        self.assertIn('liquidity_score', buy_result)
        self.assertIn('time_window_minutes', buy_result)
        
        # Verify calculations for buy side
        self.assertEqual(buy_result['pre_spread'], 1.0)  # 100.5 - 99.5
        self.assertAlmostEqual(buy_result['post_spread'], 1.2, places=10)  # 100.6 - 99.4
        self.assertAlmostEqual(buy_result['spread_change'], 0.2, places=10)  # post_spread - pre_spread
        self.assertAlmostEqual(buy_result['spread_change_percent'], 20.0, places=10)  # (post_spread - pre_spread) / pre_spread * 100
        self.assertEqual(buy_result['size_change'], -100.0)  # post_ask_size - pre_ask_size
        self.assertEqual(buy_result['size_change_percent'], -12.5)  # (post_ask_size - pre_ask_size) / pre_ask_size * 100
        self.assertEqual(buy_result['time_window_minutes'], 5)
        
        # Test sell side liquidity impact
        sell_result = self.slippage_analysis._handle_analyze_liquidity_impact({
            "pre_trade_bid_ask": self.pre_trade_bid_ask,
            "post_trade_bid_ask": self.post_trade_bid_ask,
            "side": "sell",
            "time_window": 5
        })
        
        # Verify calculations for sell side
        self.assertEqual(sell_result['pre_spread'], 1.0)  # 100.5 - 99.5
        self.assertAlmostEqual(sell_result['post_spread'], 1.2, places=10)  # 100.6 - 99.4
        self.assertAlmostEqual(sell_result['spread_change'], 0.2, places=10)  # post_spread - pre_spread
        self.assertEqual(sell_result['size_change'], -100.0)  # post_bid_size - pre_bid_size
        self.assertEqual(sell_result['size_change_percent'], -10.0)  # (post_bid_size - pre_bid_size) / pre_bid_size * 100

    def test_analyze_liquidity_impact_missing_params(self):
        """Test liquidity impact analysis with missing parameters."""
        result = self.slippage_analysis._handle_analyze_liquidity_impact({
            "pre_trade_bid_ask": self.pre_trade_bid_ask,
            "side": "buy"
        })
        
        # Verify error handling
        self.assertIn('error', result)

    def test_calculate_slippage_internal(self):
        """Test the internal _calculate_slippage method."""
        # Test buy side
        buy_amount, buy_percent = self.slippage_analysis._calculate_slippage(100.0, 100.5, "buy")
        self.assertEqual(buy_amount, 0.5)
        self.assertEqual(buy_percent, 0.5)
        
        # Test sell side
        sell_amount, sell_percent = self.slippage_analysis._calculate_slippage(100.0, 99.5, "sell")
        self.assertEqual(sell_amount, 0.5)
        self.assertEqual(sell_percent, 0.5)

    def test_analyze_market_impact_cpu(self):
        """Test the internal _analyze_market_impact_cpu method."""
        # Create pre and post trade price series
        pre_trade_prices = self.sample_prices[:50]
        post_trade_prices = self.sample_prices[50:] + 0.2  # Slight price increase after trade
        
        result = self.slippage_analysis._analyze_market_impact_cpu(
            pre_trade_prices,
            post_trade_prices,
            1000,
            10000,
            5
        )
        
        # Verify structure and computation type
        self.assertIn('computed_on', result)
        self.assertEqual(result['computed_on'], 'cpu')
        
        # Verify calculations
        self.assertIsInstance(result['price_change'], float)
        self.assertEqual(result['relative_trade_size'], 0.1)
        self.assertEqual(result['time_window_minutes'], 5)

    def test_public_api_methods(self):
        """Test the public API methods."""
        # Mock the call_endpoint method to return a predefined result
        self.slippage_analysis.call_endpoint = MagicMock(return_value={"test": "result"})
        
        # Test calculate_slippage
        result = self.slippage_analysis.calculate_slippage(
            expected_price=100.0,
            executed_price=100.5,
            side="buy",
            quantity=100
        )
        self.assertEqual(result, {"test": "result"})
        self.slippage_analysis.call_endpoint.assert_called_with(
            "calculate_slippage", 
            {
                "expected_price": 100.0,
                "executed_price": 100.5,
                "side": "buy",
                "quantity": 100,
            }
        )
        
        # Test analyze_market_impact
        result = self.slippage_analysis.analyze_market_impact(
            pre_trade_prices=self.sample_prices[:50].tolist(),
            post_trade_prices=self.sample_prices[50:].tolist(),
            trade_size=1000,
            average_volume=10000
        )
        self.assertEqual(result, {"test": "result"})
        self.slippage_analysis.call_endpoint.assert_called_with(
            "analyze_market_impact", 
            {
                "pre_trade_prices": self.sample_prices[:50].tolist(),
                "post_trade_prices": self.sample_prices[50:].tolist(),
                "trade_size": 1000,
                "average_volume": 10000,
            }
        )
        
        # Test evaluate_execution_timing
        result = self.slippage_analysis.evaluate_execution_timing(
            execution_price=100.0,
            price_series=self.sample_prices.tolist(),
            execution_index=50
        )
        self.assertEqual(result, {"test": "result"})
        self.slippage_analysis.call_endpoint.assert_called_with(
            "evaluate_execution_timing", 
            {
                "execution_price": 100.0,
                "price_series": self.sample_prices.tolist(),
                "execution_index": 50,
            }
        )
        
        # Test analyze_liquidity_impact
        result = self.slippage_analysis.analyze_liquidity_impact(
            pre_trade_bid_ask=self.pre_trade_bid_ask,
            post_trade_bid_ask=self.post_trade_bid_ask,
            side="buy"
        )
        self.assertEqual(result, {"test": "result"})
        self.slippage_analysis.call_endpoint.assert_called_with(
            "analyze_liquidity_impact", 
            {
                "pre_trade_bid_ask": self.pre_trade_bid_ask,
                "post_trade_bid_ask": self.post_trade_bid_ask,
                "side": "buy",
            }
        )

    def test_tool_registration(self):
        """Test that all required methods exist in SlippageAnalysisMCP class."""
        # Since we're mocking the tool registration process, test that the methods exist instead
        required_methods = {
            "calculate_slippage",
            "analyze_market_impact",
            "evaluate_execution_timing",
            "analyze_liquidity_impact"
        }
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.slippage_analysis, method_name))
            self.assertTrue(callable(getattr(self.slippage_analysis, method_name)))


if __name__ == "__main__":
    unittest.main()
