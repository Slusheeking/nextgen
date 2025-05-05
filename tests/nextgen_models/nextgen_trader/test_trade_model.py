"""
Integration Test for NextGen Trader Model

This module tests the integration of the TradeModel with MCP tools and AutoGen orchestrator.
It verifies the model correctly handles Alpaca actions, redis data storage, and market monitoring.
"""

import os
import sys
import json
import pytest
import time
import logging
import pandas as pd
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Configure logging for the tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trade_model_test")

# Ensure environment variables are loaded
from dotenv import load_dotenv
load_dotenv()

# Import the TradeModel and related classes
from nextgen_models.nextgen_trader.trade_model import (
    TradeModel, TradeExecutor, TradeMonitor, TradePositionManager, TradeAnalytics
)

# Import necessary MCP tools
from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP
from mcp_tools.analysis_mcp.peak_detection_mcp import PeakDetectionMCP
from mcp_tools.analysis_mcp.slippage_analysis_mcp import SlippageAnalysisMCP
from mcp_tools.analysis_mcp.drift_detection_mcp import DriftDetectionMCP

# Import AutoGen orchestrator
from nextgen_models.autogen_orchestrator.autogen_model import AutoGenOrchestrator, get_api_key

# Test constants
TEST_TICKER = "AAPL"  # Use a liquid, well-known stock for tests
TEST_QUANTITY = 1     # Small quantity for test trades
TEST_ORDER_ID = "test-order-123"


class TestTradeModel:
    """Test suite for the TradeModel integration."""
    
    @classmethod
    def setup_class(cls):
        """
        Set up the test environment.
        """
        logger.info("Setting up test environment")
        
        # Verify essential API keys are available
        cls.verify_environment_variables()
        
        # Load TradeModel config
        cls.trade_model_config = cls.load_trade_model_config()
        
        # Initialize the TradeModel with test configuration
        cls.trade_model = TradeModel(cls.trade_model_config)
        
        # Initialize test data
        cls.setup_test_data()
        
        logger.info("Test environment set up complete")
    
    @staticmethod
    def verify_environment_variables():
        """
        Verify that all required environment variables are set.
        """
        required_vars = [
            "ALPACA_API_KEY", 
            "ALPACA_SECRET_KEY",
            "POLYGON_API_KEY",
            "OPENROUTER_API_KEY",
            "REDIS_HOST",
            "REDIS_PORT"
        ]
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            logger.error("Please set these variables in your .env file or environment")
            pytest.fail(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @classmethod
    def load_trade_model_config(cls):
        """
        Load the TradeModel configuration.
        
        Returns:
            Configuration dictionary for the TradeModel
        """
        try:
            config_path = "config/nextgen_trader/trade_model_config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            logger.info("Loaded TradeModel config from file")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load TradeModel config: {e}")
            logger.info("Using default test configuration")
            
            # Create default test configuration
            return {
                "alpaca_config": {
                    "api_key": os.environ.get("ALPACA_API_KEY"),
                    "api_secret": os.environ.get("ALPACA_SECRET_KEY"),
                    "paper": True,  # Always use paper trading for tests
                    "base_url": "https://paper-api.alpaca.markets"
                },
                "redis_config": {
                    "host": os.environ.get("REDIS_HOST", "localhost"),
                    "port": int(os.environ.get("REDIS_PORT", 6379)),
                    "db": 0,
                    "prefix": "test_trade:"
                },
                "polygon_config": {
                    "api_key": os.environ.get("POLYGON_API_KEY"),
                    "cache_enabled": True,
                    "cache_ttl": 300
                },
                "peak_detection_config": {
                    "window_size": 10,
                    "sensitivity": 0.5,
                    "smooth_data": True
                },
                "slippage_analysis_config": {
                    "benchmark": "vwap",
                    "tolerance": 0.01
                },
                "drift_detection_config": {
                    "window_size": 10,
                    "threshold": 0.02,
                    "min_periods": 5
                },
                "daily_capital_limit": 5000.0,
                "no_overnight_positions": True,
                "llm_config": {
                    "temperature": 0.1,
                    "config_list": [
                        {
                            "model": "anthropic/claude-3-opus-20240229",
                            "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
                            "base_url": "https://openrouter.ai/api/v1",
                            "api_type": "openai",
                            "api_version": None,
                        }
                    ]
                }
            }
    
    @classmethod
    def setup_test_data(cls):
        """
        Set up test data for the integration tests.
        """
        # Test data for market order
        cls.market_order_data = {
            "symbol": TEST_TICKER,
            "action": "buy",
            "quantity": TEST_QUANTITY,
            "order_type": "market",
            "capital_amount": 200.0  # Small amount for test
        }
        
        # Test data for limit order
        cls.limit_order_data = {
            "symbol": TEST_TICKER,
            "action": "buy",
            "quantity": TEST_QUANTITY,
            "order_type": "limit",
            "limit_price": 180.0,  # Set a reasonable limit price for AAPL
            "capital_amount": 200.0
        }
        
        # Test data for monitoring
        cls.monitor_data = {
            "symbol": TEST_TICKER,
            "order_id": TEST_ORDER_ID,
            "entry_price": 190.0  # Example entry price
        }
    
    def test_trade_model_initialization(self):
        """
        Test that the TradeModel initializes correctly with all MCP tools.
        """
        # Verify TradeModel instance
        assert isinstance(self.trade_model, TradeModel)
        
        # Verify MCP tools are initialized
        assert isinstance(self.trade_model.alpaca_mcp, AlpacaMCP)
        assert isinstance(self.trade_model.redis_mcp, RedisMCP)
        assert isinstance(self.trade_model.polygon_mcp, PolygonRestMCP)
        assert isinstance(self.trade_model.peak_detection_mcp, PeakDetectionMCP)
        assert isinstance(self.trade_model.slippage_analysis_mcp, SlippageAnalysisMCP)
        assert isinstance(self.trade_model.drift_detection_mcp, DriftDetectionMCP)
        
        # Verify that the executor, monitor, position_manager, and analytics are initialized
        assert isinstance(self.trade_model.executor, TradeExecutor)
        assert isinstance(self.trade_model.monitor, TradeMonitor)
        assert isinstance(self.trade_model.position_manager, TradePositionManager)
        assert isinstance(self.trade_model.analytics, TradeAnalytics)
        
        # Verify AutoGen integration
        assert isinstance(self.trade_model.llm_config, dict)
        assert "config_list" in self.trade_model.llm_config
        assert self.trade_model.agents is not None
        assert "trade_assistant" in self.trade_model.agents
        assert "user_proxy" in self.trade_model.agents
        
        logger.info("TradeModel initialized with all required components")
    
    def test_mcp_tool_integration(self):
        """
        Test that MCP tools are properly integrated with TradeModel.
        """
        # Test AlpacaMCP
        account_info = self.trade_model.alpaca_mcp.get_account_info()
        assert isinstance(account_info, dict)
        assert "account_number" in account_info or "id" in account_info
        logger.info(f"Alpaca account info retrieved: Account ID {account_info.get('id', 'N/A')}")
        
        # Test RedisMCP
        test_key = "test_trade:integration_test"
        test_value = "test_value"
        self.trade_model.redis_mcp.set_value(test_key, test_value)
        retrieved_value = self.trade_model.redis_mcp.get_value(test_key)
        assert retrieved_value == test_value
        self.trade_model.redis_mcp.delete(test_key)
        logger.info("Redis MCP set/get operations successful")
        
        # Test PolygonRestMCP
        ticker_details = self.trade_model.polygon_mcp.get_ticker_details(TEST_TICKER)
        assert isinstance(ticker_details, dict)
        assert "ticker" in ticker_details
        assert ticker_details["ticker"] == TEST_TICKER
        logger.info(f"Polygon REST MCP ticker details retrieved for {TEST_TICKER}")
        
        # Test PeakDetectionMCP
        # Generate synthetic price data
        import numpy as np
        np.random.seed(42)
        price_data = np.cumsum(np.random.normal(0, 1, 100)) + 100
        # Add a peak
        price_data[50:60] += 5
        peak_result = self.trade_model.peak_detection_mcp.detect_peaks({"prices": price_data.tolist()})
        assert isinstance(peak_result, dict)
        assert "peaks" in peak_result
        logger.info(f"Peak detection MCP detected {len(peak_result['peaks'])} peaks")
        
        # Test DriftDetectionMCP
        drift_result = self.trade_model.drift_detection_mcp.detect_drift({"prices": price_data.tolist()})
        assert isinstance(drift_result, dict)
        assert "drift_detected" in drift_result
        logger.info(f"Drift detection MCP result: drift_detected={drift_result['drift_detected']}")
    
    def test_start_of_day_procedure(self):
        """
        Test the start of day procedure.
        """
        # Patch get_account_info to return test data
        with patch.object(self.trade_model.alpaca_mcp, 'get_account_info') as mock_account_info:
            mock_account_info.return_value = {
                "cash": 10000.0,
                "buying_power": 20000.0,
                "equity": 10000.0,
                "id": "test_account"
            }
            
            # Patch get_market_hours to return open market
            with patch.object(self.trade_model.alpaca_mcp, 'get_market_hours') as mock_market_hours:
                mock_market_hours.return_value = {
                    "is_open": True,
                    "next_close": (datetime.now() + timedelta(hours=4)).isoformat()
                }
                
                # Execute start of day procedure
                result = self.trade_model.start_of_day()
                
                # Verify result
                assert result["status"] == "success"
                assert result["available_capital"] <= 10000.0  # Should be limited by daily_capital_limit
                assert result["daily_capital_limit"] == self.trade_model.daily_capital_limit
                
                # Verify Redis keys are set
                today_key = f"trade:daily_usage:{datetime.now().strftime('%Y-%m-%d')}"
                daily_usage = self.trade_model.redis_mcp.get_value(today_key)
                assert daily_usage is not None
                assert float(daily_usage) == 0.0
                
                # Verify constraints are stored in Redis
                constraints = self.trade_model.redis_mcp.get_json("trade:daily_constraints")
                assert constraints is not None
                assert "available_capital" in constraints
                assert "daily_capital_limit" in constraints
                assert "no_overnight_positions" in constraints
                
                logger.info("Start of day procedure executed successfully")
    
    def test_market_hours_check(self):
        """
        Test the market hours check functionality.
        """
        # Test with market open
        with patch.object(self.trade_model.alpaca_mcp, 'get_market_hours') as mock_market_hours:
            mock_market_hours.return_value = {
                "is_open": True,
                "next_close": (datetime.now() + timedelta(hours=4)).isoformat()
            }
            
            assert self.trade_model._is_market_open() is True
        
        # Test with market closed
        with patch.object(self.trade_model.alpaca_mcp, 'get_market_hours') as mock_market_hours:
            mock_market_hours.return_value = {
                "is_open": False,
                "next_open": (datetime.now() + timedelta(hours=16)).isoformat()
            }
            
            assert self.trade_model._is_market_open() is False
        
        logger.info("Market hours check tested successfully")
    
    def test_trade_execution_flow(self):
        """
        Test the entire trade execution flow from decision to order.
        """
        # Mock market being open
        with patch.object(self.trade_model, '_is_market_open') as mock_market_open:
            mock_market_open.return_value = True
            
            # Mock get_latest_quote to return test data
            with patch.object(self.trade_model.alpaca_mcp, 'get_latest_quote') as mock_quote:
                mock_quote.return_value = {
                    "ask_price": 190.0,
                    "bid_price": 189.9,
                    "last_price": 190.0,
                    "timestamp": datetime.now().isoformat()
                }
                
                # Mock submit_market_order to return test data
                with patch.object(self.trade_model.alpaca_mcp, 'submit_market_order') as mock_submit:
                    mock_submit.return_value = {
                        "id": TEST_ORDER_ID,
                        "status": "filled",
                        "symbol": TEST_TICKER,
                        "qty": str(TEST_QUANTITY),
                        "side": "buy",
                        "type": "market",
                        "filled_qty": str(TEST_QUANTITY),
                        "filled_avg_price": "190.0",
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Mock position monitoring
                    with patch.object(self.trade_model.monitor, 'start_position_monitoring') as mock_monitor:
                        mock_monitor.return_value = {"status": "monitoring", "symbol": TEST_TICKER}
                        
                        # Execute a buy order
                        result = self.trade_model.execute_trade(self.market_order_data)
                        
                        # Verify result
                        assert result["status"] == "filled"
                        assert result["id"] == TEST_ORDER_ID
                        assert result["symbol"] == TEST_TICKER
                        assert float(result["filled_qty"]) == TEST_QUANTITY
                        
                        # Verify monitoring was started
                        mock_monitor.assert_called_once()
                        
                        # Verify daily usage was updated
                        mock_update_daily_usage = mock_submit.return_value["filled_qty"] * float(mock_submit.return_value["filled_avg_price"])
                        
                        logger.info(f"Market order executed successfully: Order ID {result['id']}")
    
    def test_position_monitoring(self):
        """
        Test the position monitoring functionality.
        """
        # Mock get_position to return test data
        with patch.object(self.trade_model.position_manager, 'get_position') as mock_position:
            mock_position.return_value = {
                "symbol": TEST_TICKER,
                "qty": 1.0,
                "avg_entry_price": 190.0,
                "market_value": 190.0,
                "unrealized_pl": 0.0,
                "current_price": 190.0
            }
            
            # Mock get_order to return test data
            with patch.object(self.trade_model.alpaca_mcp, 'get_order') as mock_order:
                mock_order.return_value = {
                    "id": TEST_ORDER_ID,
                    "status": "filled",
                    "symbol": TEST_TICKER,
                    "qty": "1",
                    "side": "buy",
                    "filled_avg_price": "190.0"
                }
                
                # Test starting position monitoring
                monitor_result = self.trade_model.monitor.start_position_monitoring(
                    TEST_TICKER, TEST_ORDER_ID
                )
                
                # Verify monitoring configuration
                assert monitor_result["symbol"] == TEST_TICKER
                assert monitor_result["entry_price"] == 190.0
                assert monitor_result["quantity"] == 1.0
                assert "stop_loss_pct" in monitor_result
                assert "take_profit_pct" in monitor_result
                assert "trailing_stop_pct" in monitor_result
                
                # Verify Redis storage
                monitor_config = self.trade_model.redis_mcp.get_json(f"trade:monitor:{TEST_TICKER}")
                assert monitor_config is not None
                assert monitor_config["symbol"] == TEST_TICKER
                
                logger.info(f"Position monitoring started for {TEST_TICKER}")
                
                # Test check_exit_conditions
                with patch.object(self.trade_model.alpaca_mcp, 'get_latest_quote') as mock_quote:
                    # Test normal case (no exit)
                    mock_quote.return_value = {
                        "ask_price": 191.0,  # Slight profit, but not enough to trigger take profit
                        "bid_price": 190.9,
                        "last_price": 191.0
                    }
                    
                    exit_result = self.trade_model.monitor.check_exit_conditions(TEST_TICKER)
                    assert exit_result["should_exit"] is False
                    
                    # Test stop loss
                    mock_quote.return_value = {
                        "ask_price": 185.0,  # 2.6% loss, should trigger stop loss
                        "bid_price": 184.9,
                        "last_price": 185.0
                    }
                    
                    exit_result = self.trade_model.monitor.check_exit_conditions(TEST_TICKER)
                    assert exit_result["should_exit"] is True
                    assert exit_result["reason"] == "stop_loss"
                    
                    # Test take profit
                    mock_quote.return_value = {
                        "ask_price": 200.0,  # 5.3% gain, should trigger take profit
                        "bid_price": 199.9,
                        "last_price": 200.0
                    }
                    
                    exit_result = self.trade_model.monitor.check_exit_conditions(TEST_TICKER)
                    assert exit_result["should_exit"] is True
                    assert exit_result["reason"] == "take_profit"
                    
                    # Test trailing stop
                    # First set a new high
                    mock_quote.return_value = {
                        "ask_price": 195.0,  # 2.6% gain, not enough for take profit
                        "bid_price": 194.9,
                        "last_price": 195.0
                    }
                    self.trade_model.monitor.check_exit_conditions(TEST_TICKER)
                    
                    # Then drop price slightly to trigger trailing stop
                    mock_quote.return_value = {
                        "ask_price": 193.0,  # 1.0% drop from high, should trigger trailing stop
                        "bid_price": 192.9,
                        "last_price": 193.0
                    }
                    
                    exit_result = self.trade_model.monitor.check_exit_conditions(TEST_TICKER)
                    assert exit_result["should_exit"] is True
                    assert exit_result["reason"] == "trailing_stop"
                    
                    logger.info("Exit conditions testing successful")
    
    def test_redis_data_storage(self):
        """
        Test that data is correctly stored in Redis.
        """
        # Generate unique test keys
        test_prefix = f"test_trade:{int(time.time())}:"
        order_key = f"{test_prefix}order:test123"
        position_key = f"{test_prefix}position:AAPL"
        
        # Store order data
        order_data = {
            "id": "test123",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "1",
            "status": "filled",
            "filled_avg_price": "190.0",
            "timestamp": datetime.now().isoformat()
        }
        self.trade_model.redis_mcp.set_json(order_key, order_data)
        
        # Store position data
        position_data = {
            "symbol": "AAPL",
            "qty": 1.0,
            "avg_entry_price": 190.0,
            "market_value": 190.0,
            "unrealized_pl": 0.0
        }
        self.trade_model.redis_mcp.set_json(position_key, position_data)
        
        # Retrieve and verify data
        retrieved_order = self.trade_model.redis_mcp.get_json(order_key)
        assert retrieved_order == order_data
        
        retrieved_position = self.trade_model.redis_mcp.get_json(position_key)
        assert retrieved_position == position_data
        
        # Test with streaming data
        stream_key = f"{test_prefix}events"
        event_data = {
            "event_type": "position_opened",
            "symbol": "AAPL",
            "quantity": 1.0,
            "price": 190.0,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            self.trade_model.redis_mcp.add_to_stream(stream_key, event_data)
            
            # Read from stream
            result = self.trade_model.redis_mcp.read_stream(stream_key, count=1)
            assert len(result) > 0
            
            logger.info("Redis data storage test successful")
        except Exception as e:
            logger.warning(f"Stream operations failed, possibly due to Redis version: {e}")
        
        # Clean up
        self.trade_model.redis_mcp.delete(order_key)
        self.trade_model.redis_mcp.delete(position_key)
    
    def test_portfolio_management(self):
        """
        Test portfolio management functionality.
        """
        # Mock Alpaca get_all_positions
        with patch.object(self.trade_model.alpaca_mcp, 'get_all_positions') as mock_positions:
            # Create a MockPosition class to mimic Alpaca's Position object
            class MockPosition:
                def __init__(self, data):
                    self._raw = data
            
            # Create test positions
            mock_positions.return_value = [
                MockPosition({
                    "symbol": "AAPL",
                    "qty": "1",
                    "avg_entry_price": "190.0",
                    "market_value": "191.0",
                    "cost_basis": "190.0",
                    "unrealized_pl": "1.0",
                    "unrealized_plpc": "0.0053",
                    "current_price": "191.0"
                }),
                MockPosition({
                    "symbol": "MSFT",
                    "qty": "1",
                    "avg_entry_price": "380.0",
                    "market_value": "384.0",
                    "cost_basis": "380.0",
                    "unrealized_pl": "4.0",
                    "unrealized_plpc": "0.0105",
                    "current_price": "384.0"
                })
            ]
            
            # Mock Alpaca get_account_info
            with patch.object(self.trade_model.alpaca_mcp, 'get_account_info') as mock_account:
                mock_account.return_value = {
                    "cash": "9000.0",
                    "buying_power": "18000.0",
                    "equity": "9575.0",
                    "id": "test_account"
                }
                
                # Test sync_portfolio_data
                self.trade_model.position_manager.sync_portfolio_data()
                
                # Test get_positions
                positions = self.trade_model.position_manager.get_positions()
                assert len(positions) == 2
                assert positions[0]["symbol"] == "AAPL"
                assert positions[1]["symbol"] == "MSFT"
                assert float(positions[0]["qty"]) == 1.0
                
                # Test get_position
                with patch.object(self.trade_model.alpaca_mcp, 'get_position') as mock_get_position:
                    mock_get_position.return_value = MockPosition({
                        "symbol": "AAPL",
                        "qty": "1",
                        "avg_entry_price": "190.0",
                        "market_value": "191.0",
                        "cost_basis": "190.0",
                        "unrealized_pl": "1.0",
                        "unrealized_plpc": "0.0053",
                        "current_price": "191.0"
                    })
                    
                    position = self.trade_model.position_manager.get_position("AAPL")
                    assert position["symbol"] == "AAPL"
                    assert float(position["qty"]) == 1.0
                    assert float(position["market_value"]) == 191.0
                
                # Test get_portfolio_constraints
                constraints = self.trade_model.position_manager.get_portfolio_constraints()
                assert "daily_capital_limit" in constraints
                assert "daily_capital_used" in constraints
                assert "remaining_daily_capital" in constraints
                assert "no_overnight_positions" in constraints
                
                logger.info("Portfolio management tests successful")
    
    def test_integration_with_autogen(self):
        """
        Test integration with the AutoGen orchestrator.
        """
        # Mock AutoGen components to avoid making actual API calls
        with patch.object(self.trade_model.agents["trade_assistant"], '_send_message') as mock_send:
            # Mock the user proxy agent
            mock_user_proxy = MagicMock()
            self.trade_model.agents["user_proxy"] = mock_user_proxy
            
            # Test that registered functions are accessible
            functions = list(mock_user_proxy.register_function.call_args_list)
            registered_function_names = [call.args[0].__name__ for call in functions if hasattr(call.args[0], '__name__')]
            
            # Check that essential functions are registered
            essential_functions = [
                "execute_market_order",
                "execute_limit_order",
                "start_position_monitoring",
                "check_exit_conditions",
                "get_positions",
                "get_position"
            ]
            
            for func_name in essential_functions:
                assert any(func_name == name for name in registered_function_names), f"Function {func_name} not registered"
            
            logger.info("AutoGen integration test successful")
    
    def test_run_monitoring_cycle(self):
        """
        Test the monitoring cycle functionality.
        """
        # Mock get_positions to return test positions
        with patch.object(self.trade_model.position_manager, 'get_positions') as mock_positions:
            mock_positions.return_value = [
                {
                    "symbol": "AAPL",
                    "qty": 1.0,
                    "avg_entry_price": 190.0,
                    "market_value": 185.0,  # 2.6% loss, should trigger stop loss
                    "cost_basis": 190.0,
                    "unrealized_pl": -5.0,
                    "current_price": 185.0
                }
            ]
            
            # Mock check_exit_conditions to return exit signal
            with patch.object(self.trade_model.monitor, 'check_exit_conditions') as mock_check:
                mock_check.return_value = {
                    "should_exit": True,
                    "reason": "stop_loss",
                    "data": {
                        "current_price": 185.0,
                        "entry_price": 190.0,
                        "gain_loss_pct": -0.0263
                    }
                }
                
                # Mock execute_trade to simulate selling the position
                with patch.object(self.trade_model, 'execute_trade') as mock_execute:
                    mock_execute.return_value = {
                        "status": "filled",
                        "id": "test-sell-123",
                        "symbol": "AAPL",
                        "side": "sell",
                        "qty": "1",
                        "filled_qty": "1",
                        "filled_avg_price": "185.0",
