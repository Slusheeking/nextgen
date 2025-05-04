"""
Test suite for the TradeModel in nextgen_models.nextgen_trader.trade_model

This test suite verifies the functionality of the TradeModel, including:
- Initialization with proper configurations
- Integration with Alpaca MCP
- Integration with Redis MCP
- Trade execution and monitoring
- Position management
- Capital management
"""

import unittest
import json
import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import time

from nextgen_models.nextgen_trader.trade_model import (
    TradeModel,
    TradeExecutor,
    TradeMonitor,
    TradePositionManager,
)
from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP


class TestTradeModel(unittest.TestCase):
    """Test case for TradeModel"""

    def setUp(self):
        """Set up test environment"""
        # Create mock configurations
        self.config = {
            "daily_capital_limit": 5000.0,
            "no_overnight_positions": True,
            "alpaca_config": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "paper_trading": True,
            },
            "redis_config": {"host": "localhost", "port": 6379, "db": 0},
            "polygon_config": {"api_key": "test_polygon_key"},
        }

        # Create mocks for MCP clients
        self.mock_alpaca_mcp = MagicMock(spec=AlpacaMCP)
        self.mock_redis_mcp = MagicMock(spec=RedisMCP)
        self.mock_polygon_mcp = MagicMock(spec=PolygonRestMCP)

        # Setup mock market hours
        self.mock_alpaca_mcp.get_market_hours.return_value = {
            "is_open": True,
            "next_close": datetime.now() + timedelta(hours=3),
        }

        # Setup mock account info
        self.mock_alpaca_mcp.get_account_info.return_value = {
            "cash": "10000.0",
            "buying_power": "20000.0",
            "portfolio_value": "30000.0",
            "equity": "30000.0",
        }

        # Patch the MCP clients
        self.alpaca_patcher = patch(
            "nextgen_models.nextgen_trader.trade_model.AlpacaMCP",
            return_value=self.mock_alpaca_mcp,
        )
        self.redis_patcher = patch(
            "nextgen_models.nextgen_trader.trade_model.RedisMCP",
            return_value=self.mock_redis_mcp,
        )
        self.polygon_patcher = patch(
            "nextgen_models.nextgen_trader.trade_model.PolygonRestMCP",
            return_value=self.mock_polygon_mcp,
        )

        # Start the patchers
        self.mock_alpaca = self.alpaca_patcher.start()
        self.mock_redis = self.redis_patcher.start()
        self.mock_polygon = self.polygon_patcher.start()

        # Create an instance of TradeModel
        self.trade_model = TradeModel(self.config)

    def tearDown(self):
        """Tear down test environment"""
        # Stop the patchers
        self.alpaca_patcher.stop()
        self.redis_patcher.stop()
        self.polygon_patcher.stop()

    def test_initialization(self):
        """Test TradeModel initialization"""
        self.assertEqual(self.trade_model.daily_capital_limit, 5000.0)
        self.assertTrue(self.trade_model.no_overnight_positions)
        self.assertIsNotNone(self.trade_model.executor)
        self.assertIsNotNone(self.trade_model.monitor)
        self.assertIsNotNone(self.trade_model.position_manager)
        self.assertIsNotNone(self.trade_model.analytics)

    def test_start_of_day(self):
        """Test start of day procedure"""
        # Setup mocks
        today_key = f"trade:daily_usage:{datetime.now().strftime('%Y-%m-%d')}"
        self.mock_redis_mcp.set_value.return_value = True

        # Call the method
        result = self.trade_model.start_of_day()

        # Verify results
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["available_capital"], 5000.0)  # Limited by daily_capital_limit

        # Verify method calls
        self.mock_redis_mcp.set_value.assert_any_call(today_key, "0")
        self.mock_alpaca_mcp.get_account_info.assert_called_once()
        self.mock_redis_mcp.set_json.assert_any_call("trade:daily_constraints", {
            "daily_capital_limit": 5000.0,
            "available_capital": 5000.0,
            "no_overnight_positions": True,
            "market_hours": self.mock_alpaca_mcp.get_market_hours(),
            "date": datetime.now().strftime("%Y-%m-%d"),
        })

    def test_execute_trade_market_open(self):
        """Test execute_trade when market is open"""
        # Setup mocks
        self.mock_alpaca_mcp.get_market_hours.return_value = {"is_open": True}
        self.mock_alpaca_mcp.get_latest_quote.return_value = {"ask_price": "150.00"}

        # Create a trade decision
        trade_decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "order_type": "market",
            "capital_amount": 1600.0,
        }

        # Set up executor mock
        self.trade_model.executor.execute_market_order = MagicMock(
            return_value={
                "status": "filled",
                "filled_qty": 10,
                "filled_avg_price": 152.50,
            }
        )

        # Call the method
        result = self.trade_model.execute_trade(trade_decision)

        # Verify results
        self.trade_model.executor.execute_market_order.assert_called_once_with(
            "AAPL", 10, "buy"
        )
        self.assertEqual(result["status"], "filled")

    def test_execute_trade_market_closed(self):
        """Test execute_trade when market is closed"""
        # Setup mocks
        self.mock_alpaca_mcp.get_market_hours.return_value = {"is_open": False}

        # Create a trade decision
        trade_decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "order_type": "market",
            "capital_amount": 1500.0,
        }

        # Call the method
        result = self.trade_model.execute_trade(trade_decision)

        # Verify results
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "Market closed")

    def test_execute_trade_missing_parameters(self):
        """Test execute_trade with missing parameters"""
        # Setup mocks
        self.mock_alpaca_mcp.get_market_hours.return_value = {"is_open": True}

        # Create an incomplete trade decision
        trade_decision = {
            "symbol": "AAPL",
            "action": "buy",
            # Missing quantity
            "order_type": "market",
            "capital_amount": 1500.0,
        }

        # Call the method
        result = self.trade_model.execute_trade(trade_decision)

        # Verify results
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "Missing required parameters")

    def test_execute_trade_missing_capital_amount(self):
        """Test execute_trade with missing capital amount for buy order"""
        # Setup mocks
        self.mock_alpaca_mcp.get_market_hours.return_value = {"is_open": True}

        # Create a trade decision without capital amount
        trade_decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "order_type": "market",
            # Missing capital_amount
        }

        # Call the method
        result = self.trade_model.execute_trade(trade_decision)

        # Verify results
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "Missing capital_amount for buy order")

    def test_execute_trade_insufficient_capital(self):
        """Test execute_trade with insufficient daily capital"""
        # Setup mocks
        self.mock_alpaca_mcp.get_market_hours.return_value = {"is_open": True}
        
        # Mock daily usage higher than limit
        self.trade_model.position_manager.get_daily_usage = MagicMock(return_value=4900.0)

        # Create a trade decision
        trade_decision = {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 10,
            "order_type": "market",
            "capital_amount": 1500.0,  # Exceeds available (5000 - 4900 = 100)
        }

        # Call the method
        result = self.trade_model.execute_trade(trade_decision)

        # Verify results
        self.assertEqual(result["status"], "failed")
        self.assertTrue("Insufficient daily capital" in result["reason"])

    def test_run_monitoring_cycle(self):
        """Test the monitoring cycle for active positions"""
        # Setup mocks
        # Mock positions
        mock_positions = [
            {"symbol": "AAPL", "qty": 10, "avg_entry_price": 150.0},
            {"symbol": "MSFT", "qty": 5, "avg_entry_price": 200.0},
        ]
        self.trade_model.position_manager.get_positions = MagicMock(return_value=mock_positions)

        # Mock exit conditions check
        self.trade_model.monitor.check_exit_conditions = MagicMock(
            side_effect=[
                {"should_exit": True, "reason": "stop_loss"},  # AAPL should exit
                {"should_exit": False},  # MSFT should not exit
            ]
        )

        # Mock execute_trade
        self.trade_model.execute_trade = MagicMock(
            return_value={"status": "filled", "proceeds": 1450.0}
        )

        # Mock notify_decision_model
        self.trade_model.notify_decision_model = MagicMock(return_value=True)

        # Call the method
        result = self.trade_model.run_monitoring_cycle()

        # Verify results
        self.assertEqual(result["positions_checked"], 2)
        self.assertEqual(result["exit_signals"], 1)
        self.assertEqual(result["positions_closed"], 1)
        self.assertEqual(result["errors"], 0)

        # Verify method calls
        self.trade_model.monitor.check_exit_conditions.assert_any_call("AAPL")
        self.trade_model.monitor.check_exit_conditions.assert_any_call("MSFT")
        self.trade_model.execute_trade.assert_called_once()
        self.trade_model.notify_decision_model.assert_called_once()

        # Verify trade decision parameters
        trade_call_args = self.trade_model.execute_trade.call_args[0][0]
        self.assertEqual(trade_call_args["symbol"], "AAPL")
        self.assertEqual(trade_call_args["action"], "sell")
        self.assertEqual(trade_call_args["quantity"], 10.0)  # Should be float
        self.assertEqual(trade_call_args["order_type"], "market")


class TestTradeExecutor(unittest.TestCase):
    """Test case for TradeExecutor"""

    def setUp(self):
        """Set up test environment"""
        # Create mock TradeModel
        self.mock_trade_model = MagicMock()
        self.mock_trade_model.alpaca_mcp = MagicMock(spec=AlpacaMCP)
        self.mock_trade_model.redis_mcp = MagicMock(spec=RedisMCP)
        self.mock_trade_model.logger = MagicMock()
        self.mock_trade_model.position_manager = MagicMock()
        self.mock_trade_model.monitor = MagicMock()

        # Create an instance of TradeExecutor
        self.trade_executor = TradeExecutor(self.mock_trade_model)

    def test_execute_market_order_buy(self):
        """Test executing a buy market order"""
        # Setup mocks
        self.mock_trade_model.alpaca_mcp.submit_market_order.return_value = {
            "id": "test_order_id",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "type": "market",
            "status": "filled",
            "filled_qty": 10,
            "filled_avg_price": 150.0,
            "created_at": datetime.now().isoformat(),
            "filled_at": datetime.now().isoformat(),
        }

        # Call the method
        result = self.trade_executor.execute_market_order("AAPL", 10, "buy")

        # Verify results
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["filled_qty"], 10)
        self.assertEqual(result["filled_avg_price"], 150.0)

        # Verify method calls
        self.mock_trade_model.alpaca_mcp.submit_market_order.assert_called_once_with(
            symbol="AAPL", qty=10, side="buy", time_in_force="day"
        )
        self.mock_trade_model.redis_mcp.set_json.assert_called_once()
        self.mock_trade_model.redis_mcp.add_to_sorted_set.assert_called_once()
        self.mock_trade_model.monitor.start_position_monitoring.assert_called_once_with(
            "AAPL", "test_order_id"
        )
        self.mock_trade_model.position_manager.update_daily_usage.assert_called_once_with(
            10 * 150.0
        )

    def test_execute_market_order_sell(self):
        """Test executing a sell market order"""
        # Setup mocks
        self.mock_trade_model.alpaca_mcp.submit_market_order.return_value = {
            "id": "test_order_id",
            "symbol": "AAPL",
            "side": "sell",
            "qty": 10,
            "type": "market",
            "status": "filled",
            "filled_qty": 10,
            "filled_avg_price": 150.0,
            "created_at": datetime.now().isoformat(),
            "filled_at": datetime.now().isoformat(),
        }

        # Call the method
        result = self.trade_executor.execute_market_order("AAPL", 10, "sell")

        # Verify results
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["filled_qty"], 10)
        self.assertEqual(result["filled_avg_price"], 150.0)
        self.assertEqual(result["proceeds"], 1500.0)  # 10 * 150.0

        # Verify method calls
        self.mock_trade_model.alpaca_mcp.submit_market_order.assert_called_once_with(
            symbol="AAPL", qty=10, side="sell", time_in_force="day"
        )
        self.mock_trade_model.redis_mcp.set_json.assert_called_once()
        self.mock_trade_model.redis_mcp.add_to_sorted_set.assert_called_once()
        self.mock_trade_model.position_manager.update_daily_usage.assert_called_once_with(
            -1500.0
        )
        # Monitor should not be called for sell orders
        self.mock_trade_model.monitor.start_position_monitoring.assert_not_called()

    def test_execute_limit_order(self):
        """Test executing a limit order"""
        # Setup mocks
        self.mock_trade_model.alpaca_mcp.submit_limit_order.return_value = {
            "id": "test_order_id",
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "type": "limit",
            "limit_price": 150.0,
            "status": "accepted",
            "created_at": datetime.now().isoformat(),
        }

        # Call the method
        result = self.trade_executor.execute_limit_order("AAPL", 10, "buy", 150.0)

        # Verify results
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["limit_price"], 150.0)

        # Verify method calls
        self.mock_trade_model.alpaca_mcp.submit_limit_order.assert_called_once_with(
            symbol="AAPL", qty=10, side="buy", limit_price=150.0, time_in_force="day"
        )
        self.mock_trade_model.redis_mcp.set_json.assert_called_once()
        self.mock_trade_model.redis_mcp.add_to_sorted_set.assert_called_once()
        self.mock_trade_model.position_manager.update_daily_usage.assert_called_once_with(
            10 * 150.0
        )


class TestTradeMonitor(unittest.TestCase):
    """Test case for TradeMonitor"""

    def setUp(self):
        """Set up test environment"""
        # Create mock TradeModel
        self.mock_trade_model = MagicMock()
        self.mock_trade_model.alpaca_mcp = MagicMock(spec=AlpacaMCP)
        self.mock_trade_model.redis_mcp = MagicMock(spec=RedisMCP)
        self.mock_trade_model.drift_detection_mcp = MagicMock()
        self.mock_trade_model.logger = MagicMock()
        self.mock_trade_model.position_manager = MagicMock()
        self.mock_trade_model.no_overnight_positions = True

        # Create an instance of TradeMonitor
        self.trade_monitor = TradeMonitor(self.mock_trade_model)

    def test_start_position_monitoring(self):
        """Test starting position monitoring"""
        # Setup mocks
        self.mock_trade_model.position_manager.get_position.return_value = {
            "symbol": "AAPL",
            "qty": 10,
            "avg_entry_price": 150.0,
        }
        self.mock_trade_model.alpaca_mcp.get_order.return_value = {
            "id": "test_order_id",
            "symbol": "AAPL",
            "side": "buy",
            "filled_avg_price": 150.0,
        }
        self.mock_trade_model.redis_mcp.set_json.return_value = True

        # Call the method
        result = self.trade_monitor.start_position_monitoring("AAPL", "test_order_id")

        # Verify results
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["entry_price"], 150.0)
        self.assertEqual(result["quantity"], 10.0)

        # Verify method calls
        self.mock_trade_model.position_manager.get_position.assert_called_once_with(
            "AAPL"
        )
        self.mock_trade_model.alpaca_mcp.get_order.assert_called_once_with(
            "test_order_id"
        )
        self.mock_trade_model.redis_mcp.set_json.assert_called_once()

    def test_check_exit_conditions_stop_loss(self):
        """Test checking exit conditions with stop loss triggered"""
        # Setup mocks
        # Create monitor config with low stop loss
        monitor_config = {
            "symbol": "AAPL",
            "entry_price": 150.0,
            "high_since_entry": 150.0,
            "low_since_entry": 145.0,
            "stop_loss_pct": 0.02,  # 2%
            "take_profit_pct": 0.05,  # 5%
            "trailing_stop_pct": 0.01,  # 1%
        }
        self.trade_monitor.active_monitors = {"AAPL": monitor_config}

        # Mock current price below stop loss
        self.mock_trade_model.alpaca_mcp.get_latest_quote.return_value = {
            "ask_price": 145.0,  # Below stop loss (150 * 0.98 = 147)
        }

        # Call the method
        result = self.trade_monitor.check_exit_conditions("AAPL")

        # Verify results
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "stop_loss")
        self.assertAlmostEqual(
            result["data"]["gain_loss_pct"], -0.0333, places=3
        )  # (145 - 150) / 150

    def test_check_exit_conditions_take_profit(self):
        """Test checking exit conditions with take profit triggered"""
        # Setup mocks
        # Create monitor config
        monitor_config = {
            "symbol": "AAPL",
            "entry_price": 150.0,
            "high_since_entry": 160.0,
            "low_since_entry": 150.0,
            "stop_loss_pct": 0.02,  # 2%
            "take_profit_pct": 0.05,  # 5%
            "trailing_stop_pct": 0.01,  # 1%
        }
        self.trade_monitor.active_monitors = {"AAPL": monitor_config}

        # Mock current price above take profit
        self.mock_trade_model.alpaca_mcp.get_latest_quote.return_value = {
            "ask_price": 160.0,  # Above take profit (150 * 1.05 = 157.5)
        }

        # Call the method
        result = self.trade_monitor.check_exit_conditions("AAPL")

        # Verify results
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "take_profit")
        self.assertAlmostEqual(result["data"]["gain_loss_pct"], 0.0667, places=3)  # (160 - 150) / 150

    def test_check_exit_conditions_trailing_stop(self):
        """Test checking exit conditions with trailing stop triggered"""
        # Setup mocks
        # Create monitor config with high water mark
        monitor_config = {
            "symbol": "AAPL",
            "entry_price": 150.0,
            "high_since_entry": 160.0,  # High water mark
            "low_since_entry": 150.0,
            "stop_loss_pct": 0.02,  # 2%
            "take_profit_pct": 0.05,  # 5%
            "trailing_stop_pct": 0.01,  # 1%
        }
        self.trade_monitor.active_monitors = {"AAPL": monitor_config}

        # Mock current price dropped from high but still in profit
        self.mock_trade_model.alpaca_mcp.get_latest_quote.return_value = {
            "ask_price": 158.0,  # Below high by more than 1% (160 * 0.99 = 158.4)
        }

        # Call the method
        result = self.trade_monitor.check_exit_conditions("AAPL")

        # Verify results
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "trailing_stop")
        self.assertAlmostEqual(result["data"]["gain_loss_pct"], 0.0533, places=3)  # (158 - 150) / 150
        self.assertAlmostEqual(
            result["data"]["drawdown_from_peak"], -0.0125, places=3
        )  # (158 - 160) / 160

    def test_check_exit_conditions_market_close(self):
        """Test checking exit conditions with approaching market close"""
        # Setup mocks
        # Create monitor config
        monitor_config = {
            "symbol": "AAPL",
            "entry_price": 150.0,
            "high_since_entry": 152.0,
            "low_since_entry": 149.0,
            "stop_loss_pct": 0.02,  # 2%
            "take_profit_pct": 0.05,  # 5%
            "trailing_stop_pct": 0.01,  # 1%
        }
        self.trade_monitor.active_monitors = {"AAPL": monitor_config}

        # Mock current price (no triggering of other exits)
        self.mock_trade_model.alpaca_mcp.get_latest_quote.return_value = {
            "ask_price": 152.0,
        }

        # Mock approaching market close
        # The _is_approaching_market_close method would check if market close is within 15 minutes
        self.trade_monitor._is_approaching_market_close = MagicMock(return_value=True)

        # Call the method
        result = self.trade_monitor.check_exit_conditions("AAPL")

        # Verify results
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "market_close")


class TestTradePositionManager(unittest.TestCase):
    """Test case for TradePositionManager"""

    def setUp(self):
        """Set up test environment"""
        # Create mock TradeModel
        self.mock_trade_model = MagicMock()
        self.mock_trade_model.alpaca_mcp = MagicMock(spec=AlpacaMCP)
        self.mock_trade_model.redis_mcp = MagicMock(spec=RedisMCP)
        self.mock_trade_model.logger = MagicMock()

        # Patch the sync_portfolio_data method to avoid calling it during init
        with patch.object(
            TradePositionManager, "sync_portfolio_data", return_value=True
        ):
            # Create an instance of TradePositionManager
            self.position_manager = TradePositionManager(self.mock_trade_model)

    def test_get_positions(self):
        """Test getting positions"""
        # Setup mocks
        # First return None to simulate cache miss, then provide data from Alpaca
        self.mock_trade_model.redis_mcp.get_json.side_effect = [None]
        
        # Mock Alpaca response
        mock_position_objs = [
            MagicMock(_raw={
                "symbol": "AAPL",
                "qty": "10",
                "avg_entry_price": "150.00",
                "market_value": "1550.00",
                "cost_basis": "1500.00",
                "unrealized_pl": "50.00",
                "unrealized_plpc": "0.0333",
                "current_price": "155.00",
            }),
            MagicMock(_raw={
                "symbol": "MSFT",
                "qty": "5",
                "avg_entry_price": "200.00",
                "market_value": "1025.00",
                "cost_basis": "1000.00",
                "unrealized_pl": "25.00",
                "unrealized_plpc": "0.025",
                "current_price": "205.00",
            }),
        ]
        self.mock_trade_model.alpaca_mcp.get_all_positions.return_value = mock_position_objs

        # Call the method
        result = self.position_manager.get_positions()

        # Verify results
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["symbol"], "AAPL")
        self.assertEqual(result[0]["qty"], 10.0)  # Should be converted to float
        self.assertEqual(result[0]["avg_entry_price"], 150.0)
        self.assertEqual(result[1]["symbol"], "MSFT")
        self.assertEqual(result[1]["qty"], 5.0)

        # Verify method calls
        self.mock_trade_model.redis_mcp.get_json.assert_called_once()
        self.mock_trade_model.alpaca_mcp.get_all_positions.assert_called_once()
        self.mock_trade_model.redis_mcp.set_json.assert_called()

    def test_get_position(self):
        """Test getting a specific position"""
        # Setup mocks
        # First return None to simulate cache miss, then provide data from Alpaca
        self.mock_trade_model.redis_mcp.get_json.return_value = None
        
        # Mock Alpaca response
        mock_position = MagicMock(_raw={
            "symbol": "AAPL",
            "qty": "10",
            "avg_entry_price": "150.00",
            "market_value": "1550.00",
            "cost_basis": "1500.00",
            "unrealized_pl": "50.00",
            "unrealized_plpc": "0.0333",
            "current_price": "155.00",
        })
        self.mock_trade_model.alpaca_mcp.get_position.return_value = mock_position

        # Call the method
        result = self.position_manager.get_position("AAPL")

        # Verify results
        self.assertEqual(result["symbol"], "AAPL")
        self.assertEqual(result["qty"], 10.0)  # Should be converted to float
        self.assertEqual(result["avg_entry_price"], 150.0)

        # Verify method calls
        self.mock_trade_model.redis_mcp.get_json.assert_called_once()
        self.mock_trade_model.alpaca_mcp.get_position.assert_called_once_with("AAPL")
        self.mock_trade_model.redis_mcp.set_json.assert_called_once()

    def test_update_daily_usage(self):
        """Test updating daily capital usage"""
        # Setup mocks
        self.mock_trade_model.redis_mcp.increment_float.return_value = 1500.0
        
        # Call the method
        result = self.position_manager.update_daily_usage(1500.0)
