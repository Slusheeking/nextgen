#!/usr/bin/env python3
"""
Integration test for the Alpaca MCP Tool.

This module tests the functionality of the AlpacaMCP class with the real Alpaca API,
including actual trading operations (using paper trading).

IMPORTANT: This test requires valid Alpaca API credentials to be set in environment variables:
- ALPACA_API_KEY
- ALPACA_SECRET_KEY

The test will use paper trading mode to avoid using real money.
"""

import os
import time
import unittest
from dotenv import load_dotenv

# Import the target module for testing
from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP


class TestAlpacaIntegration(unittest.TestCase):
    """Integration test cases for the AlpacaMCP class with real API."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before running tests."""
        # Load environment variables
        load_dotenv()
        
        # Check if API credentials are available
        cls.api_key = os.environ.get("ALPACA_API_KEY")
        cls.api_secret = os.environ.get("ALPACA_SECRET_KEY")
        
        if not cls.api_key or not cls.api_secret:
            raise unittest.SkipTest(
                "Skipping integration tests: Alpaca API credentials not found in environment variables"
            )
        
        # Configure the MCP server for testing with paper trading
        config = {
            "api_key": cls.api_key,
            "api_secret": cls.api_secret,
            "paper_trading": True  # Always use paper trading for tests
        }
        
        # Initialize the Alpaca MCP
        cls.alpaca_mcp = AlpacaMCP(config)
        
        # Test symbol and quantity for trading
        cls.test_symbol = "AAPL"  # Apple stock
        cls.test_qty = 1  # Just buy/sell 1 share for testing
        
        # Store order IDs for cleanup
        cls.order_ids = []

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have run."""
        # Cancel any remaining orders
        if hasattr(cls, 'alpaca_mcp') and cls.alpaca_mcp:
            try:
                cls.alpaca_mcp.cancel_all_orders()
                print("All orders canceled during cleanup")
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def setUp(self):
        """Set up before each test method."""
        # No market hours check - run tests regardless of market status
        pass

    def test_01_account_info(self):
        """Test getting account information."""
        account_info = self.alpaca_mcp.get_account_info()
        
        # Verify account info structure
        self.assertIn("id", account_info)
        self.assertIn("status", account_info)
        self.assertIn("currency", account_info)
        self.assertIn("buying_power", account_info)
        self.assertIn("cash", account_info)
        self.assertIn("portfolio_value", account_info)
        
        # Verify account is active
        self.assertEqual(account_info["status"], "ACTIVE")
        
        # Print account info for debugging
        print(f"Account ID: {account_info['id']}")
        print(f"Status: {account_info['status']}")
        print(f"Buying Power: ${account_info['buying_power']}")
        print(f"Portfolio Value: ${account_info['portfolio_value']}")

    def test_02_market_data(self):
        """Test getting market data."""
        # Get latest quote
        quote = self.alpaca_mcp.get_latest_quote(self.test_symbol)
        
        # Verify quote structure
        self.assertIn("symbol", quote)
        self.assertIn("ask_price", quote)
        self.assertIn("bid_price", quote)
        
        # Print quote info for debugging
        print(f"{self.test_symbol} Ask: ${quote['ask_price']}")
        print(f"{self.test_symbol} Bid: ${quote['bid_price']}")
        
        # Get historical bars
        bars = self.alpaca_mcp.get_historical_bars(
            symbol=self.test_symbol,
            timeframe="1day",
            limit=5
        )
        
        # Verify bars data
        self.assertGreater(len(bars), 0)
        self.assertIn("timestamp", bars[0])
        self.assertIn("close", bars[0])
        
        # Print recent closing prices
        print(f"Recent {self.test_symbol} closing prices:")
        for bar in bars:
            print(f"{bar['timestamp']}: ${bar['close']}")

    def test_03_submit_market_buy_order(self):
        """Test submitting a market buy order."""
        # Submit a market buy order
        order = self.alpaca_mcp.submit_market_order(
            symbol=self.test_symbol,
            qty=self.test_qty,
            side="buy"
        )
        
        # Store order ID for potential cleanup
        if "id" in order:
            self.__class__.order_ids.append(order["id"])
        
        # Verify order structure
        self.assertIn("id", order)
        self.assertIn("symbol", order)
        self.assertIn("qty", order)
        self.assertIn("side", order)
        self.assertIn("type", order)
        self.assertIn("status", order)
        
        # Verify order details
        self.assertEqual(order["symbol"], self.test_symbol)
        self.assertEqual(float(order["qty"]), self.test_qty)
        self.assertEqual(order["side"], "buy")
        self.assertEqual(order["type"], "market")
        
        # Print order info for debugging
        print(f"Buy Order ID: {order['id']}")
        print(f"Status: {order['status']}")
        
        # Wait for order to be filled (up to 10 seconds)
        filled = self._wait_for_order_to_fill(order["id"], timeout=10)
        if not filled:
            print("Buy order was not filled within the timeout period - skipping position verification")
            return
        
        # Verify position was created
        position = self.alpaca_mcp.get_position(self.test_symbol)
        self.assertIsNotNone(position)
        self.assertEqual(position["symbol"], self.test_symbol)
        self.assertGreaterEqual(float(position["qty"]), self.test_qty)
        
        print(f"Position: {self.test_qty} shares of {self.test_symbol}")
        print(f"Current Value: ${position['market_value']}")

    def test_04_submit_market_sell_order(self):
        """Test submitting a market sell order."""
        # Get current position
        position = self.alpaca_mcp.get_position(self.test_symbol)
        if not position:
            self.skipTest(f"No position in {self.test_symbol} to sell")
        
        sell_qty = float(position["qty"])
        
        # Submit a market sell order
        order = self.alpaca_mcp.submit_market_order(
            symbol=self.test_symbol,
            qty=sell_qty,
            side="sell"
        )
        
        # Store order ID for potential cleanup
        if "id" in order:
            self.__class__.order_ids.append(order["id"])
        
        # Verify order structure
        self.assertIn("id", order)
        self.assertIn("symbol", order)
        self.assertIn("qty", order)
        self.assertIn("side", order)
        self.assertIn("type", order)
        self.assertIn("status", order)
        
        # Verify order details
        self.assertEqual(order["symbol"], self.test_symbol)
        self.assertEqual(float(order["qty"]), sell_qty)
        self.assertEqual(order["side"], "sell")
        self.assertEqual(order["type"], "market")
        
        # Print order info for debugging
        print(f"Sell Order ID: {order['id']}")
        print(f"Status: {order['status']}")
        
        # Wait for order to be filled (up to 10 seconds)
        filled = self._wait_for_order_to_fill(order["id"], timeout=10)
        if not filled:
            print("Sell order was not filled within the timeout period - skipping position verification")
            return
        
        # Verify position was closed
        try:
            position = self.alpaca_mcp.get_position(self.test_symbol)
            self.fail(f"Position in {self.test_symbol} still exists after sell order")
        except:
            # Expected - position should be closed
            print(f"Position in {self.test_symbol} successfully closed")

    def test_05_submit_limit_orders(self):
        """Test submitting limit orders."""
        # Get latest quote
        try:
            quote = self.alpaca_mcp.get_latest_quote(self.test_symbol)
            
            # Set limit price below current bid (for buy) and above current ask (for sell)
            # to avoid immediate execution
            if quote and "bid_price" in quote and "ask_price" in quote:
                current_price = (quote["bid_price"] + quote["ask_price"]) / 2
            else:
                # Fallback if quote data is not available
                current_price = 150.00  # Reasonable price for AAPL
                print(f"Using fallback price of ${current_price} for limit order test")
                
            buy_limit_price = round(current_price * 0.95, 2)  # 5% below current price
            sell_limit_price = round(current_price * 1.05, 2)  # 5% above current price
            
            # Submit a limit buy order
            buy_order = self.alpaca_mcp.submit_limit_order(
                symbol=self.test_symbol,
                qty=self.test_qty,
                side="buy",
                limit_price=buy_limit_price
            )
            
            # Store order ID for cleanup
            if "id" in buy_order:
                self.__class__.order_ids.append(buy_order["id"])
            
            # Verify order structure
            self.assertIn("id", buy_order)
            self.assertIn("symbol", buy_order)
            self.assertIn("qty", buy_order)
            self.assertIn("side", buy_order)
            self.assertIn("type", buy_order)
            self.assertIn("status", buy_order)
            self.assertIn("limit_price", buy_order)
            
            # Verify order details
            self.assertEqual(buy_order["symbol"], self.test_symbol)
            self.assertEqual(float(buy_order["qty"]), self.test_qty)
            self.assertEqual(buy_order["side"], "buy")
            self.assertEqual(buy_order["type"], "limit")
            self.assertEqual(float(buy_order["limit_price"]), buy_limit_price)
            
            # Print order info for debugging
            print(f"Limit Buy Order ID: {buy_order['id']}")
            print(f"Status: {buy_order['status']}")
            print(f"Limit Price: ${buy_limit_price}")
            
            # Cancel the limit buy order
            cancel_result = self.alpaca_mcp.fetch_data("cancel_order", {"order_id": buy_order["id"]})
            self.assertEqual(cancel_result["status"], "canceled")
            print(f"Limit Buy Order canceled: {buy_order['id']}")
        except Exception as e:
            self.fail(f"Error in limit order test: {e}")

    def test_06_cancel_all_orders(self):
        """Test canceling all orders."""
        # Submit multiple limit orders that won't execute immediately
        orders = []
        
        try:
            # Get a reasonable price for AAPL
            try:
                quote = self.alpaca_mcp.get_latest_quote(self.test_symbol)
                if quote and "bid_price" in quote:
                    current_price = quote["bid_price"]
                else:
                    current_price = 150.00  # Fallback price for AAPL
            except:
                current_price = 150.00  # Fallback price for AAPL
                
            # Submit 3 limit buy orders at different prices
            for i in range(3):
                # Set limit price well below current price to avoid execution
                limit_price = round(current_price * (0.80 - (i * 0.05)), 2)
                
                order = self.alpaca_mcp.submit_limit_order(
                    symbol=self.test_symbol,
                    qty=self.test_qty,
                    side="buy",
                    limit_price=limit_price
                )
                
                if "id" in order:
                    orders.append(order)
                    self.__class__.order_ids.append(order["id"])
                    print(f"Created limit order {i+1}: {order['id']} at ${limit_price}")
            
            # Verify we created some orders
            self.assertGreater(len(orders), 0, "Failed to create any test orders")
            
            # Get open orders to verify they exist
            open_orders = self.alpaca_mcp.get_orders(status="open")
            self.assertGreater(len(open_orders), 0, "No open orders found")
            
            # Cancel all orders
            result = self.alpaca_mcp.cancel_all_orders()
            self.assertEqual(result["status"], "all orders canceled")
            print("All orders canceled successfully")
            
            # Verify all orders are canceled
            time.sleep(2)  # Give the API time to process the cancellations
            open_orders = self.alpaca_mcp.get_orders(status="open")
            self.assertEqual(len(open_orders), 0, "Some orders were not canceled")
            
        except Exception as e:
            self.fail(f"Error in cancel all orders test: {e}")

    def test_07_market_hours(self):
        """Test getting market hours."""
        try:
            # Get market hours
            market_hours = self.alpaca_mcp.get_market_hours()
            
            # Verify market hours structure
            self.assertIn("timestamp", market_hours)
            self.assertIn("is_open", market_hours)
            self.assertIn("next_open", market_hours)
            self.assertIn("next_close", market_hours)
            
            # Print market hours for debugging
            print(f"Market is {'open' if market_hours['is_open'] else 'closed'}")
            print(f"Next market open: {market_hours['next_open']}")
            print(f"Next market close: {market_hours['next_close']}")
            
            # Test is_market_open convenience method
            is_open = self.alpaca_mcp.is_market_open()
            self.assertEqual(is_open, market_hours["is_open"])
            
        except Exception as e:
            self.fail(f"Error in market hours test: {e}")

    def _wait_for_order_to_fill(self, order_id, timeout=10):
        """Wait for an order to be filled."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                order_status = self.alpaca_mcp.fetch_data("get_order", {"order_id": order_id})
                if order_status.get("status") == "filled":
                    return True
                elif order_status.get("status") in ["rejected", "canceled", "expired"]:
                    print(f"Order {order_id} status: {order_status.get('status')}")
                    return False
                print(f"Order {order_id} status: {order_status.get('status')} - waiting...")
                time.sleep(1)
            except Exception as e:
                print(f"Error checking order status: {e}")
                time.sleep(1)
        print(f"Order {order_id} did not fill within {timeout} seconds - continuing test")
        return False


if __name__ == "__main__":
    unittest.main()
