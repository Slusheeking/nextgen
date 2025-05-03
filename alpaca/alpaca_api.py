"""
Alpaca Markets API integration for trading and market data.

A production-ready client for accessing Alpaca Markets API to submit orders,
manage positions, and retrieve market data for algorithmic trading systems.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, Tuple

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
from loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

# Load environment variables
load_dotenv()

# Setup traditional logging as fallback
logger = logging.getLogger(__name__)

class AlpacaClient:
    """
    Production client for the Alpaca Markets API with comprehensive observability.
    """
    
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None, paper: bool = True):
        """
        Initialize the Alpaca client.
        
        Args:
            api_key: API key for Alpaca (defaults to ALPACA_API_KEY environment variable)
            secret_key: Secret key for Alpaca (defaults to ALPACA_SECRET_KEY environment variable)
            paper: Whether to use paper trading (defaults to True)
        """
        # Initialize observability tools
        self.loki = LokiManager(service_name="alpaca")
        self.prom = PrometheusManager(service_name="alpaca")
        
        # Create metrics
        self.request_counter = self.prom.create_counter(
            "alpaca_api_requests_total", 
            "Total count of Alpaca API requests",
            ["endpoint", "action", "status"]
        )
        
        self.request_latency = self.prom.create_histogram(
            "alpaca_api_request_duration_seconds",
            "Alpaca API request duration in seconds",
            ["endpoint", "action"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.order_counter = self.prom.create_counter(
            "alpaca_orders_total",
            "Total count of orders submitted to Alpaca",
            ["symbol", "side", "type", "status"]
        )
        
        self.position_gauge = self.prom.create_gauge(
            "alpaca_position_value",
            "Current value of positions held in Alpaca",
            ["symbol", "side"]
        )
        
        self.account_gauge = self.prom.create_gauge(
            "alpaca_account_value",
            "Current account value metrics",
            ["metric"]
        )
        
        # Initialize API client
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            log_msg = "Missing Alpaca API credentials - API calls will fail"
            logger.warning(log_msg)
            self.loki.warning(log_msg, component="alpaca")
            return
            
        # Set the base URL based on paper trading preference
        self.paper = paper
        
        # Initialize trading client
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            
            # Initialize market data client
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key
            )
            
            logger.info("Alpaca client initialized")
            self.loki.info("Alpaca client initialized", 
                         component="alpaca", 
                         paper_trading=str(self.paper))
            
            # Verify API connectivity and update account metrics
            self._verify_api_connectivity()
            self._update_account_metrics()
            
        except Exception as e:
            error_msg = f"Failed to initialize Alpaca client: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="initialization")
            
            # Track initialization errors
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="initialization",
                action="connect",
                status="error"
            )

    def _verify_api_connectivity(self) -> bool:
        """
        Verify API connectivity by getting account information.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            self.trading_client.get_account()
            elapsed = time.time() - start_time
            
            log_msg = f"Alpaca API connectivity verified in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         paper_trading=str(self.paper),
                         duration=f"{elapsed:.4f}")
            
            # Record successful connectivity check
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="connectivity_check",
                status="success"
            )
            
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="account",
                action="connectivity_check"
            )
            
            return True
            
        except Exception as e:
            error_msg = f"API connectivity check failed: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="connectivity",
                          paper_trading=str(self.paper))
            
            # Record failed connectivity check
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="connectivity_check",
                status="error"
            )
            
            return False

    def _update_account_metrics(self) -> None:
        """Update Prometheus metrics with current account information"""
        try:
            start_time = time.time()
            account = self.trading_client.get_account()
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="account",
                action="get_account"
            )
            
            # Set account metrics in Prometheus
            self.prom.set_gauge("alpaca_account_value", float(account.equity), metric="equity")
            self.prom.set_gauge("alpaca_account_value", float(account.buying_power), metric="buying_power")
            self.prom.set_gauge("alpaca_account_value", float(account.cash), metric="cash")
            
            buying_power_pct = (float(account.buying_power) / float(account.equity)) if float(account.equity) > 0 else 0
            self.prom.set_gauge("alpaca_account_value", buying_power_pct, metric="buying_power_pct")
            
            # Log account metrics
            self.loki.info("Updated account metrics", 
                         component="alpaca", 
                         equity=str(account.equity),
                         buying_power=str(account.buying_power),
                         cash=str(account.cash))
            
            # Increment successful account request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="get_account",
                status="success"
            )
        
        except Exception as e:
            error_msg = f"Failed to update account metrics: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="account_metrics")
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="get_account",
                status="error"
            )

    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information
        """
        try:
            start_time = time.time()
            account = self.trading_client.get_account()
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="account",
                action="get_account"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="get_account",
                status="success"
            )
            
            log_msg = f"Account information retrieved in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}")
            
            # Return account info as dictionary
            return account._raw
            
        except Exception as e:
            error_msg = f"Failed to get account information: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_account")
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="get_account",
                status="error"
            )
            
            return {"error": str(e)}

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of position dictionaries
        """
        try:
            start_time = time.time()
            positions = self.trading_client.get_all_positions()
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="positions",
                action="get_all"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="positions",
                action="get_all",
                status="success"
            )
            
            position_count = len(positions)
            log_msg = f"Retrieved {position_count} positions in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         position_count=position_count)
            
            # Update position metrics in Prometheus
            for position in positions:
                side = "long" if float(position.qty) > 0 else "short"
                self.prom.set_gauge(
                    "alpaca_position_value",
                    float(position.market_value),
                    symbol=position.symbol,
                    side=side
                )
            
            # Return positions as list of dictionaries
            return [position._raw for position in positions]
            
        except Exception as e:
            error_msg = f"Failed to get positions: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_positions")
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="positions",
                action="get_all",
                status="error"
            )
            
            return []

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: The stock symbol
            
        Returns:
            Position dictionary or error
        """
        try:
            start_time = time.time()
            position = self.trading_client.get_position(symbol)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="positions",
                action="get_single"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="positions",
                action="get_single",
                status="success"
            )
            
            log_msg = f"Retrieved position for {symbol} in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         symbol=symbol,
                         quantity=position.qty,
                         value=position.market_value)
            
            # Update position metric in Prometheus
            side = "long" if float(position.qty) > 0 else "short"
            self.prom.set_gauge(
                "alpaca_position_value",
                float(position.market_value),
                symbol=position.symbol,
                side=side
            )
            
            # Return position as dictionary
            return position._raw
            
        except Exception as e:
            error_msg = f"Failed to get position for {symbol}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_position",
                          symbol=symbol)
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="positions",
                action="get_single",
                status="error"
            )
            
            return {"error": str(e)}

    def submit_market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day") -> Dict[str, Any]:
        """
        Submit a market order.
        
        Args:
            symbol: The stock symbol
            qty: Quantity to buy/sell
            side: "buy" or "sell"
            time_in_force: Time in force (default "day")
            
        Returns:
            Order information dictionary
        """
        try:
            # Convert parameters to proper enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY
            if time_in_force.lower() == "gtc":
                tif = TimeInForce.GTC
            elif time_in_force.lower() == "ioc":
                tif = TimeInForce.IOC
            
            # Create market order request
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif
            )
            
            # Submit order and track timing
            start_time = time.time()
            order = self.trading_client.submit_order(order_request)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="orders",
                action="submit_market"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="submit_market",
                status="success"
            )
            
            # Increment order counter
            self.prom.increment_counter(
                "alpaca_orders_total",
                1,
                symbol=symbol,
                side=side.lower(),
                type="market",
                status="submitted"
            )
            
            log_msg = f"Submitted {side} market order for {qty} shares of {symbol} in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         symbol=symbol,
                         quantity=str(qty),
                         side=side,
                         order_id=order.id)
            
            # Update account metrics after order submission
            self._update_account_metrics()
            
            # Return order information as dictionary
            return order._raw
            
        except Exception as e:
            error_msg = f"Failed to submit market order for {symbol}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="submit_market_order",
                          symbol=symbol,
                          quantity=str(qty),
                          side=side)
            
            # Increment error counters
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="submit_market",
                status="error"
            )
            
            self.prom.increment_counter(
                "alpaca_orders_total",
                1,
                symbol=symbol,
                side=side.lower(),
                type="market",
                status="error"
            )
            
            return {"error": str(e)}

    def submit_limit_order(self, symbol: str, qty: float, price: float, side: str, time_in_force: str = "day") -> Dict[str, Any]:
        """
        Submit a limit order.
        
        Args:
            symbol: The stock symbol
            qty: Quantity to buy/sell
            price: Limit price
            side: "buy" or "sell"
            time_in_force: Time in force (default "day")
            
        Returns:
            Order information dictionary
        """
        try:
            # Convert parameters to proper enums
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            tif = TimeInForce.DAY
            if time_in_force.lower() == "gtc":
                tif = TimeInForce.GTC
            elif time_in_force.lower() == "ioc":
                tif = TimeInForce.IOC
            
            # Create limit order request
            order_request = LimitOrderRequest(
                symbol=symbol,
                limit_price=price,
                qty=qty,
                side=order_side,
                time_in_force=tif
            )
            
            # Submit order and track timing
            start_time = time.time()
            order = self.trading_client.submit_order(order_request)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="orders",
                action="submit_limit"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="submit_limit",
                status="success"
            )
            
            # Increment order counter
            self.prom.increment_counter(
                "alpaca_orders_total",
                1,
                symbol=symbol,
                side=side.lower(),
                type="limit",
                status="submitted"
            )
            
            log_msg = f"Submitted {side} limit order for {qty} shares of {symbol} at ${price} in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         symbol=symbol,
                         quantity=str(qty),
                         price=str(price),
                         side=side,
                         order_id=order.id)
            
            # Update account metrics after order submission
            self._update_account_metrics()
            
            # Return order information as dictionary
            return order._raw
            
        except Exception as e:
            error_msg = f"Failed to submit limit order for {symbol}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="submit_limit_order",
                          symbol=symbol,
                          quantity=str(qty),
                          price=str(price),
                          side=side)
            
            # Increment error counters
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="submit_limit",
                status="error"
            )
            
            self.prom.increment_counter(
                "alpaca_orders_total",
                1,
                symbol=symbol,
                side=side.lower(),
                type="limit",
                status="error"
            )
            
            return {"error": str(e)}

    def get_orders(self, status: str = "open") -> List[Dict[str, Any]]:
        """
        Get all orders with a specific status.
        
        Args:
            status: Order status (default "open")
            
        Returns:
            List of order dictionaries
        """
        try:
            start_time = time.time()
            
            if status.lower() == "open":
                orders = self.trading_client.get_orders()
            elif status.lower() == "closed":
                orders = self.trading_client.get_orders(status="closed")
            else:
                orders = self.trading_client.get_orders(status="all")
                
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="orders",
                action="get_" + status.lower()
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="get_" + status.lower(),
                status="success"
            )
            
            order_count = len(orders)
            log_msg = f"Retrieved {order_count} {status} orders in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         order_count=order_count,
                         status=status)
            
            # Return orders as list of dictionaries
            return [order._raw for order in orders]
            
        except Exception as e:
            error_msg = f"Failed to get {status} orders: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_orders",
                          status=status)
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="get_" + status.lower(),
                status="error"
            )
            
            return []

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order by ID.
        
        Args:
            order_id: The order ID to cancel
            
        Returns:
            Success or error status dictionary
        """
        try:
            start_time = time.time()
            self.trading_client.cancel_order_by_id(order_id)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="orders",
                action="cancel"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="cancel",
                status="success"
            )
            
            # Increment order counter for cancellations
            self.prom.increment_counter(
                "alpaca_orders_total",
                1,
                symbol="unknown",  # We don't have the symbol at this point
                side="unknown",    # We don't have the side at this point
                type="unknown",    # We don't have the type at this point
                status="cancelled"
            )
            
            log_msg = f"Cancelled order {order_id} in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         order_id=order_id)
            
            # Update account metrics after order cancellation
            self._update_account_metrics()
            
            return {"status": "success", "order_id": order_id}
            
        except Exception as e:
            error_msg = f"Failed to cancel order {order_id}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="cancel_order",
                          order_id=order_id)
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="orders",
                action="cancel",
                status="error"
            )
            
            return {"error": str(e), "order_id": order_id}

    def get_historical_bars(self, symbol: str, timeframe: str = "1Day", 
                           start: Optional[str] = None, end: Optional[str] = None, 
                           limit: int = 100) -> pd.DataFrame:
        """
        Get historical bar data for a symbol.
        
        Args:
            symbol: The stock symbol
            timeframe: Bar timeframe (e.g., "1Min", "1Hour", "1Day")
            start: Start date in ISO format (e.g., "2023-01-01")
            end: End date in ISO format (e.g., "2023-12-31")
            limit: Maximum number of bars to return
            
        Returns:
            DataFrame with historical bar data
        """
        try:
            # Convert timeframe string to TimeFrame enum
            tf_mapping = {
                "1min": TimeFrame.MINUTE,
                "1hour": TimeFrame.HOUR,
                "1day": TimeFrame.DAY,
                "1week": TimeFrame.WEEK,
                "1month": TimeFrame.MONTH
            }
            
            # Default to daily if timeframe not recognized
            tf = tf_mapping.get(timeframe.lower(), TimeFrame.DAY)
            
            # Create request
            request_params = {"symbol_or_symbols": symbol, "timeframe": tf, "limit": limit}
            
            if start:
                request_params["start"] = pd.Timestamp(start, tz="America/New_York")
            if end:
                request_params["end"] = pd.Timestamp(end, tz="America/New_York")
                
            bars_request = StockBarsRequest(**request_params)
            
            # Fetch bars and track timing
            start_time = time.time()
            bars = self.data_client.get_stock_bars(bars_request)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="data",
                action="get_bars"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="data",
                action="get_bars",
                status="success"
            )
            
            # Convert to DataFrame
            if symbol in bars:
                df = bars[symbol].df
                bar_count = len(df)
                
                log_msg = f"Retrieved {bar_count} {timeframe} bars for {symbol} in {elapsed:.4f}s"
                logger.info(log_msg)
                self.loki.info(log_msg, 
                             component="alpaca", 
                             duration=f"{elapsed:.4f}",
                             symbol=symbol,
                             timeframe=timeframe,
                             bar_count=bar_count)
                
                return df
            else:
                log_msg = f"No bars returned for {symbol}"
                logger.warning(log_msg)
                self.loki.warning(log_msg, 
                                component="alpaca", 
                                symbol=symbol,
                                timeframe=timeframe)
                
                return pd.DataFrame()
            
        except Exception as e:
            error_msg = f"Failed to get historical bars for {symbol}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_bars",
                          symbol=symbol,
                          timeframe=timeframe)
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="data",
                action="get_bars",
                status="error"
            )
            
            return pd.DataFrame()
            
    def get_market_clock(self) -> Dict[str, Any]:
        """
        Get the current market clock information.
        
        Returns:
            Dictionary with market clock information
        """
        try:
            start_time = time.time()
            clock = self.trading_client.get_clock()
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="market",
                action="get_clock"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="market",
                action="get_clock",
                status="success"
            )
            
            is_open = clock.is_open
            next_open = clock.next_open
            next_close = clock.next_close
            
            # Create a gauge for market status
            self.prom.set_gauge(
                "alpaca_market_status",
                1 if is_open else 0,
                metric="is_open"
            )
            
            log_msg = f"Market is {'open' if is_open else 'closed'}, next open: {next_open}, next close: {next_close}"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         is_open=str(is_open),
                         next_open=str(next_open),
                         next_close=str(next_close))
            
            # Return clock information as dictionary
            return clock._raw
            
        except Exception as e:
            error_msg = f"Failed to get market clock: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_market_clock")
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="market",
                action="get_clock",
                status="error"
            )
            
            return {"error": str(e)}
    
    def get_calendar(self, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get market calendar for a specified date range.
        
        Args:
            start: Start date in ISO format (e.g., "2023-01-01")
            end: End date in ISO format (e.g., "2023-12-31")
            
        Returns:
            List of calendar day dictionaries
        """
        try:
            # Prepare parameters
            params = {}
            if start:
                params["start"] = pd.Timestamp(start).date().isoformat()
            if end:
                params["end"] = pd.Timestamp(end).date().isoformat()
                
            start_time = time.time()
            calendar = self.trading_client.get_calendar(**params)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="market",
                action="get_calendar"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="market",
                action="get_calendar",
                status="success"
            )
            
            days_count = len(calendar)
            log_msg = f"Retrieved {days_count} calendar days in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         days_count=days_count)
            
            # Return calendar information as list of dictionaries
            return [day._raw for day in calendar]
            
        except Exception as e:
            error_msg = f"Failed to get market calendar: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_calendar")
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="market",
                action="get_calendar",
                status="error"
            )
            
            return []
    
    def close_all_positions(self) -> Dict[str, Any]:
        """
        Close (liquidate) all open positions.
        
        Returns:
            Dictionary with results of close operations
        """
        try:
            start_time = time.time()
            
            # Get current positions
            positions = self.trading_client.get_all_positions()
            position_count = len(positions)
            
            # Close all positions
            closed_positions = self.trading_client.close_all_positions(cancel_orders=True)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="positions",
                action="close_all"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="positions",
                action="close_all",
                status="success"
            )
            
            log_msg = f"Closed {position_count} positions in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         position_count=position_count)
            
            # Update account metrics after closing positions
            self._update_account_metrics()
            
            # Return results
            return {"status": "success", "closed_positions": closed_positions}
            
        except Exception as e:
            error_msg = f"Failed to close all positions: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="close_all_positions")
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="positions",
                action="close_all",
                status="error"
            )
            
            return {"error": str(e)}
    
    def get_account_activities(self, activity_type: str = "FILL", 
                              start: Optional[str] = None, 
                              end: Optional[str] = None, 
                              limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get account activity history.
        
        Args:
            activity_type: Type of activity (e.g., "FILL", "TRANS", "DIV", etc.)
            start: Start date in ISO format (e.g., "2023-01-01")
            end: End date in ISO format (e.g., "2023-12-31")
            limit: Maximum number of activities to return
            
        Returns:
            List of activity dictionaries
        """
        try:
            # Prepare parameters
            params = {"activity_types": activity_type}
            
            if start:
                params["start"] = pd.Timestamp(start)
            if end:
                params["end"] = pd.Timestamp(end)
            
            if limit:
                params["page_size"] = min(limit, 100)  # API allows max 100 per page
                
            start_time = time.time()
            activities = self.trading_client.get_activities(**params)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="account",
                action="get_activities"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="get_activities",
                status="success"
            )
            
            activity_count = len(activities)
            log_msg = f"Retrieved {activity_count} {activity_type} activities in {elapsed:.4f}s"
            logger.info(log_msg)
            self.loki.info(log_msg, 
                         component="alpaca", 
                         duration=f"{elapsed:.4f}",
                         activity_type=activity_type,
                         activity_count=activity_count)
            
            # Return activities as list of dictionaries
            return [activity._raw for activity in activities]
            
        except Exception as e:
            error_msg = f"Failed to get account activities: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_activities",
                          activity_type=activity_type)
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="account",
                action="get_activities",
                status="error"
            )
            
            return []
    def get_quotes(self, symbol: str, start: Optional[str] = None, 
                  end: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
        """
        Get historical quotes for a symbol.
        
        Args:
            symbol: The stock symbol
            start: Start date in ISO format (e.g., "2023-01-01")
            end: End date in ISO format (e.g., "2023-12-31")
            limit: Maximum number of quotes to return
            
        Returns:
            DataFrame with quote data
        """
        try:
            # Create request
            request_params = {"symbol_or_symbols": symbol, "limit": limit}
            
            if start:
                request_params["start"] = pd.Timestamp(start, tz="America/New_York")
            if end:
                request_params["end"] = pd.Timestamp(end, tz="America/New_York")
                
            quotes_request = StockQuotesRequest(**request_params)
            
            # Fetch quotes and track timing
            start_time = time.time()
            quotes = self.data_client.get_stock_quotes(quotes_request)
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "alpaca_api_request_duration_seconds",
                elapsed,
                endpoint="data",
                action="get_quotes"
            )
            
            # Increment successful request counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="data",
                action="get_quotes",
                status="success"
            )
            
            # Convert to DataFrame
            if symbol in quotes:
                df = quotes[symbol].df
                quote_count = len(df)
                
                log_msg = f"Retrieved {quote_count} quotes for {symbol} in {elapsed:.4f}s"
                logger.info(log_msg)
                self.loki.info(log_msg, 
                             component="alpaca", 
                             duration=f"{elapsed:.4f}",
                             symbol=symbol,
                             quote_count=quote_count)
                
                return df
            else:
                log_msg = f"No quotes returned for {symbol}"
                logger.warning(log_msg)
                self.loki.warning(log_msg, 
                                component="alpaca", 
                                symbol=symbol)
                
                return pd.DataFrame()
                
        except Exception as e:
            error_msg = f"Failed to get quotes for {symbol}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="alpaca", 
                          error_type="get_quotes",
                          symbol=symbol)
            
            # Increment error counter
            self.prom.increment_counter(
                "alpaca_api_requests_total",
                1,
                endpoint="data",
                action="get_quotes",
                status="error"
            )
            
            return pd.DataFrame()
