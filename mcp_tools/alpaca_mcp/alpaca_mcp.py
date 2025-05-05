"""
Alpaca MCP Server

This module implements a Model Context Protocol (MCP) server for Alpaca,
providing access to trading functionality, account information, and market data.
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Alpaca Trading API
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopOrderRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame

# Import internal modules
from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring


class AlpacaMCP(BaseMCPServer):
    """
    MCP server for Alpaca trading platform.

    This server provides access to Alpaca for trading, account information,
    and market data in the trading system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Alpaca MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - api_key: Alpaca API key (overrides environment variable)
                - api_secret: Alpaca API secret (overrides environment variable)
                - base_url: Alpaca API base URL (default: https://api.alpaca.markets)
                - paper_trading: Whether to use paper trading (default: True)
        """
        super().__init__(name="alpaca_mcp", config=config)

        # Load configuration - if no config provided, try to load from standard location
        if config is None:
            config_path = os.path.join("config", "alpaca_mcp", "alpaca_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        self.config = json.load(f)
                    self.logger.info(f"Configuration loaded from {config_path}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Error parsing configuration file {config_path}: {e}")
                    self.config = {}
                except Exception as e:
                    self.logger.error(f"Error loading configuration file {config_path}: {e}")
                    self.config = {}
            else:
                self.logger.warning(f"No configuration provided and standard config file not found at {config_path}")
                self.config = {}
        else:
            self.config = config

        # Initialize monitoring
        self.monitor, self.metrics = setup_monitoring(
            service_name="alpaca-mcp",
            enable_prometheus=True,
            enable_loki=True,
            default_labels={"component": "alpaca_mcp"},
        )
        if self.monitor:
            self.monitor.log_info(
                "AlpacaMCP initialized", component="alpaca_mcp", action="initialization"
            )

        # Initialize Alpaca clients
        self.trading_client, self.data_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        self.logger.info("AlpacaMCP initialized")

    def fetch_data(
        self, endpoint_name: str, params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Fetch data from an endpoint.

        Args:
            endpoint_name: Name of the endpoint to fetch data from
            params: Parameters to pass to the endpoint

        Returns:
            Fetched data

        Raises:
            ValueError: If the endpoint doesn't exist
        """
        if endpoint_name not in self.endpoints:
            self.logger.error(f"Endpoint not found: {endpoint_name}")
            return {"error": f"Endpoint not found: {endpoint_name}"}

        endpoint = self.endpoints[endpoint_name]
        handler = endpoint.get("handler")

        if not handler:
            self.logger.error(f"No handler for endpoint: {endpoint_name}")
            return {"error": f"No handler for endpoint: {endpoint_name}"}

        params = params or {}

        # Check required parameters
        required_params = endpoint.get("required_params", [])
        for param in required_params:
            if param not in params:
                self.logger.error(f"Missing required parameter: {param}")
                return {"error": f"Missing required parameter: {param}"}

        # Call the handler
        self.logger.info(f"Fetching data from endpoint: {endpoint_name}")
        try:
            result = handler(params)
            return result
        except Exception as e:
            self.logger.error(f"Error fetching data from endpoint {endpoint_name}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error fetching data from endpoint {endpoint_name}: {e}",
                    component="alpaca_mcp",
                    action="fetch_data_error",
                    error=str(e),
                )
            return {"error": str(e)}

    def _initialize_client(self) -> Tuple[TradingClient, StockHistoricalDataClient]:
        """
        Initialize the Alpaca clients.

        Returns:
            Tuple of (TradingClient, StockHistoricalDataClient)
        """
        try:
            # Get Alpaca configuration
            api_key = self.config.get("api_key") or os.environ.get("ALPACA_API_KEY")
            api_secret = self.config.get("api_secret") or os.environ.get(
                "ALPACA_SECRET_KEY"
            )
            paper_trading = self.config.get("paper_trading", True)

            if not api_key or not api_secret:
                self.logger.warning(
                    "Alpaca API credentials not provided - API calls will fail"
                )
                return None, None

            # Create Alpaca clients
            trading_client = TradingClient(api_key, api_secret, paper=paper_trading)
            data_client = StockHistoricalDataClient(api_key, api_secret)

            # Test connection by getting account info
            account = trading_client.get_account()
            self.logger.info(
                f"Connected to Alpaca account: {account.id} (Paper: {paper_trading})"
            )

            return trading_client, data_client

        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Failed to initialize Alpaca client: {e}",
                    component="alpaca_mcp",
                    action="connection_error",
                    error=str(e),
                )
            return None, None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Alpaca.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            # Account endpoints
            "get_account": {
                "description": "Get account information",
                "category": "account",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_get_account,
            },
            "get_positions": {
                "description": "Get current positions",
                "category": "account",
                "required_params": [],
                "optional_params": ["symbol"],
                "handler": self._handle_get_positions,
            },
            "get_portfolio_history": {
                "description": "Get portfolio history",
                "category": "account",
                "required_params": [],
                "optional_params": [
                    "period",
                    "timeframe",
                    "date_start",
                    "date_end",
                    "extended_hours",
                ],
                "handler": self._handle_get_portfolio_history,
            },
            # Order endpoints
            "get_orders": {
                "description": "Get orders",
                "category": "orders",
                "required_params": [],
                "optional_params": [
                    "status",
                    "limit",
                    "after",
                    "until",
                    "direction",
                    "nested",
                ],
                "handler": self._handle_get_orders,
            },
            "get_order": {
                "description": "Get order by ID",
                "category": "orders",
                "required_params": ["order_id"],
                "optional_params": [],
                "handler": self._handle_get_order,
            },
            "submit_market_order": {
                "description": "Submit a market order",
                "category": "orders",
                "required_params": ["symbol", "qty", "side"],
                "optional_params": [
                    "time_in_force",
                    "extended_hours",
                    "client_order_id",
                ],
                "handler": self._handle_submit_market_order,
            },
            "submit_limit_order": {
                "description": "Submit a limit order",
                "category": "orders",
                "required_params": ["symbol", "qty", "side", "limit_price"],
                "optional_params": [
                    "time_in_force",
                    "extended_hours",
                    "client_order_id",
                ],
                "handler": self._handle_submit_limit_order,
            },
            "submit_stop_order": {
                "description": "Submit a stop order",
                "category": "orders",
                "required_params": ["symbol", "qty", "side", "stop_price"],
                "optional_params": [
                    "time_in_force",
                    "extended_hours",
                    "client_order_id",
                ],
                "handler": self._handle_submit_stop_order,
            },
            "cancel_order": {
                "description": "Cancel an order",
                "category": "orders",
                "required_params": ["order_id"],
                "optional_params": [],
                "handler": self._handle_cancel_order,
            },
            "cancel_all_orders": {
                "description": "Cancel all orders",
                "category": "orders",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_cancel_all_orders,
            },
            # Market data endpoints
            "get_bars": {
                "description": "Get historical bars",
                "category": "market_data",
                "required_params": ["symbol", "timeframe"],
                "optional_params": ["start", "end", "limit", "adjustment"],
                "handler": self._handle_get_bars,
            },
            "get_quotes": {
                "description": "Get historical quotes",
                "category": "market_data",
                "required_params": ["symbol"],
                "optional_params": ["start", "end", "limit"],
                "handler": self._handle_get_quotes,
            },
            "get_latest_trade": {
                "description": "Get latest trade",
                "category": "market_data",
                "required_params": ["symbol"],
                "optional_params": [],
                "handler": self._handle_get_latest_trade,
            },
            "get_latest_quote": {
                "description": "Get latest quote",
                "category": "market_data",
                "required_params": ["symbol"],
                "optional_params": [],
                "handler": self._handle_get_latest_quote,
            },
            # Asset endpoints
            "get_assets": {
                "description": "Get assets",
                "category": "assets",
                "required_params": [],
                "optional_params": ["status", "asset_class"],
                "handler": self._handle_get_assets,
            },
            "get_asset": {
                "description": "Get asset by symbol",
                "category": "assets",
                "required_params": ["symbol"],
                "optional_params": [],
                "handler": self._handle_get_asset,
            },
            # Calendar endpoints
            "get_calendar": {
                "description": "Get market calendar",
                "category": "calendar",
                "required_params": [],
                "optional_params": ["start", "end"],
                "handler": self._handle_get_calendar,
            },
            "get_clock": {
                "description": "Get market clock",
                "category": "calendar",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_get_clock,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Alpaca MCP."""
        self.register_tool(self.get_account_info)
        self.register_tool(self.get_buying_power)
        self.register_tool(self.get_positions)
        self.register_tool(self.get_position)
        self.register_tool(self.get_orders)
        self.register_tool(self.submit_market_order)
        self.register_tool(self.submit_limit_order)
        self.register_tool(self.cancel_all_orders)
        self.register_tool(self.get_market_hours)
        self.register_tool(self.get_historical_bars)
        self.register_tool(self.get_latest_quote)

    # Handler methods for specific endpoints

    def _handle_get_account(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_account endpoint."""
        try:
            account = self.trading_client.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "last_equity": float(account.last_equity),
                "last_maintenance_margin": float(account.last_maintenance_margin),
                "multiplier": account.multiplier,
                "daytrade_count": account.daytrade_count,
                "daytrading_buying_power": float(account.daytrading_buying_power),
                "regt_buying_power": float(account.regt_buying_power),
            }
        except Exception as e:
            self.logger.error(f"Error getting account information: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting account information: {e}",
                    component="alpaca_mcp",
                    action="get_account_error",
                    error=str(e),
                )
            return {"error": f"Failed to get account information: {str(e)}"}

    def _handle_get_positions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_positions endpoint."""
        symbol = params.get("symbol")

        try:
            if symbol:
                position = self.trading_client.get_position(symbol)
                return self._format_position(position)
            else:
                positions = self.trading_client.get_all_positions()
                return {
                    "positions": [
                        self._format_position(position) for position in positions
                    ],
                    "count": len(positions),
                }
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting positions: {e}",
                    component="alpaca_mcp",
                    action="get_positions_error",
                    error=str(e),
                )
            return {"error": f"Failed to get positions: {str(e)}"}

    def _format_position(self, position) -> Dict[str, Any]:
        """Format a position object into a dictionary."""
        return {
            "symbol": position.symbol,
            "qty": float(position.qty),
            "avg_entry_price": float(position.avg_entry_price),
            "market_value": float(position.market_value),
            "cost_basis": float(position.cost_basis),
            "unrealized_pl": float(position.unrealized_pl),
            "unrealized_plpc": float(position.unrealized_plpc),
            "current_price": float(position.current_price),
            "lastday_price": float(position.lastday_price),
            "change_today": float(position.change_today),
            "side": "long" if float(position.qty) > 0 else "short",
        }

    def _handle_get_portfolio_history(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_portfolio_history endpoint."""
        period = params.get("period")
        timeframe = params.get("timeframe")
        date_start = params.get("date_start")
        date_end = params.get("date_end")
        extended_hours = params.get("extended_hours", False)

        try:
            history = self.trading_client.get_portfolio_history(
                period=period,
                timeframe=timeframe,
                date_start=date_start,
                date_end=date_end,
                extended_hours=extended_hours,
            )

            # Format the response
            result = {
                "timestamp": history.timestamp,
                "equity": history.equity,
                "profit_loss": history.profit_loss,
                "profit_loss_pct": history.profit_loss_pct,
                "base_value": history.base_value,
                "timeframe": history.timeframe,
            }

            return result
        except Exception as e:
            self.logger.error(f"Error getting portfolio history: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting portfolio history: {e}",
                    component="alpaca_mcp",
                    action="get_portfolio_history_error",
                    error=str(e),
                )
            return {"error": f"Failed to get portfolio history: {str(e)}"}

    def _handle_get_orders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_orders endpoint."""
        status = params.get("status")
        limit = params.get("limit")
        after = params.get("after")
        until = params.get("until")
        direction = params.get("direction")
        nested = params.get("nested", True)

        try:
            orders = self.trading_client.get_orders(
                status=status,
                limit=limit,
                after=after,
                until=until,
                direction=direction,
                nested=nested,
            )

            return {
                "orders": [self._format_order(order) for order in orders],
                "count": len(orders),
            }
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting orders: {e}",
                    component="alpaca_mcp",
                    action="get_orders_error",
                    error=str(e),
                )
            return {"error": f"Failed to get orders: {str(e)}"}

    def _handle_get_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_order endpoint."""
        order_id = params.get("order_id")

        if not order_id:
            return {"error": "Missing required parameter: order_id"}

        try:
            # Use get_order_by_id instead of get_orders with filter
            order = self.trading_client.get_order_by_id(order_id)
            if not order:
                return {"error": f"Order not found: {order_id}"}
            
            # Return the formatted order
            return self._format_order(order)
        except Exception as e:
            self.logger.error(f"Error getting order {order_id}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting order {order_id}: {e}",
                    component="alpaca_mcp",
                    action="get_order_error",
                    error=str(e),
                )
            return {"error": f"Failed to get order: {str(e)}"}

    def _format_order(self, order) -> Dict[str, Any]:
        """Format an order object into a dictionary."""
        return {
            "id": order.id,
            "client_order_id": order.client_order_id,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "updated_at": order.updated_at.isoformat() if order.updated_at else None,
            "submitted_at": order.submitted_at.isoformat()
            if order.submitted_at
            else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "expired_at": order.expired_at.isoformat() if order.expired_at else None,
            "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
            "failed_at": order.failed_at.isoformat() if order.failed_at else None,
            "asset_id": order.asset_id,
            "symbol": order.symbol,
            "asset_class": order.asset_class,
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else None,
            "type": order.type,
            "side": order.side,
            "time_in_force": order.time_in_force,
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "filled_avg_price": float(order.filled_avg_price)
            if order.filled_avg_price
            else None,
            "status": order.status,
            "extended_hours": order.extended_hours,
            "legs": [self._format_order(leg) for leg in order.legs]
            if hasattr(order, "legs") and order.legs
            else None,
        }

    def _handle_submit_market_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle submit_market_order endpoint."""
        symbol = params.get("symbol")
        qty = params.get("qty")
        side = params.get("side")
        time_in_force = params.get("time_in_force", "day")
        client_order_id = params.get("client_order_id")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}
        if qty is None:
            return {"error": "Missing required parameter: qty"}
        if not side:
            return {"error": "Missing required parameter: side"}

        try:
            # Create the order request
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce(time_in_force),
                client_order_id=client_order_id,
            )

            # Submit the order
            order = self.trading_client.submit_order(order_request)

            return self._format_order(order)
        except Exception as e:
            self.logger.error(f"Error submitting market order: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error submitting market order: {e}",
                    component="alpaca_mcp",
                    action="submit_market_order_error",
                    error=str(e),
                )
            return {"error": f"Failed to submit market order: {str(e)}"}

    def _handle_submit_limit_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle submit_limit_order endpoint."""
        symbol = params.get("symbol")
        qty = params.get("qty")
        side = params.get("side")
        limit_price = params.get("limit_price")
        time_in_force = params.get("time_in_force", "day")
        client_order_id = params.get("client_order_id")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}
        if qty is None:
            return {"error": "Missing required parameter: qty"}
        if not side:
            return {"error": "Missing required parameter: side"}
        if limit_price is None:
            return {"error": "Missing required parameter: limit_price"}

        try:
            # Create the order request
            order_request = LimitOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                limit_price=float(limit_price),
                time_in_force=TimeInForce(time_in_force),
                client_order_id=client_order_id,
            )

            # Submit the order
            order = self.trading_client.submit_order(order_request)

            return self._format_order(order)
        except Exception as e:
            self.logger.error(f"Error submitting limit order: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error submitting limit order: {e}",
                    component="alpaca_mcp",
                    action="submit_limit_order_error",
                    error=str(e),
                )
            return {"error": f"Failed to submit limit order: {str(e)}"}

    def _handle_submit_stop_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle submit_stop_order endpoint."""
        symbol = params.get("symbol")
        qty = params.get("qty")
        side = params.get("side")
        stop_price = params.get("stop_price")
        time_in_force = params.get("time_in_force", "day")
        client_order_id = params.get("client_order_id")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}
        if qty is None:
            return {"error": "Missing required parameter: qty"}
        if not side:
            return {"error": "Missing required parameter: side"}
        if stop_price is None:
            return {"error": "Missing required parameter: stop_price"}

        try:
            # Create the order request
            order_request = StopOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
                stop_price=float(stop_price),
                time_in_force=TimeInForce(time_in_force),
                client_order_id=client_order_id,
            )

            # Submit the order
            order = self.trading_client.submit_order(order_request)

            return self._format_order(order)
        except Exception as e:
            self.logger.error(f"Error submitting stop order: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error submitting stop order: {e}",
                    component="alpaca_mcp",
                    action="submit_stop_order_error",
                    error=str(e),
                )
            return {"error": f"Failed to submit stop order: {str(e)}"}

    def _handle_cancel_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cancel_order endpoint."""
        order_id = params.get("order_id")

        if not order_id:
            return {"error": "Missing required parameter: order_id"}

        try:
            self.trading_client.cancel_order(order_id)
            return {"order_id": order_id, "status": "canceled"}
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error canceling order {order_id}: {e}",
                    component="alpaca_mcp",
                    action="cancel_order_error",
                    error=str(e),
                )
            return {"error": f"Failed to cancel order: {str(e)}"}

    def _handle_cancel_all_orders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cancel_all_orders endpoint."""
        try:
            self.trading_client.cancel_orders()
            return {"status": "all orders canceled"}
        except Exception as e:
            self.logger.error(f"Error canceling all orders: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error canceling all orders: {e}",
                    component="alpaca_mcp",
                    action="cancel_all_orders_error",
                    error=str(e),
                )
            return {"error": f"Failed to cancel all orders: {str(e)}"}

    def _handle_get_bars(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_bars endpoint."""
        symbol = params.get("symbol")
        timeframe = params.get("timeframe")
        start = params.get("start")
        end = params.get("end")
        limit = params.get("limit")
        adjustment = params.get("adjustment", "raw")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}
        if not timeframe:
            return {"error": "Missing required parameter: timeframe"}

        try:
            # Convert timeframe string to TimeFrame enum
            tf_mapping = {
                "1min": TimeFrame.Minute,
                "5min": TimeFrame.Minute_5,
                "15min": TimeFrame.Minute_15,
                "1hour": TimeFrame.Hour,
                "1day": TimeFrame.Day,
                "1week": TimeFrame.Week,
                "1month": TimeFrame.Month,
            }

            tf = tf_mapping.get(timeframe.lower())
            if not tf:
                return {"error": f"Invalid timeframe: {timeframe}"}

            # Create the request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
                limit=limit,
                adjustment=adjustment,
            )

            # Get the bars
            bars = self.data_client.get_stock_bars(request)

            # Format the response
            if symbol in bars:
                # Single symbol response
                symbol_bars = bars[symbol]
                result = []

                for bar in symbol_bars:
                    result.append(
                        {
                            "timestamp": bar.timestamp.isoformat(),
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume),
                            "vwap": float(bar.vwap) if hasattr(bar, "vwap") else None,
                            "trade_count": int(bar.trade_count)
                            if hasattr(bar, "trade_count")
                            else None,
                        }
                    )

                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": result,
                    "count": len(result),
                }
            else:
                # Empty response
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": [],
                    "count": 0,
                }
        except Exception as e:
            self.logger.error(f"Error getting bars for {symbol}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting bars for {symbol}: {e}",
                    component="alpaca_mcp",
                    action="get_bars_error",
                    error=str(e),
                )
            return {"error": f"Failed to get bars: {str(e)}"}

    def _handle_get_quotes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_quotes endpoint."""
        symbol = params.get("symbol")
        start = params.get("start")
        end = params.get("end")
        limit = params.get("limit")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}

        try:
            # Create the request
            request = StockQuotesRequest(
                symbol_or_symbols=symbol, start=start, end=end, limit=limit
            )

            # Get the quotes
            quotes = self.data_client.get_stock_quotes(request)

            # Format the response
            if symbol in quotes:
                # Single symbol response
                symbol_quotes = quotes[symbol]
                result = []

                for quote in symbol_quotes:
                    result.append(
                        {
                            "timestamp": quote.timestamp.isoformat(),
                            "ask_price": float(quote.ask_price),
                            "ask_size": int(quote.ask_size),
                            "bid_price": float(quote.bid_price),
                            "bid_size": int(quote.bid_size),
                            "conditions": quote.conditions
                            if hasattr(quote, "conditions")
                            else None,
                        }
                    )

                return {"symbol": symbol, "quotes": result, "count": len(result)}
            else:
                # Empty response
                return {"symbol": symbol, "quotes": [], "count": 0}
        except Exception as e:
            self.logger.error(f"Error getting quotes for {symbol}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting quotes for {symbol}: {e}",
                    component="alpaca_mcp",
                    action="get_quotes_error",
                    error=str(e),
                )
            return {"error": f"Failed to get quotes: {str(e)}"}

    def _handle_get_latest_trade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_latest_trade endpoint."""
        symbol = params.get("symbol")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}

        try:
            # Get the latest trade
            trade = self.data_client.get_latest_trade(symbol)

            # Format the response
            return {
                "symbol": symbol,
                "timestamp": trade.timestamp.isoformat(),
                "price": float(trade.price),
                "size": int(trade.size),
                "exchange": trade.exchange,
                "conditions": trade.conditions
                if hasattr(trade, "conditions")
                else None,
                "tape": trade.tape if hasattr(trade, "tape") else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting latest trade for {symbol}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting latest trade for {symbol}: {e}",
                    component="alpaca_mcp",
                    action="get_latest_trade_error",
                    error=str(e),
                )
            return {"error": f"Failed to get latest trade: {str(e)}"}

    def _handle_get_latest_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_latest_quote endpoint."""
        symbol = params.get("symbol")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}

        try:
            # Use get_stock_quotes with limit=1 instead of get_latest_quote
            from alpaca.data.requests import StockQuotesRequest
            from datetime import datetime, timedelta
            
            # Get quotes from the last hour
            end = datetime.now()
            start = end - timedelta(hours=1)
            
            request = StockQuotesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                limit=1  # Just get the latest one
            )
            
            quotes = self.data_client.get_stock_quotes(request)
            
            if symbol not in quotes or not quotes[symbol]:
                return {"error": f"No quotes found for {symbol}"}
            
            # Get the latest quote (should be only one due to limit=1)
            quote = quotes[symbol][-1]
            
            # Format the response
            return {
                "symbol": symbol,
                "timestamp": quote.timestamp.isoformat(),
                "ask_price": float(quote.ask_price),
                "ask_size": int(quote.ask_size),
                "bid_price": float(quote.bid_price),
                "bid_size": int(quote.bid_size),
                "conditions": quote.conditions
                if hasattr(quote, "conditions")
                else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting latest quote for {symbol}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting latest quote for {symbol}: {e}",
                    component="alpaca_mcp",
                    action="get_latest_quote_error",
                    error=str(e),
                )
            return {"error": f"Failed to get latest quote: {str(e)}"}

    def _handle_get_assets(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_assets endpoint."""
        status = params.get("status")
        asset_class = params.get("asset_class")

        try:
            assets = self.trading_client.get_all_assets(
                status=status, asset_class=asset_class
            )

            # Format the response
            result = []
            for asset in assets:
                result.append(
                    {
                        "id": asset.id,
                        "class": asset.class_,
                        "exchange": asset.exchange,
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "status": asset.status,
                        "tradable": asset.tradable,
                        "marginable": asset.marginable,
                        "shortable": asset.shortable,
                        "easy_to_borrow": asset.easy_to_borrow,
                        "fractionable": asset.fractionable,
                    }
                )

            return {"assets": result, "count": len(result)}
        except Exception as e:
            self.logger.error(f"Error getting assets: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting assets: {e}",
                    component="alpaca_mcp",
                    action="get_assets_error",
                    error=str(e),
                )
            return {"error": f"Failed to get assets: {str(e)}"}

    def _handle_get_asset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_asset endpoint."""
        symbol = params.get("symbol")

        if not symbol:
            return {"error": "Missing required parameter: symbol"}

        try:
            asset = self.trading_client.get_asset(symbol)

            # Format the response
            return {
                "id": asset.id,
                "class": asset.class_,
                "exchange": asset.exchange,
                "symbol": asset.symbol,
                "name": asset.name,
                "status": asset.status,
                "tradable": asset.tradable,
                "marginable": asset.marginable,
                "shortable": asset.shortable,
                "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable,
            }
        except Exception as e:
            self.logger.error(f"Error getting asset {symbol}: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting asset {symbol}: {e}",
                    component="alpaca_mcp",
                    action="get_asset_error",
                    error=str(e),
                )
            return {"error": f"Failed to get asset: {str(e)}"}

    def _handle_get_calendar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_calendar endpoint."""
        start = params.get("start")
        end = params.get("end")

        try:
            calendar = self.trading_client.get_calendar(start=start, end=end)

            # Format the response
            result = []
            for day in calendar:
                result.append(
                    {
                        "date": day.date.isoformat(),
                        "open": day.open.isoformat(),
                        "close": day.close.isoformat(),
                        "session_open": day.session_open.isoformat(),
                        "session_close": day.session_close.isoformat(),
                    }
                )

            return {"calendar": result, "count": len(result)}
        except Exception as e:
            self.logger.error(f"Error getting calendar: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting calendar: {e}",
                    component="alpaca_mcp",
                    action="get_calendar_error",
                    error=str(e),
                )
            return {"error": f"Failed to get calendar: {str(e)}"}

    def _handle_get_clock(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_clock endpoint."""
        try:
            clock = self.trading_client.get_clock()

            # Format the response
            return {
                "timestamp": clock.timestamp.isoformat(),
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat(),
                "next_close": clock.next_close.isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error getting clock: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Error getting clock: {e}",
                    component="alpaca_mcp",
                    action="get_clock_error",
                    error=str(e),
                )
            return {"error": f"Failed to get clock: {str(e)}"}

    # Public API methods for models to use directly

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.

        Returns:
            Dictionary with account information
        """
        return self.fetch_data("get_account", {})

    def get_buying_power(self) -> float:
        """
        Get available buying power.

        Returns:
            Available buying power as a float
        """
        account = self.fetch_data("get_account", {})
        if "error" in account:
            return 0.0
        return float(account.get("buying_power", 0.0))

    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions.

        Returns:
            List of position dictionaries
        """
        result = self.fetch_data("get_positions", {})
        if "error" in result:
            return []
        return result.get("positions", [])

    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions (alias for get_positions).

        Returns:
            List of position dictionaries
        """
        return self.get_positions()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position dictionary or None if not found
        """
        result = self.fetch_data("get_positions", {"symbol": symbol})
        if "error" in result or "symbol" not in result:
            return None
        return result

    def get_orders(self, status: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get orders.

        Args:
            status: Order status filter (open, closed, all)
            limit: Maximum number of orders to return

        Returns:
            List of order dictionaries
        """
        params = {}
        if status:
            params["status"] = status
        if limit:
            params["limit"] = limit

        result = self.fetch_data("get_orders", params)
        if "error" in result:
            return []
        return result.get("orders", [])

    def submit_market_order(
        self, symbol: str, qty: float, side: str, time_in_force: str = "day"
    ) -> Dict[str, Any]:
        """
        Submit a market order.

        Args:
            symbol: Stock symbol
            qty: Quantity of shares
            side: Order side (buy or sell)
            time_in_force: Time in force (day, gtc, opg, cls, ioc, fok)

        Returns:
            Order information
        """
        params = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "time_in_force": time_in_force,
        }
        return self.fetch_data("submit_market_order", params)

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Dict[str, Any]:
        """
        Submit a limit order.

        Args:
            symbol: Stock symbol
            qty: Quantity of shares
            side: Order side (buy or sell)
            limit_price: Limit price
            time_in_force: Time in force (day, gtc, opg, cls, ioc, fok)

        Returns:
            Order information
        """
        params = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "limit_price": limit_price,
            "time_in_force": time_in_force,
        }
        return self.fetch_data("submit_limit_order", params)

    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Status information
        """
        return self.fetch_data("cancel_all_orders", {})

    def get_market_hours(self) -> Dict[str, Any]:
        """
        Get current market hours information.

        Returns:
            Dictionary with market hours information
        """
        return self.fetch_data("get_clock", {})

    def is_market_open(self) -> bool:
        """
        Check if the market is currently open.

        Returns:
            True if market is open, False otherwise
        """
        clock = self.fetch_data("get_clock", {})
        if "error" in clock:
            return False
        return clock.get("is_open", False)

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1day",
        start: str = None,
        end: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1min, 5min, 15min, 1hour, 1day, 1week, 1month)
            start: Start time in ISO format
            end: End time in ISO format
            limit: Maximum number of bars to return

        Returns:
            List of bar dictionaries
        """
        params = {"symbol": symbol, "timeframe": timeframe}

        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if limit:
            params["limit"] = limit

        result = self.fetch_data("get_bars", params)
        if "error" in result:
            return []
        return result.get("bars", [])

    def get_latest_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest quote for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Quote dictionary or None if not found
        """
        result = self.fetch_data("get_latest_quote", {"symbol": symbol})
        if "error" in result:
            return None
        return result
