r"""
Trading MCP Server

This module implements a Model Context Protocol (MCP) server for trading platform interactions,
initially focusing on Alpaca, providing access to trading functionality,
account information, and market data.
"""

import os
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import Alpaca Trading API
try:
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
    HAVE_ALPACA = True
except ImportError:
    HAVE_ALPACA = False
    print("Warning: alpaca-trade-api not installed. Trading features will be unavailable.")


# Import internal modules
from mcp_tools.base_mcp_server import BaseMCPServer
# Assuming monitoring setup might be centralized or removed if not needed here
# from monitoring import setup_monitoring


class TradingMCP(BaseMCPServer):
    """
    MCP server for trading platform interactions (initially Alpaca).

    Provides access to trading, account information, and market data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Trading MCP server.

        Args:
            config: Optional configuration dictionary. May contain Alpaca settings:
                - api_key: Alpaca API key (overrides environment variable)
                - api_secret: Alpaca API secret (overrides environment variable)
                - base_url: Alpaca API base URL (default: https://api.alpaca.markets)
                - paper_trading: Whether to use paper trading (default: True)
        """
        # Load config from standard path if not provided
        if config is None:
            config_path = os.path.join("config", "trading_mcp", "trading_mcp_config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    print(f"Configuration loaded from {config_path}") # Use print if logger not ready
                except Exception as e:
                    print(f"Error loading config from {config_path}: {e}")
                    config = {}
            else:
                print(f"Warning: Config file not found at {config_path}. Using default settings.")
                config = {}

        # Initialize base server *after* loading config
        super().__init__(name="trading_mcp", config=config)

        # Initialize Alpaca clients (only if library is available)
        self.trading_client = None
        self.data_client = None
        if HAVE_ALPACA:
            self._initialize_alpaca_client()
        else:
            self.logger.warning("Alpaca library not found, Alpaca features disabled.")

        # Register tools (only if client initialized successfully)
        if self.trading_client and self.data_client:
            self._register_alpaca_tools()
        else:
            self.logger.warning("Alpaca client not initialized, tools not registered.")

        self.logger.info("TradingMCP initialized")

    def _initialize_alpaca_client(self):
        """Initialize the Alpaca clients."""
        start_time = time.time()
        self.logger.info("Initializing Alpaca clients")
        
        try:
            # Get Alpaca configuration from self.config (already loaded)
            alpaca_config = self.config.get("alpaca", {}) # Expect settings under "alpaca" key
            api_key = alpaca_config.get("api_key") or os.environ.get("ALPACA_API_KEY")
            api_secret = alpaca_config.get("api_secret") or os.environ.get("ALPACA_SECRET_KEY")
            paper_trading = alpaca_config.get("paper_trading", True)

            if not api_key or not api_secret:
                self.logger.warning("Alpaca API credentials not provided - API calls will fail")
                self.logger.counter("trading_mcp.credential_failures")
                return

            # Create Alpaca clients
            client_start_time = time.time()
            self.trading_client = TradingClient(api_key, api_secret, paper=paper_trading)
            self.data_client = StockHistoricalDataClient(api_key, api_secret)
            client_init_time = (time.time() - client_start_time) * 1000
            self.logger.timing("trading_mcp.client_init_time_ms", client_init_time)

            # Test connection by getting account info
            try:
                acct_start_time = time.time()
                account = self.trading_client.get_account()
                acct_query_time = (time.time() - acct_start_time) * 1000
                self.logger.timing("trading_mcp.account_query_time_ms", acct_query_time)
                
                self.logger.info(f"Connected to Alpaca account: {account.id} (Paper: {paper_trading})")
                self.logger.counter("trading_mcp.successful_connections")
                
                # Track account status metrics
                try:
                    equity_value = float(account.equity)
                    buying_power = float(account.buying_power)
                    self.logger.gauge("trading_mcp.account_equity", equity_value)
                    self.logger.gauge("trading_mcp.account_buying_power", buying_power)
                except (ValueError, AttributeError):
                    self.logger.warning("Failed to parse account numeric values for metrics")
            except Exception as acct_e:
                self.logger.error(f"Error getting Alpaca account: {acct_e}")
                self.logger.counter("trading_mcp.account_access_failures")
                self.trading_client = None
                self.data_client = None
                raise  # Re-raise to be caught by outer exception handler

        except Exception as e:
            self.logger.error(f"Failed to initialize Alpaca client: {e}", exc_info=True)
            self.logger.counter("trading_mcp.client_init_failures")
            self.trading_client = None
            self.data_client = None
            
        # Log initialization time
        total_init_time = (time.time() - start_time) * 1000
        self.logger.timing("trading_mcp.alpaca_init_total_time_ms", total_init_time)
        
        # Report status as a gauge metric
        client_status = 1 if (self.trading_client and self.data_client) else 0
        self.logger.gauge("trading_mcp.alpaca_client_status", client_status)

    def _register_alpaca_tools(self):
        """Register tools specific to Alpaca functionality."""
        # Account Tools
        self.register_tool(self.get_account_info, "get_account_info", "Get Alpaca account information.")
        self.register_tool(self.get_buying_power, "get_buying_power", "Get available Alpaca buying power.")
        self.register_tool(self.get_positions, "get_positions", "Get all current Alpaca positions.")
        self.register_tool(self.get_position, "get_position", "Get Alpaca position for a specific symbol.")
        self.register_tool(self.get_portfolio_history, "get_portfolio_history", "Get Alpaca portfolio history.")
        
        # Register additional monitoring tools
        self.register_tool(self.get_trading_client_health, "get_trading_client_health", 
                         "Get health status of Alpaca trading client connection.")

        # Order Tools
        self.register_tool(self.get_orders, "get_orders", "Get Alpaca orders (filter by status, limit).")
        self.register_tool(self.get_order_by_id, "get_order_by_id", "Get a specific Alpaca order by its ID.")
        self.register_tool(self.submit_market_order, "submit_market_order", "Submit a market order via Alpaca.")
        self.register_tool(self.submit_limit_order, "submit_limit_order", "Submit a limit order via Alpaca.")
        self.register_tool(self.submit_stop_order, "submit_stop_order", "Submit a stop order via Alpaca.")
        self.register_tool(self.cancel_order, "cancel_order", "Cancel a specific Alpaca order by ID.")
        self.register_tool(self.cancel_all_orders, "cancel_all_orders", "Cancel all open Alpaca orders.")

        # Market Data Tools
        self.register_tool(self.get_historical_bars, "get_historical_bars", "Get historical bars for a symbol from Alpaca.")
        self.register_tool(self.get_historical_quotes, "get_historical_quotes", "Get historical quotes for a symbol from Alpaca.")
        self.register_tool(self.get_latest_trade, "get_latest_trade", "Get the latest trade for a symbol from Alpaca.")
        self.register_tool(self.get_latest_quote, "get_latest_quote", "Get the latest quote for a symbol from Alpaca.")
        
        # Asset Tools
        self.register_tool(self.get_assets, "get_assets", "Get available assets from Alpaca (filter by status/class).")
        self.register_tool(self.get_asset, "get_asset", "Get information for a specific asset symbol from Alpaca.")
        
        if HAVE_ALPACA:
            def _str_to_timeframe(self, timeframe_str: str) -> Optional[TimeFrame]:
                """Convert a string timeframe to Alpaca TimeFrame enum."""
                tf_mapping = {
                    "1min": TimeFrame.Minute, "minute": TimeFrame.Minute,
                    "5min": TimeFrame.Minute_5,
                    "15min": TimeFrame.Minute_15,
                    "1hour": TimeFrame.Hour, "hour": TimeFrame.Hour,
                    "1day": TimeFrame.Day, "day": TimeFrame.Day,
                    "1week": TimeFrame.Week, "week": TimeFrame.Week,
                    "1month": TimeFrame.Month, "month": TimeFrame.Month,
                }
                return tf_mapping.get(timeframe_str.lower())
            
            # Assign the method to the instance
            self._str_to_timeframe = _str_to_timeframe.__get__(self, self.__class__)

    def _check_trading_client(self) -> bool:
        """
        Check if trading client is initialized.
        
        Returns:
            bool: True if client is available, False otherwise
        """
        if self.trading_client is None:
            self.logger.error("Trading client is not initialized. API calls will fail.")
            self.logger.counter("trading_mcp.missing_client_errors")
            return False
        return True
        
    def _check_data_client(self) -> bool:
        """
        Check if data client is initialized.
        
        Returns:
            bool: True if client is available, False otherwise
        """
        if self.data_client is None:
            self.logger.error("Data client is not initialized. API calls will fail.")
            self.logger.counter("trading_mcp.missing_client_errors")
            return False
        return True
        
    def get_trading_client_health(self) -> Dict[str, Any]:
        """
        Get health status of Alpaca trading client connection.
        
        Returns:
            Dictionary containing health status metrics
        """
        start_time = time.time()
        operation_id = f"health_{int(start_time)}"
        
        self.logger.info(f"Performing trading client health check - ID: {operation_id}")
        
        # Initialize health report structure
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "check_id": operation_id,
            "status": "unknown",
            "client_initialized": bool(self.trading_client and self.data_client),
            "connection_status": "unknown",
            "response_times_ms": {},
            "metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        # Early return if clients not initialized
        if not health_report["client_initialized"]:
            health_report["status"] = "critical"
            health_report["connection_status"] = "disconnected"
            health_report["issues"].append("Trading and/or data clients not initialized")
            health_report["recommendations"].append("Check API credentials and connectivity")
            
            # Log this health status
            self.logger.warning(f"Trading client health check failed - clients not initialized - ID: {operation_id}")
            self.logger.gauge("trading_mcp.health_status", 0)  # 0 = critical
            
            # Calculate total duration
            health_report["duration_ms"] = round((time.time() - start_time) * 1000, 2)
            return health_report
        
        # Test account connectivity (most reliable way to verify connection)
        try:
            acct_start_time = time.time()
            account = self.trading_client.get_account()
            acct_query_time = (time.time() - acct_start_time) * 1000
            health_report["response_times_ms"]["account_query"] = round(acct_query_time, 2)
            
            # Connection is successful
            health_report["connection_status"] = "connected"
            
            # Extract some account metrics for health report
            try:
                equity_value = float(account.equity)
                buying_power = float(account.buying_power)
                health_report["metrics"]["account_equity"] = equity_value
                health_report["metrics"]["account_buying_power"] = buying_power
                
                # Add to the health metrics (gauge values)
                self.logger.gauge("trading_mcp.account_equity", equity_value)
                self.logger.gauge("trading_mcp.account_buying_power", buying_power)
            except (ValueError, AttributeError) as e:
                health_report["issues"].append(f"Failed to extract account metrics: {str(e)}")
            
            # Test order listing for additional connectivity check
            try:
                orders_start_time = time.time()
                recent_orders = self.trading_client.get_orders(limit=5)
                orders_query_time = (time.time() - orders_start_time) * 1000
                health_report["response_times_ms"]["orders_query"] = round(orders_query_time, 2)
                health_report["metrics"]["recent_orders_count"] = len(recent_orders) if recent_orders else 0
            except Exception as e:
                health_report["issues"].append(f"Failed to query recent orders: {str(e)}")
                health_report["recommendations"].append("Check API permissions for order access")
            
            # Add slow response detection
            slow_threshold_ms = 500  # Threshold for "slow" responses
            for check_name, response_time in health_report["response_times_ms"].items():
                if response_time > slow_threshold_ms:
                    health_report["issues"].append(f"Slow response for {check_name}: {response_time}ms")
            
            # Determine overall status based on checks
            if health_report["issues"]:
                if len(health_report["issues"]) > 2:
                    health_report["status"] = "degraded"
                    self.logger.gauge("trading_mcp.health_status", 1)  # 1 = degraded
                else:
                    health_report["status"] = "warning"
                    self.logger.gauge("trading_mcp.health_status", 2)  # 2 = warning
            else:
                health_report["status"] = "healthy"
                self.logger.gauge("trading_mcp.health_status", 3)  # 3 = healthy
            
        except Exception as e:
            health_report["status"] = "critical"
            health_report["connection_status"] = "error"
            health_report["issues"].append(f"Failed to connect to Alpaca API: {str(e)}")
            health_report["recommendations"].append("Check API credentials, network connectivity, and Alpaca service status")
            self.logger.error(f"Trading client health check failed: {str(e)} - ID: {operation_id}")
            self.logger.gauge("trading_mcp.health_status", 0)  # 0 = critical
            self.logger.counter("trading_mcp.health_check_failures")
        
        # Calculate total health check duration
        total_duration = (time.time() - start_time) * 1000
        health_report["duration_ms"] = round(total_duration, 2)
        self.logger.timing("trading_mcp.health_check_duration_ms", total_duration)
        
        # Log health check completion
        log_method = self.logger.info if health_report["status"] in ["healthy", "warning"] else self.logger.warning
        log_method(f"Trading client health check completed - Status: {health_report['status']} - ID: {operation_id}")
        
        return health_report


    # --- Helper Methods ---

    def _ensure_client(self) -> bool:
        """Check if the Alpaca client is initialized."""
        if not self.trading_client or not self.data_client:
            self.logger.error("Alpaca client not initialized or connection failed.")
            return False
        return True

    def _format_api_error(self, e: Exception, action: str) -> Dict[str, str]:
        """Format API errors consistently."""
        self.logger.error(f"Alpaca API error during {action}: {e}", exc_info=True)
        # Consider adding specific Alpaca error code parsing if available
        return {"error": f"Alpaca API error: {str(e)}"}

    def _format_position(self, position) -> Dict[str, Any]:
        """Format an Alpaca position object into a dictionary."""
        # Ensure values are serializable (float, int, str)
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

    def _format_order(self, order) -> Dict[str, Any]:
        """Format an Alpaca order object into a dictionary."""
        return {
            "id": order.id,
            "client_order_id": order.client_order_id,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "updated_at": order.updated_at.isoformat() if order.updated_at else None,
            "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
            "filled_at": order.filled_at.isoformat() if order.filled_at else None,
            "expired_at": order.expired_at.isoformat() if order.expired_at else None,
            "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
            "failed_at": order.failed_at.isoformat() if order.failed_at else None,
            "asset_id": order.asset_id,
            "symbol": order.symbol,
            "asset_class": str(order.asset_class), # Enum to string
            "qty": float(order.qty) if order.qty else None,
            "filled_qty": float(order.filled_qty) if order.filled_qty else None,
            "type": str(order.type), # Enum to string
            "side": str(order.side), # Enum to string
            "time_in_force": str(order.time_in_force), # Enum to string
            "limit_price": float(order.limit_price) if order.limit_price else None,
            "stop_price": float(order.stop_price) if order.stop_price else None,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "status": str(order.status), # Enum to string
            "extended_hours": order.extended_hours,
            "legs": [self._format_order(leg) for leg in order.legs] if hasattr(order, "legs") and order.legs else None,
        }

    # --- Tool Implementations ---

    # Account Tools
    def get_account_info(self) -> Dict[str, Any]:
        """Get Alpaca account information."""
        start_time = time.time()
        operation_id = f"acct_{int(start_time)}"
        self.logger.info(f"Getting account info - Operation ID: {operation_id}")
        
        if not self._ensure_client():
            self.logger.counter("trading_mcp.account_info_failures")
            error_duration = (time.time() - start_time) * 1000
            self.logger.timing("trading_mcp.account_info_error_ms", error_duration)
            return {
                "error": "Alpaca client not available.",
                "_monitoring": {
                    "operation_id": operation_id,
                    "duration_ms": round(error_duration, 2),
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
        
        try:
            # Request account info with timing
            request_start_time = time.time()
            account = self.trading_client.get_account()
            request_duration = (time.time() - request_start_time) * 1000
            self.logger.timing("trading_mcp.account_request_ms", request_duration)
            
            # Track account metrics
            try:
                equity_value = float(account.equity)
                buying_power = float(account.buying_power)
                self.logger.gauge("trading_mcp.account_equity", equity_value)
                self.logger.gauge("trading_mcp.account_buying_power", buying_power)
                
                # Calculate margin usage percentage
                if float(account.portfolio_value) > 0:
                    margin_usage = (float(account.initial_margin) / float(account.portfolio_value)) * 100
                    self.logger.gauge("trading_mcp.margin_usage_percent", margin_usage)
            except (ValueError, AttributeError) as e:
                self.logger.warning(f"Error calculating account metrics: {str(e)}")
            
            # Format necessary fields to be JSON serializable
            result = {
                "id": account.id,
                "status": str(account.status),
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
            
            # Calculate and log total duration
            total_duration = (time.time() - start_time) * 1000
            self.logger.timing("trading_mcp.account_info_total_ms", total_duration)
            self.logger.counter("trading_mcp.account_info_success")
            
            # Add monitoring info to result
            result["_monitoring"] = {
                "operation_id": operation_id,
                "duration_ms": round(total_duration, 2),
                "request_ms": round(request_duration, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error getting account info: {error_msg}")
            self.logger.counter("trading_mcp.account_info_failures")
            
            # Calculate and log error duration
            error_duration = (time.time() - start_time) * 1000
            self.logger.timing("trading_mcp.account_info_error_ms", error_duration)
            
            error_response = self._format_api_error(e, "get_account_info")
            
            # Add monitoring info to error response
            error_response["_monitoring"] = {
                "operation_id": operation_id,
                "duration_ms": round(error_duration, 2),
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return error_response

    def get_buying_power(self) -> Dict[str, Any]:
        """Get available Alpaca buying power."""
        account_info = self.get_account_info()
        if "error" in account_info:
            return account_info
        return {"buying_power": account_info.get("buying_power", 0.0)}

    def get_positions(self) -> Dict[str, Any]:
        """Get all current Alpaca positions."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            positions = self.trading_client.get_all_positions()
            return {
                "positions": [self._format_position(p) for p in positions],
                "count": len(positions),
            }
        except Exception as e:
            return self._format_api_error(e, "get_positions")

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """Get Alpaca position for a specific symbol."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            position = self.trading_client.get_position(symbol)
            return self._format_position(position)
        except Exception as e:
            # Handle case where position doesn't exist gracefully
            if "position not found" in str(e).lower():
                 self.logger.info(f"No position found for symbol: {symbol}")
                 return {"symbol": symbol, "qty": 0.0} # Return zero position
            return self._format_api_error(e, f"get_position for {symbol}")

    def get_portfolio_history(self, period: Optional[str] = None, timeframe: Optional[str] = None,
                              date_start: Optional[str] = None, date_end: Optional[str] = None,
                              extended_hours: Optional[bool] = False) -> Dict[str, Any]:
        """Get Alpaca portfolio history."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            history = self.trading_client.get_portfolio_history(
                period=period, timeframe=timeframe, date_start=date_start,
                date_end=date_end, extended_hours=extended_hours
            )
            # Format the response to be JSON serializable
            return {
                "timestamp": [ts.isoformat() for ts in history.timestamp],
                "equity": [float(e) if e is not None else None for e in history.equity],
                "profit_loss": [float(pl) if pl is not None else None for pl in history.profit_loss],
                "profit_loss_pct": [float(plp) if plp is not None else None for plp in history.profit_loss_pct],
                "base_value": float(history.base_value),
                "timeframe": str(history.timeframe),
            }
        except Exception as e:
            return self._format_api_error(e, "get_portfolio_history")


    # Order Tools
    def get_orders(self, status: Optional[str] = None, limit: Optional[int] = 50,
                   after: Optional[str] = None, until: Optional[str] = None,
                   direction: Optional[str] = None, nested: Optional[bool] = True) -> Dict[str, Any]:
        """Get Alpaca orders (filter by status, limit, date range, direction)."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            orders = self.trading_client.get_orders(
                status=status, limit=limit, after=after, until=until,
                direction=direction, nested=nested
            )
            return {
                "orders": [self._format_order(o) for o in orders],
                "count": len(orders),
            }
        except Exception as e:
            return self._format_api_error(e, "get_orders")

    def get_order_by_id(self, order_id: str) -> Dict[str, Any]:
        """Get a specific Alpaca order by its ID."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return self._format_order(order)
        except Exception as e:
             if "order not found" in str(e).lower():
                  return {"error": f"Order ID '{order_id}' not found."}
             return self._format_api_error(e, f"get_order_by_id for {order_id}")

    def submit_market_order(self, symbol: str, qty: float, side: str,
                            time_in_force: str = "day", client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit a market order via Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            order_side = OrderSide(side.lower())
            tif = TimeInForce(time_in_force.lower())
            order_request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=order_side, time_in_force=tif,
                client_order_id=client_order_id
            )
            order = self.trading_client.submit_order(order_request)
            self.logger.info(f"Market order submitted: {order.id} for {symbol}")
            return self._format_order(order)
        except Exception as e:
            return self._format_api_error(e, "submit_market_order")

    def submit_limit_order(self, symbol: str, qty: float, side: str, limit_price: float,
                           time_in_force: str = "day", client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit a limit order via Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            order_side = OrderSide(side.lower())
            tif = TimeInForce(time_in_force.lower())
            order_request = LimitOrderRequest(
                symbol=symbol, qty=qty, side=order_side, limit_price=limit_price,
                time_in_force=tif, client_order_id=client_order_id
            )
            order = self.trading_client.submit_order(order_request)
            self.logger.info(f"Limit order submitted: {order.id} for {symbol}")
            return self._format_order(order)
        except Exception as e:
            return self._format_api_error(e, "submit_limit_order")

    def submit_stop_order(self, symbol: str, qty: float, side: str, stop_price: float,
                          time_in_force: str = "day", client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Submit a stop order via Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            order_side = OrderSide(side.lower())
            tif = TimeInForce(time_in_force.lower())
            order_request = StopOrderRequest(
                symbol=symbol, qty=qty, side=order_side, stop_price=stop_price,
                time_in_force=tif, client_order_id=client_order_id
            )
            order = self.trading_client.submit_order(order_request)
            self.logger.info(f"Stop order submitted: {order.id} for {symbol}")
            return self._format_order(order)
        except Exception as e:
            return self._format_api_error(e, "submit_stop_order")

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a specific Alpaca order by ID."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            self.trading_client.cancel_order(order_id)
            self.logger.info(f"Order canceled: {order_id}")
            return {"order_id": order_id, "status": "canceled"}
        except Exception as e:
            # Handle case where order might already be closed/canceled
            if "order is not cancelable" in str(e).lower():
                 self.logger.warning(f"Order {order_id} is not cancelable (already closed/canceled?).")
                 # Attempt to get order status to confirm
                 order_status = self.get_order_by_id(order_id)
                 if "error" in order_status:
                      return {"error": f"Failed to cancel order {order_id} and could not retrieve status: {str(e)}"}
                 else:
                      return {"order_id": order_id, "status": order_status.get("status", "not_cancelable")}
            return self._format_api_error(e, f"cancel_order for {order_id}")

    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel all open Alpaca orders."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            cancel_responses = self.trading_client.cancel_orders() # Returns list of order statuses or empty list
            canceled_ids = [resp.id for resp in cancel_responses if resp.status == 207] # 207 Multi-Status might indicate success
            failed_ids = [resp.id for resp in cancel_responses if resp.status != 207]
            self.logger.info(f"Attempted to cancel all orders. Success/Partial: {len(canceled_ids)}, Failed: {len(failed_ids)}")
            return {"status": "attempted cancel all", "canceled_count": len(canceled_ids), "failed_count": len(failed_ids)}
        except Exception as e:
            return self._format_api_error(e, "cancel_all_orders")


    # Market Data Tools
    def get_historical_bars(self, symbol: str, timeframe: str, start: Optional[str] = None,
                            end: Optional[str] = None, limit: Optional[int] = 100,
                            adjustment: str = "raw") -> Dict[str, Any]:
        """Get historical bars for a symbol from Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        tf = self._str_to_timeframe(timeframe)
        if not tf: return {"error": f"Invalid timeframe: {timeframe}"}

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=tf, start=start, end=end,
                limit=limit, adjustment=adjustment
            )
            bars = self.data_client.get_stock_bars(request)
            # Format the response
            if symbol in bars:
                symbol_bars = bars[symbol]
                result = [{
                    "timestamp": bar.timestamp.isoformat(), "open": float(bar.open),
                    "high": float(bar.high), "low": float(bar.low), "close": float(bar.close),
                    "volume": int(bar.volume),
                    "vwap": float(bar.vwap) if hasattr(bar, "vwap") and bar.vwap is not None else None,
                    "trade_count": int(bar.trade_count) if hasattr(bar, "trade_count") and bar.trade_count is not None else None,
                } for bar in symbol_bars]
                return {"symbol": symbol, "timeframe": timeframe, "bars": result, "count": len(result)}
            else:
                return {"symbol": symbol, "timeframe": timeframe, "bars": [], "count": 0}
        except Exception as e:
            return self._format_api_error(e, f"get_historical_bars for {symbol}")

    def get_historical_quotes(self, symbol: str, start: Optional[str] = None,
                              end: Optional[str] = None, limit: Optional[int] = 100) -> Dict[str, Any]:
        """Get historical quotes for a symbol from Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            request = StockQuotesRequest(
                symbol_or_symbols=symbol, start=start, end=end, limit=limit
            )
            quotes = self.data_client.get_stock_quotes(request)
            # Format the response
            if symbol in quotes:
                symbol_quotes = quotes[symbol]
                result = [{
                    "timestamp": quote.timestamp.isoformat(), "ask_price": float(quote.ask_price),
                    "ask_size": int(quote.ask_size), "bid_price": float(quote.bid_price),
                    "bid_size": int(quote.bid_size),
                    "conditions": quote.conditions if hasattr(quote, "conditions") else None,
                } for quote in symbol_quotes]
                return {"symbol": symbol, "quotes": result, "count": len(result)}
            else:
                return {"symbol": symbol, "quotes": [], "count": 0}
        except Exception as e:
            return self._format_api_error(e, f"get_historical_quotes for {symbol}")

    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """Get the latest trade for a symbol from Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            trade = self.data_client.get_stock_latest_trade(symbol)
            return {
                "symbol": symbol, "timestamp": trade.timestamp.isoformat(),
                "price": float(trade.price), "size": int(trade.size),
                "exchange": trade.exchange,
                "conditions": trade.conditions if hasattr(trade, "conditions") else None,
                "tape": trade.tape if hasattr(trade, "tape") else None,
            }
        except Exception as e:
            return self._format_api_error(e, f"get_latest_trade for {symbol}")

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Get the latest quote for a symbol from Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            quote = self.data_client.get_stock_latest_quote(symbol)
            return {
                "symbol": symbol, "timestamp": quote.timestamp.isoformat(),
                "ask_price": float(quote.ask_price), "ask_size": int(quote.ask_size),
                "bid_price": float(quote.bid_price), "bid_size": int(quote.bid_size),
                "conditions": quote.conditions if hasattr(quote, "conditions") else None,
            }
        except Exception as e:
            return self._format_api_error(e, f"get_latest_quote for {symbol}")


    # Asset Tools
    def get_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """Get available assets from Alpaca (filter by status/class)."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            assets = self.trading_client.get_all_assets(status=status, asset_class=asset_class)
            result = [{
                "id": asset.id, "class": str(asset.asset_class), "exchange": asset.exchange,
                "symbol": asset.symbol, "name": asset.name, "status": str(asset.status),
                "tradable": asset.tradable, "marginable": asset.marginable,
                "shortable": asset.shortable, "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable,
            } for asset in assets]
            return {"assets": result, "count": len(result)}
        except Exception as e:
            return self._format_api_error(e, "get_assets")

    def get_asset(self, symbol: str) -> Dict[str, Any]:
        """Get information for a specific asset symbol from Alpaca."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            asset = self.trading_client.get_asset(symbol)
            return {
                "id": asset.id, "class": str(asset.asset_class), "exchange": asset.exchange,
                "symbol": asset.symbol, "name": asset.name, "status": str(asset.status),
                "tradable": asset.tradable, "marginable": asset.marginable,
                "shortable": asset.shortable, "easy_to_borrow": asset.easy_to_borrow,
                "fractionable": asset.fractionable,
            }
        except Exception as e:
             if "asset not found" in str(e).lower():
                  return {"error": f"Asset '{symbol}' not found."}
             return self._format_api_error(e, f"get_asset for {symbol}")


    # Calendar Tools
    def get_calendar(self, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
        """Get the Alpaca market calendar for a date range."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            calendar = self.trading_client.get_calendar(start=start, end=end)
            result = [{
                "date": day.date.isoformat(), "open": day.open.isoformat(),
                "close": day.close.isoformat(),
                "session_open": day.session_open.isoformat(),
                "session_close": day.session_close.isoformat(),
            } for day in calendar]
            return {"calendar": result, "count": len(result)}
        except Exception as e:
            return self._format_api_error(e, "get_calendar")

    def get_clock(self) -> Dict[str, Any]:
        """Get the current Alpaca market clock status."""
        if not self._ensure_client(): return {"error": "Alpaca client not available."}
        try:
            clock = self.trading_client.get_clock()
            return {
                "timestamp": clock.timestamp.isoformat(), "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat(), "next_close": clock.next_close.isoformat(),
            }
        except Exception as e:
            return self._format_api_error(e, "get_clock")

    def is_market_open(self) -> Dict[str, Any]:
        """Check if the market is currently open according to Alpaca."""
        clock_info = self.get_clock()
        if "error" in clock_info:
            return clock_info # Propagate error
        return {"is_open": clock_info.get("is_open", False)}
