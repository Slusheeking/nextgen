"""
Polygon.io REST API MCP Server

This module implements a Model Context Protocol (MCP) server for the Polygon.io
REST API data source, providing access to various market data endpoints.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import time
import requests
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from monitoring.system_monitor import MonitoringManager

from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP


class PolygonRestMCP(BaseDataMCP):
    """
    MCP server for Polygon.io REST API.

    This server provides access to Polygon.io REST API endpoints through a
    unified interface with dynamic endpoint selection and configuration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Polygon.io REST API MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - api_key: Polygon.io API key (overrides environment variable)
                - base_url: Base URL for REST API
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 300)
        """
        super().__init__(name="polygon_rest_mcp", config=config)

        # Initialize monitoring
        self.monitor = MonitoringManager(
            service_name="polygon-rest-mcp"
        )
        self.monitor.log_info(
            "PolygonRestMCP initialized",
            component="polygon_rest_mcp",
            action="initialization",
        )

        # Initialize the REST client
        self.rest_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        self.logger.info(
            "PolygonRestMCP initialized with %d endpoints", len(self.endpoints)
        )
        self.monitor.log_info(
            f"PolygonRestMCP initialized with {len(self.endpoints)} endpoints",
            component="polygon_rest_mcp",
            action="init_endpoints",
        )

    def _initialize_client(self) -> Union[Dict[str, Any], None]:
        """
        Initialize the Polygon REST client configuration.

        Returns:
            Client configuration or None if initialization fails
        """
        try:
            # Get API key using base class method that uses our env_loader
            api_key = self.get_api_key("polygon")
            base_url = self.config.get("base_url", "https://api.polygon.io")

            if not api_key:
                self.logger.warning("No Polygon API key provided - API calls will fail")

            #
            # Return the configuration directly - no need for a separate client
            # class
            return {"api_key": api_key, "base_url": base_url}

        except Exception as e:
            self.logger.error(f"Failed to initialize Polygon REST client: {e}")
            if hasattr(self, "monitor") and self.monitor:
                self.monitor.log_error(
                    f"Failed to initialize Polygon REST client: {e}",
                    component="polygon_rest_mcp",
                    action="client_init_error",
                    error=str(e),
                )
            return None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Polygon.io REST API.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            # Stock Data
            "ticker_details": {
                "path": "/v3/reference/tickers/{ticker}",
                "method": "GET",
                "description": "Get ticker details including company information",
                "category": "reference",
                "required_params": ["ticker"],
                "optional_params": ["date"],
                "handler": self._handle_ticker_details,
            },
            "daily_bars": {
                "path": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}",
                "method": "GET",
                "description": "Get aggregate (bars) for a ticker over a date range",
                "category": "market_data",
                "required_params": ["ticker", "from", "to"],
                "optional_params": [
                    "multiplier",
                    "timespan",
                    "adjusted",
                    "sort",
                    "limit",
                ],
                "default_values": {
                    "multiplier": "1",
                    "timespan": "day",
                    "adjusted": "true",
                    "sort": "asc",
                    "limit": "5000",
                },
                "handler": self._handle_daily_bars,
            },
            "previous_close": {
                "path": "/v2/aggs/ticker/{ticker}/prev",
                "method": "GET",
                "description": "Get the previous day's open, high, low, and close for a ticker",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": ["adjusted"],
                "default_values": {"adjusted": "true"},
                "handler": self._handle_previous_close,
            },
            "market_status": {
                "path": "/v1/marketstatus/now",
                "method": "GET",
                "description": "Get current market status (open/closed)",
                "category": "market_info",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_market_status,
            },
            "market_holidays": {
                "path": "/v1/marketstatus/upcoming",
                "method": "GET",
                "description": "Get upcoming market holidays and special events",
                "category": "market_info",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_market_holidays,
            },
            "stock_splits": {
                "path": "/v3/reference/splits",
                "method": "GET",
                "description": "Get stock splits for a ticker",
                "category": "corporate_actions",
                "required_params": ["ticker"],
                "optional_params": ["execution_date", "reverse_split", "limit"],
                "default_values": {"limit": "100"},
                "handler": self._handle_stock_splits,
            },
            "stock_dividends": {
                "path": "/v3/reference/dividends",
                "method": "GET",
                "description": "Get stock dividends for a ticker",
                "category": "corporate_actions",
                "required_params": ["ticker"],
                "optional_params": [
                    "ex_dividend_date",
                    "dividend_type",
                    "frequency",
                    "limit",
                ],
                "default_values": {"limit": "100"},
                "handler": self._handle_stock_dividends,
            },
            "ticker_types": {
                "path": "/v3/reference/tickers/types",
                "method": "GET",
                "description": "Get ticker types and asset classes",
                "category": "reference",
                "required_params": [],
                "optional_params": ["asset_class", "locale"],
                "handler": self._handle_ticker_types,
            },
            "ticker_list": {
                "path": "/v3/reference/tickers",
                "method": "GET",
                "description": "List all tickers with optional filtering",
                "category": "reference",
                "required_params": [],
                "optional_params": [
                    "ticker",
                    "type",
                    "market",
                    "exchange",
                    "cusip",
                    "cik",
                    "date",
                    "search",
                    "active",
                    "sort",
                    "order",
                    "limit",
                ],
                "default_values": {
                    "active": "true",
                    "sort": "ticker",
                    "order": "asc",
                    "limit": "100",
                },
                "handler": self._handle_ticker_list,
            },
            "stock_snapshot": {
                "path": "/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}",
                "method": "GET",
                "description": "Get a snapshot of a stock ticker",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": [],
                "handler": self._handle_stock_snapshot,
            },
            "market_movers": {
                "path": "/v2/snapshot/locale/us/markets/stocks/{direction}",
                "method": "GET",
                "description": "Get market movers (gainers or losers)",
                "category": "market_data",
                "required_params": ["direction"],
                "optional_params": ["limit", "include_otc"],
                "default_values": {"limit": "20", "include_otc": "false"},
                "handler": self._handle_market_movers,
            },
            # New endpoints from example
            "grouped_daily_bars": {
                "path": "/v2/aggs/grouped/locale/us/market/stocks/{date}",
                "method": "GET",
                "description": "Get grouped daily bars for entire market for a specific date",
                "category": "market_data",
                "required_params": ["date"],
                "optional_params": ["adjusted", "include_otc"],
                "default_values": {"adjusted": "true", "include_otc": "false"},
                "handler": self._handle_grouped_daily_bars,
            },
            "daily_open_close": {
                "path": "/v1/open-close/{ticker}/{date}",
                "method": "GET",
                "description": "Get daily open, close, high, and low for a specific ticker and date",
                "category": "market_data",
                "required_params": ["ticker", "date"],
                "optional_params": ["adjusted"],
                "default_values": {"adjusted": "true"},
                "handler": self._handle_daily_open_close,
            },
            "trades": {
                "path": "/v3/trades/{ticker}",
                "method": "GET",
                "description": "Get trades for a ticker symbol",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": [
                    "timestamp",
                    "timestamp_lt",
                    "timestamp_lte",
                    "timestamp_gt",
                    "timestamp_gte",
                    "limit",
                    "sort",
                    "order",
                ],
                "default_values": {
                    "limit": "100",
                    "sort": "timestamp",
                    "order": "desc",
                },
                "handler": self._handle_trades,
            },
            "last_trade": {
                "path": "/v2/last/trade/{ticker}",
                "method": "GET",
                "description": "Get the most recent trade for a ticker symbol",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": [],
                "handler": self._handle_last_trade,
            },
            "quotes": {
                "path": "/v3/quotes/{ticker}",
                "method": "GET",
                "description": "Get quotes for a ticker symbol",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": [
                    "timestamp",
                    "timestamp_lt",
                    "timestamp_lte",
                    "timestamp_gt",
                    "timestamp_gte",
                    "limit",
                    "sort",
                    "order",
                ],
                "default_values": {
                    "limit": "100",
                    "sort": "timestamp",
                    "order": "desc",
                },
                "handler": self._handle_quotes,
            },
            "last_quote": {
                "path": "/v2/last/nbbo/{ticker}",
                "method": "GET",
                "description": "Get the most recent quote for a ticker symbol",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": [],
                "handler": self._handle_last_quote,
            },
            "universal_snapshot": {
                "path": "/v3/snapshot",
                "method": "GET",
                "description": "Get universal snapshots for multiple assets of a specific type",
                "category": "market_data",
                "required_params": ["type"],
                "optional_params": ["ticker_any_of", "order", "limit", "sort"],
                "default_values": {"limit": "50", "sort": "ticker", "order": "asc"},
                "handler": self._handle_universal_snapshot,
            },
            "snapshot_all": {
                "path": "/v2/snapshot/locale/us/markets/{market_type}/tickers",
                "method": "GET",
                "description": "Get a snapshot of all tickers in a market",
                "category": "market_data",
                "required_params": ["market_type"],
                "optional_params": ["tickers", "include_otc"],
                "default_values": {"include_otc": "false"},
                "handler": self._handle_snapshot_all,
            },
            "option_snapshot": {
                "path": "/v3/snapshot/options/{underlying_asset}/{option_contract}",
                "method": "GET",
                "description": "Get snapshot for a specific option contract",
                "category": "options",
                "required_params": ["underlying_asset", "option_contract"],
                "optional_params": [],
                "handler": self._handle_option_snapshot,
            },
            "exchanges": {
                "path": "/v3/reference/exchanges",
                "method": "GET",
                "description": "List exchanges known by Polygon.io",
                "category": "reference",
                "required_params": [],
                "optional_params": ["asset_class", "locale"],
                "handler": self._handle_exchanges,
            },
            "conditions": {
                "path": "/v3/reference/conditions",
                "method": "GET",
                "description": "List conditions used by Polygon.io",
                "category": "reference",
                "required_params": [],
                "optional_params": ["asset_class", "data_type", "id", "sip"],
                "handler": self._handle_conditions,
            },
            "stock_financials": {
                "path": "/v3/reference/financials",
                "method": "GET",
                "description": "Get fundamental financial data for companies",
                "category": "fundamentals",
                "required_params": [],
                "optional_params": [
                    "ticker",
                    "cik",
                    "company_name",
                    "company_name_search",
                    "sic",
                    "filing_date",
                    "filing_date_lt",
                    "filing_date_lte",
                    "filing_date_gt",
                    "filing_date_gte",
                    "period_of_report_date",
                    "period_of_report_date_lt",
                    "period_of_report_date_lte",
                    "period_of_report_date_gt",
                    "period_of_report_date_gte",
                    "timeframe",
                    "include_sources",
                    "limit",
                    "sort",
                    "order",
                ],
                "default_values": {
                    "limit": "100",
                    "sort": "filing_date",
                    "order": "desc",
                    "timeframe": "quarterly",
                },
                "handler": self._handle_stock_financials,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Polygon REST API."""
        self.register_tool(self.get_stock_price)
        self.register_tool(self.get_company_info)
        self.register_tool(self.get_historical_data)
        self.register_tool(self.get_market_status)
        self.register_tool(self.get_ticker_list)
        self.register_tool(self.get_trades)
        self.register_tool(self.get_quotes)
        self.register_tool(self.get_last_trade)
        self.register_tool(self.get_last_quote)
        self.register_tool(self.get_daily_open_close)
        self.register_tool(self.get_stock_financials)
        self.register_tool(self.get_market_movers)

    def _execute_endpoint_fetch(
        self, endpoint_config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the fetch operation based on endpoint configuration.

        Args:
            endpoint_config: Configuration for the endpoint
            params: Parameters for the request

        Returns:
            Fetched data

        Raises:
            Exception: If the fetch fails
        """
        # Merge provided params with default values
        merged_params = {**endpoint_config.get("default_values", {}), **params}

        # Get the handler function for this endpoint
        handler = endpoint_config.get("handler")
        if handler and callable(handler):
            return handler(merged_params)

        # Default handling if no specific handler
        path = endpoint_config.get("path")
        method = endpoint_config.get("method", "GET")

        # Format the path with path parameters
        try:
            formatted_path = path.format(**merged_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        # Extract query parameters (those not used in path formatting)
        query_params = {
            k: v for k, v in merged_params.items() if "{" + k + "}" not in path
        }

        # Add API key to query params
        query_params["apiKey"] = self.api_key

        # Make the request
        url = f"{self.base_url}{formatted_path}"

        # Add request tracking headers
        headers = {
            "User-Agent": f"FinGPT/NextGen-{os.environ.get('VERSION', '1.0.0')}",
            "X-Request-Source": "fingpt-mcp-rest",
        }

        # Execute request with retry logic
        retry_count = 0
        max_retries = 3
        retry_delay = 1.0  # seconds

        while retry_count <= max_retries:
            try:
                if method == "GET":
                    response = requests.get(
                        url, params=query_params, headers=headers, timeout=15
                    )
                elif method == "POST":
                    response = requests.post(
                        url, json=query_params, headers=headers, timeout=15
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_count += 1
                    if retry_count <= max_retries:
                        sleep_time = retry_delay * (
                            2 ** (retry_count - 1)
                        )  # Exponential backoff
                        self.logger.warning(
                            f"Rate limit exceeded, retrying in {sleep_time}s (attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue

                # Handle other error codes
                raise Exception(
                    f"API request failed: {response.status_code} - {response.text}"
                )
            except requests.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    sleep_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Request failed, retrying in {sleep_time}s (attempt {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    raise Exception(f"Request failed after {max_retries} retries: {e}")

    # Handler methods for specific endpoints

    def _handle_ticker_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ticker details endpoint."""
        return self._execute_polygon_request("/v3/reference/tickers/{ticker}", params)

    def _handle_daily_bars(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle daily bars endpoint."""
        # Ensure from/to dates are properly formatted
        for date_field in ["from", "to"]:
            if date_field in params and not isinstance(params[date_field], str):
                params[date_field] = params[date_field].strftime("%Y-%m-%d")

        path = "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}"
        return self._execute_polygon_request(path, params)

    def _handle_previous_close(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle previous close endpoint."""
        return self._execute_polygon_request("/v2/aggs/ticker/{ticker}/prev", params)

    def _handle_market_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market status endpoint."""
        return self._execute_polygon_request("/v1/marketstatus/now", params)

    def _handle_market_holidays(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market holidays endpoint."""
        return self._execute_polygon_request("/v1/marketstatus/upcoming", params)

    def _handle_stock_splits(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock splits endpoint."""
        return self._execute_polygon_request("/v3/reference/splits", params)

    def _handle_stock_dividends(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock dividends endpoint."""
        return self._execute_polygon_request("/v3/reference/dividends", params)

    def _handle_ticker_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ticker types endpoint."""
        return self._execute_polygon_request("/v3/reference/tickers/types", params)

    def _handle_ticker_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ticker list endpoint."""
        return self._execute_polygon_request("/v3/reference/tickers", params)

    def _handle_stock_snapshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock snapshot endpoint."""
        path = "/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        return self._execute_polygon_request(path, params)

    def _handle_market_movers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market movers endpoint."""
        direction = params.pop("direction", "gainers")
        if direction not in ["gainers", "losers"]:
            raise ValueError("Direction must be either 'gainers' or 'losers'")

        path = f"/v2/snapshot/locale/us/markets/stocks/{direction}"
        return self._execute_polygon_request(path, params)

    # New handler methods for additional endpoints

    def _handle_grouped_daily_bars(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle grouped daily bars endpoint."""
        # Ensure date is properly formatted
        if "date" in params and not isinstance(params["date"], str):
            params["date"] = params["date"].strftime("%Y-%m-%d")

        path = "/v2/aggs/grouped/locale/us/market/stocks/{date}"
        return self._execute_polygon_request(path, params)

    def _handle_daily_open_close(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle daily open close endpoint."""
        # Ensure date is properly formatted
        if "date" in params and not isinstance(params["date"], str):
            params["date"] = params["date"].strftime("%Y-%m-%d")

        path = "/v1/open-close/{ticker}/{date}"
        return self._execute_polygon_request(path, params)

    def _handle_trades(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trades endpoint."""
        # Format timestamp parameters if needed
        for ts_param in [
            "timestamp",
            "timestamp_lt",
            "timestamp_lte",
            "timestamp_gt",
            "timestamp_gte",
        ]:
            if ts_param in params and not isinstance(params[ts_param], str):
                if isinstance(params[ts_param], (int, float)):
                    # Already a timestamp, leave as is
                    pass
                else:
                    # Convert datetime to ISO format
                    params[ts_param] = params[ts_param].isoformat()

        path = "/v3/trades/{ticker}"
        return self._execute_polygon_request(path, params)

    def _handle_last_trade(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle last trade endpoint."""
        path = "/v2/last/trade/{ticker}"
        return self._execute_polygon_request(path, params)

    def _handle_quotes(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quotes endpoint."""
        # Format timestamp parameters if needed
        for ts_param in [
            "timestamp",
            "timestamp_lt",
            "timestamp_lte",
            "timestamp_gt",
            "timestamp_gte",
        ]:
            if ts_param in params and not isinstance(params[ts_param], str):
                if isinstance(params[ts_param], (int, float)):
                    # Already a timestamp, leave as is
                    pass
                else:
                    # Convert datetime to ISO format
                    params[ts_param] = params[ts_param].isoformat()

        path = "/v3/quotes/{ticker}"
        return self._execute_polygon_request(path, params)

    def _handle_last_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle last quote endpoint."""
        path = "/v2/last/nbbo/{ticker}"
        return self._execute_polygon_request(path, params)

    def _handle_universal_snapshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle universal snapshot endpoint."""
        # Convert ticker_any_of list to comma-separated string if needed
        if "ticker_any_of" in params and isinstance(params["ticker_any_of"], list):
            params["ticker_any_of"] = ",".join(params["ticker_any_of"])

        path = "/v3/snapshot"
        return self._execute_polygon_request(path, params)

    def _handle_snapshot_all(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle snapshot all endpoint."""
        # Convert tickers list to comma-separated string if needed
        if "tickers" in params and isinstance(params["tickers"], list):
            params["tickers"] = ",".join(params["tickers"])

        path = "/v2/snapshot/locale/us/markets/{market_type}/tickers"
        return self._execute_polygon_request(path, params)

    def _handle_option_snapshot(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle option snapshot endpoint."""
        path = "/v3/snapshot/options/{underlying_asset}/{option_contract}"
        return self._execute_polygon_request(path, params)

    def _handle_exchanges(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle exchanges endpoint."""
        path = "/v3/reference/exchanges"
        return self._execute_polygon_request(path, params)

    def _handle_conditions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conditions endpoint."""
        path = "/v3/reference/conditions"
        return self._execute_polygon_request(path, params)

    def _handle_stock_financials(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock financials endpoint."""
        # Format date parameters if needed
        for date_param in [
            "filing_date",
            "filing_date_lt",
            "filing_date_lte",
            "filing_date_gt",
            "filing_date_gte",
            "period_of_report_date",
            "period_of_report_date_lt",
            "period_of_report_date_lte",
            "period_of_report_date_gt",
            "period_of_report_date_gte",
        ]:
            if date_param in params and not isinstance(params[date_param], str):
                params[date_param] = params[date_param].strftime("%Y-%m-%d")

        path = "/v3/reference/financials"
        return self._execute_polygon_request(path, params)

    def _execute_polygon_request(
        self, path: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a request to Polygon.io API."""
        # Format path with parameters
        try:
            formatted_path = path.format(**params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        # Extract query parameters
        query_params = {k: v for k, v in params.items() if "{" + k + "}" not in path}

        # Add API key
        query_params["apiKey"] = self.api_key

        # Make request
        url = f"{self.base_url}{formatted_path}"

        # Add request tracking headers
        headers = {
            "User-Agent": f"FinGPT/NextGen-{os.environ.get('VERSION', '1.0.0')}",
            "X-Request-Source": "fingpt-mcp-rest",
        }

        # Execute request with retry logic
        retry_count = 0
        max_retries = 3
        retry_delay = 1.0  # seconds

        while retry_count <= max_retries:
            try:
                response = requests.get(
                    url, params=query_params, headers=headers, timeout=15
                )

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    retry_count += 1
                    if retry_count <= max_retries:
                        sleep_time = retry_delay * (
                            2 ** (retry_count - 1)
                        )  # Exponential backoff
                        self.logger.warning(
                            f"Rate limit exceeded, retrying in {sleep_time}s (attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue

                # Handle other error codes
                raise Exception(
                    f"API request failed: {response.status_code} - {response.text}"
                )
            except requests.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    sleep_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Request failed, retrying in {sleep_time}s (attempt {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    raise Exception(f"Request failed after {max_retries} retries: {e}")

    # Public API methods for models to use directly

    def get_stock_price(
        self, ticker: str, include_after_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Get current stock price and basic information.

        Args:
            ticker: Stock ticker symbol
            include_after_hours: Whether to include after-hours price

        Returns:
            Dictionary with stock price information
        """
        if include_after_hours:
            # Use snapshot which includes after-hours
            return self.fetch_data("stock_snapshot", {"ticker": ticker})
        else:
            # Use previous close
            return self.fetch_data("previous_close", {"ticker": ticker})

    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information and fundamentals.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with company information
        """
        return self.fetch_data("ticker_details", {"ticker": ticker})

    def get_historical_data(
        self, ticker: str, period: str = "1mo", interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get historical price data.

        Args:
            ticker: Stock ticker symbol
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd")
            interval: Data interval ("1d", "1h", "15m")

        Returns:
            Historical price data
        """
        # Convert period to from/to dates
        to_date = datetime.now()

        if period == "1d":
            from_date = to_date - timedelta(days=1)
        elif period == "5d":
            from_date = to_date - timedelta(days=5)
        elif period == "1mo":
            from_date = to_date - timedelta(days=30)
        elif period == "3mo":
            from_date = to_date - timedelta(days=90)
        elif period == "6mo":
            from_date = to_date - timedelta(days=180)
        elif period == "1y":
            from_date = to_date - timedelta(days=365)
        elif period == "2y":
            from_date = to_date - timedelta(days=2 * 365)
        elif period == "5y":
            from_date = to_date - timedelta(days=5 * 365)
        elif period == "ytd":
            from_date = datetime(to_date.year, 1, 1)
        else:
            raise ValueError(f"Invalid period: {period}")

        # Convert interval to multiplier/timespan
        if interval == "1d":
            multiplier = "1"
            timespan = "day"
        elif interval == "1h":
            multiplier = "1"
            timespan = "hour"
        elif interval == "15m":
            multiplier = "15"
            timespan = "minute"
        else:
            raise ValueError(f"Invalid interval: {interval}")

        return self.fetch_data(
            "daily_bars",
            {
                "ticker": ticker,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "multiplier": multiplier,
                "timespan": timespan,
            },
        )

    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status (open/closed).

        Returns:
            Market status information
        """
        return self.fetch_data("market_status", {})

    def get_ticker_list(
        self, market: str = "stocks", active: bool = True, limit: int = 100
    ) -> Dict[str, Any]:
        """
        Get a list of tickers.

        Args:
            market: Market type ("stocks", "crypto", "fx", "indices")
            active: Whether to only include active tickers
            limit: Maximum number of tickers to return

        Returns:
            List of tickers
        """
        return self.fetch_data(
            "ticker_list",
            {"market": market, "active": str(active).lower(), "limit": str(limit)},
        )

    # New public API methods for additional endpoints

    def get_trades(self, ticker: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get recent trades for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of trades to return

        Returns:
            Dictionary with trade data
        """
        return self.fetch_data("trades", {"ticker": ticker, "limit": str(limit)})

    def get_quotes(self, ticker: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get recent quotes for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of quotes to return

        Returns:
            Dictionary with quote data
        """
        return self.fetch_data("quotes", {"ticker": ticker, "limit": str(limit)})

    def get_last_trade(self, ticker: str) -> Dict[str, Any]:
        """
        Get the most recent trade for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with last trade data
        """
        return self.fetch_data("last_trade", {"ticker": ticker})

    def get_last_quote(self, ticker: str) -> Dict[str, Any]:
        """
        Get the most recent quote for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with last quote data
        """
        return self.fetch_data("last_quote", {"ticker": ticker})

    def get_daily_open_close(
        self, ticker: str, date: Union[str, datetime]
    ) -> Dict[str, Any]:
        """
        Get daily open, close, high, and low for a specific ticker and date.

        Args:
            ticker: Stock ticker symbol
            date: Date to get data for (YYYY-MM-DD format or datetime object)

        Returns:
            Dictionary with daily OHLC data
        """
        if isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        return self.fetch_data("daily_open_close", {"ticker": ticker, "date": date})

    def get_stock_financials(
        self, ticker: str = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get fundamental financial data for companies.

        Args:
            ticker: Optional stock ticker symbol
            limit: Maximum number of financial reports to return

        Returns:
            Dictionary with financial data
        """
        params = {"limit": str(limit)}
        if ticker:
            params["ticker"] = ticker

        return self.fetch_data("stock_financials", params)

    def get_market_movers(
        self, category: str = "gainers", limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get top market movers (gainers or losers) using the Polygon REST API.
        Args:
            category: "gainers" or "losers"
            limit: Number of movers to return
        Returns:
            Dictionary with market movers data (may include "results" key)
        """
        direction = category if category in ["gainers", "losers"] else "gainers"
        try:
            data = self.fetch_data(
                "market_movers", {"direction": direction, "limit": str(limit)}
            )
            return data
        except Exception as e:
            self.logger.error(f"Error getting market movers: {e}")
            if self.monitor:
                self.monitor.log_error(
                    f"Error getting market movers: {e}",
                    component="polygon_rest_mcp",
                    action="market_movers_error",
                    error=str(e),
                )
            return {}

    @property
    def api_key(self) -> str:
        """Get the API key from the client configuration."""
        if not self.rest_client:
            return ""
        return self.rest_client.get("api_key", "")

    @property
    def base_url(self) -> str:
        """Get the base URL from the client configuration."""
        if not self.rest_client:
            return "https://api.polygon.io"
        return self.rest_client.get("base_url", "https://api.polygon.io").rstrip("/")
