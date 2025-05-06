"""
Polygon.io WebSocket MCP Server

This module implements a Model Context Protocol (MCP) server for the Polygon.io
WebSocket API, providing access to real-time market data streams.
"""

import os
import json
import asyncio
import importlib
# Import time module for performance monitoring
import time
from uuid import uuid4
from typing import Dict, List, Any, Optional, Union

# Try to import required dependencies
try:
    import websockets
except ImportError:
    websockets = None

try:
    import dotenv
except ImportError:
    dotenv = None

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP

# Load environment variables
dotenv.load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')


class PolygonWsMCP(BaseDataMCP):
    """
    MCP server for Polygon.io WebSocket API.

    This server provides access to real-time market data streams through Polygon.io
    WebSocket API with dynamic channel selection and management of connections.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Polygon.io WebSocket MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - api_key: Polygon.io API key (overrides environment variable)
                - ws_url: WebSocket URL
                - max_connections: Maximum number of concurrent connections
                - ping_interval: How often to ping connections to keep them alive
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 300)
        """
        super().__init__(name="polygon_ws_mcp", config=config)

        # Initialize monitoring/logger
        self.logger = NetdataLogger(component_name="polygon-ws-mcp")
        self.logger.info("PolygonWsMCP initialized")

        # Initialize the WS client
        self.ws_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Active connections tracking
        self.active_connections = {}
        self.connection_tasks = {}

        # WebSocket connection settings
        self.max_connections = self.config.get("max_connections", 5)
        self.ping_interval = self.config.get("ping_interval", 30)

        # Data buffers for each subscription
        self.data_buffers = {}
        self.buffer_limit = self.config.get("buffer_limit", 1000)

        # Register specific tools
        self._register_specific_tools()

        self.logger.info(
            f"PolygonWsMCP initialized with {len(self.endpoints)} stream types"
        )
        # Removed self.monitor.log_info for endpoints

    def _initialize_client(self) -> Dict[str, Any]:
        """
        Initialize the Polygon WebSocket client configuration.

        Returns:
            Client configuration dictionary or None if initialization fails
        """
        try:
            # Prioritize environment variable for API key
            api_key = os.environ.get("POLYGON_API_KEY") or self.config.get("api_key")
            ws_url = self.config.get("ws_url", "wss://socket.polygon.io/stocks")

            if not api_key:
                self.logger.error("No Polygon API key provided - WebSocket connections will fail")
                return None

            self.logger.info(f"Loaded Polygon API key: {api_key[:4]}...{api_key[-4:]}")

            return {"api_key": api_key, "ws_url": ws_url}

        except Exception as e:
            self.logger.error(f"Failed to initialize Polygon WebSocket client: {e}")
            return None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available stream types for Polygon.io WebSocket API.

        Returns:
            Dictionary mapping stream types to their configurations
        """
        return {
            "trades": {
                "channel_prefix": "T.",
                "description": "Real-time trade data",
                "category": "market_data",
                "required_params": ["symbols"],
                "optional_params": ["buffer_size"],
                "default_values": {"buffer_size": "100"},
                "handler": self._handle_trades_stream,
            },
            "quotes": {
                "channel_prefix": "Q.",
                "description": "Real-time quote data (bid/ask updates)",
                "category": "market_data",
                "required_params": ["symbols"],
                "optional_params": ["buffer_size"],
                "default_values": {"buffer_size": "100"},
                "handler": self._handle_quotes_stream,
            },
            "minute_bars": {
                "channel_prefix": "AM.",
                "description": "Real-time aggregate minute bars",
                "category": "market_data",
                "required_params": ["symbols"],
                "optional_params": ["buffer_size"],
                "default_values": {"buffer_size": "100"},
                "handler": self._handle_minute_bars_stream,
            },
            "second_bars": {
                "channel_prefix": "A.",
                "description": "Real-time aggregate second bars",
                "category": "market_data",
                "required_params": ["symbols"],
                "optional_params": ["buffer_size"],
                "default_values": {"buffer_size": "100"},
                "handler": self._handle_second_bars_stream,
            },
            "status": {
                "channel_prefix": "status",
                "description": "Connection status messages",
                "category": "system",
                "required_params": [],
                "optional_params": ["buffer_size"],
                "default_values": {"buffer_size": "10"},
                "handler": self._handle_status_stream,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Polygon WebSocket API."""
        self.register_tool(self.subscribe_to_stream)
        self.register_tool(self.unsubscribe_from_stream)
        self.register_tool(self.get_latest_trades)
        self.register_tool(self.get_latest_quotes)
        self.register_tool(self.get_active_subscriptions)
        self.register_tool(self.get_latest_minute_bars)
        self.register_tool(self.get_latest_second_bars)
        self.register_tool(self.get_connection_status)

    def _execute_endpoint_fetch(
        self, endpoint_config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the WebSocket stream operation based on endpoint configuration.

        For WebSocket, this doesn't immediately fetch data but rather sets up
        a subscription that will receive data over time.

        Args:
            endpoint_config: Configuration for the stream
            params: Parameters for the subscription

        Returns:
            Status information about the subscription

        Raises:
            Exception: If the subscription setup fails
        """
        # Get the handler function for this endpoint
        handler = endpoint_config.get("handler")
        if handler and callable(handler):
            return handler(params)

        # Default handling if no specific handler
        stream_type = endpoint_config.get("stream_type")
        if not stream_type:
            raise ValueError("No stream_type specified for endpoint")

        # Check if symbols are required and provided
        if (
            "symbols" in endpoint_config.get("required_params", [])
            and "symbols" not in params
        ):
            raise ValueError("Required parameter 'symbols' not provided")

        # Generate a subscription ID
        subscription_id = f"{stream_type}_{uuid4().hex[:8]}"

        # Create a new subscription
        return self._create_subscription(subscription_id, stream_type, params)

    async def _connect_and_subscribe(
        self, subscription_id: str, stream_type: str, symbols: List[str]
    ):
        """
        Establish WebSocket connection and subscribe to channels.

        Args:
            subscription_id: Unique ID for this subscription
            stream_type: Type of data stream
            symbols: List of symbols to subscribe to
        """
        try:
            # Connect to WebSocket
            ws = await websockets.connect(
                self.ws_url,
                ping_interval=self.ping_interval,
                ping_timeout=10,
                close_timeout=5,
            )

            # Keep track of this connection
            self.active_connections[subscription_id] = ws
            self.logger.gauge("active_connections", len(self.active_connections))

            # Authenticate
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            auth_response = await ws.recv()

            # Parse authentication response
            auth_data = json.loads(auth_response)
            if not self._is_auth_successful(auth_data):
                self.logger.error(f"Authentication failed: {auth_response}")
                await ws.close()
                del self.active_connections[subscription_id]
                return

            # Subscribe to channels
            endpoint_config = self.endpoints.get(stream_type)
            if not endpoint_config:
                self.logger.error(f"Invalid stream type: {stream_type}")
                await ws.close()
                del self.active_connections[subscription_id]
                return

            prefix = endpoint_config.get("channel_prefix", "")
            channels = [f"{prefix}{symbol}" for symbol in symbols]

            for channel in channels:
                await ws.send(json.dumps({"action": "subscribe", "params": channel}))

            # Start message handler
            self.data_buffers[subscription_id] = []
            asyncio.create_task(self._handle_messages(ws, subscription_id, stream_type))

            self.logger.info(f"Subscribed to {len(channels)} {stream_type} channels")
            self.logger.counter("ws_subscribe_count", 1)

        except Exception as e:
            self.logger.error(f"Error in WebSocket connection: {e}")
            if subscription_id in self.active_connections:
                await self.active_connections[subscription_id].close()
                del self.active_connections[subscription_id]

    async def _handle_messages(self, ws, subscription_id: str, stream_type: str):
        """
        Process messages from a WebSocket connection.

        Args:
            ws: WebSocket connection
            subscription_id: Unique ID for this subscription
            stream_type: Type of data stream
        """
        buffer = self.data_buffers.get(subscription_id, [])
        buffer_limit = int(self.config.get("buffer_limit", 1000))

        try:
            while True:
                try:
                    # Wait for a message with timeout
                    message = await asyncio.wait_for(ws.recv(), timeout=60)
                    self.logger.counter("ws_message_receive_count", 1)
                    self.logger.gauge("ws_message_size_bytes", len(message.encode("utf-8")))

                    # Start timing the message processing
                    time_module = time  # Local reference to time module
                    process_start_time = time_module.time()
                    
                    # Parse and process the message
                    data = json.loads(message)

                    # Process message based on type
                    if isinstance(data, list):
                        for item in data:
                            # Add to buffer, respecting limit
                            buffer.append(item)
                            if len(buffer) > buffer_limit:
                                buffer.pop(0)
                    else:
                        # Add to buffer, respecting limit
                        buffer.append(data)
                        if len(buffer) > buffer_limit:
                            buffer.pop(0)
                            
                    # Measure and record message processing time
                    processing_time = time_module.time() - process_start_time
                    self.logger.timing("message_processing_time_ms", processing_time * 1000)

                except asyncio.TimeoutError:
                    #
                    # No message in timeout period, send ping to check
                    # connection
                    self.logger.debug("No message received in 60 seconds, sending ping")
                    pong = await ws.ping()
                    await asyncio.wait_for(pong, timeout=10)
                    self.logger.debug("Received pong response, connection still active")

                except websockets.ConnectionClosed as e:
                    self.logger.warning(f"WebSocket connection closed: {e}")
                    break

        except Exception as e:
            self.logger.error(f"Error in message handling: {e}")
            self.logger.counter("ws_error_count", 1)
        finally:
            # Clean up connection
            if subscription_id in self.active_connections:
                await ws.close()
                del self.active_connections[subscription_id]
                self.logger.gauge("active_connections", len(self.active_connections))
                self.logger.info(f"Closed WebSocket connection for {subscription_id}")

    def _is_auth_successful(
        self, auth_data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> bool:
        """
        Check if authentication was successful.

        Args:
            auth_data: Authentication response data

        Returns:
            Whether authentication was successful
        """
        if isinstance(auth_data, list):
            for msg in auth_data:
                if msg.get("ev") == "status" and msg.get("status") == "auth_success":
                    return True
        elif isinstance(auth_data, dict):
            if (
                auth_data.get("ev") == "status"
                and auth_data.get("status") == "auth_success"
            ):
                return True
        return False

    def _create_subscription(
        self, subscription_id: str, stream_type: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a new subscription to a data stream.

        Args:
            subscription_id: Unique ID for this subscription
            stream_type: Type of data stream
            params: Parameters for the subscription

        Returns:
            Status information about the subscription
        """
        # Extract symbols
        symbols = params.get("symbols", [])
        if isinstance(symbols, str):
            symbols = [sym.strip() for sym in symbols.split(",")]

        # Check connection limit
        if len(self.active_connections) >= self.max_connections:
            self.logger.warning(
                f"Maximum WebSocket connections reached ({self.max_connections})"
            )
            return {"error": f"Maximum connections reached ({self.max_connections})"}

        # Start connection in background
        loop = asyncio.get_event_loop()
        task = loop.create_task(
            self._connect_and_subscribe(subscription_id, stream_type, symbols)
        )
        self.connection_tasks[subscription_id] = task

        # Return subscription information
        return {
            "subscription_id": subscription_id,
            "stream_type": stream_type,
            "symbols": symbols,
            "status": "connecting",
            "message": f"Establishing WebSocket connection for {stream_type} data",
        }

    # Handler methods for specific stream types

    def _handle_trades_stream(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle trades stream subscription."""
        # Generate subscription ID
        subscription_id = f"trades_{uuid4().hex[:8]}"
        return self._create_subscription(subscription_id, "trades", params)

    def _handle_quotes_stream(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quotes stream subscription."""
        subscription_id = f"quotes_{uuid4().hex[:8]}"
        return self._create_subscription(subscription_id, "quotes", params)

    def _handle_minute_bars_stream(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle minute bars stream subscription."""
        subscription_id = f"minute_bars_{uuid4().hex[:8]}"
        return self._create_subscription(subscription_id, "minute_bars", params)

    def _handle_second_bars_stream(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle second bars stream subscription."""
        subscription_id = f"second_bars_{uuid4().hex[:8]}"
        return self._create_subscription(subscription_id, "second_bars", params)

    def _handle_status_stream(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle status stream subscription."""
        subscription_id = f"status_{uuid4().hex[:8]}"
        return self._create_subscription(subscription_id, "status", params)

    # Public API methods for models to use directly

    def subscribe_to_stream(
        self, stream_type: str, symbols: Union[str, List[str]], buffer_size: int = 100
    ) -> Dict[str, Any]:
        """
        Subscribe to a real-time data stream.

        Args:
            stream_type: Type of stream ("trades", "quotes", "minute_bars", "second_bars")
            symbols: Symbol or list of symbols to subscribe to
            buffer_size: Size of the data buffer

        Returns:
            Subscription information
        """
        # Format symbols parameter
        if isinstance(symbols, str):
            # Convert comma-separated string to list
            if "," in symbols:
                symbols = [sym.strip() for sym in symbols.split(",")]
            else:
                symbols = [symbols]

        # Check stream type
        if stream_type not in self.endpoints:
            self.logger.error(f"Invalid stream type: {stream_type}")
            return {
                "error": f"Invalid stream type: {stream_type}",
                "available_stream_types": list(self.endpoints.keys()),
            }

        # Make subscription
        return self.fetch_data(
            stream_type, {"symbols": symbols, "buffer_size": buffer_size}
        )

    def unsubscribe_from_stream(self, subscription_id: str) -> Dict[str, Any]:
        """
        Unsubscribe from a data stream.

        Args:
            subscription_id: ID of the subscription to cancel

        Returns:
            Status information
        """
        if subscription_id not in self.active_connections:
            return {"error": f"Subscription not found: {subscription_id}"}

        # Schedule connection closure
        loop = asyncio.get_event_loop()
        loop.create_task(self._close_subscription(subscription_id))

        return {"status": "closing", "subscription_id": subscription_id}

    async def _close_subscription(self, subscription_id: str):
        """Close a WebSocket subscription."""
        if subscription_id in self.active_connections:
            ws = self.active_connections[subscription_id]
            await ws.close()
            del self.active_connections[subscription_id]

            # Clean up related data
            if subscription_id in self.data_buffers:
                del self.data_buffers[subscription_id]
            if subscription_id in self.connection_tasks:
                del self.connection_tasks[subscription_id]

            self.logger.info(f"Closed subscription: {subscription_id}")

    def get_latest_trades(
        self, subscription_id: str = None, symbol: str = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get latest trades from a subscription.

        Args:
            subscription_id: Specific subscription to get data from,
                          if not provided, will search for any trade subscription
            symbol: Filter by symbol
            limit: Maximum number of trades to return

        Returns:
            Latest trade data
        """
        # Find matching subscription
        if subscription_id is None and symbol is not None:
            # Find subscription for this symbol
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("trades_"):
                    subscription_id = sub_id
                    break

        if subscription_id is None:
            # Find any trades subscription
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("trades_"):
                    subscription_id = sub_id
                    break

        if subscription_id not in self.data_buffers:
            return {"error": "No active trades subscription found"}

        # Get trade data from buffer
        buffer = self.data_buffers[subscription_id]

        # Filter by symbol if requested
        if symbol is not None:
            trades = [t for t in buffer if t.get("sym") == symbol]
        else:
            trades = buffer

        # Return limited number of trades
        return {"trades": trades[-limit:], "subscription_id": subscription_id}

    def get_latest_quotes(
        self, subscription_id: str = None, symbol: str = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get latest quotes from a subscription.

        Args:
            subscription_id: Specific subscription to get data from,
                          if not provided, will search for any quote subscription
            symbol: Filter by symbol
            limit: Maximum number of quotes to return

        Returns:
            Latest quote data
        """
        # Find matching subscription
        if subscription_id is None and symbol is not None:
            # Find subscription for this symbol
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("quotes_"):
                    subscription_id = sub_id
                    break

        if subscription_id is None:
            # Find any quotes subscription
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("quotes_"):
                    subscription_id = sub_id
                    break

        if subscription_id not in self.data_buffers:
            return {"error": "No active quotes subscription found"}

        # Get quote data from buffer
        buffer = self.data_buffers[subscription_id]

        # Filter by symbol if requested
        if symbol is not None:
            quotes = [q for q in buffer if q.get("sym") == symbol]
        else:
            quotes = buffer

        # Return limited number of quotes
        return {"quotes": quotes[-limit:], "subscription_id": subscription_id}

    def get_active_subscriptions(self) -> Dict[str, Any]:
        """
        Get information about active subscriptions.

        Returns:
            Dictionary with active subscription information
        """
        subscriptions = []

        for sub_id, ws in self.active_connections.items():
            # Determine stream type from subscription ID
            stream_type = "unknown"
            if sub_id.startswith("trades_"):
                   stream_type = "trades"
            elif sub_id.startswith("quotes_"):
                stream_type = "quotes"
            elif sub_id.startswith("minute_bars_"):
                stream_type = "minute_bars"
            elif sub_id.startswith("second_bars_"):
                stream_type = "second_bars"
            elif sub_id.startswith("status_"):
                stream_type = "status"

            # Get buffer size
            buffer_size = len(self.data_buffers.get(sub_id, []))

            subscriptions.append(
                {
                    "subscription_id": sub_id,
                    "stream_type": stream_type,
                    "buffer_size": buffer_size,
                    "status": "active",
                }
            )

        return {"subscriptions": subscriptions, "count": len(subscriptions)}

    def get_latest_minute_bars(
        self, subscription_id: str = None, symbol: str = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get latest minute bars from a subscription.

        Args:
            subscription_id: Specific subscription to get data from,
                          if not provided, will search for any minute bars subscription
            symbol: Filter by symbol
            limit: Maximum number of bars to return

        Returns:
            Latest minute bars data
        """
        # Find matching subscription
        if subscription_id is None and symbol is not None:
            # Find subscription for this symbol
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("minute_bars_"):
                    subscription_id = sub_id
                    break

        if subscription_id is None:
            # Find any minute bars subscription
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("minute_bars_"):
                    subscription_id = sub_id
                    break

        if subscription_id not in self.data_buffers:
            return {"error": "No active minute bars subscription found"}

        # Get minute bars data from buffer
        buffer = self.data_buffers[subscription_id]

        # Filter by symbol if requested
        if symbol is not None:
            bars = [b for b in buffer if b.get("sym") == symbol]
        else:
            bars = buffer

        # Return limited number of bars
        return {"minute_bars": bars[-limit:], "subscription_id": subscription_id}

    def get_latest_second_bars(
        self, subscription_id: str = None, symbol: str = None, limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get latest second bars from a subscription.

        Args:
            subscription_id: Specific subscription to get data from,
                          if not provided, will search for any second bars subscription
            symbol: Filter by symbol
            limit: Maximum number of bars to return

        Returns:
            Latest second bars data
        """
        # Find matching subscription
        if subscription_id is None and symbol is not None:
            # Find subscription for this symbol
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("second_bars_"):
                    subscription_id = sub_id
                    break

        if subscription_id is None:
            # Find any second bars subscription
            for sub_id, ws in self.active_connections.items():
                if sub_id.startswith("second_bars_"):
                    subscription_id = sub_id
                    break

        if subscription_id not in self.data_buffers:
            return {"error": "No active second bars subscription found"}

        # Get second bars data from buffer
        buffer = self.data_buffers[subscription_id]

        # Filter by symbol if requested
        if symbol is not None:
            bars = [b for b in buffer if b.get("sym") == symbol]
        else:
            bars = buffer

        # Return limited number of bars
        return {"second_bars": bars[-limit:], "subscription_id": subscription_id}

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get status of all WebSocket connections.

        Returns:
            Dictionary with connection status information
        """
        connections = []

        for sub_id, ws in self.active_connections.items():
            # Determine stream type from subscription ID
            stream_type = "unknown"
            if sub_id.startswith("trades_"):
                stream_type = "trades"
            elif sub_id.startswith("quotes_"):
                stream_type = "quotes"
            elif sub_id.startswith("minute_bars_"):
                stream_type = "minute_bars"
            elif sub_id.startswith("second_bars_"):
                stream_type = "second_bars"
            elif sub_id.startswith("status_"):
                stream_type = "status"

            # Get connection state
            state = "connected"
            if ws.closed:
                state = "closed"

            connections.append(
                {
                    "subscription_id": sub_id,
                    "stream_type": stream_type,
                    "state": state,
                    "buffer_size": len(self.data_buffers.get(sub_id, [])),
                }
            )

        return {
            "connections": connections,
            "count": len(connections),
            "max_connections": self.max_connections,
        }

    @property
    def api_key(self) -> str:
        """Get the API key from the client configuration."""
        if not self.ws_client:
            return ""
        return self.ws_client.get("api_key", "")

    @property
    def ws_url(self) -> str:
        """Get the WebSocket URL from the client configuration."""
        if not self.ws_client:
            return "wss://socket.polygon.io/stocks"
        return self.ws_client.get("ws_url", "wss://socket.polygon.io/stocks")
