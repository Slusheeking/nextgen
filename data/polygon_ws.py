"""
Polygon.io WebSocket API integration for real-time financial market data.

A production-ready client for accessing Polygon.io WebSocket API to stream
real-time market data for algorithmic trading systems.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union, Callable

import websockets
from dotenv import load_dotenv
from loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

# Load environment variables
load_dotenv()

# Setup traditional logging as fallback
logger = logging.getLogger(__name__)

class PolygonWebSocketClient:
    """
    Production client for the Polygon.io WebSocket API.
    """

    def __init__(self, api_key: Optional[str] = None, ws_url: str = "wss://socket.polygon.io/stocks"):
        """
        Initialize the Polygon.io WebSocket API client.
        
        Args:
            api_key: API key for Polygon.io (defaults to POLYGON_API_KEY environment variable)
            ws_url: WebSocket URL for Polygon.io API (defaults to wss://socket.polygon.io/stocks)
        """
        # Initialize observability tools
        self.loki = LokiManager(service_name="data-polygon-ws")
        self.prom = PrometheusManager(service_name="data-polygon-ws")
        
        # Create metrics
        self.connections_gauge = self.prom.create_gauge(
            "polygon_ws_active_connections",
            "Number of active Polygon.io WebSocket connections",
            ["channel_type"]
        )
        
        self.message_counter = self.prom.create_counter(
            "polygon_ws_messages_total", 
            "Total count of Polygon.io WebSocket messages received",
            ["channel_type", "event_type", "symbol"]
        )
        
        self.error_counter = self.prom.create_counter(
            "polygon_ws_errors_total", 
            "Total count of Polygon.io WebSocket errors",
            ["error_type"]
        )
        
        self.message_latency = self.prom.create_histogram(
            "polygon_ws_message_processing_seconds",
            "Polygon.io WebSocket message processing time in seconds",
            ["event_type"],
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        )
        
        # Initialize API client
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            log_msg = "No Polygon API key provided - WebSocket connections will fail"
            logger.warning(log_msg)
            self.loki.warning(log_msg, component="polygon_ws")
            
        self.ws_url = ws_url
        
        # Active connections to keep track of
        self.active_connections = set()
        
        # Define the optimal channels for different data retrieval purposes
        self.optimal_channels = {
            "trades": {
                "channel": "T.{symbol}",
                "method": "SUBSCRIBE",
                "description": "Real-time trades"
            },
            "quotes": {
                "channel": "Q.{symbol}",
                "method": "SUBSCRIBE",
                "description": "Real-time quotes"
            },
            "minute_bars": {
                "channel": "AM.{symbol}",
                "method": "SUBSCRIBE",
                "description": "Real-time aggregate (minute bars)"
            },
            "nbbo_quotes": {
                "channel": "C.{symbol}",
                "method": "SUBSCRIBE", 
                "description": "Real-time NBBO quote"
            }
        }
        
        # Active connection tracking data
        self.connection_start_times = {}
        
        logger.info("PolygonWebSocketClient initialized")
        self.loki.info("PolygonWebSocketClient initialized", component="polygon_ws")

    async def connect(self, 
                     channel_type: str, 
                     symbols: List[str], 
                     message_handler: Optional[Callable[[Dict[str, Any]], None]] = None,
                     error_handler: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Connect to a Polygon WebSocket stream for specific symbols and channel type.
        
        Args:
            channel_type: The type of channel to connect to ("trades", "quotes", "minute_bars", "nbbo_quotes")
            symbols: List of stock symbols to subscribe to
            message_handler: Optional callback function to process received messages
            error_handler: Optional callback function to handle errors
            
        Returns:
            Status information about the connection
        """
        if channel_type not in self.optimal_channels:
            logger.error(f"Unknown channel type: {channel_type}")
            if error_handler:
                error_handler(f"Unknown channel type: {channel_type}")
            return {"error": f"Unknown channel type: {channel_type}"}
            
        channel_template = self.optimal_channels[channel_type]["channel"]
        
        # Default message handler if none provided
        if message_handler is None:
            message_handler = self._default_message_handler
            
        # Default error handler if none provided
        if error_handler is None:
            error_handler = self._default_error_handler
        
        # Create subscription messages for each symbol
        subscriptions = []
        for symbol in symbols:
            channel = channel_template.format(symbol=symbol)
            subscriptions.append(channel)
            
        logger.info(f"Connecting to {self.ws_url} and subscribing to {len(subscriptions)} channels")
        self.loki.info(f"Connecting to {self.ws_url} and subscribing to {len(subscriptions)} channels", 
                     component="polygon_ws", 
                     channel_type=channel_type,
                     symbol_count=len(symbols))
        
        try:
            # Connect to WebSocket with heartbeat management
            ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,  # Send ping every 30 seconds
                ping_timeout=10,   # Wait 10 seconds for pong response
                close_timeout=5    # Wait 5 seconds for graceful close
            )
            
            # Store active connection for potential cleanup
            self.active_connections.add(ws)
            
            # Store connection start time
            self.connection_start_times[ws] = time.time()
            
            # Update active connections gauge
            self.prom.set_gauge(
                "polygon_ws_active_connections",
                len(self.active_connections),
                channel_type=channel_type
            )
            
            # Authenticate
            auth_start_time = time.time()
            await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            auth_response = await ws.recv()
            auth_time = time.time() - auth_start_time
            
            logger.info(f"Authentication response: {auth_response}")
            self.loki.info(f"Authentication response received in {auth_time:.4f}s", 
                         component="polygon_ws", 
                         auth_time=f"{auth_time:.4f}")
            
            auth_data = json.loads(auth_response)
            if not self._is_successful_auth(auth_data):
                error_msg = f"Authentication failed: {auth_response}"
                logger.error(error_msg)
                self.loki.error(error_msg, 
                              component="polygon_ws", 
                              error_type="auth_failure")
                
                # Increment error counter
                self.prom.increment_counter(
                    "polygon_ws_errors_total",
                    1,
                    error_type="auth_failure"
                )
                
                if error_handler:
                    error_handler(error_msg)
                    
                await ws.close()
                self.active_connections.remove(ws)
                if ws in self.connection_start_times:
                    del self.connection_start_times[ws]
                    
                # Update active connections gauge
                self.prom.set_gauge(
                    "polygon_ws_active_connections",
                    len(self.active_connections),
                    channel_type=channel_type
                )
                
                return {"error": error_msg}
            
            # Subscribe to channels
            for channel in subscriptions:
                await ws.send(json.dumps({"action": "subscribe", "params": channel}))
            logger.info(f"Subscribed to {len(subscriptions)} channels")
            self.loki.info(f"Subscribed to {len(subscriptions)} channels", 
                         component="polygon_ws", 
                         channel_type=channel_type,
                         channels=subscriptions)
            
            # Create task to process messages
            asyncio.create_task(
                self._process_messages(ws, message_handler, error_handler)
            )
            
            success_result = {
                "status": "connected",
                "channel_type": channel_type,
                "symbols": symbols,
                "subscriptions": subscriptions
            }
            
            self.loki.info("WebSocket connection established successfully", 
                         component="polygon_ws", 
                         channel_type=channel_type,
                         symbol_count=len(symbols))
            
            return success_result
            
        except Exception as e:
            error_msg = f"Error connecting to WebSocket: {str(e)}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="polygon_ws", 
                          error_type="connection_failure")
            
            # Increment error counter
            self.prom.increment_counter(
                "polygon_ws_errors_total",
                1,
                error_type="connection_failure"
            )
            
            if error_handler:
                error_handler(error_msg)
                
            return {"error": error_msg}
    
    def _is_successful_auth(self, auth_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> bool:
        """Check if authentication was successful based on response."""
        if isinstance(auth_data, list):
            for msg in auth_data:
                if msg.get("ev") == "status" and msg.get("status") == "auth_success":
                    return True
        elif isinstance(auth_data, dict):
            if auth_data.get("ev") == "status" and auth_data.get("status") == "auth_success":
                return True
        return False
    
    async def _process_messages(self, 
                               ws: websockets.WebSocketClientProtocol, 
                               message_handler: Callable[[Dict[str, Any]], None],
                               error_handler: Callable[[str], None]) -> None:
        """
        Process incoming messages from WebSocket connection.
        
        Args:
            ws: WebSocket connection
            message_handler: Callback function to process messages
            error_handler: Callback function to handle errors
        """
        try:
            while True:
                try:
                    # Set timeout to detect connection issues
                    message = await asyncio.wait_for(ws.recv(), timeout=60)
                    
                    # Start timing message processing
                    process_start = time.time()
                    
                    # Parse and process the message
                    data = json.loads(message)
                    
                    # Process message based on type
                    if isinstance(data, list):
                        for item in data:
                            # Track the message
                            event_type = item.get("ev", "unknown")
                            symbol = item.get("sym", "unknown")
                            
                            # Record message receipt in Prometheus
                            self.prom.increment_counter(
                                "polygon_ws_messages_total",
                                1,
                                channel_type=self._get_channel_type_from_event(event_type),
                                event_type=event_type,
                                symbol=symbol
                            )
                            
                            # Process the message
                            message_handler(item)
                    else:
                        # Track the message
                        event_type = data.get("ev", "unknown")
                        symbol = data.get("sym", "unknown")
                        
                        # Record message receipt in Prometheus
                        self.prom.increment_counter(
                            "polygon_ws_messages_total",
                            1,
                            channel_type=self._get_channel_type_from_event(event_type),
                            event_type=event_type,
                            symbol=symbol
                        )
                        
                        # Process the message
                        message_handler(data)
                    
                    # Record message processing time
                    processing_time = time.time() - process_start
                    self.prom.observe_histogram(
                        "polygon_ws_message_processing_seconds",
                        processing_time,
                        event_type=event_type
                    )
                        
                except asyncio.TimeoutError:
                    # No message in timeout period, send ping to check connection
                    logger.debug("No message received in 60 seconds, sending ping")
                    self.loki.debug("No message received in 60 seconds, sending ping", 
                                  component="polygon_ws")
                    pong = await ws.ping()
                    await asyncio.wait_for(pong, timeout=10)
                    logger.debug("Received pong response, connection still active")
                    self.loki.debug("Received pong response, connection still active", 
                                  component="polygon_ws")
                except websockets.ConnectionClosed as e:
                    logger.warning(f"WebSocket connection closed: {e}")
                    self.loki.warning(f"WebSocket connection closed: {e}", 
                                    component="polygon_ws", 
                                    close_code=str(e.code),
                                    close_reason=e.reason)
                    
                    # Record the error
                    self.prom.increment_counter(
                        "polygon_ws_errors_total",
                        1,
                        error_type="connection_closed"
                    )
                    
                    if error_handler:
                        error_handler(f"WebSocket connection closed: {e}")
                    break
                
        except Exception as e:
            error_msg = f"Error in message processing: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="polygon_ws", 
                          error_type="message_processing")
            
            # Record the error
            self.prom.increment_counter(
                "polygon_ws_errors_total",
                1,
                error_type="message_processing"
            )
            
            if error_handler:
                error_handler(error_msg)
        finally:
            # Clean up connection if still active
            if ws in self.active_connections:
                # Calculate connection duration for metrics
                connection_duration = 0
                if ws in self.connection_start_times:
                    connection_duration = time.time() - self.connection_start_times[ws]
                    del self.connection_start_times[ws]
                
                self.active_connections.remove(ws)
                await ws.close()
                
                # Update active connections gauge
                channel_type = "unknown"  # As we're in a generic handler
                self.prom.set_gauge(
                    "polygon_ws_active_connections",
                    len(self.active_connections),
                    channel_type=channel_type
                )
                
                logger.info(f"WebSocket connection closed after {connection_duration:.1f} seconds")
                self.loki.info(f"WebSocket connection closed after {connection_duration:.1f} seconds", 
                             component="polygon_ws",
                             connection_duration=f"{connection_duration:.1f}")
    
    def _get_channel_type_from_event(self, event_type: str) -> str:
        """Map event type to channel type for metrics."""
        if event_type == "T":
            return "trades"
        elif event_type == "Q":
            return "quotes"
        elif event_type == "AM":
            return "minute_bars"
        elif event_type == "C":
            return "nbbo_quotes"
        else:
            return "unknown"
    
    def _default_message_handler(self, message: Dict[str, Any]) -> None:
        """Default handler for WebSocket messages if none provided."""
        event_type = message.get("ev", "unknown")
        symbol = message.get("sym", "unknown")
        
        # Print summary log
        logger.info(f"Received {event_type} event for {symbol}")
        
        # Print detailed info based on event type
        if event_type == "T":  # Trade
            trade_info = f"Trade: {symbol} price={message.get('p')} size={message.get('s')}"
            logger.info(trade_info)
            self.loki.info(trade_info, 
                         component="polygon_ws", 
                         event_type="trade",
                         symbol=symbol,
                         price=message.get('p'),
                         size=message.get('s'))
            
        elif event_type == "Q":  # Quote
            quote_info = f"Quote: {symbol} bid={message.get('bp')}x{message.get('bs')} ask={message.get('ap')}x{message.get('as')}"
            logger.info(quote_info)
            self.loki.info(quote_info, 
                         component="polygon_ws", 
                         event_type="quote",
                         symbol=symbol,
                         bid_price=message.get('bp'),
                         bid_size=message.get('bs'),
                         ask_price=message.get('ap'),
                         ask_size=message.get('as'))
            
        elif event_type == "AM":  # Minute aggregate/bar
            bar_info = f"Minute Bar: {symbol} open={message.get('o')} high={message.get('h')} low={message.get('l')} close={message.get('c')} volume={message.get('v')}"
            logger.info(bar_info)
            self.loki.info(bar_info, 
                         component="polygon_ws", 
                         event_type="minute_bar",
                         symbol=symbol,
                         open=message.get('o'),
                         high=message.get('h'),
                         low=message.get('l'),
                         close=message.get('c'),
                         volume=message.get('v'))
    
    def _default_error_handler(self, error_message: str) -> None:
        """Default handler for WebSocket errors if none provided."""
        logger.error(f"WebSocket error: {error_message}")
        self.loki.error(f"WebSocket error: {error_message}", 
                      component="polygon_ws", 
                      error_type="websocket_error")
        
        # Increment error counter
        self.prom.increment_counter(
            "polygon_ws_errors_total",
            1,
            error_type="websocket_error"
        )
            
    async def close_all_connections(self) -> None:
        """Close all active WebSocket connections."""
        connection_count = len(self.active_connections)
        logger.info(f"Closing {connection_count} active WebSocket connections")
        self.loki.info(f"Closing {connection_count} active WebSocket connections", 
                     component="polygon_ws", 
                     connection_count=connection_count)
        
        for ws in list(self.active_connections):
            try:
                # Calculate connection duration for metrics
                connection_duration = 0
                if ws in self.connection_start_times:
                    connection_duration = time.time() - self.connection_start_times[ws]
                    del self.connection_start_times[ws]
                
                await ws.close()
                self.active_connections.remove(ws)
                
                # Record connection duration
                self.loki.info(f"WebSocket connection closed after {connection_duration:.1f} seconds", 
                             component="polygon_ws",
                             connection_duration=f"{connection_duration:.1f}")
                
            except Exception as e:
                error_msg = f"Error closing WebSocket connection: {e}"
                logger.error(error_msg)
                self.loki.error(error_msg, 
                              component="polygon_ws", 
                              error_type="close_failure")
                
                # Increment error counter
                self.prom.increment_counter(
                    "polygon_ws_errors_total",
                    1,
                    error_type="close_failure"
                )
        
        # Update active connections gauge for all channel types
        for channel_type in self.optimal_channels.keys():
            self.prom.set_gauge(
                "polygon_ws_active_connections",
                0,  # We've closed all connections
                channel_type=channel_type
            )
