"""
Redis MCP Server

This module implements a Model Context Protocol (MCP) server for Redis,
providing access to Redis for state management, caching, and pub/sub messaging.
It connects to the official Redis server with integrated monitoring.
"""

import os
import time
import json
import redis
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from mcp_tools.base_mcp_server import BaseMCPServer
from monitoring import setup_monitoring
from monitoring.system_monitor import MonitoringManager


class RedisMCP(BaseMCPServer):
    """
    MCP server for Redis.

    This server provides access to Redis for state management, caching,
    and pub/sub messaging in the trading system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Redis MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - host: Redis host (default: localhost)
                - port: Redis port (default: 6379)
                - db: Redis database (default: 0)
                - password: Redis password (default: None)
                - decode_responses: Decode responses (default: True)
        """
        # Load environment variables
        load_dotenv()

        super().__init__(name="redis_mcp", config=config)

        # Initialize monitoring
        self.monitor, self.metrics = self._setup_monitoring()

        # Initialize Redis client
        self.redis_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        self.logger.info("RedisMCP initialized with integrated Redis server")
        if self.monitor:
            self.monitor.log_info(
                "RedisMCP initialized", component="redis_mcp", action="initialization"
            )

    def _initialize_client(self) -> redis.Redis:
        """
        Initialize the Redis client to connect to the official Redis server.

        Returns:
            Redis client
        """
        try:
            # Get Redis configuration from environment variables or config
            host = self.config.get("host", os.getenv("REDIS_HOST", "localhost"))
            port = int(self.config.get("port", os.getenv("REDIS_PORT", "6379")))
            db = int(self.config.get("db", os.getenv("REDIS_DB", "0")))
            password = self.config.get("password", os.getenv("REDIS_PASSWORD", None))
            decode_responses = self.config.get("decode_responses", True)

            # Create Redis client
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_timeout=self.config.get("socket_timeout", 5),
                socket_connect_timeout=self.config.get("socket_connect_timeout", 5),
                socket_keepalive=self.config.get("socket_keepalive", True),
                retry_on_timeout=self.config.get("retry_on_timeout", True),
            )

            # Test connection
            client.ping()
            self.logger.info(f"Connected to Redis at {host}:{port}/{db}")

            if self.monitor:
                self.monitor.log_info(
                    f"Connected to Redis at {host}:{port}/{db}",
                    component="redis_mcp",
                    action="connection",
                    host=host,
                    port=port,
                    db=db,
                )

            return client

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            if self.monitor:
                self.monitor.log_error(
                    f"Failed to initialize Redis client: {e}",
                    component="redis_mcp",
                    action="connection_error",
                    error=str(e),
                )
            self.logger.warning("Using dummy client for testing/development")
            return None

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Redis.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            # Basic key-value operations
            "get": {
                "description": "Get a value from Redis",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_get,
            },
            "set": {
                "description": "Set a value in Redis",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": ["expiry"],
                "handler": self._handle_set,
            },
            "delete": {
                "description": "Delete a key from Redis",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_delete,
            },
            # Hash operations
            "hget": {
                "description": "Get a value from a hash",
                "category": "state",
                "required_params": ["key", "field"],
                "optional_params": [],
                "handler": self._handle_hget,
            },
            "hset": {
                "description": "Set a value in a hash",
                "category": "state",
                "required_params": ["key", "field", "value"],
                "optional_params": [],
                "handler": self._handle_hset,
            },
            "hgetall": {
                "description": "Get all fields and values from a hash",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_hgetall,
            },
            "hincrby": {
                "description": "Increment a value in a hash",
                "category": "state",
                "required_params": ["key", "field", "amount"],
                "optional_params": [],
                "handler": self._handle_hincrby,
            },
            # List operations
            "lpush": {
                "description": "Push a value to the left of a list",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": [],
                "handler": self._handle_lpush,
            },
            "rpush": {
                "description": "Push a value to the right of a list",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": [],
                "handler": self._handle_rpush,
            },
            "lrange": {
                "description": "Get a range of values from a list",
                "category": "state",
                "required_params": ["key", "start", "stop"],
                "optional_params": [],
                "handler": self._handle_lrange,
            },
            # Set operations
            "sadd": {
                "description": "Add a value to a set",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": [],
                "handler": self._handle_sadd,
            },
            "smembers": {
                "description": "Get all members of a set",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_smembers,
            },
            # Sorted set operations
            "zadd": {
                "description": "Add a value to a sorted set",
                "category": "state",
                "required_params": ["key", "score", "value"],
                "optional_params": [],
                "handler": self._handle_zadd,
            },
            "zrange": {
                "description": "Get a range of values from a sorted set",
                "category": "state",
                "required_params": ["key", "start", "stop"],
                "optional_params": ["withscores"],
                "handler": self._handle_zrange,
            },
            "zrevrange": {
                "description": "Get a range of values from a sorted set in reverse order",
                "category": "state",
                "required_params": ["key", "start", "stop"],
                "optional_params": ["withscores"],
                "handler": self._handle_zrevrange,
            },
            # Pub/Sub operations
            "publish": {
                "description": "Publish a message to a channel",
                "category": "pubsub",
                "required_params": ["channel", "message"],
                "optional_params": [],
                "handler": self._handle_publish,
            },
            # JSON operations
            "json_set": {
                "description": "Set a JSON value in Redis",
                "category": "state",
                "required_params": ["key", "json_data"],
                "optional_params": ["expiry"],
                "handler": self._handle_json_set,
            },
            "json_get": {
                "description": "Get a JSON value from Redis",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_json_get,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Redis MCP."""
        self.register_tool(self.get_value)
        self.register_tool(self.set_value)
        self.register_tool(self.delete_value)
        self.register_tool(self.get_hash)
        self.register_tool(self.set_hash)
        self.register_tool(self.get_list)
        self.register_tool(self.add_to_list)
        self.register_tool(self.get_sorted_set)
        self.register_tool(self.add_to_sorted_set)
        self.register_tool(self.publish_message)
        self.register_tool(self.set_json)
        self.register_tool(self.get_json)
        self.register_tool(self.increment_hash_value)

    def _setup_monitoring(self):
        """
        Set up monitoring for Redis MCP.

        Returns:
            Tuple of (MonitoringManager, metrics_dict)
        """
        try:
            # Get configuration from environment variables or config
            enable_prometheus = self.config.get("enable_prometheus", False)  # Default to False to avoid port conflicts
            
            # Set up monitoring using the new monitoring system
            monitor = MonitoringManager(service_name="redis_mcp")
            
            # Create a metrics dictionary for compatibility with existing code
            metrics = {}
            
            # Only try to register custom metrics if Prometheus is enabled
            if enable_prometheus:
                try:
                    # Use valid metric names (no hyphens)
                    metrics["redis_operations_total"] = "redis_operations_total"
                    metrics["redis_operation_duration_seconds"] = "redis_operation_duration_seconds"
                except Exception as e:
                    self.logger.warning(f"Could not register custom metrics: {e}")
            
            self.logger.info("Monitoring initialized successfully")
            return monitor, metrics
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            return None, {}

    def _record_metrics(self, operation: str, duration: float, status: str = "success"):
        """Record metrics for Redis operations."""
        if not self.monitor:
            return

        # Increment operation counter
        self.monitor.increment_counter(
            "redis_operations_total", 1, operation=operation, status=status
        )

        # Record operation duration
        self.monitor.observe_histogram(
            "redis_operation_duration_seconds", duration, operation=operation
        )

    def _log_operation(
        self, operation: str, key: str, status: str, error: Optional[str] = None
    ):
        """Log Redis operations."""
        if not self.monitor:
            return

        log_data = {
            "component": "redis_mcp",
            "action": operation,
            "key": key,
            "status": status,
        }

        if error:
            log_data["error"] = error
            self.monitor.log_error(
                f"Redis operation {operation} failed: {error}", **log_data
            )
        else:
            self.monitor.log_info(f"Redis operation {operation} completed", **log_data)

    def _handle_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get endpoint."""
        key = params.get("key")
        if not key:
            return {"error": "Missing required parameter: key"}

        try:
            start_time = time.time()
            value = self.redis_client.get(key)
            duration = time.time() - start_time

            # Record metrics and logs
            self._record_metrics("get", duration)
            self._log_operation("get", key, "success")

            return {"key": key, "value": value}
        except Exception as e:
            self.logger.error(f"Error getting value for key {key}: {e}")

            # Record error metrics and logs
            self._record_metrics("get", time.time() - start_time, "error")
            self._log_operation("get", key, "error", str(e))

            return {"error": f"Failed to get value: {str(e)}"}

    def _handle_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set endpoint."""
        key = params.get("key")
        value = params.get("value")
        expiry = params.get("expiry")

        if not key:
            return {"error": "Missing required parameter: key"}
        if value is None:
            return {"error": "Missing required parameter: value"}

        try:
            start_time = time.time()
            if expiry:
                self.redis_client.setex(key, int(expiry), value)
            else:
                self.redis_client.set(key, value)
            duration = time.time() - start_time

            # Record metrics and logs
            self._record_metrics("set", duration)
            self._log_operation("set", key, "success")

            return {"key": key, "value": value, "status": "success"}
        except Exception as e:
            self.logger.error(f"Error setting value for key {key}: {e}")

            # Record error metrics and logs
            self._record_metrics("set", time.time() - start_time, "error")
            self._log_operation("set", key, "error", str(e))

            return {"error": f"Failed to set value: {str(e)}"}

    def _handle_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delete endpoint."""
        key = params.get("key")
        if not key:
            return {"error": "Missing required parameter: key"}

        try:
            result = self.redis_client.delete(key)
            return {"key": key, "deleted": result > 0}
        except Exception as e:
            self.logger.error(f"Error deleting key {key}: {e}")
            return {"error": f"Failed to delete key: {str(e)}"}

    def _handle_hget(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hget endpoint."""
        key = params.get("key")
        field = params.get("field")

        if not key:
            return {"error": "Missing required parameter: key"}
        if not field:
            return {"error": "Missing required parameter: field"}

        try:
            value = self.redis_client.hget(key, field)
            return {"key": key, "field": field, "value": value}
        except Exception as e:
            self.logger.error(f"Error getting hash field {field} for key {key}: {e}")
            return {"error": f"Failed to get hash field: {str(e)}"}

    def _handle_hset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hset endpoint."""
        key = params.get("key")
        field = params.get("field")
        value = params.get("value")

        if not key:
            return {"error": "Missing required parameter: key"}
        if not field:
            return {"error": "Missing required parameter: field"}
        if value is None:
            return {"error": "Missing required parameter: value"}

        try:
            self.redis_client.hset(key, field, value)
            return {"key": key, "field": field, "value": value, "status": "success"}
        except Exception as e:
            self.logger.error(f"Error setting hash field {field} for key {key}: {e}")
            return {"error": f"Failed to set hash field: {str(e)}"}

    def _handle_hgetall(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hgetall endpoint."""
        key = params.get("key")
        if not key:
            return {"error": "Missing required parameter: key"}

        try:
            values = self.redis_client.hgetall(key)
            return {"key": key, "values": values}
        except Exception as e:
            self.logger.error(f"Error getting all hash fields for key {key}: {e}")
            return {"error": f"Failed to get all hash fields: {str(e)}"}

    def _handle_hincrby(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hincrby endpoint."""
        key = params.get("key")
        field = params.get("field")
        amount = params.get("amount")

        if not key:
            return {"error": "Missing required parameter: key"}
        if not field:
            return {"error": "Missing required parameter: field"}
        if amount is None:
            return {"error": "Missing required parameter: amount"}

        try:
            result = self.redis_client.hincrby(key, field, int(amount))
            return {"key": key, "field": field, "value": result}
        except Exception as e:
            self.logger.error(
                f"Error incrementing hash field {field} for key {key}: {e}"
            )
            return {"error": f"Failed to increment hash field: {str(e)}"}

    def _handle_lpush(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle lpush endpoint."""
        key = params.get("key")
        value = params.get("value")

        if not key:
            return {"error": "Missing required parameter: key"}
        if value is None:
            return {"error": "Missing required parameter: value"}

        try:
            result = self.redis_client.lpush(key, value)
            return {"key": key, "value": value, "length": result}
        except Exception as e:
            self.logger.error(f"Error pushing value to list {key}: {e}")
            return {"error": f"Failed to push value to list: {str(e)}"}

    def _handle_rpush(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle rpush endpoint."""
        key = params.get("key")
        value = params.get("value")

        if not key:
            return {"error": "Missing required parameter: key"}
        if value is None:
            return {"error": "Missing required parameter: value"}

        try:
            result = self.redis_client.rpush(key, value)
            return {"key": key, "value": value, "length": result}
        except Exception as e:
            self.logger.error(f"Error pushing value to list {key}: {e}")
            return {"error": f"Failed to push value to list: {str(e)}"}

    def _handle_lrange(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle lrange endpoint."""
        key = params.get("key")
        start = params.get("start")
        stop = params.get("stop")

        if not key:
            return {"error": "Missing required parameter: key"}
        if start is None:
            return {"error": "Missing required parameter: start"}
        if stop is None:
            return {"error": "Missing required parameter: stop"}

        try:
            values = self.redis_client.lrange(key, int(start), int(stop))
            return {
                "key": key,
                "start": int(start),
                "stop": int(stop),
                "values": values,
            }
        except Exception as e:
            self.logger.error(f"Error getting range from list {key}: {e}")
            return {"error": f"Failed to get range from list: {str(e)}"}

    def _handle_sadd(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sadd endpoint."""
        key = params.get("key")
        value = params.get("value")

        if not key:
            return {"error": "Missing required parameter: key"}
        if value is None:
            return {"error": "Missing required parameter: value"}

        try:
            result = self.redis_client.sadd(key, value)
            return {"key": key, "value": value, "added": result > 0}
        except Exception as e:
            self.logger.error(f"Error adding value to set {key}: {e}")
            return {"error": f"Failed to add value to set: {str(e)}"}

    def _handle_smembers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle smembers endpoint."""
        key = params.get("key")
        if not key:
            return {"error": "Missing required parameter: key"}

        try:
            members = self.redis_client.smembers(key)
            return {"key": key, "members": list(members)}
        except Exception as e:
            self.logger.error(f"Error getting members of set {key}: {e}")
            return {"error": f"Failed to get members of set: {str(e)}"}

    def _handle_zadd(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle zadd endpoint."""
        key = params.get("key")
        score = params.get("score")
        value = params.get("value")

        if not key:
            return {"error": "Missing required parameter: key"}
        if score is None:
            return {"error": "Missing required parameter: score"}
        if value is None:
            return {"error": "Missing required parameter: value"}

        try:
            result = self.redis_client.zadd(key, {value: float(score)})
            return {
                "key": key,
                "value": value,
                "score": float(score),
                "added": result > 0,
            }
        except Exception as e:
            self.logger.error(f"Error adding value to sorted set {key}: {e}")
            return {"error": f"Failed to add value to sorted set: {str(e)}"}

    def _handle_zrange(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle zrange endpoint."""
        key = params.get("key")
        start = params.get("start")
        stop = params.get("stop")
        withscores = params.get("withscores", False)

        if not key:
            return {"error": "Missing required parameter: key"}
        if start is None:
            return {"error": "Missing required parameter: start"}
        if stop is None:
            return {"error": "Missing required parameter: stop"}

        try:
            if withscores:
                values = self.redis_client.zrange(
                    key, int(start), int(stop), withscores=True
                )
                # Convert to a more JSON-friendly format
                result = []
                for value, score in values:
                    result.append({"value": value, "score": score})
                return {
                    "key": key,
                    "start": int(start),
                    "stop": int(stop),
                    "values": result,
                }
            else:
                values = self.redis_client.zrange(key, int(start), int(stop))
                return {
                    "key": key,
                    "start": int(start),
                    "stop": int(stop),
                    "values": values,
                }
        except Exception as e:
            self.logger.error(f"Error getting range from sorted set {key}: {e}")
            return {"error": f"Failed to get range from sorted set: {str(e)}"}

    def _handle_zrevrange(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle zrevrange endpoint."""
        key = params.get("key")
        start = params.get("start")
        stop = params.get("stop")
        withscores = params.get("withscores", False)

        if not key:
            return {"error": "Missing required parameter: key"}
        if start is None:
            return {"error": "Missing required parameter: start"}
        if stop is None:
            return {"error": "Missing required parameter: stop"}

        try:
            if withscores:
                values = self.redis_client.zrevrange(
                    key, int(start), int(stop), withscores=True
                )
                # Convert to a more JSON-friendly format
                result = []
                for value, score in values:
                    result.append({"value": value, "score": score})
                return {
                    "key": key,
                    "start": int(start),
                    "stop": int(stop),
                    "values": result,
                }
            else:
                values = self.redis_client.zrevrange(key, int(start), int(stop))
                return {
                    "key": key,
                    "start": int(start),
                    "stop": int(stop),
                    "values": values,
                }
        except Exception as e:
            self.logger.error(f"Error getting reverse range from sorted set {key}: {e}")
            return {"error": f"Failed to get reverse range from sorted set: {str(e)}"}

    def _handle_publish(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle publish endpoint."""
        channel = params.get("channel")
        message = params.get("message")

        if not channel:
            return {"error": "Missing required parameter: channel"}
        if message is None:
            return {"error": "Missing required parameter: message"}

        try:
            result = self.redis_client.publish(channel, message)
            return {"channel": channel, "message": message, "receivers": result}
        except Exception as e:
            self.logger.error(f"Error publishing message to channel {channel}: {e}")
            return {"error": f"Failed to publish message: {str(e)}"}

    def _handle_json_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle json_set endpoint."""
        key = params.get("key")
        json_data = params.get("json_data")
        expiry = params.get("expiry")

        if not key:
            return {"error": "Missing required parameter: key"}
        if json_data is None:
            return {"error": "Missing required parameter: json_data"}

        try:
            # Convert to JSON string if it's not already a string
            if not isinstance(json_data, str):
                json_string = json.dumps(json_data)
            else:
                json_string = json_data

            if expiry:
                self.redis_client.setex(key, int(expiry), json_string)
            else:
                self.redis_client.set(key, json_string)

            return {"key": key, "status": "success"}
        except Exception as e:
            self.logger.error(f"Error setting JSON value for key {key}: {e}")
            return {"error": f"Failed to set JSON value: {str(e)}"}

    def _handle_json_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle json_get endpoint."""
        key = params.get("key")
        if not key:
            return {"error": "Missing required parameter: key"}

        try:
            json_string = self.redis_client.get(key)
            if json_string:
                json_data = json.loads(json_string)
                return {"key": key, "data": json_data}
            else:
                return {"key": key, "data": None}
        except Exception as e:
            self.logger.error(f"Error getting JSON value for key {key}: {e}")
            return {"error": f"Failed to get JSON value: {str(e)}"}

    # Public API methods for models to use directly

    def fetch_data(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch data from an endpoint.

        Args:
            endpoint: Endpoint name
            params: Parameters for the endpoint

        Returns:
            Result of the endpoint handler
        """
        if endpoint not in self.endpoints:
            return {"error": f"Endpoint not found: {endpoint}"}

        handler = self.endpoints[endpoint]["handler"]
        return handler(params)

    def get_value(self, key: str) -> Any:
        """
        Get a value from Redis.

        Args:
            key: Key to get

        Returns:
            Value or None if key doesn't exist
        """
        return self.fetch_data("get", {"key": key}).get("value")

    def set_value(self, key: str, value: Any, expiry: Optional[int] = None) -> bool:
        """
        Set a value in Redis.

        Args:
            key: Key to set
            value: Value to set
            expiry: Optional expiry time in seconds

        Returns:
            True if successful, False otherwise
        """
        params = {"key": key, "value": value}
        if expiry:
            params["expiry"] = expiry
        return "error" not in self.fetch_data("set", params)

    def delete_value(self, key: str) -> bool:
        """
        Delete a key from Redis.

        Args:
            key: Key to delete

        Returns:
            True if key was deleted, False otherwise
        """
        return self.fetch_data("delete", {"key": key}).get("deleted", False)

    def get_hash(self, key: str) -> Dict[str, Any]:
        """
        Get all fields and values from a hash.

        Args:
            key: Hash key

        Returns:
            Dictionary of field-value pairs
        """
        return self.fetch_data("hgetall", {"key": key}).get("values", {})

    def set_hash(self, key: str, field_values: Dict[str, Any]) -> bool:
        """
        Set multiple fields in a hash.

        Args:
            key: Hash key
            field_values: Dictionary of field-value pairs

        Returns:
            True if successful, False otherwise
        """
        success = True
        for field, value in field_values.items():
            result = self.fetch_data(
                "hset", {"key": key, "field": field, "value": value}
            )
            if "error" in result:
                success = False
        return success

    def get_list(self, key: str, start: int = 0, stop: int = -1) -> List[Any]:
        """
        Get a range of values from a list.

        Args:
            key: List key
            start: Start index (default: 0)
            stop: Stop index (default: -1, meaning the last element)

        Returns:
            List of values
        """
        return self.fetch_data(
            "lrange", {"key": key, "start": start, "stop": stop}
        ).get("values", [])

    def add_to_list(self, key: str, value: Any, prepend: bool = False) -> int:
        """
        Add a value to a list.

        Args:
            key: List key
            value: Value to add
            prepend: Whether to prepend the value (default: False)

        Returns:
            Length of the list after the operation
        """
        if prepend:
            return self.fetch_data("lpush", {"key": key, "value": value}).get(
                "length", 0
            )
        else:
            return self.fetch_data("rpush", {"key": key, "value": value}).get(
                "length", 0
            )

    def get_sorted_set(
        self,
        key: str,
        start: int = 0,
        stop: int = -1,
        reverse: bool = True,
        with_scores: bool = True,
    ) -> List[Any]:
        """
        Get a range of values from a sorted set.

        Args:
            key: Sorted set key
            start: Start index (default: 0)
            stop: Stop index (default: -1, meaning the last element)
            reverse: Whether to return in reverse order (default: True)
            with_scores: Whether to include scores (default: True)

        Returns:
            List of values or list of value-score pairs
        """
        params = {"key": key, "start": start, "stop": stop, "withscores": with_scores}
        if reverse:
            return self.fetch_data("zrevrange", params).get("values", [])
        else:
            return self.fetch_data("zrange", params).get("values", [])

    def add_to_sorted_set(self, key: str, value: Any, score: float) -> bool:
        """
        Add a value to a sorted set.

        Args:
            key: Sorted set key
            value: Value to add
            score: Score for the value

        Returns:
            True if the value was added, False otherwise
        """
        return self.fetch_data(
            "zadd", {"key": key, "value": value, "score": score}
        ).get("added", False)

    def publish_message(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: Channel to publish to
            message: Message to publish

        Returns:
            Number of clients that received the message
        """
        return self.fetch_data("publish", {"channel": channel, "message": message}).get(
            "receivers", 0
        )

    def set_json(self, key: str, data: Any, expiry: Optional[int] = None) -> bool:
        """
        Set a JSON value in Redis.

        Args:
            key: Key to set
            data: JSON-serializable data
            expiry: Optional expiry time in seconds

        Returns:
            True if successful, False otherwise
        """
        params = {"key": key, "json_data": data}
        if expiry:
            params["expiry"] = expiry
        return "error" not in self.fetch_data("json_set", params)

    def get_json(self, key: str) -> Any:
        """
        Get a JSON value from Redis.

        Args:
            key: Key to get

        Returns:
            Deserialized JSON data or None if key doesn't exist
        """
        return self.fetch_data("json_get", {"key": key}).get("data")

    def increment_hash_value(self, key: str, field: str, amount: int = 1) -> int:
        """
        Increment a value in a hash.

        Args:
            key: Hash key
            field: Field to increment
            amount: Amount to increment by (default: 1)

        Returns:
            New value after incrementing
        """
        return self.fetch_data(
            "hincrby", {"key": key, "field": field, "amount": amount}
        ).get("value", 0)

    def expire(self, key: str, seconds: int) -> bool:
        """
        Set a key's time to live in seconds.

        Args:
            key: Key to set expiry on
            seconds: Time to live in seconds

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.redis_client:
                result = self.redis_client.expire(key, seconds)
                return result
            return False
        except Exception as e:
            self.logger.error(f"Error setting expiry for key {key}: {e}")
            return False

    def increment_float(self, key: str, amount: float) -> float:
        """
        Increment a float value stored at key.

        Args:
            key: Key to increment
            amount: Amount to increment by

        Returns:
            New value after incrementing
        """
        try:
            if self.redis_client:
                result = self.redis_client.incrbyfloat(key, amount)
                return float(result)
            return 0.0
        except Exception as e:
            self.logger.error(f"Error incrementing float for key {key}: {e}")
            return 0.0

    def get(self, key: str) -> str:
        """
        Get the value of a key.

        Args:
            key: Key to get

        Returns:
            Value as string or None if key doesn't exist
        """
        try:
            if self.redis_client:
                return self.redis_client.get(key)
            return None
        except Exception as e:
            self.logger.error(f"Error getting value for key {key}: {e}")
            return None
