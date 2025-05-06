"""
Redis MCP Server

This module implements a Model Context Protocol (MCP) server for Redis,
providing access to Redis for state management, caching, and pub/sub messaging.
It connects to the official Redis server with integrated monitoring.
"""

import os
import time
import json
import importlib
from typing import Dict, List, Any, Optional

# Direct imports instead of dynamic loading
try:
    import redis
except ImportError:
    redis = None
    print("Warning: Redis package not found or import failed.")

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector

# Import local Redis server
from local_redis.redis_server import RedisServer

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer


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
        # Initialize monitoring first so we can use it for logging
        self.logger = NetdataLogger(component_name="redis_mcp")
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.metrics_collector.start()
        
        self.logger.info("Initializing RedisMCP")
        
        super().__init__(name="redis_mcp", config=config)

        self.config_path = os.path.join("config", "redis_mcp", "redis_mcp_config.json")
        self.config = self._load_config()
        
        # Extract and prepare Redis server configuration to pass to the server
        self.redis_server_config = {
            "connection": {
                "host": self.config.get("host", "${REDIS_HOST:localhost}"),
                "port": self.config.get("port", "${REDIS_PORT:6379}"),
                "db": self.config.get("db", "${REDIS_DB:0}"),
                "password": self.config.get("password", "${REDIS_PASSWORD:}"),
                "decode_responses": self.config.get("decode_responses", True),
                "socket_timeout": self.config.get("socket_timeout", 5),
                "socket_connect_timeout": self.config.get("socket_connect_timeout", 5),
                "socket_keepalive": self.config.get("socket_keepalive", True),
                "retry_on_timeout": self.config.get("retry_on_timeout", True)
            }
        }
        
        self.logger.info("Configuration loaded", config_path=self.config_path)
            
        # Set up config file monitoring
        self._last_config_check = time.time()
        self._config_check_interval = 30  # Check for config changes every 30 seconds
        self._last_config_modified = os.path.getmtime(self.config_path) if os.path.exists(self.config_path) else 0
        
        # Initialize Redis client
        self.redis_client = self._initialize_client()
        self.logger.info("Redis client initialized")

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints() or {}
        self.logger.info("Endpoints initialized", endpoint_count=len(self.endpoints))

        # Register specific tools
        self._register_specific_tools()
        self.logger.info("Specific tools registered")

        self.logger.info("RedisMCP initialization completed",
                        component="redis_mcp", action="initialization")

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the config file.
        
        Returns:
            Dictionary containing configuration values
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
                return config
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing configuration file {self.config_path}: {e}")
                return {}
            except Exception as e:
                self.logger.error(f"Error loading configuration file {self.config_path}: {e}")
                return {}
        else:
            self.logger.warning(f"Configuration file not found at {self.config_path}")
            return {}
            
    def _check_config_changes(self):
        """
        Check if the configuration file has changed and reload it if necessary.
        """
        current_time = time.time()
        if current_time - self._last_config_check > self._config_check_interval:
            self._last_config_check = current_time
            if os.path.exists(self.config_path):
                modified_time = os.path.getmtime(self.config_path)
                if modified_time > self._last_config_modified:
                    self._last_config_modified = modified_time
                    self.logger.info(f"Configuration file changed, reloading from {self.config_path}")
                    new_config = self._load_config()
                    if new_config:
                        self.config = new_config
                        # Reinitialize client if connection settings changed
                        self.redis_client = self._initialize_client()

    def _initialize_client(self):
        """
        Initialize the Redis client by getting it from the local Redis server instance.
        In synthetic data mode, returns a mock client to avoid actual Redis connections.

        Returns:
            Redis client instance from the local Redis server, or a mock client if in synthetic mode,
            or None if unavailable and not in synthetic mode.
        """
        # Check if we're in synthetic data mode
        data_source = os.getenv('E2E_DATA_SOURCE')
        if data_source == 'synthetic':
            self.logger.info("Using synthetic data mode - returning mock Redis client")
            return self._create_mock_redis_client()
            
        self.logger.info("Attempting to get Redis client from local Redis server")
        try:
            # Get the singleton instance of the local Redis server
            # Pass the config to the RedisServer if needed, though it might load its own
            # based on its __init__ logic. Assuming RedisServer handles its own config loading.
            redis_server_instance = RedisServer.get_instance(config=self.redis_server_config)

            # Attempt to get the connected Redis client from the server instance with retries
            client = None
            retries = 0
            max_retries = 5  # Configure the number of retries
            retry_delay = 1  # Configure the delay between retries in seconds

            while retries < max_retries:
                try:
                    client = redis_server_instance.get_client()
                    if client:
                        self.logger.info("Successfully obtained Redis client from local Redis server")
                        return client
                except Exception as e:
                    self.logger.warning(f"Attempt {retries + 1} to get Redis client failed: {e}. Retrying in {retry_delay} seconds.")
                    time.sleep(retry_delay)
                    retries += 1

            # If loop finishes without getting a client
            self.logger.error(f"Failed to obtain Redis client from local Redis server after {max_retries} retries.")
            # Check if we should fallback to mock client
            if os.getenv('FALLBACK_TO_MOCK_REDIS', '').lower() in ('true', '1', 'yes'):
                self.logger.warning("Falling back to mock Redis client as fallback is enabled")
                return self._create_mock_redis_client()
            # Otherwise return None
            return None

        except Exception as e:
            self.logger.error(f"An unexpected error occurred while trying to get Redis client from local Redis server: {e}",
                            error=str(e))
            # Check if we should fallback to mock client
            if os.getenv('FALLBACK_TO_MOCK_REDIS', '').lower() in ('true', '1', 'yes'):
                self.logger.warning("Falling back to mock Redis client due to exception")
                return self._create_mock_redis_client()
            # Otherwise return None
            return None
            
    def _create_mock_redis_client(self):
        """
        Create a mock Redis client for synthetic data testing.
        This mock client implements a basic in-memory key-value store to simulate Redis.
        
        Returns:
            Mock Redis client object
        """
        self.logger.info("Creating mock Redis client for synthetic testing")
        
        # Create a simple in-memory mock Redis client
        class MockRedisClient:
            def __init__(self):
                self.data = {}  # Simple key-value store
                self.hashes = {}  # For hash operations
                self.lists = {}  # For list operations
                self.sets = {}  # For set operations
                self.sorted_sets = {}  # For sorted set operations
                
            def ping(self):
                return True
                
            def get(self, key):
                return self.data.get(key)
                
            def set(self, key, value, ex=None):
                self.data[key] = value
                return True
                
            def delete(self, *keys):
                count = 0
                for key in keys:
                    if key in self.data:
                        del self.data[key]
                        count += 1
                return count
                
            def hget(self, name, key):
                if name not in self.hashes:
                    return None
                return self.hashes[name].get(key)
                
            def hset(self, name, key, value):
                if name not in self.hashes:
                    self.hashes[name] = {}
                self.hashes[name][key] = value
                return 1
                
            def hgetall(self, name):
                return self.hashes.get(name, {})
                
            def hincrby(self, name, key, amount=1):
                if name not in self.hashes:
                    self.hashes[name] = {}
                if key not in self.hashes[name]:
                    self.hashes[name][key] = 0
                self.hashes[name][key] += amount
                return self.hashes[name][key]
                
            def lpush(self, name, *values):
                if name not in self.lists:
                    self.lists[name] = []
                for value in values:
                    self.lists[name].insert(0, value)
                return len(self.lists[name])
                
            def rpush(self, name, *values):
                if name not in self.lists:
                    self.lists[name] = []
                for value in values:
                    self.lists[name].append(value)
                return len(self.lists[name])
                
            def lrange(self, name, start, end):
                if name not in self.lists:
                    return []
                # Handle negative indices
                if end == -1:
                    end = len(self.lists[name])
                return self.lists[name][start:end+1]
                
            def sadd(self, name, *values):
                if name not in self.sets:
                    self.sets[name] = set()
                old_len = len(self.sets[name])
                for value in values:
                    self.sets[name].add(value)
                return len(self.sets[name]) - old_len
                
            def smembers(self, name):
                return self.sets.get(name, set())
                
            def zadd(self, name, mapping):
                if name not in self.sorted_sets:
                    self.sorted_sets[name] = {}
                old_len = len(self.sorted_sets[name])
                for member, score in mapping.items():
                    self.sorted_sets[name][member] = score
                return len(self.sorted_sets[name]) - old_len
                
            def zrange(self, name, start, end, withscores=False):
                if name not in self.sorted_sets:
                    return []
                # Convert to list and sort by score
                items = list(self.sorted_sets[name].items())
                items.sort(key=lambda x: x[1])
                
                # Handle negative indices
                if end == -1:
                    end = len(items)
                    
                items = items[start:end+1]
                
                if withscores:
                    return items
                else:
                    return [item[0] for item in items]
                
            def zrevrange(self, name, start, end, withscores=False):
                if name not in self.sorted_sets:
                    return []
                # Convert to list and sort by score (reversed)
                items = list(self.sorted_sets[name].items())
                items.sort(key=lambda x: x[1], reverse=True)
                
                # Handle negative indices
                if end == -1:
                    end = len(items)
                    
                items = items[start:end+1]
                
                if withscores:
                    return items
                else:
                    return [item[0] for item in items]
                    
            def publish(self, channel, message):
                # Mock publish always returns 0 receivers in synthetic mode
                return 0
                
            def expire(self, key, seconds):
                # This is a mock, so we don't actually implement expiration
                # Always return true if the key exists
                return key in self.data
                
            def info(self):
                # Return synthetic info for monitoring purposes
                return {
                    "connected_clients": 1,
                    "used_memory": 1000,
                    "used_memory_human": "1K",
                    "total_commands_processed": 100,
                    "keyspace_hits": 50,
                    "keyspace_misses": 10
                }
                
        self.logger.info("Mock Redis client created successfully")
        return MockRedisClient()

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Redis.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        self.logger.info("Initializing endpoints")
        endpoints = {
            # Basic key-value operations
            "get": {
                "description": "Get a value from Redis",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_get
            },
            "ping": {
                "description": "Ping the Redis server",
                "category": "system",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_ping
            },
            "set": {
                "description": "Set a value in Redis",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": ["expiry"],
                "handler": self._handle_set
            },
            "delete": {
                "description": "Delete a key from Redis",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_delete
            },
            # Hash operations
            "hget": {
                "description": "Get a value from a hash",
                "category": "state",
                "required_params": ["key", "field"],
                "optional_params": [],
                "handler": self._handle_hget
            },
            "hset": {
                "description": "Set a value in a hash",
                "category": "state",
                "required_params": ["key", "field", "value"],
                "optional_params": [],
                "handler": self._handle_hset
            },
            "hgetall": {
                "description": "Get all fields and values from a hash",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_hgetall
            },
            "hincrby": {
                "description": "Increment a value in a hash",
                "category": "state",
                "required_params": ["key", "field", "amount"],
                "optional_params": [],
                "handler": self._handle_hincrby
            },
            # List operations
            "lpush": {
                "description": "Push a value to the left of a list",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": [],
                "handler": self._handle_lpush
            },
            "rpush": {
                "description": "Push a value to the right of a list",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": [],
                "handler": self._handle_rpush
            },
            "lrange": {
                "description": "Get a range of values from a list",
                "category": "state",
                "required_params": ["key", "start", "stop"],
                "optional_params": [],
                "handler": self._handle_lrange
            },
            # Set operations
            "sadd": {
                "description": "Add a value to a set",
                "category": "state",
                "required_params": ["key", "value"],
                "optional_params": [],
                "handler": self._handle_sadd
            },
            "smembers": {
                "description": "Get all members of a set",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_smembers
            },
            # Sorted set operations
            "zadd": {
                "description": "Add a value to a sorted set",
                "category": "state",
                "required_params": ["key", "score", "value"],
                "optional_params": [],
                "handler": self._handle_zadd
            },
            "zrange": {
                "description": "Get a range of values from a sorted set",
                "category": "state",
                "required_params": ["key", "start", "stop"],
                "optional_params": ["withscores"],
                "handler": self._handle_zrange
            },
            "zrevrange": {
                "description": "Get a range of values from a sorted set in reverse order",
                "category": "state",
                "required_params": ["key", "start", "stop"],
                "optional_params": ["withscores"],
                "handler": self._handle_zrevrange
            },
            # Pub/Sub operations
            "publish": {
                "description": "Publish a message to a channel",
                "category": "pubsub",
                "required_params": ["channel", "message"],
                "optional_params": [],
                "handler": self._handle_publish
            },
            # JSON operations
            "json_set": {
                "description": "Set a JSON value in Redis",
                "category": "state",
                "required_params": ["key", "json_data"],
                "optional_params": ["expiry"],
                "handler": self._handle_json_set
            },
            "json_get": {
                "description": "Get a JSON value from Redis",
                "category": "state",
                "required_params": ["key"],
                "optional_params": [],
                "handler": self._handle_json_get
            }
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
        self.register_tool(self.ping)
    def _record_metrics(self, operation: str, duration: float, status: str = "success"):
        """Record metrics for Redis operations."""
        # Record operation timing
        self.logger.timing(f"redis_operation_{operation}", duration * 1000)  # Convert to ms
        
        # Record operation status as gauge
        if status == "success":
            self.logger.gauge(f"redis_operation_{operation}_success", 1)
        else:
            self.logger.gauge(f"redis_operation_{operation}_error", 1)

    def _log_operation(
        self, operation: str, key: str, status: str, error: Optional[str] = None
    ):
        """Log Redis operations."""
        log_data = {
            "component": "redis_mcp",
            "action": operation,
            "key": key,
            "status": status,
        }

        if error:
            log_data["error"] = error
            self.logger.error(f"Redis operation {operation} failed: {error}", **log_data)
        else:
            self.logger.info(f"Redis operation {operation} completed", **log_data)

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
            self.logger.timing("redis_get_ms", duration * 1000)
            self.logger.info("Redis GET operation completed", key=key, duration_ms=duration * 1000)

            return {"key": key, "value": value}
        except Exception as e:
            self.logger.error(f"Error getting value for key {key}: {e}", key=key, error=str(e))
            return {"error": f"Failed to get value: {str(e)}"}

    def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ping endpoint."""
        try:
            self.redis_client.ping()
            return {"status": "success", "message": "PONG"}
        except Exception as e:
            self.logger.error(f"Error pinging Redis server: {e}")
            return {"error": f"Failed to ping Redis server: {str(e)}"}

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
            self.logger.timing("redis_set_ms", duration * 1000)
            self.logger.info("Redis SET operation completed", 
                           key=key, 
                           has_expiry=(expiry is not None),
                           duration_ms=duration * 1000)

            return {"key": key, "value": value, "status": "success"}
        except Exception as e:
            self.logger.error(f"Error setting value for key {key}: {e}", 
                             key=key, 
                             error=str(e))
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
        json_data = params.get("data")  # Changed from json_data to data
        expiry = params.get("expiry")

        if not key:
            return {"error": "Missing required parameter: key"}
        if json_data is None:
            return {"error": "Missing required parameter: data"}

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
        self.logger.debug(f"Fetching data for endpoint: {endpoint}")
        if endpoint not in self.endpoints:
            self.logger.error(f"Endpoint not found: {endpoint}")
            return {"error": f"Endpoint not found: {endpoint}"}

        endpoint_config = self.endpoints[endpoint]
        handler = endpoint_config.get("handler")
        if not handler:
            self.logger.error(f"Handler not found for endpoint: {endpoint}")
            return {"error": f"Handler not found for endpoint: {endpoint}"}

        try:
            result = handler(params)
            self.logger.debug(f"Data fetched successfully for endpoint: {endpoint}")
            return result
        except Exception as e:
            self.logger.error(f"Error fetching data for endpoint {endpoint}: {str(e)}")
            return {"error": f"Error fetching data: {str(e)}"}

    def get_value(self, key: str) -> Dict[str, Any]:
        """
        Get a value from Redis.

        Args:
            key: Key to get

        Returns:
            Dictionary containing the value or None if key doesn't exist
        """
        result = self.fetch_data("get", {"key": key})
        
        # Ensure the result is always a dictionary with a 'value' key
        if isinstance(result, dict):
            if 'value' in result:
                # If the result is already a dictionary with a 'value' key, return it directly.
                return result
            elif 'error' in result:
                # If the result is an error dictionary, return a dictionary with value as None and the error.
                return {"value": None, "error": result["error"]}
            else:
                # If the result is a dictionary but doesn't have a 'value' key,
                # assume the entire dictionary is the value.
                self.logger.warning(f"Received dictionary without 'value' key from fetch_data for key {key}. Using entire dict as value.")
                return {"value": result}
        elif result is not None:
            # If the result is not a dictionary but is not None (e.g., string, bytes, number),
            # wrap it in a dictionary with the 'value' key.
            self.logger.debug(f"Received non-dict, non-None result type from fetch_data for key {key}: {type(result)}. Wrapping in value dict.")
            return {"value": result}
        else:
            # If the result is None, return a dictionary with value as None.
            self.logger.debug(f"Received None result from fetch_data for key {key}. Returning value as None.")
            return {"value": None}

    def set_value(self, key: str, value: Any, expiry: Optional[int] = None) -> dict:
        """
        Set a value in Redis.

        Args:
            key: Key to set
            value: Value to set
            expiry: Optional expiry time in seconds

        Returns:
            Dict with operation status and error (if any)
        """
        params = {"key": key, "value": value}
        if expiry:
            params["expiry"] = expiry
        result = self.fetch_data("set", params)
        if "error" in result:
            return {"success": False, "error": result["error"]}
        return {"success": True}

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

    def set_json(self, key: str, data: Any = None, value: Any = None, expiry: Optional[int] = None) -> bool:
        """
        Set a JSON value in Redis.
        
        Args:
            key: Key to set
            data: JSON-serializable data (preferred parameter)
            value: Alternative parameter for data (for compatibility)
            expiry: Optional expiry time in seconds
            
        Returns:
            True if successful, False otherwise
        """
        # Use data if provided, otherwise fall back to value
        store_data = data if data is not None else value
        params = {"key": key, "data": store_data}
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
            
    def delete_key(self, key: str) -> Dict[str, Any]:
        """
        Delete a key from Redis (alias for delete_value for backwards compatibility).
        
        Args:
            key: Key to delete
            
        Returns:
            Dictionary with operation status
        """
        deleted = self.delete_value(key)
        return {"success": deleted, "key": key}

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

    def ping(self) -> dict:
        """
        Ping the Redis server to check connectivity.

        Returns:
            Dictionary with status and message.
        """
        return self.fetch_data("ping", {})

