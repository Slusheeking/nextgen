"""
Redis Server Module

This module implements a Redis server with integrated monitoring using NetdataLogger
for metrics collection and logging. It provides a centralized Redis instance for the 
NextGen FinGPT system with automatic metrics collection and performance monitoring.

Features:
- Singleton pattern to ensure only one Redis server instance
- Integrated monitoring with NetdataLogger
- Automatic metrics collection for Redis operations
- Configuration via JSON files or environment variables
- Thread-safe operation recording and logging
"""

import os
import time
import re
import json
#
# Import the Redis client library with an alias to avoid conflict with our
# local redis package
import redis as redis_client
import logging
import threading
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector


class RedisServer:
    """
    Redis Server with integrated monitoring using NetdataLogger.

    This class provides a centralized Redis instance for the NextGen FinGPT system,
    with integrated logging and metrics using NetdataLogger. It implements the
    singleton pattern to ensure only one Redis server instance is running at a time.
    
    Features:
    - Automatic metrics collection for Redis operations
    - Performance monitoring with NetdataLogger
    - Thread-safe operation recording and logging
    - Configuration via JSON files or environment variables
    """

    _instance = None  # Singleton instance

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None):
        """
        Get or create the singleton instance of RedisServer.

        Args:
            config: Optional configuration dictionary to override environment variables

        Returns:
            RedisServer instance
        """
        if cls._instance is None:
            cls._instance = RedisServer(config)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Redis server with integrated monitoring.

        Args:
            config: Optional configuration dictionary to override environment variables
        """
        # Load environment variables
        load_dotenv()

        # Initialize logging first for early debugging
        self.logger = logging.getLogger("redis_server")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Load configuration from file if no config provided
        if config is None:
            config_path = os.path.join("config", "local_redis", "redis_server_config.json")
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
        self.monitor, self.metrics = self._setup_monitoring()

        # Initialize Redis client
        self.redis_client = self._initialize_redis()

        # Register metrics
        if self.monitor:
            self._register_metrics()
            self._start_metrics_collection()

        self.logger.info("Redis server initialized with integrated monitoring")
        if self.monitor:
            self.monitor.info("Redis server initialized")

    def _setup_monitoring(self):
        """
        Set up monitoring for Redis server using NetdataLogger.

        Returns:
            Tuple of (NetdataLogger, metrics_dict)
        """
        try:
            # Get monitoring settings from config
            monitoring_config = self.config.get("monitoring", {})
            
            # Extract monitoring parameters
            metrics_interval = int(monitoring_config.get("metrics_interval", 15))
            
            # Set up monitoring with NetdataLogger
            logger = NetdataLogger(component_name="redis-server")
            
            # Initialize system metrics collector
            metrics_collector = SystemMetricsCollector(logger)
            metrics_collector.start()
            
            # Store metrics interval for use in metrics collection
            self.metrics_interval = metrics_interval
            self.metrics_collector = metrics_collector

            # Create an empty metrics dictionary for compatibility
            metrics = {}

            self.logger.info("Monitoring initialized successfully with NetdataLogger")
            return logger, metrics
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            return None, {}

    def _initialize_redis(self):
        """
        Initialize the Redis client.

        Returns:
            Redis client instance
        """
        # Get connection settings from config
        connection_config = self.config.get("connection", {})

        # Extract connection parameters with environment variable fallbacks
        # Use self._get_config_value helper to parse ${ENV_VAR:default} format
        host = self._get_config_value(connection_config.get("host", "${REDIS_HOST:localhost}"))
        # Get a list of ports to try. Prioritize a list from config, then a single port from config/env, then a default list.
        ports_to_try_config = connection_config.get("ports", [])
        if not ports_to_try_config:
             single_port_config = self._get_config_value(connection_config.get("port", "${REDIS_PORT:6379}"))
             try:
                 ports_to_try_config = [int(single_port_config)]
             except ValueError:
                 self.logger.error(f"Invalid port configuration: {single_port_config}. Using default ports.")
                 ports_to_try_config = [6379, 6380, 6381, 6382, 6383] # Default ports

        db = int(self._get_config_value(connection_config.get("db", "${REDIS_DB:0}")))

        # Extract additional connection parameters
        socket_timeout = float(connection_config.get("socket_timeout", 5.0))
        socket_connect_timeout = float(connection_config.get("socket_connect_timeout", 10.0))
        retry_on_timeout = bool(connection_config.get("retry_on_timeout", True))
        decode_responses = bool(connection_config.get("decode_responses", True))

        client = None
        for port in ports_to_try_config:
            try:
                # Log connection details (excluding password)
                self.logger.info(f"Attempting to connect to Redis at host={host}, port={port}, db={db}")

                # Get password from environment variable
                password = self._get_config_value(connection_config.get("password", "${REDIS_PASSWORD:}"))
                
                # Create Redis client
                temp_client = redis_client.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password if password else None, # Use password from environment if available
                    decode_responses=decode_responses,
                    socket_timeout=socket_timeout,
                    socket_connect_timeout=socket_connect_timeout,
                    socket_keepalive=True,
                    retry_on_timeout=retry_on_timeout
                )

                # Test connection
                temp_client.ping()
                self.logger.info(f"Successfully connected to Redis at {host}:{port}/{db}")

                if self.monitor:
                    self.monitor.info(
                        f"Connected to Redis at {host}:{port}/{db}",
                        host=host,
                        port=port,
                        db=db
                    )
                client = temp_client
                break # Connection successful, break the loop

            except redis_client.exceptions.ConnectionError as e:
                self.logger.warning(f"Failed to connect to Redis at {host}:{port}/{db}: {e}. Trying next port...")
                if self.monitor:
                    self.monitor.warning(
                        f"Failed to connect to Redis at {host}:{port}/{db}",
                        host=host,
                        port=port,
                        db=db,
                        error=str(e)
                    )
                time.sleep(1) # Wait a bit before trying the next port
                continue # Try the next port
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while connecting to Redis at {host}:{port}/{db}: {e}. Trying next port...")
                if self.monitor:
                    self.monitor.error(
                        f"An unexpected error occurred while connecting to Redis at {host}:{port}/{db}",
                        host=host,
                        port=port,
                        db=db,
                        error=str(e)
                    )
                time.sleep(1) # Wait a bit before trying the next port
                continue # Try the next port

        if client is None:
            self.logger.error("Failed to connect to Redis on any of the specified ports.")
            if self.monitor:
                self.monitor.error("Failed to connect to Redis on any of the specified ports.")
            # Return a dummy client or None if no connection was successful
            self.logger.warning("Returning None as no Redis connection was successful.")
            return None

        return client

    def _register_metrics(self):
        """Register metrics for Redis monitoring using NetdataLogger."""
        if not self.monitor:
            return
            
        # With NetdataLogger, we don't need to pre-register metrics
        # They are created on-the-fly when used
        self.logger.info("Redis metrics registered with NetdataLogger")

    def _start_metrics_collection(self):
        """Start a background thread to collect Redis metrics using NetdataLogger."""
        if not self.monitor or not self.redis_client:
            return

        def collect_metrics():
            while True:
                try:
                    # Get Redis info
                    info = self.redis_client.info()

                    # Update metrics using NetdataLogger
                    self.monitor.gauge("redis_connected_clients", info.get("connected_clients", 0))
                    self.monitor.gauge("redis_used_memory", info.get("used_memory", 0))
                    self.monitor.gauge("redis_total_commands_processed", info.get("total_commands_processed", 0))
                    self.monitor.gauge("redis_keyspace_hits", info.get("keyspace_hits", 0))
                    self.monitor.gauge("redis_keyspace_misses", info.get("keyspace_misses", 0))

                    # Log to monitoring
                    self.monitor.info(
                        "Redis metrics collected",
                        connected_clients=info.get("connected_clients", 0),
                        used_memory_human=info.get("used_memory_human", "N/A"),
                        total_commands_processed=info.get("total_commands_processed", 0)
                    )
                except Exception as e:
                    self.logger.error(f"Error collecting Redis metrics: {e}")
                    if self.monitor:
                        self.monitor.error(
                            f"Error collecting Redis metrics: {e}",
                            error=str(e)
                        )

                # Sleep for configured interval (defaults to 15 seconds)
                time.sleep(getattr(self, 'metrics_interval', 15))

        # Start metrics collection thread
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
        self.logger.info("Redis metrics collection started")

    def get_client(self):
        """
        Get the Redis client instance.

        Returns:
            Redis client instance
        """
        return self.redis_client

    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all available tools on this MCP server.

        Returns:
            List of tool information dictionaries
        """
        # Define the tools provided by this Redis server
        tools = [
            {"name": "ping", "description": "Check the connectivity to the Redis server."},
            {"name": "get", "description": "Get the value of a key."},
            {"name": "set", "description": "Set the value of a key."},
            {"name": "delete", "description": "Delete a key."},
            {"name": "keys", "description": "Find all keys matching a pattern."},
            {"name": "set_json", "description": "Set the value of a key as JSON."},
            {"name": "get_json", "description": "Get the value of a key as JSON."},
            {"name": "xadd", "description": "Add a new message to a stream."},
            {"name": "xread", "description": "Read messages from streams."},
            {"name": "xreadgroup", "description": "Read messages from streams using a consumer group."},
            {"name": "xgroup_create", "description": "Create a consumer group for a stream."},
            {"name": "xack", "description": "Acknowledge pending messages in a stream."},
            {"name": "zadd", "description": "Add a member with a score to a sorted set."},
            {"name": "zrangebyscore", "description": "Get members from a sorted set within a score range."},
            {"name": "zrevrange", "description": "Get members from a sorted set in reverse order."},
            {"name": "smembers", "description": "Get all members of a set."},
        ]
        self.logger.info(f"Listing {len(tools)} tools")
        return tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific tool provided by this MCP server.

        Args:
            tool_name: The name of the tool to call.
            arguments: A dictionary of arguments for the tool.

        Returns:
            The result of the tool execution.
        """
        start_time = time.time()
        self.logger.info(f"Calling tool: {tool_name} with arguments: {list(arguments.keys())}")

        try:
            result = None
            if tool_name == "ping":
                result = self.redis_client.ping()
                result = {"result": "PONG" if result else "FAIL"} # Wrap boolean result

            elif tool_name == "get":
                key = arguments.get("key")
                if key is None: raise ValueError("Missing 'key' argument for get")
                result = {"value": self.redis_client.get(key)}

            elif tool_name == "set":
                key = arguments.get("key")
                value = arguments.get("value")
                expiry = arguments.get("expiry")
                if key is None or value is None: raise ValueError("Missing 'key' or 'value' argument for set")
                self.redis_client.set(key, value, ex=expiry)
                result = {"status": "success"}

            elif tool_name == "delete":
                key = arguments.get("key")
                if key is None: raise ValueError("Missing 'key' argument for delete")
                deleted_count = self.redis_client.delete(key)
                result = {"deleted_count": deleted_count}

            elif tool_name == "keys":
                pattern = arguments.get("pattern")
                if pattern is None: raise ValueError("Missing 'pattern' argument for keys")
                # Note: KEYS can be slow on large databases. Use SCAN in production.
                keys = self.redis_client.keys(pattern)
                result = {"keys": keys}

            elif tool_name == "set_json":
                key = arguments.get("key")
                value = arguments.get("value")
                expiry = arguments.get("expiry")
                if key is None or value is None: raise ValueError("Missing 'key' or 'value' argument for set_json")
                # Use RedisJSON module if available
                if hasattr(self.redis_client, 'json'):
                    self.redis_client.json().set(key, "$", json.dumps(value))
                    if expiry:
                        self.redis_client.expire(key, expiry)
                    result = {"status": "success"}
                else:
                    # Fallback to standard set if RedisJSON is not available
                    self.logger.warning("RedisJSON module not available, falling back to standard SET for set_json")
                    self.redis_client.set(key, json.dumps(value), ex=expiry)
                    result = {"status": "success_fallback"}

            elif tool_name == "get_json":
                key = arguments.get("key")
                if key is None: raise ValueError("Missing 'key' argument for get_json")
                # Use RedisJSON module if available
                if hasattr(self.redis_client, 'json'):
                    json_value = self.redis_client.json().get(key)
                    result = {"value": json.loads(json_value) if json_value else None}
                else:
                    # Fallback to standard get and json load
                    self.logger.warning("RedisJSON module not available, falling back to standard GET for get_json")
                    value = self.redis_client.get(key)
                    result = {"value": json.loads(value) if value else None}

            elif tool_name == "xadd":
                stream = arguments.get("stream")
                data = arguments.get("data")
                if stream is None or data is None: raise ValueError("Missing 'stream' or 'data' argument for xadd")
                message_id = self.redis_client.xadd(stream, data)
                result = {"message_id": message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id}

            elif tool_name == "xread":
                streams = arguments.get("streams")
                count = arguments.get("count")
                block = arguments.get("block")
                if streams is None: raise ValueError("Missing 'streams' argument for xread")
                # Prepare stream IDs - use latest ID for each stream
                stream_ids = {stream: '>' for stream in streams}
                messages = self.redis_client.xread(stream_ids, count=count, block=block)
                # Format the result to match expected structure
                formatted_messages = []
                if messages:
                    for stream_name_bytes, stream_messages in messages:
                        stream_name = stream_name_bytes.decode('utf-8')
                        formatted_stream_messages = []
                        for msg_id_bytes, msg_data_bytes in stream_messages:
                            msg_id = msg_id_bytes.decode('utf-8')
                            # Decode message data bytes to string, then parse JSON
                            msg_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in msg_data_bytes.items()}
                            # Attempt to parse JSON values if they look like JSON strings
                            for k, v in msg_data.items():
                                if v.strip().startswith('{') and v.strip().endswith('}'):
                                    try:
                                        msg_data[k] = json.loads(v)
                                    except json.JSONDecodeError:
                                        pass # Keep as string if not valid JSON
                            formatted_stream_messages.append((msg_id, msg_data))
                        formatted_messages.append((stream_name, formatted_stream_messages))
                result = {"messages": formatted_messages}

            elif tool_name == "xreadgroup":
                group = arguments.get("group")
                consumer = arguments.get("consumer")
                streams = arguments.get("streams")
                ids = arguments.get("ids")
                count = arguments.get("count")
                block = arguments.get("block")
                if group is None or consumer is None or streams is None or ids is None:
                    raise ValueError("Missing required arguments for xreadgroup")
                
                # Prepare stream IDs
                stream_ids = {stream: id_val for stream, id_val in zip(streams, ids)}
                
                messages = self.redis_client.xreadgroup(group, consumer, stream_ids, count=count, block=block)
                
                # Format the result
                formatted_messages = []
                if messages:
                    for stream_name_bytes, stream_messages in messages:
                        stream_name = stream_name_bytes.decode('utf-8')
                        formatted_stream_messages = []
                        for msg_id_bytes, msg_data_bytes in stream_messages:
                            msg_id = msg_id_bytes.decode('utf-8')
                            # Decode message data bytes to string, then parse JSON
                            msg_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in msg_data_bytes.items()}
                            # Attempt to parse JSON values if they look like JSON strings
                            for k, v in msg_data.items():
                                if v.strip().startswith('{') and v.strip().endswith('}'):
                                    try:
                                        msg_data[k] = json.loads(v)
                                    except json.JSONDecodeError:
                                        pass # Keep as string if not valid JSON
                            formatted_stream_messages.append((msg_id, msg_data))
                        formatted_messages.append((stream_name, formatted_stream_messages))
                result = {"messages": formatted_messages}

            elif tool_name == "xgroup_create":
                stream = arguments.get("stream")
                group = arguments.get("group")
                id_val = arguments.get("id")
                mkstream = arguments.get("mkstream", False)
                if stream is None or group is None or id_val is None:
                    raise ValueError("Missing required arguments for xgroup_create")
                
                # Use mkstream argument if supported by the redis-py version
                if mkstream and hasattr(self.redis_client, 'xgroup_create') and 'mkstream' in self.redis_client.xgroup_create.__code__.co_varnames:
                     success = self.redis_client.xgroup_create(stream, group, id_val, mkstream=mkstream)
                else:
                     # Fallback for older versions or if mkstream is not used
                     success = self.redis_client.xgroup_create(stream, group, id_val)
                     if mkstream:
                         self.logger.warning("mkstream argument not supported or used in xgroup_create, stream must exist.")
                         
                result = {"success": success}

            elif tool_name == "xack":
                stream = arguments.get("stream")
                group = arguments.get("group")
                ids = arguments.get("ids")
                if stream is None or group is None or ids is None:
                    raise ValueError("Missing required arguments for xack")
                
                acked_count = self.redis_client.xack(stream, group, *ids)
                result = {"acked_count": acked_count}

            elif tool_name == "zadd":
                key = arguments.get("key")
                score = arguments.get("score")
                member = arguments.get("member")
                if key is None or score is None or member is None:
                    raise ValueError("Missing required arguments for zadd")
                
                added_count = self.redis_client.zadd(key, {member: score})
                result = {"added_count": added_count}

            elif tool_name == "zrangebyscore":
                key = arguments.get("key")
                min_score = arguments.get("min")
                max_score = arguments.get("max")
                if key is None or min_score is None or max_score is None:
                    raise ValueError("Missing required arguments for zrangebyscore")
                
                members = self.redis_client.zrangebyscore(key, min_score, max_score)
                result = {"members": [m.decode('utf-8') for m in members]}

            elif tool_name == "zrevrange":
                key = arguments.get("key")
                start = arguments.get("start")
                stop = arguments.get("stop")
                if key is None or start is None or stop is None:
                    raise ValueError("Missing required arguments for zrevrange")
                
                members = self.redis_client.zrevrange(key, start, stop)
                result = {"members": [m.decode('utf-8') for m in members]}

            elif tool_name == "smembers":
                key = arguments.get("key")
                if key is None: raise ValueError("Missing 'key' argument for smembers")
                members = self.redis_client.smembers(key)
                result = {"members": [m.decode('utf-8') for m in members]}

            else:
                raise ValueError(f"Tool not found: {tool_name}")

            duration = time.time() - start_time
            self.record_operation(tool_name, duration, "success")
            self.log_operation(tool_name, arguments.get("key", "N/A"), "success")
            return result

        except Exception as e:
            duration = time.time() - start_time
            self.record_operation(tool_name, duration, "error")
            self.log_operation(tool_name, arguments.get("key", "N/A"), "error", str(e))
            return {"error": str(e)}


    def record_operation(self, operation: str, duration: float, status: str = "success"):
        """
        Record a Redis operation for metrics using NetdataLogger.

        Args:
            operation: Name of the operation (e.g., "get", "set")
            duration: Duration of the operation in seconds
            status: Status of the operation ("success" or "error")
        """
        if not self.monitor:
            return

        # Record operation timing (convert to milliseconds)
        self.monitor.timing(f"redis_operation_{operation}_ms", duration * 1000)
        
        # Record operation status as gauge
        if status == "success":
            self.monitor.gauge(f"redis_operation_{operation}_success", 1)
        else:
            self.monitor.gauge(f"redis_operation_{operation}_error", 1)

    def log_operation(self, operation: str, key: str, status: str, error: Optional[str] = None):
        """
        Log a Redis operation using NetdataLogger.

        Args:
            operation: Name of the operation (e.g., "get", "set")
            key: Redis key being operated on
            status: Status of the operation ("success" or "error")
            error: Error message if status is "error"
        """
        if not self.monitor:
            return

        if error:
            self.monitor.error(f"Redis operation {operation} failed: {error}", 
                              operation=operation,
                              key=key,
                              status=status,
                              error=error)
        else:
            self.monitor.info(f"Redis operation {operation} completed", 
                             operation=operation,
                             key=key,
                             status=status)
            
    def _get_config_value(self, value: str) -> str:
        """
        Parse a configuration value that may contain environment variable references.
        Supports the format ${ENV_VAR:default_value} where default_value is used
        if ENV_VAR is not set.
        
        Args:
            value: The configuration value to parse
            
        Returns:
            The parsed value with environment variables replaced
        """
        if not isinstance(value, str):
            return value
            
        # Match ${ENV_VAR:default} pattern
        pattern = r'\${([^:{}]+)(?::([^{}]*))?}'
        
        def replace_env_var(match):
            env_var = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(env_var, default)
            
        # Replace all occurrences of the pattern
        return re.sub(pattern, replace_env_var, value)


# Example usage
if __name__ == "__main__":
    # Get Redis server instance
    redis_server = RedisServer.get_instance()

    # Get Redis client
    redis_client = redis_server.get_client()

    if redis_client:
        # Perform some operations
        start_time = time.time()
        try:
            redis_client.set("test_key", "test_value")
            redis_server.record_operation("set", time.time() - start_time)
            redis_server.log_operation("set", "test_key", "success")

            start_time = time.time()
            value = redis_client.get("test_key")
            redis_server.record_operation("get", time.time() - start_time)
            redis_server.log_operation("get", "test_key", "success")

            print(f"Retrieved value: {value}")
        except Exception as e:
            redis_server.record_operation("set", time.time() - start_time, "error")
            redis_server.log_operation("set", "test_key", "error", str(e))
            print(f"Error: {e}")

    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Redis server shutting down")
