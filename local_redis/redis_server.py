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
from typing import Dict, Any, Optional
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
        try:
            # Get connection settings from config
            connection_config = self.config.get("connection", {})
            
            # Extract connection parameters with environment variable fallbacks
            # Use self._get_config_value helper to parse ${ENV_VAR:default} format
            host = self._get_config_value(connection_config.get("host", "${REDIS_HOST:localhost}"))
            port = int(self._get_config_value(connection_config.get("port", "${REDIS_PORT:6379}")))
            db = int(self._get_config_value(connection_config.get("db", "${REDIS_DB:0}")))
            password = self._get_config_value(connection_config.get("password", "${REDIS_PASSWORD:}"))
            
            # Extract additional connection parameters
            socket_timeout = float(connection_config.get("socket_timeout", 5.0))
            socket_connect_timeout = float(connection_config.get("socket_connect_timeout", 10.0))
            retry_on_timeout = bool(connection_config.get("retry_on_timeout", True))
            decode_responses = bool(connection_config.get("decode_responses", True))
            
            # Log connection details (excluding password)
            self.logger.info(f"Initializing Redis client with host={host}, port={port}, db={db}")
            
            # Create Redis client
            client = redis_client.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                socket_keepalive=True,
                retry_on_timeout=retry_on_timeout
            )

            # Test connection
            client.ping()
            self.logger.info(f"Connected to Redis at {host}:{port}/{db}")

            if self.monitor:
                self.monitor.info(
                    f"Connected to Redis at {host}:{port}/{db}",
                    host=host,
                    port=port,
                    db=db
                )

            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            if self.monitor:
                self.monitor.error(
                    f"Failed to initialize Redis client: {e}",
                    error=str(e)
                )

            # Return a dummy client for testing/development
            self.logger.warning("Using dummy Redis client for testing/development")
            return None

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
