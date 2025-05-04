"""
Redis Server Module

This module implements a Redis server with Loki logging and Prometheus monitoring integration.
It provides a centralized Redis instance for the NextGen FinGPT system.
"""

import os
import time
import redis
import logging
import threading
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from prometheus.loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

class RedisServer:
    """
    Redis Server with Loki logging and Prometheus monitoring integration.
    
    This class provides a centralized Redis instance for the NextGen FinGPT system,
    with integrated logging to Loki and metrics to Prometheus.
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
        Initialize the Redis server with Loki and Prometheus integration.
        
        Args:
            config: Optional configuration dictionary to override environment variables
        """
        # Load environment variables
        load_dotenv()
        
        # Store configuration
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("redis_server")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Initialize Loki logger
        self.loki_logger = self._setup_loki_logger()
        
        # Initialize Prometheus metrics
        self.prometheus = self._setup_prometheus()
        
        # Initialize Redis client
        self.redis_client = self._initialize_redis()
        
        # Register metrics
        self._register_metrics()
        
        # Start metrics collection thread
        self._start_metrics_collection()
        
        self.logger.info("Redis server initialized with Loki logging and Prometheus monitoring")
        if self.loki_logger:
            self.loki_logger.info("Redis server initialized", component="redis_server", action="initialization")
    
    def _setup_loki_logger(self) -> Optional[LokiManager]:
        """
        Set up Loki logger for Redis server.
        
        Returns:
            LokiManager instance or None if setup fails
        """
        try:
            loki_url = self.config.get("loki_url") or os.getenv("LOKI_URL")
            if not loki_url:
                self.logger.warning("LOKI_URL not set, Loki logging disabled")
                return None
            
            loki_logger = LokiManager(
                service_name="redis-server",
                loki_url=loki_url,
                loki_username=self.config.get("loki_username") or os.getenv("LOKI_USERNAME"),
                loki_password=self.config.get("loki_password") or os.getenv("LOKI_PASSWORD")
            )
            
            self.logger.info(f"Loki logging enabled at {loki_url}")
            return loki_logger
        except Exception as e:
            self.logger.error(f"Failed to initialize Loki logger: {e}")
            return None
    
    def _setup_prometheus(self) -> Optional[PrometheusManager]:
        """
        Set up Prometheus metrics for Redis server.
        
        Returns:
            PrometheusManager instance or None if setup fails
        """
        try:
            metrics_port = self.config.get("metrics_port") or os.getenv("PROMETHEUS_METRICS_PORT")
            if not metrics_port:
                self.logger.warning("PROMETHEUS_METRICS_PORT not set, using default port 8010")
                metrics_port = 8010
            
            prometheus = PrometheusManager(
                service_name="redis-server",
                expose_metrics=True,
                metrics_port=int(metrics_port),
                pushgateway_url=self.config.get("pushgateway_url") or os.getenv("PUSHGATEWAY_URL")
            )
            
            self.logger.info(f"Prometheus metrics enabled on port {metrics_port}")
            return prometheus
        except Exception as e:
            self.logger.error(f"Failed to initialize Prometheus metrics: {e}")
            return None
    
    def _initialize_redis(self) -> redis.Redis:
        """
        Initialize the Redis client.
        
        Returns:
            Redis client instance
        """
        try:
            # Get Redis configuration from environment variables or config
            host = self.config.get("host") or os.getenv("REDIS_HOST", "localhost")
            port = int(self.config.get("port") or os.getenv("REDIS_PORT", "6379"))
            db = int(self.config.get("db") or os.getenv("REDIS_DB", "0"))
            password = self.config.get("password") or os.getenv("REDIS_PASSWORD")
            
            # Create Redis client
            client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            
            # Test connection
            client.ping()
            self.logger.info(f"Connected to Redis at {host}:{port}/{db}")
            
            if self.loki_logger:
                self.loki_logger.info(
                    f"Connected to Redis at {host}:{port}/{db}",
                    component="redis_server",
                    action="connection",
                    host=host,
                    port=port,
                    db=db
                )
            
            return client
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis client: {e}")
            if self.loki_logger:
                self.loki_logger.error(
                    f"Failed to initialize Redis client: {e}",
                    component="redis_server",
                    action="connection_error",
                    error=str(e)
                )
            
            # Return a dummy client for testing/development
            self.logger.warning("Using dummy Redis client for testing/development")
            return None
    
    def _register_metrics(self):
        """Register Prometheus metrics for Redis monitoring."""
        if not self.prometheus:
            return
        
        # Create metrics
        self.prometheus.create_gauge(
            "connected_clients",
            "Number of client connections",
        )
        
        self.prometheus.create_gauge(
            "used_memory",
            "Used memory in bytes",
        )
        
        self.prometheus.create_gauge(
            "total_commands_processed",
            "Total number of commands processed",
        )
        
        self.prometheus.create_gauge(
            "keyspace_hits",
            "Number of successful lookups of keys in the main dictionary",
        )
        
        self.prometheus.create_gauge(
            "keyspace_misses",
            "Number of failed lookups of keys in the main dictionary",
        )
        
        self.prometheus.create_counter(
            "operations_total",
            "Total number of Redis operations",
            ["operation", "status"]
        )
        
        self.prometheus.create_histogram(
            "operation_duration_seconds",
            "Duration of Redis operations in seconds",
            ["operation"]
        )
    
    def _start_metrics_collection(self):
        """Start a background thread to collect Redis metrics."""
        if not self.prometheus or not self.redis_client:
            return
        
        def collect_metrics():
            while True:
                try:
                    # Get Redis info
                    info = self.redis_client.info()
                    
                    # Update metrics
                    self.prometheus.set_gauge("connected_clients", info.get("connected_clients", 0))
                    self.prometheus.set_gauge("used_memory", info.get("used_memory", 0))
                    self.prometheus.set_gauge("total_commands_processed", info.get("total_commands_processed", 0))
                    self.prometheus.set_gauge("keyspace_hits", info.get("keyspace_hits", 0))
                    self.prometheus.set_gauge("keyspace_misses", info.get("keyspace_misses", 0))
                    
                    # Log to Loki
                    if self.loki_logger:
                        self.loki_logger.info(
                            "Redis metrics collected",
                            component="redis_server",
                            action="metrics_collection",
                            connected_clients=info.get("connected_clients", 0),
                            used_memory_human=info.get("used_memory_human", "N/A"),
                            total_commands_processed=info.get("total_commands_processed", 0)
                        )
                except Exception as e:
                    self.logger.error(f"Error collecting Redis metrics: {e}")
                    if self.loki_logger:
                        self.loki_logger.error(
                            f"Error collecting Redis metrics: {e}",
                            component="redis_server",
                            action="metrics_collection_error",
                            error=str(e)
                        )
                
                # Sleep for 15 seconds
                time.sleep(15)
        
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
        self.logger.info("Redis metrics collection started")
    
    def get_client(self) -> redis.Redis:
        """
        Get the Redis client instance.
        
        Returns:
            Redis client instance
        """
        return self.redis_client
    
    def record_operation(self, operation: str, duration: float, status: str = "success"):
        """
        Record a Redis operation for metrics.
        
        Args:
            operation: Name of the operation (e.g., "get", "set")
            duration: Duration of the operation in seconds
            status: Status of the operation ("success" or "error")
        """
        if not self.prometheus:
            return
        
        # Increment operation counter
        self.prometheus.increment_counter(
            "operations_total",
            1,
            operation=operation,
            status=status
        )
        
        # Record operation duration
        self.prometheus.observe_histogram(
            "operation_duration_seconds",
            duration,
            operation=operation
        )
    
    def log_operation(self, operation: str, key: str, status: str, error: Optional[str] = None):
        """
        Log a Redis operation to Loki.
        
        Args:
            operation: Name of the operation (e.g., "get", "set")
            key: Redis key being operated on
            status: Status of the operation ("success" or "error")
            error: Error message if status is "error"
        """
        if not self.loki_logger:
            return
        
        log_data = {
            "component": "redis_server",
            "action": operation,
            "key": key,
            "status": status
        }
        
        if error:
            log_data["error"] = error
            self.loki_logger.error(f"Redis operation {operation} failed: {error}", **log_data)
        else:
            self.loki_logger.info(f"Redis operation {operation} completed", **log_data)


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
