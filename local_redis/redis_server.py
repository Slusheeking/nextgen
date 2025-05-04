"""
Redis Server Module

This module implements a Redis server with integrated monitoring using the consolidated
monitoring module that provides both Loki logging and Prometheus metrics.
It provides a centralized Redis instance for the NextGen FinGPT system.
"""

import os
import time
# Import the Redis client library with an alias to avoid conflict with our local redis package
import redis as redis_client
import logging
import threading
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import the consolidated monitoring module
from monitoring import setup_monitoring

class RedisServer:
    """
    Redis Server with integrated monitoring.
    
    This class provides a centralized Redis instance for the NextGen FinGPT system,
    with integrated logging and metrics using the consolidated monitoring module.
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
        
        # Store configuration
        self.config = config or {}
        
        # Set up logging
        self.logger = logging.getLogger("redis_server")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
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
            self.monitor.log_info("Redis server initialized", component="redis_server", action="initialization")
    
    def _setup_monitoring(self):
        """
        Set up monitoring for Redis server.
        
        Returns:
            Tuple of (MonitoringManager, metrics_dict)
        """
        try:
            # Get configuration from environment variables or config
            enable_prometheus = self.config.get("enable_prometheus", True)
            enable_loki = self.config.get("enable_loki", True)
            
            # Set up monitoring
            monitor, metrics = setup_monitoring(
                service_name="redis-server",
                enable_prometheus=enable_prometheus,
                enable_loki=enable_loki,
                default_labels={"component": "redis_server"}
            )
            
            self.logger.info("Monitoring initialized successfully")
            return monitor, metrics
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
            # Get Redis configuration from environment variables or config
            host = self.config.get("host") or os.getenv("REDIS_HOST", "localhost")
            port = int(self.config.get("port") or os.getenv("REDIS_PORT", "6379"))
            db = int(self.config.get("db") or os.getenv("REDIS_DB", "0"))
            password = self.config.get("password") or os.getenv("REDIS_PASSWORD")
            
            # Create Redis client
            client = redis_client.Redis(
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
            
            if self.monitor:
                self.monitor.log_info(
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
            if self.monitor:
                self.monitor.log_error(
                    f"Failed to initialize Redis client: {e}",
                    component="redis_server",
                    action="connection_error",
                    error=str(e)
                )
            
            # Return a dummy client for testing/development
            self.logger.warning("Using dummy Redis client for testing/development")
            return None
    
    def _register_metrics(self):
        """Register metrics for Redis monitoring."""
        if not self.monitor:
            return
        
        # Create metrics if they don't already exist
        if 'connected_clients' not in self.metrics:
            self.metrics['connected_clients'] = self.monitor.create_gauge(
                "connected_clients",
                "Number of client connections"
            )
        
        if 'used_memory' not in self.metrics:
            self.metrics['used_memory'] = self.monitor.create_gauge(
                "used_memory",
                "Used memory in bytes"
            )
        
        if 'total_commands_processed' not in self.metrics:
            self.metrics['total_commands_processed'] = self.monitor.create_gauge(
                "total_commands_processed",
                "Total number of commands processed"
            )
        
        if 'keyspace_hits' not in self.metrics:
            self.metrics['keyspace_hits'] = self.monitor.create_gauge(
                "keyspace_hits",
                "Number of successful lookups of keys in the main dictionary"
            )
        
        if 'keyspace_misses' not in self.metrics:
            self.metrics['keyspace_misses'] = self.monitor.create_gauge(
                "keyspace_misses",
                "Number of failed lookups of keys in the main dictionary"
            )
        
        if 'operations_total' not in self.metrics:
            self.metrics['operations_total'] = self.monitor.create_counter(
                "operations_total",
                "Total number of Redis operations",
                ["operation", "status"]
            )
        
        if 'operation_duration_seconds' not in self.metrics:
            self.metrics['operation_duration_seconds'] = self.monitor.create_histogram(
                "operation_duration_seconds",
                "Duration of Redis operations in seconds",
                ["operation"]
            )
    
    def _start_metrics_collection(self):
        """Start a background thread to collect Redis metrics."""
        if not self.monitor or not self.redis_client:
            return
        
        def collect_metrics():
            while True:
                try:
                    # Get Redis info
                    info = self.redis_client.info()
                    
                    # Update metrics
                    self.monitor.set_gauge("connected_clients", info.get("connected_clients", 0))
                    self.monitor.set_gauge("used_memory", info.get("used_memory", 0))
                    self.monitor.set_gauge("total_commands_processed", info.get("total_commands_processed", 0))
                    self.monitor.set_gauge("keyspace_hits", info.get("keyspace_hits", 0))
                    self.monitor.set_gauge("keyspace_misses", info.get("keyspace_misses", 0))
                    
                    # Log to monitoring
                    self.monitor.log_info(
                        "Redis metrics collected",
                        component="redis_server",
                        action="metrics_collection",
                        connected_clients=info.get("connected_clients", 0),
                        used_memory_human=info.get("used_memory_human", "N/A"),
                        total_commands_processed=info.get("total_commands_processed", 0)
                    )
                except Exception as e:
                    self.logger.error(f"Error collecting Redis metrics: {e}")
                    if self.monitor:
                        self.monitor.log_error(
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
    
    def get_client(self):
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
        if not self.monitor:
            return
        
        # Increment operation counter
        self.monitor.increment_counter("operations_total", 1, operation=operation, status=status)
        
        # Record operation duration
        self.monitor.observe_histogram("operation_duration_seconds", duration, operation=operation)
    
    def log_operation(self, operation: str, key: str, status: str, error: Optional[str] = None):
        """
        Log a Redis operation.
        
        Args:
            operation: Name of the operation (e.g., "get", "set")
            key: Redis key being operated on
            status: Status of the operation ("success" or "error")
            error: Error message if status is "error"
        """
        if not self.monitor:
            return
        
        log_data = {
            "component": "redis_server",
            "action": operation,
            "key": key,
            "status": status
        }
        
        if error:
            log_data["error"] = error
            self.monitor.log_error(f"Redis operation {operation} failed: {error}", **log_data)
        else:
            self.monitor.log_info(f"Redis operation {operation} completed", **log_data)


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
