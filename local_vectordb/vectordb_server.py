"""
Vector Database Server Module

This module implements a vector database server with integrated monitoring using the consolidated
monitoring module that provides both Loki logging and Prometheus metrics.
It provides a centralized Vector DB instance for the NextGen system based on ChromaDB.
"""

import os
import time
import re
import json
import logging
import threading
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Import ChromaDB - the vector database we'll be using
try:
    import chromadb
    HAVE_CHROMA = True
except ImportError:
    HAVE_CHROMA = False
    chromadb = None

# Import will be done dynamically in _setup_monitoring to handle import errors gracefully
# from monitoring.netdata_logger import NetdataLogger


class VectorDBServer:
    """
    Vector Database Server with integrated monitoring.

    This class provides a centralized vector database instance for the NextGen system,
    with integrated logging and metrics using the consolidated monitoring module.
    """

    _instance = None  # Singleton instance

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None):
        """
        Get or create the singleton instance of VectorDBServer.

        Args:
            config: Optional configuration dictionary to override environment variables

        Returns:
            VectorDBServer instance
        """
        if cls._instance is None:
            cls._instance = VectorDBServer(config)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Vector DB server with integrated monitoring.

        Args:
            config: Optional configuration dictionary to override environment variables
        """
        # Load environment variables
        load_dotenv()

        # Initialize logging first for early debugging
        self.logger = logging.getLogger("vectordb_server")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Load configuration from file if no config provided
        if config is None:
            config_path = os.path.join("config", "local_vectordb", "vectordb_server_config.json")
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

        # Initialize configuration values
        connection_config = self.config.get("connection", {})
        server_settings = self.config.get("server_settings", {})
        
        # Extract connection parameters with environment variable fallbacks
        self.db_path = self._get_config_value(connection_config.get("db_path", "${VECTORDB_PATH:./vector_db_storage}"))
        self.host = self._get_config_value(connection_config.get("host", "${VECTORDB_HOST:localhost}"))
        self.port = int(self._get_config_value(connection_config.get("port", "${VECTORDB_PORT:8000}")))
        self.use_http_server = self._get_config_value(connection_config.get("use_http_server", "${VECTORDB_USE_HTTP_SERVER:false}")).lower() == "true"
        
        # Initialize the default collection name (MUST come before _initialize_db)
        self.default_collection_name = server_settings.get("default_collection", "financial_context")

        # Initialize Vector DB client
        self.client = self._initialize_db()
        self.collections = {}

        # Register metrics
        if self.monitor:
            self._register_metrics()
            self._start_metrics_collection()

        self.logger.info("Vector DB server initialized with integrated monitoring")
        if self.monitor:
            self.monitor.info(
                "Vector DB server initialized", 
                component="vectordb_server", 
                action="initialization"
            )

    def _setup_monitoring(self):
        """
        Set up monitoring for Vector DB server using NetdataLogger.

        Returns:
            Tuple of (NetdataLogger instance, metrics_dict)
        """
        try:
            # Get configuration from environment variables or config
            monitoring_config = self.config.get("monitoring", {})
            log_dir = monitoring_config.get("log_dir") or os.getenv("LOG_DIR")
            statsd_host = monitoring_config.get("statsd_host") or os.getenv("STATSD_HOST", "localhost")
            statsd_port = int(monitoring_config.get("statsd_port") or os.getenv("STATSD_PORT", "8125"))
            
            # Check if NetdataLogger is properly imported
            try:
                from monitoring.netdata_logger import NetdataLogger
                
                # Initialize the NetdataLogger
                monitor = NetdataLogger(
                    component_name="vectordb-server",
                    log_dir=log_dir,
                    statsd_host=statsd_host,
                    statsd_port=statsd_port
                )
                
                # Test the logger
                monitor.info("VectorDB monitoring initialized")
                self.logger.info("NetdataLogger initialized successfully")
                
                # Define a metrics dictionary for consistent metric naming
                metrics = {
                    'collection_count': "collection_count",
                    'embeddings_count': "embeddings_count",
                    'operations_total': "operations_total",
                    'operation_duration_seconds': "operation_duration_seconds"
                }
                
                # Start system metrics collection
                try:
                    monitor.start_system_metrics(interval=15)
                    self.logger.info("System metrics collection started")
                except Exception as e:
                    self.logger.warning(f"Failed to start system metrics collection: {e}")
                
                return monitor, metrics
                
            except ImportError:
                self.logger.warning("NetdataLogger not found, falling back to dummy monitor")
                
                # Create a dummy monitor in case NetdataLogger is not available
                class DummyMonitor:
                    def __getattr__(self, name):
                        return lambda *args, **kwargs: None
                
                return DummyMonitor(), {}
                
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring: {e}")
            
            # Create a dummy monitor in case of any failure
            class DummyMonitor:
                def __getattr__(self, name):
                    return lambda *args, **kwargs: None
                
            return DummyMonitor(), {}

    def _initialize_db(self):
        """
        Initialize the Vector Database client.

        Returns:
            ChromaDB client instance
        """
        if not HAVE_CHROMA:
            self.logger.error(
                "ChromaDB library not found. Install with: pip install chromadb"
            )
            return None

        try:
            # Ensure db_path directory exists
            os.makedirs(self.db_path, exist_ok=True)
            
            self.logger.info(f"Initializing vector database at {self.db_path}")
            
            # Create client based on configuration
            if self.use_http_server:
                # HTTP client (connecting to a ChromaDB server)
                client = chromadb.HttpClient(host=self.host, port=self.port)
                self.logger.info(f"Using HTTP client to connect to ChromaDB at {self.host}:{self.port}")
            else:
                # Persistent client (local storage)
                client = chromadb.PersistentClient(path=self.db_path)
                self.logger.info(f"Using persistent client at {self.db_path}")
            
            # Test connection by getting or creating the default collection
            self._get_collection(client, self.default_collection_name)
            
            if self.monitor:
                self.monitor.info(
                    "Vector DB connection established",
                    component="vectordb_server",
                    action="connection",
                    path=self.db_path
                )
            
            return client
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector database: {e}")
            if self.monitor:
                self.monitor.error(
                    f"Failed to initialize vector database: {e}",
                    component="vectordb_server",
                    action="connection_error", 
                    error=str(e)
                )
            
            # Return a dummy client for testing/development
            self.logger.warning("Using dummy ChromaDB client for testing/development")
            return None

    def _get_collection(self, client, name: str):
        """Get or create a collection in ChromaDB."""
        if not client:
            return None
            
        try:
            # Initialize collections dict if it doesn't exist
            if not hasattr(self, 'collections'):
                self.collections = {}
                
            # If collection exists in cache, return it
            if name in self.collections:
                return self.collections[name]
                
            # Otherwise get or create it
            collection = client.get_or_create_collection(name)
            self.collections[name] = collection
            
            self.logger.info(f"Accessed collection: {name}")
            return collection
        except Exception as e:
            self.logger.error(f"Error accessing collection '{name}': {e}")
            return None

    def _register_metrics(self):
        """Register metrics for Vector DB monitoring."""
        if not self.monitor:
            return

        # Define the metrics dictionary for consistent naming
        # NetdataLogger doesn't require pre-registration, but we use this for naming consistency
        self.metrics['collection_count'] = "collection_count"
        self.metrics['embeddings_count'] = "embeddings_count"
        self.metrics['operations_total'] = "operations_total"
        self.metrics['operation_duration_seconds'] = "operation_duration_seconds"
        
        # Log that metrics have been registered
        self.logger.info("VectorDB metrics registered")

    def _start_metrics_collection(self):
        """Start a background thread to collect Vector DB metrics."""
        if not self.client or not self.monitor:
            return

        def collect_metrics():
            # Track monitoring failures to avoid excessive error logging
            monitoring_failures = 0
            max_failures = 3  # After this many failures, stop trying to use monitoring
            
            while True:
                try:
                    # Get all collections
                    collections = self.client.list_collections()
                    collection_count = len(collections)
                    
                    # Log to file logger - this always works
                    self.logger.info(f"Vector DB metrics collected - {collection_count} collections")
                    
                    # Only try to use monitoring if we haven't had too many failures
                    if monitoring_failures < max_failures:
                        try:
                            # Update collection count metric using NetdataLogger gauge method
                            self.monitor.gauge("collection_count", collection_count)
                            
                            # Update embeddings count for each collection
                            total_embeddings = 0
                            for collection in collections:
                                try:
                                    # Get collection by name
                                    coll = self._get_collection(self.client, collection.name)
                                    if coll:
                                        # Get count of items in collection
                                        count = coll.count()
                                        total_embeddings += count
                                        
                                        # Record per-collection embedding count
                                        self.monitor.gauge(
                                            f"embeddings_count_{collection.name}", 
                                            count
                                        )
                                except Exception as e:
                                    # Only log to standard logger, not monitoring
                                    self.logger.error(f"Error getting count for collection {collection.name}: {e}")
                            
                            # Record total embeddings across all collections
                            self.monitor.gauge("embeddings_total", total_embeddings)
                            
                            # Log metrics collection success
                            self.monitor.info(
                                "Vector DB metrics collected",
                                component="vectordb_server",
                                action="metrics_collection",
                                collection_count=collection_count,
                                total_embeddings=total_embeddings
                            )
                            
                        except Exception as e:
                            monitoring_failures += 1
                            self.logger.error(f"Error updating monitoring metrics: {e}")
                            if monitoring_failures >= max_failures:
                                self.logger.warning("Too many monitoring failures, disabling monitoring metrics")
                
                except Exception as e:
                    self.logger.error(f"Error collecting vector DB metrics: {e}")

                # Sleep for 15 seconds
                time.sleep(15)

        # Start metrics collection thread
        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()
        self.logger.info("Vector DB metrics collection started")

    def get_client(self):
        """
        Get the ChromaDB client instance.

        Returns:
            ChromaDB client instance
        """
        return self.client
        
    def get_collection(self, name: str):
        """
        Get a specific collection from the vector database.
        
        Args:
            name: Name of the collection
            
        Returns:
            ChromaDB collection
        """
        return self._get_collection(self.client, name)
        
    def list_collections(self):
        """
        List all collections in the vector database.
        
        Returns:
            List of collection names
        """
        if not self.client:
            return []
            
        try:
            collections = self.client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
            
    def record_operation(self, operation: str, duration: float, status: str = "success"):
        """
        Record a Vector DB operation for metrics.

        Args:
            operation: Name of the operation (e.g., "add", "query")
            duration: Duration of the operation in seconds
            status: Status of the operation ("success" or "error")
        """
        if not self.monitor:
            return

        try:
            # Increment operation counter - NetdataLogger uses counter() method
            self.monitor.counter(
                f"{self.metrics.get('operations_total', 'operations_total')}_{status}",
                1
            )
            
            # Record operation duration - NetdataLogger uses timing() for durations
            # Convert seconds to milliseconds for timing method
            self.monitor.timing(
                f"{self.metrics.get('operation_duration_seconds', 'operation_duration_seconds')}_{operation}",
                duration * 1000  # Convert to milliseconds
            )
            
            # Also record as histogram if needed for distribution stats
            self.monitor.histogram(
                f"vectordb_{operation}_duration",
                duration * 1000  # Convert to milliseconds
            )
        except Exception as e:
            self.logger.error(f"Error recording operation metrics: {e}")

    def log_operation(self, operation: str, collection: str, status: str, error: Optional[str] = None):
        """
        Log a Vector DB operation.

        Args:
            operation: Name of the operation (e.g., "add", "query")
            collection: Collection being operated on
            status: Status of the operation ("success" or "error")
            error: Error message if status is "error"
        """
        if not self.monitor:
            return

        try:
            # Create structured log data
            log_data = {
                "component": "vectordb_server",
                "action": operation,
                "collection": collection,
                "status": status
            }

            if error:
                log_data["error"] = error
                # NetdataLogger uses error() method
                self.monitor.error(
                    f"Vector DB operation {operation} failed", 
                    **log_data
                )
            else:
                # NetdataLogger uses info() method
                self.monitor.info(
                    f"Vector DB operation {operation} completed", 
                    **log_data
                )
        except Exception as e:
            self.logger.error(f"Error logging operation: {e}")
            
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
    # Get Vector DB server instance
    vector_db_server = VectorDBServer.get_instance()

    # Get Vector DB client
    client = vector_db_server.get_client()

    if client:
        # Perform some operations
        start_time = time.time()
        try:
            # Create a test collection
            test_collection = vector_db_server.get_collection("test_collection")
            
            # Add some test data
            test_collection.add(
                documents=["This is a test document", "Another test document"],
                metadatas=[{"source": "test"}, {"source": "test"}],
                ids=["id1", "id2"]
            )
            
            vector_db_server.record_operation("add", time.time() - start_time)
            vector_db_server.log_operation("add", "test_collection", "success")

            # Query the collection
            start_time = time.time()
            results = test_collection.query(
                query_texts=["test document"], 
                n_results=2
            )
            
            vector_db_server.record_operation("query", time.time() - start_time)
            vector_db_server.log_operation("query", "test_collection", "success")

            print(f"Query results: {results}")
            print(f"Collections: {vector_db_server.list_collections()}")
            
        except Exception as e:
            vector_db_server.record_operation("operation", time.time() - start_time, "error")
            vector_db_server.log_operation("operation", "test_collection", "error", str(e))
            print(f"Error: {e}")

    # Keep the server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Vector DB server shutting down")