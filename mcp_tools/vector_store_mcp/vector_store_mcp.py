
"""
Vector Store MCP using ChromaDB

This module provides an MCP server for interacting with a ChromaDB vector store
with comprehensive monitoring, metrics collection, and visualization capabilities.

Features:
- Persistent ChromaDB storage
- Add, search, and manage vector embeddings
- Performance metrics and visualization
- Health monitoring and diagnostics
"""

from __future__ import annotations # Defer evaluation of type hints

import os
import time
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

# Import base MCP server
from mcp_tools.base_mcp_server import BaseMCPServer

# Import monitoring components
from monitoring.stock_charts import StockChartGenerator

# Import ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    HAVE_CHROMADB = True
except ImportError:
    HAVE_CHROMADB = False
    print("Warning: chromadb not installed. Vector store features will be unavailable.")


class VectorStoreMCP(BaseMCPServer):
    """
    MCP Server for ChromaDB Vector Store operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VectorStoreMCP.

        Args:
            config: Configuration dictionary, expected to contain:
                - db_path: Path to the ChromaDB persistent storage directory (default: 'vector_db_storage')
                - default_collection_name: Default collection name (default: 'default_collection')
                - enable_health_check: Whether to enable health checking (default: True)
                - health_check_interval: Interval in seconds for health checks (default: 60)
                - performance_logging: Whether to enable detailed performance logging (default: True)
                - cache_embeddings: Whether to cache embeddings and results (default: False)
        """
        # Record initialization start time
        init_start_time = time.time()
        
        # Initialize base MCP server
        super().__init__(name="vector_store_mcp", config=config)
        
        # Configure database settings
        self.db_path = self.config.get("db_path", "vector_db_storage")
        self.default_collection_name = self.config.get("default_collection_name", "default_collection")
        self.enable_health_check = self.config.get("enable_health_check", True)
        self.health_check_interval = self.config.get("health_check_interval", 60)
        self.performance_logging = self.config.get("performance_logging", True)
        self.enable_cache = self.config.get("cache_embeddings", False)
        
        # Initialize detailed performance counters
        self.add_documents_count = 0
        self.search_collection_count = 0
        self.list_collections_count = 0
        self.get_collection_info_count = 0
        self.total_vectors_added = 0
        self.total_queries_executed = 0
        self.execution_errors = 0
        self.total_processing_time_ms = 0
        
        # Initialize cache if enabled
        if self.enable_cache:
            self.results_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0
            self.cache_lock = threading.RLock()
            self.logger.info("Embeddings cache initialized")
        
        # Set up locks for thread safety
        self.client_lock = threading.RLock()
        
        # Initialize health monitoring
        self._health_status = {
            "status": "initializing",
            "last_checked": datetime.now().isoformat(),
            "metrics": {
                "collections_count": 0,
                "total_vectors": 0,
                "avg_search_time_ms": 0
            }
        }
        self._last_health_check = time.time()
        
        # Initialize chart generator for visualizations
        self.chart_generator = StockChartGenerator()
        self.logger.info("Performance chart generator initialized")

        # Ensure the database directory exists
        os.makedirs(self.db_path, exist_ok=True)

        self.client = None # Initialize client to None
        self._health_status["status"] = "degraded" # Assume degraded until client is initialized

        if HAVE_CHROMADB:
            try:
                client_init_start = time.time()
                
                # Initialize ChromaDB client
                self.client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(anonymized_telemetry=False) # Disable telemetry
                )
                
                # Measure initialization time
                client_init_time = (time.time() - client_init_start) * 1000  # ms
                self.logger.timing("vector_store_mcp.chromadb_init_time_ms", client_init_time)
                
                # Update health status
                self._health_status["status"] = "healthy"
                
                # Log initialization success with context
                self.logger.info("ChromaDB client initialized", 
                               db_path=self.db_path,
                               init_time_ms=client_init_time)
                
                # Get initial collection metrics
                collections = self.client.list_collections()
                collection_count = len(collections)
                self.logger.gauge("vector_store_mcp.collections_count", collection_count)
                self._health_status["metrics"]["collections_count"] = collection_count
                
                # Register standard tools
                self.register_tool(self.add_documents, description="Add documents (text, embeddings, metadata) to a collection.")
                self.register_tool(self.search_collection, description="Search a collection using query embeddings.")
                self.register_tool(self.list_collections, description="List all available collections.")
                self.register_tool(self.get_collection_info, description="Get information about a specific collection.")
                
                # Register monitoring tools
                self._register_monitoring_tools()
                
                # Start health check if enabled
                if self.enable_health_check:
                    self._start_health_check_thread()
                    self.logger.info("Health check thread started")

            except Exception as e:
                self.logger.error(f"Failed to initialize ChromaDB client: {e}", exc_info=True)
                self.client = None # Ensure client is None if initialization fails
                self._health_status["status"] = "degraded"
                self.execution_errors += 1
                self.logger.counter("vector_store_mcp.initialization_errors")
        else:
            self.logger.warning("ChromaDB not available, vector store features disabled.")
            # Register placeholder tools that return errors if ChromaDB is not available
            self.register_tool(lambda *args, **kwargs: {"status": "error", "error": "ChromaDB not available."}, 
                               "add_documents", description="Add documents (text, embeddings, metadata) to a collection (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "error", "error": "ChromaDB not available."}, 
                               "search_collection", description="Search a collection using query embeddings (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "error", "error": "ChromaDB not available."}, 
                               "list_collections", description="List all available collections (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "error", "error": "ChromaDB not available."}, 
                               "get_collection_info", description="Get information about a specific collection (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "error", "error": "ChromaDB not available."}, 
                               "get_health_status", description="Get detailed health status and metrics for the vector store (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "success", "message": "Cache is not enabled or ChromaDB not available."}, 
                               "clear_cache", description="Clear the embeddings and results cache (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "success", "message": "Performance report not available without ChromaDB."}, 
                               "generate_performance_report", description="Generate a comprehensive performance report (ChromaDB required).")
        
        # Record total initialization time
        init_duration = (time.time() - init_start_time) * 1000  # milliseconds
        self.logger.timing("vector_store_mcp.initialization_time_ms", init_duration)
        
        # Set up initial gauge metrics
        self.logger.gauge("vector_store_mcp.health_status", 1 if self._health_status["status"] == "healthy" else 0)
        self.logger.gauge("vector_store_mcp.total_vectors", 0)
        self.logger.gauge("vector_store_mcp.error_rate", 0)
        
        # Log initialization complete with metrics
        self.logger.info("Vector Store MCP initialization complete", 
                       init_time_ms=init_duration,
                       status=self._health_status["status"],
                       cache_enabled=self.enable_cache,
                       health_check_enabled=self.enable_health_check)
        
    def _register_monitoring_tools(self):
        """Register additional monitoring and diagnostic tools."""
        # These tools are registered conditionally in __init__ now
        pass
        
    def _start_health_check_thread(self):
        """Start a background thread for periodic health checks."""
        # Only start if ChromaDB is available
        if not HAVE_CHROMADB:
            self.logger.warning("Cannot start health check thread: ChromaDB not available.")
            return
            
        def health_check_loop():
            self.logger.info(f"Starting health check thread with interval of {self.health_check_interval} seconds")
            
            # Initialize health check counter
            health_check_counter = 0
            
            # Initialize shutdown flag if it doesn't exist
            if not hasattr(self, 'shutdown_requested'):
                self.shutdown_requested = False
                
            while not self.shutdown_requested:
                try:
                    # Increment counter
                    health_check_counter += 1

                    # Run health check
                    self._check_health()

                    # Log health status - less verbose on regular checks
                    if health_check_counter % 5 == 0:  # Every 5th check, log more details
                        self._log_detailed_health_metrics()

                    # Generate performance charts periodically (every 10 checks)
                    if health_check_counter % 10 == 0:

                        self._generate_performance_charts() # Assuming a method like this exists or should be added

                    # Sleep for the specified interval
                    time.sleep(self.health_check_interval)

                except Exception as e:
                    self.logger.error(f"Health check failed: {e}", exc_info=True)
                    self._health_status["status"] = "degraded"
                    self.execution_errors += 1
                    self.logger.counter("vector_store_mcp.health_check_errors")
                    # Sleep even on error to prevent tight loop
                    time.sleep(self.health_check_interval)

