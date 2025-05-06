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

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

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
                
                # Initialize ChromaDB client with HTTP client
                # Assuming the ChromaDB server is running on localhost:8000
                self.client = chromadb.HttpClient(
                    host="localhost",
                    port=8000,
                    settings=Settings(anonymized_telemetry=False) # Disable telemetry
                )
                
                # Measure initialization time
                client_init_time = (time.time() - client_init_start) * 1000  # ms
                self.logger.timing("vector_store_mcp.chromadb_init_time_ms", client_init_time)
                
                # Update health status
                self._health_status["status"] = "healthy"
                
                # Log initialization success with context
                self.logger.info("ChromaDB client initialized", 
                               host="localhost",
                               port=8000,
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
                self.register_tool(self.create_collection, description="Create a new collection.")
                self.register_tool(self.delete_collection, description="Delete a collection.")
                
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
                               "create_collection", description="Create a new collection (ChromaDB required).")
            self.register_tool(lambda *args, **kwargs: {"status": "error", "error": "ChromaDB not available."}, 
                               "delete_collection", description="Delete a collection (ChromaDB required).")
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

    # Placeholder methods for registered tools and internal helpers
    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Placeholder for adding documents to a collection."""
        self.logger.info(f"Placeholder: add_documents called for collection {collection_name} with {len(documents)} documents")
        # Increment counter
        self.add_documents_count += len(documents)
        self.total_vectors_added += sum(len(doc.get("embeddings", [])) for doc in documents if "embeddings" in doc) # Assuming embeddings are lists
        self.logger.counter("vector_store_mcp.add_documents_count", len(documents))
        self.logger.gauge("vector_store_mcp.total_vectors", self.total_vectors_added)
        return {"status": "success", "message": "add_documents placeholder executed"}

    def search_collection(self, collection_name: str, query_embeddings: List[List[float]], n_results: int = 10) -> Dict[str, Any]:
        """Placeholder for searching a collection."""
        self.logger.info(f"Placeholder: search_collection called for collection {collection_name} with {len(query_embeddings)} queries")
        # Increment counter
        self.search_collection_count += len(query_embeddings)
        self.total_queries_executed += len(query_embeddings)
        self.logger.counter("vector_store_mcp.search_collection_count", len(query_embeddings))
        return {"status": "success", "results": [], "processing_time_ms": 0.0}

    def list_collections(self) -> Dict[str, Any]:
        """Placeholder for listing collections."""
        self.logger.info("Placeholder: list_collections called")
        # Increment counter
        self.list_collections_count += 1
        self.logger.counter("vector_store_mcp.list_collections_count")
        return {"status": "success", "collections": []}

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Placeholder for getting collection info."""
        self.logger.info(f"Placeholder: get_collection_info called for collection {collection_name}")
        # Increment counter
        self.get_collection_info_count += 1
        self.logger.counter("vector_store_mcp.get_collection_info_count")
        return {"status": "success", "name": collection_name, "id": "placeholder_id", "count": 0}

    def create_collection(self, collection_name: str) -> Dict[str, Any]:
        """Placeholder for creating a collection."""
        self.logger.info(f"Placeholder: create_collection called for collection {collection_name}")
        return {"status": "success", "message": f"Collection '{collection_name}' placeholder created"}

    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Placeholder for deleting a collection."""
        self.logger.info(f"Placeholder: delete_collection called for collection {collection_name}")
        return {"status": "success", "message": f"Collection '{collection_name}' placeholder deleted"}

    def get_health_status(self) -> Dict[str, Any]:
        """Placeholder for getting health status."""
        self.logger.info("Placeholder: get_health_status called")
        # Update health status metrics
        self._update_health_status()
        return {"status": self._health_status["status"], "models_loaded": self._health_status["models_loaded"], "metrics": self._health_status["metrics"]}

    def clear_cache(self) -> Dict[str, Any]:
        """Placeholder for clearing cache."""
        self.logger.info("Placeholder: clear_cache called")
        if self.enable_cache:
            with self.cache_lock:
                count = len(self.results_cache)
                self.results_cache.clear()
                self.logger.info(f"Cleared {count} items from cache")
                return {"status": "success", "items_removed": count, "remaining_items": 0}
        else:
            self.logger.info("Cache is not enabled, nothing to clear")
            return {"status": "success", "message": "Cache is not enabled"}

    def generate_performance_report(self) -> Dict[str, Any]:
        """Placeholder for generating performance report."""
        self.logger.info("Placeholder: generate_performance_report called")
        # Calculate average response time if data exists
        avg_response_time = self.total_processing_time_ms / (self.add_documents_count + self.search_collection_count) if (self.add_documents_count + self.search_collection_count) > 0 else 0
        return {
            "status": "success",
            "report": {
                "add_documents_count": self.add_documents_count,
                "search_collection_count": self.search_collection_count,
                "total_vectors_added": self.total_vectors_added,
                "total_queries_executed": self.total_queries_executed,
                "execution_errors": self.execution_errors,
                "total_processing_time_ms": self.total_processing_time_ms,
                "average_processing_time_ms": avg_response_time,
                "cache_hits": self.cache_hits if self.enable_cache else "N/A",
                "cache_misses": self.cache_misses if self.enable_cache else "N/A",
            }
        }

    def _check_health(self):
        """Placeholder for internal health check logic."""
        self.logger.debug("Placeholder: _check_health executed")
        # In a real implementation, this would check connection to ChromaDB, etc.
        # For now, just update metrics based on internal counters
        self._health_status["metrics"]["collections_count"] = len(self.client.list_collections()) if self.client else 0
        self._health_status["metrics"]["total_vectors"] = self.total_vectors_added # Simple placeholder
        self._health_status["metrics"]["avg_search_time_ms"] = 0 # Placeholder
        self._health_status["last_checked"] = datetime.now().isoformat()
        self.logger.gauge("vector_store_mcp.health_status", 1 if self._health_status["status"] == "healthy" else 0)
        self.logger.gauge("vector_store_mcp.total_vectors", self._health_status["metrics"]["total_vectors"])
        self.logger.gauge("vector_store_mcp.error_rate", (self.execution_errors / self.request_count) * 100 if self.request_count > 0 else 0)


    def _log_detailed_health_metrics(self):
        """Placeholder for logging detailed health metrics."""
        self.logger.debug("Placeholder: _log_detailed_health_metrics executed")
        # Log current state of counters and metrics
        self.logger.info("Vector Store MCP Detailed Health Metrics",
                       status=self._health_status["status"],
                       collections_count=self._health_status["metrics"]["collections_count"],
                       total_vectors=self._health_status["metrics"]["total_vectors"],
                       add_documents_count=self.add_documents_count,
                       search_collection_count=self.search_collection_count,
                       execution_errors=self.execution_errors,
                       cache_hits=self.cache_hits if self.enable_cache else "N/A",
                       cache_misses=self.cache_misses if self.enable_cache else "N/A")


    def _generate_performance_charts(self):
        """Placeholder for generating performance charts."""
        self.logger.debug("Placeholder: _generate_performance_charts executed")
        # In a real implementation, this would use self.chart_generator to create charts
        pass
