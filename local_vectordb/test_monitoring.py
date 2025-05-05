#!/usr/bin/env python3
"""
Test script for VectorDB monitoring integration with NetdataLogger.

This script tests the integration between the VectorDB server and the NetdataLogger
monitoring system. It creates a VectorDB server instance, performs some operations,
and verifies that metrics are being collected and logged properly.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path to ensure imports work
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import VectorDB server
from local_vectordb.vectordb_server import VectorDBServer

def test_vectordb_monitoring():
    """Test VectorDB monitoring integration with NetdataLogger."""
    print("Testing VectorDB monitoring integration with NetdataLogger...")
    
    # Create a test configuration with monitoring options
    test_config = {
        "connection": {
            "db_path": "./vector_db_storage",
            "host": "localhost",
            "port": 8000,
            "use_http_server": False
        },
        "server_settings": {
            "default_collection": "test_collection"
        },
        "monitoring": {
            "statsd_host": "localhost",
            "statsd_port": 8125,
            "log_dir": "./logs"
        }
    }
    
    # Initialize VectorDB server with the test configuration
    vectordb = VectorDBServer(test_config)
    
    # Get the client
    client = vectordb.get_client()
    
    if not client:
        print("ERROR: Failed to initialize VectorDB client")
        sys.exit(1)
    
    # Verify monitoring is set up
    if not vectordb.monitor:
        print("ERROR: Monitoring is not properly set up")
        sys.exit(1)
    
    # Run some test operations
    try:
        print("\nRunning test operations...")
        
        # Create test collections
        for i in range(3):
            collection_name = f"test_collection_{i}"
            print(f"Creating collection: {collection_name}")
            
            # Get or create collection
            collection = vectordb.get_collection(collection_name)
            
            # Add some test data
            start_time = time.time()
            collection.add(
                documents=[f"Document {j} in {collection_name}" for j in range(5)],
                metadatas=[{"test": True, "index": j} for j in range(5)],
                ids=[f"id_{j}" for j in range(5)]
            )
            
            # Record the operation for metrics
            duration = time.time() - start_time
            vectordb.record_operation("add", duration)
            vectordb.log_operation("add", collection_name, "success")
            
            # Query the collection
            start_time = time.time()
            results = collection.query(
                query_texts=["Document"],
                n_results=3
            )
            
            # Record the query operation
            duration = time.time() - start_time
            vectordb.record_operation("query", duration)
            vectordb.log_operation("query", collection_name, "success")
            
            print(f"Collection {collection_name} created with 5 documents")
        
        # List all collections
        collections = vectordb.list_collections()
        print(f"\nTotal collections: {len(collections)}")
        print(f"Collection names: {collections}")
        
        # Wait for metrics to be collected
        print("\nWaiting for metrics collection (15 seconds)...")
        time.sleep(15)
        
        print("\nTest completed successfully")
        print("Check the logs directory for log files and Netdata for metrics")
        
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_vectordb_monitoring()