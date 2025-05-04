#!/usr/bin/env python3
"""
Test Vector Database Setup

This script tests the Vector Database setup by connecting to it and
performing basic operations. It can be used to verify that the
vector database is properly installed and functioning.
"""

import os
import sys
import time
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the Vector DB server
from local_vectordb.vectordb_server import VectorDBServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vectordb_test")

def test_connection():
    """Test basic connection to the Vector Database."""
    logger.info("Testing connection to Vector Database...")
    
    try:
        # Get Vector DB server instance
        vector_db_server = VectorDBServer.get_instance()
        
        # Get Vector DB client
        client = vector_db_server.get_client()
        
        if client:
            logger.info("✓ Successfully connected to Vector Database")
            return True, client
        else:
            logger.error("✗ Failed to get Vector DB client")
            return False, None
    except Exception as e:
        logger.error(f"✗ Error connecting to Vector Database: {e}")
        return False, None

def test_collections(client):
    """Test creating and listing collections."""
    logger.info("Testing collections functionality...")
    
    try:
        # Get Vector DB server instance
        vector_db_server = VectorDBServer.get_instance()
        
        # List collections
        collections = vector_db_server.list_collections()
        logger.info(f"Existing collections: {collections}")
        
        # Create test collection
        test_collection_name = f"test_collection_{int(time.time())}"
        test_collection = vector_db_server.get_collection(test_collection_name)
        
        # Verify collection was created
        updated_collections = vector_db_server.list_collections()
        if test_collection_name in updated_collections:
            logger.info(f"✓ Successfully created collection: {test_collection_name}")
            return True, test_collection
        else:
            logger.error(f"✗ Failed to create collection: {test_collection_name}")
            return False, None
    except Exception as e:
        logger.error(f"✗ Error testing collections: {e}")
        return False, None

def test_embeddings(collection):
    """Test adding and querying embeddings."""
    logger.info("Testing embeddings functionality...")
    
    try:
        # Create some test data
        test_docs = [
            "This is the first test document",
            "Here is another test document",
            "And a third test document with different content"
        ]
        test_metadata = [
            {"source": "test", "id": 1},
            {"source": "test", "id": 2},
            {"source": "test", "id": 3}
        ]
        test_ids = [f"id{i+1}" for i in range(len(test_docs))]
        
        # Add embeddings
        collection.add(
            documents=test_docs,
            metadatas=test_metadata,
            ids=test_ids
        )
        
        logger.info(f"✓ Successfully added {len(test_docs)} embeddings")
        
        # Query the collection
        results = collection.query(
            query_texts=["test document"],
            n_results=2
        )
        
        if results and len(results.get('ids', [])[0]) > 0:
            logger.info(f"✓ Successfully queried collection, found {len(results.get('ids', [])[0])} results")
            logger.info(f"Query results: {results}")
            return True
        else:
            logger.error("✗ Query returned no results")
            return False
    except Exception as e:
        logger.error(f"✗ Error testing embeddings: {e}")
        return False

def test_cleanup(collection_name):
    """Clean up test resources."""
    logger.info("Cleaning up test resources...")
    
    try:
        # Get Vector DB server instance
        vector_db_server = VectorDBServer.get_instance()
        
        # Get Vector DB client
        client = vector_db_server.get_client()
        
        # Delete test collection
        if client:
            client.delete_collection(collection_name)
            
            # Verify collection was deleted
            updated_collections = vector_db_server.list_collections()
            if collection_name not in updated_collections:
                logger.info(f"✓ Successfully deleted collection: {collection_name}")
                return True
            else:
                logger.warning(f"⚠ Failed to delete collection: {collection_name}")
                return False
    except Exception as e:
        logger.warning(f"⚠ Error during cleanup: {e}")
        return False

def run_tests():
    """Run all tests."""
    start_time = time.time()
    test_results = {}
    
    # Test 1: Connection
    logger.info("=== Test 1: Connection ===")
    connection_success, client = test_connection()
    test_results["connection"] = connection_success
    
    if not connection_success:
        logger.error("Cannot proceed with further tests without connection")
        return test_results
    
    # Test 2: Collections
    logger.info("\n=== Test 2: Collections ===")
    collections_success, collection = test_collections(client)
    test_results["collections"] = collections_success
    
    if not collections_success:
        logger.error("Cannot proceed with further tests without collections")
        return test_results
    
    # Test 3: Embeddings
    logger.info("\n=== Test 3: Embeddings ===")
    embeddings_success = test_embeddings(collection)
    test_results["embeddings"] = embeddings_success
    
    # Clean up
    logger.info("\n=== Cleanup ===")
    collection_name = collection.name if collection else None
    if collection_name:
        cleanup_success = test_cleanup(collection_name)
        test_results["cleanup"] = cleanup_success
    
    # Summary
    duration = time.time() - start_time
    logger.info(f"\n=== Test Summary ({duration:.2f}s) ===")
    all_passed = all(test_results.values())
    
    for test, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {test}")
    
    if all_passed:
        logger.info("\n✓✓✓ All tests passed! The Vector Database is working correctly.")
    else:
        logger.error("\n✗✗✗ Some tests failed. Please check the logs for details.")
    
    return test_results

if __name__ == "__main__":
    logger.info(f"Starting Vector Database tests at {datetime.now().isoformat()}")
    results = run_tests()
    sys.exit(0 if all(results.values()) else 1)