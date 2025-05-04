#!/usr/bin/env python3
"""
Test Vector DB MCP Integration

This script demonstrates how to use the VectorDBMCP with our local vector database.
"""

import os
import sys
import time
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the VectorDBMCP
from mcp_tools.analysis_mcp.vector_db_mcp import VectorDBMCP

def test_vectordb_mcp():
    """Test the VectorDBMCP with our local vector database."""
    print("\n=== Testing VectorDBMCP with Local Vector Database ===\n")
    
    # Configure to use local ChromaDB
    config = {
        "db_type": "chroma",
        "db_path": "./vector_db_storage",
        "enable_prometheus": False,
        "enable_loki": False
    }
    
    # Initialize VectorDBMCP
    print("1. Initializing VectorDBMCP...")
    vector_db_mcp = VectorDBMCP(config)
    
    # List available collections
    print("\n2. Listing available collections...")
    collections_result = vector_db_mcp.list_collections()
    collections = collections_result.get("collections", [])
    print(f"Found {len(collections)} collections: {collections}")
    
    # Create a test collection with timestamp to make it unique
    test_collection = f"test_collection_{int(time.time())}"
    
    print(f"\n3. Working with collection: {test_collection}")
    
    # Create test embeddings (3-dimensional for simplicity)
    embeddings = [
        [0.1, 0.2, 0.3],  # Document 1
        [0.4, 0.5, 0.6],  # Document 2
        [0.7, 0.8, 0.9],  # Document 3
    ]
    
    # Create test metadata
    metadata = [
        {"source": "test", "category": "finance", "importance": "high"},
        {"source": "test", "category": "technology", "importance": "medium"},
        {"source": "test", "category": "health", "importance": "low"},
    ]
    
    # Create test IDs
    ids = ["doc1", "doc2", "doc3"]
    
    # Add the embeddings
    print("\n4. Adding embeddings to the collection...")
    add_result = vector_db_mcp.add_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        ids=ids,
        collection_name=test_collection
    )
    print(f"Add result: {add_result}")
    
    # Query similar embeddings
    print("\n5. Querying for similar embeddings...")
    query_embedding = [0.2, 0.3, 0.4]  # More similar to the first document
    
    query_result = vector_db_mcp.query_similar(
        query_embeddings=[query_embedding],
        n_results=3,
        collection_name=test_collection,
        include=["metadatas", "documents", "distances"]
    )
    
    print(f"Query result: {query_result}")
    
    # Get document by ID
    print("\n6. Getting document by ID...")
    get_result = vector_db_mcp.get_embedding(
        ids=["doc1"],
        collection_name=test_collection,
        include=["metadatas", "embeddings"]
    )
    
    print(f"Get result: {get_result}")
    
    # Test delete
    print("\n7. Deleting document...")
    delete_result = vector_db_mcp.delete_embeddings(
        ids=["doc3"],
        collection_name=test_collection
    )
    
    print(f"Delete result: {delete_result}")
    
    # List collections again to confirm
    print("\n8. Listing collections again...")
    collections_result = vector_db_mcp.list_collections()
    collections = collections_result.get("collections", [])
    print(f"All collections: {collections}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_vectordb_mcp()