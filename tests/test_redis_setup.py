#!/usr/bin/env python3
"""
Redis Setup Test Script

This script tests the Redis server setup and MCP integration to ensure everything works correctly.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Make sure we can import modules from the project
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("redis_test")

# Import the Redis MCP client
from mcp_tools.db_mcp.redis_mcp import RedisMCP

def test_redis_connection():
    """Test basic Redis connection."""
    logger.info("Testing Redis connection...")
    
    # Create Redis MCP client
    redis_mcp = RedisMCP()
    
    # Check if Redis client is initialized
    if not redis_mcp.redis_client:
        logger.error("Failed to initialize Redis client")
        return False
    
    # Test ping
    try:
        redis_mcp.redis_client.ping()
        logger.info("Redis server is responding to ping")
        return True
    except Exception as e:
        logger.error(f"Redis server is not responding to ping: {e}")
        return False

def test_redis_operations():
    """Test basic Redis operations."""
    logger.info("Testing Redis operations...")
    
    # Create Redis MCP client
    redis_mcp = RedisMCP()
    
    # Check if Redis client is initialized
    if not redis_mcp.redis_client:
        logger.error("Failed to initialize Redis client")
        return False
    
    # Test set operation
    test_key = "test:redis:setup"
    test_value = f"Test value at {time.time()}"
    
    try:
        # Set value
        set_result = redis_mcp.fetch_data("set", {"key": test_key, "value": test_value})
        if "error" in set_result:
            logger.error(f"Failed to set value: {set_result['error']}")
            return False
        logger.info(f"Successfully set value for key {test_key}")
        
        # Get value
        get_result = redis_mcp.fetch_data("get", {"key": test_key})
        if "error" in get_result:
            logger.error(f"Failed to get value: {get_result['error']}")
            return False
        
        retrieved_value = get_result.get("value")
        if retrieved_value != test_value:
            logger.error(f"Value mismatch: expected '{test_value}', got '{retrieved_value}'")
            return False
        logger.info(f"Successfully retrieved value for key {test_key}")
        
        # Delete value
        delete_result = redis_mcp.fetch_data("delete", {"key": test_key})
        if "error" in delete_result:
            logger.error(f"Failed to delete value: {delete_result['error']}")
            return False
        logger.info(f"Successfully deleted key {test_key}")
        
        return True
    except Exception as e:
        logger.error(f"Error during Redis operations: {e}")
        return False

def test_redis_json():
    """Test Redis JSON operations."""
    logger.info("Testing Redis JSON operations...")
    
    # Create Redis MCP client
    redis_mcp = RedisMCP()
    
    # Check if Redis client is initialized
    if not redis_mcp.redis_client:
        logger.error("Failed to initialize Redis client")
        return False
    
    # Test JSON operations
    test_key = "test:redis:json"
    test_data = {
        "name": "Redis Test",
        "timestamp": time.time(),
        "nested": {
            "field1": "value1",
            "field2": 42,
            "field3": [1, 2, 3, 4, 5]
        }
    }
    
    try:
        # Set JSON
        set_result = redis_mcp.fetch_data("json_set", {"key": test_key, "json_data": test_data})
        if "error" in set_result:
            logger.error(f"Failed to set JSON: {set_result['error']}")
            return False
        logger.info(f"Successfully set JSON for key {test_key}")
        
        # Get JSON
        get_result = redis_mcp.fetch_data("json_get", {"key": test_key})
        if "error" in get_result:
            logger.error(f"Failed to get JSON: {get_result['error']}")
            return False
        
        retrieved_data = get_result.get("data")
        if not retrieved_data or retrieved_data.get("name") != test_data["name"]:
            logger.error(f"JSON data mismatch: expected name '{test_data['name']}', got '{retrieved_data.get('name') if retrieved_data else 'None'}'")
            return False
        logger.info(f"Successfully retrieved JSON for key {test_key}")
        
        # Delete JSON
        delete_result = redis_mcp.fetch_data("delete", {"key": test_key})
        if "error" in delete_result:
            logger.error(f"Failed to delete JSON: {delete_result['error']}")
            return False
        logger.info(f"Successfully deleted JSON key {test_key}")
        
        return True
    except Exception as e:
        logger.error(f"Error during Redis JSON operations: {e}")
        return False

def test_redis_mcp_methods():
    """Test Redis MCP direct methods."""
    logger.info("Testing Redis MCP direct methods...")
    
    # Create Redis MCP client
    redis_mcp = RedisMCP()
    
    # Check if Redis client is initialized
    if not redis_mcp.redis_client:
        logger.error("Failed to initialize Redis client")
        return False
    
    # Test direct methods
    test_key = "test:redis:direct"
    test_value = f"Direct method test at {time.time()}"
    
    try:
        # Set value
        set_result = redis_mcp.set_value(test_key, test_value)
        if not set_result:
            logger.error("Failed to set value using direct method")
            return False
        logger.info(f"Successfully set value using direct method for key {test_key}")
        
        # Get value
        retrieved_value = redis_mcp.get_value(test_key)
        if retrieved_value != test_value:
            logger.error(f"Value mismatch using direct method: expected '{test_value}', got '{retrieved_value}'")
            return False
        logger.info(f"Successfully retrieved value using direct method for key {test_key}")
        
        # Delete value
        delete_result = redis_mcp.delete_value(test_key)
        if not delete_result:
            logger.error("Failed to delete value using direct method")
            return False
        logger.info(f"Successfully deleted key using direct method {test_key}")
        
        return True
    except Exception as e:
        logger.error(f"Error during Redis MCP direct methods: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting Redis setup tests...")
    
    # Test Redis connection
    connection_result = test_redis_connection()
    if not connection_result:
        logger.error("Redis connection test failed")
        return 1
    
    # Test Redis operations
    operations_result = test_redis_operations()
    if not operations_result:
        logger.error("Redis operations test failed")
        return 1
    
    # Test Redis JSON operations
    json_result = test_redis_json()
    if not json_result:
        logger.error("Redis JSON operations test failed")
        return 1
    
    # Test Redis MCP direct methods
    mcp_result = test_redis_mcp_methods()
    if not mcp_result:
        logger.error("Redis MCP direct methods test failed")
        return 1
    
    logger.info("All Redis tests passed successfully!")
    logger.info("Redis server is properly set up and integrated with the MCP client")
    return 0

if __name__ == "__main__":
    sys.exit(main())