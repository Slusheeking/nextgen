#!/usr/bin/env python3
"""
Redis Setup Test Script

This script tests the Redis server setup by performing basic operations
and verifying that the Redis server is running and accessible.
"""

import os
import sys
import time
import logging
import json
import redis
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("redis_test")


def test_redis_connection():
    """Test basic Redis connection."""
    logger.info("Testing Redis connection...")

    try:
        # Get Redis configuration from environment variables
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        db = int(os.getenv("REDIS_DB", "0"))
        password = os.getenv("REDIS_PASSWORD")

        # Create Redis client
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5
        )

        # Test connection with ping
        response = client.ping()
        logger.info(f"Redis ping response: {response}")

        if response:
            logger.info(f"Successfully connected to Redis at {host}:{port}/{db}")
            return client
        else:
            logger.error("Redis ping failed")
            return None
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        return None


def test_basic_operations(client):
    """Test basic Redis operations."""
    if not client:
        logger.error("Cannot test operations: Redis client is None")
        return False

    logger.info("Testing basic Redis operations...")

    try:
        # Generate a unique test key
        test_key = f"test:redis:setup:{datetime.now().strftime('%Y%m%d%H%M%S')}"
        test_value = "Redis is working!"

        # Test SET operation
        client.set(test_key, test_value)
        logger.info(f"SET {test_key} = {test_value}")

        # Test GET operation
        retrieved_value = client.get(test_key)
        logger.info(f"GET {test_key} = {retrieved_value}")

        if retrieved_value != test_value:
            logger.error(f"Value mismatch: expected '{test_value}', got '{retrieved_value}'")
            return False

        # Test hash operations
        hash_key = f"{test_key}:hash"
        client.hset(hash_key, "field1", "value1")
        client.hset(hash_key, "field2", "value2")
        logger.info(f"HSET {hash_key} field1=value1, field2=value2")

        hash_values = client.hgetall(hash_key)
        logger.info(f"HGETALL {hash_key} = {hash_values}")

        if hash_values.get("field1") != "value1" or hash_values.get("field2") != "value2":
            logger.error(f"Hash value mismatch: {hash_values}")
            return False

        # Test list operations
        list_key = f"{test_key}:list"
        client.rpush(list_key, "item1", "item2", "item3")
        logger.info(f"RPUSH {list_key} item1, item2, item3")

        list_values = client.lrange(list_key, 0, -1)
        logger.info(f"LRANGE {list_key} 0 -1 = {list_values}")

        if list_values != ["item1", "item2", "item3"]:
            logger.error(f"List value mismatch: {list_values}")
            return False

        # Test JSON operations (using string SET/GET)
        json_key = f"{test_key}:json"
        json_data = {"name": "Redis", "type": "database", "features": ["fast", "versatile", "durable"]}
        client.set(json_key, json.dumps(json_data))
        logger.info(f"SET {json_key} = {json_data}")

        retrieved_json = client.get(json_key)
        parsed_json = json.loads(retrieved_json)
        logger.info(f"GET {json_key} = {parsed_json}")

        if parsed_json != json_data:
            logger.error(f"JSON value mismatch: {parsed_json}")
            return False

        # Clean up test keys
        client.delete(test_key, hash_key, list_key, json_key)
        logger.info(f"Cleaned up test keys")

        logger.info("All basic Redis operations completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Error testing Redis operations: {e}")
        return False


def test_redis_server_class():
    """Test the RedisServer class."""
    logger.info("Testing RedisServer class...")

    try:
        #
        # Import the RedisServer class directly to avoid conflict with redis
        # package
        import importlib.util
        spec = importlib.util.spec_from_file_location("redis_server", os.path.join(os.path.dirname(__file__), "redis_server.py"))
        redis_server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(redis_server_module)
        RedisServer = redis_server_module.RedisServer

        # Get Redis server instance
        redis_server = RedisServer.get_instance()

        # Get Redis client
        redis_client = redis_server.get_client()

        if redis_client:
            # Test connection
            response = redis_client.ping()
            logger.info(f"RedisServer ping response: {response}")

            if response:
                logger.info("Successfully connected to Redis using RedisServer class")

                # Test a basic operation
                test_key = f"test:redis:server:{datetime.now().strftime('%Y%m%d%H%M%S')}"
                test_value = "RedisServer is working!"

                # Record operation start time
                start_time = time.time()

                # Set a value
                redis_client.set(test_key, test_value)

                # Record operation metrics
                redis_server.record_operation("set", time.time() - start_time)
                redis_server.log_operation("set", test_key, "success")

                # Get the value
                start_time = time.time()
                retrieved_value = redis_client.get(test_key)
                redis_server.record_operation("get", time.time() - start_time)
                redis_server.log_operation("get", test_key, "success")

                logger.info(f"Retrieved value: {retrieved_value}")

                # Clean up
                redis_client.delete(test_key)

                return True
            else:
                logger.error("RedisServer ping failed")
                return False
        else:
            logger.error("Failed to get Redis client from RedisServer")
            return False
    except Exception as e:
        logger.error(f"Error testing RedisServer class: {e}")
        return False


def test_redis_mcp():
    """Test the Redis MCP."""
    logger.info("Testing Redis MCP...")

    try:
        # Import the RedisMCP class
        from mcp_tools.db_mcp.redis_mcp import RedisMCP

        # Create Redis MCP instance
        redis_mcp = RedisMCP()

        # Test if Redis MCP is connected
        if redis_mcp.redis_client:
            logger.info("Redis MCP is connected to Redis")

            # Test basic operations
            test_key = f"test:redis:mcp:{datetime.now().strftime('%Y%m%d%H%M%S')}"
            test_value = "Redis MCP is working!"

            # Set a value
            result = redis_mcp.set_value(test_key, test_value)
            logger.info(f"Set value result: {result}")

            # Get the value
            retrieved_value = redis_mcp.get_value(test_key)
            logger.info(f"Retrieved value: {retrieved_value}")

            if retrieved_value != test_value:
                logger.error(f"Value mismatch: expected '{test_value}', got '{retrieved_value}'")
                return False

            # Test JSON operations
            json_key = f"{test_key}:json"
            json_data = {"name": "Redis MCP", "type": "MCP server", "features": ["state management", "caching"]}

            # Set JSON
            result = redis_mcp.set_json(json_key, json_data)
            logger.info(f"Set JSON result: {result}")

            # Get JSON
            retrieved_json = redis_mcp.get_json(json_key)
            logger.info(f"Retrieved JSON: {retrieved_json}")

            if retrieved_json != json_data:
                logger.error(f"JSON value mismatch: {retrieved_json}")
                return False

            # Clean up
            redis_mcp.delete_value(test_key)
            redis_mcp.delete_value(json_key)

            logger.info("Redis MCP tests completed successfully!")
            return True
        else:
            logger.error("Redis MCP is not connected to Redis")
            return False
    except Exception as e:
        logger.error(f"Error testing Redis MCP: {e}")
        return False


def main():
    """Main entry point for the Redis setup test."""
    logger.info("Starting Redis setup test...")

    # Test Redis connection
    client = test_redis_connection()
    if not client:
        logger.error("Redis connection test failed")
        sys.exit(1)

    # Test basic operations
    if not test_basic_operations(client):
        logger.error("Redis basic operations test failed")
        sys.exit(1)

    # Test RedisServer class
    if not test_redis_server_class():
        logger.error("RedisServer class test failed")
        sys.exit(1)

    # Test Redis MCP
    if not test_redis_mcp():
        logger.error("Redis MCP test failed")
        sys.exit(1)

    logger.info("All Redis tests passed successfully!")
    logger.info("Redis is properly set up and running!")

if __name__ == "__main__":
    main()
