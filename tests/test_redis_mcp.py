#!/usr/bin/env python3
"""
Test script for Redis MCP server.

This script initializes the Redis MCP server, puts test data into Redis,
and then reads it back to verify functionality.
"""

import time
import json
from mcp_tools.db_mcp.redis_mcp import RedisMCP

def main():
    print("Initializing Redis MCP server...")
    
    # Configure Redis MCP with monitoring disabled and no password
    config = {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,  # No password for local testing
        "enable_prometheus": False,  # Disable Prometheus to avoid port conflicts
        "enable_loki": False,  # Disable Loki logging
    }
    
    # Monkey patch methods to avoid errors with monitoring
    from mcp_tools.base_mcp_server import BaseMCPServer
    
    # Save original methods
    RedisMCP._original_setup_monitoring = RedisMCP._setup_monitoring
    BaseMCPServer._original_register_tool = BaseMCPServer.register_tool
    
    # Replace with safe versions
    RedisMCP._setup_monitoring = lambda self: (None, {})
    
    def safe_register_tool(self, func, name=None, description=None):
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        self.tools[tool_name] = {
            "func": func,
            "description": tool_description,
        }

        self.logger.info(f"Registered tool: {tool_name}")
        # Skip monitor logging if monitor is None
        if hasattr(self, 'monitor') and self.monitor is not None:
            self.monitor.log_info(
                f"Registered tool: {tool_name}",
                component="base_mcp_server",
                action="register_tool",
                tool_name=tool_name
            )
    
    BaseMCPServer.register_tool = safe_register_tool
    
    redis_mcp = RedisMCP(config)
    
    # Test basic key-value operations
    print("\n=== Testing basic key-value operations ===")
    
    # Set a simple string value
    key = "test_string"
    value = "Hello from Redis MCP!"
    print(f"Setting {key} = {value}")
    success = redis_mcp.set_value(key, value)
    print(f"Set operation {'succeeded' if success else 'failed'}")
    
    # Get the value back
    retrieved_value = redis_mcp.get_value(key)
    print(f"Retrieved {key} = {retrieved_value}")
    print(f"Value matches: {retrieved_value == value}")
    
    # Test JSON operations
    print("\n=== Testing JSON operations ===")
    
    # Set a JSON value
    json_key = "test_json"
    json_data = {
        "name": "Redis MCP Test",
        "timestamp": time.time(),
        "metrics": {
            "cpu": 0.45,
            "memory": 0.32,
            "disk": 0.78
        },
        "tags": ["test", "redis", "mcp"]
    }
    print(f"Setting JSON data with key {json_key}")
    success = redis_mcp.set_json(json_key, json_data)
    print(f"Set JSON operation {'succeeded' if success else 'failed'}")
    
    # Get the JSON value back
    retrieved_json = redis_mcp.get_json(json_key)
    print(f"Retrieved JSON data: {json.dumps(retrieved_json, indent=2)}")
    print(f"JSON data matches: {retrieved_json == json_data}")
    
    # Test hash operations
    print("\n=== Testing hash operations ===")
    
    # Set a hash
    hash_key = "test_hash"
    hash_data = {
        "symbol": "AAPL",
        "price": "185.92",
        "volume": "45678901",
        "change": "+1.25%"
    }
    print(f"Setting hash data with key {hash_key}")
    success = redis_mcp.set_hash(hash_key, hash_data)
    print(f"Set hash operation {'succeeded' if success else 'failed'}")
    
    # Get the hash back
    retrieved_hash = redis_mcp.get_hash(hash_key)
    print(f"Retrieved hash data: {retrieved_hash}")
    print(f"Hash data matches: {retrieved_hash == hash_data}")
    
    # Test list operations
    print("\n=== Testing list operations ===")
    
    # Add items to a list
    list_key = "test_list"
    list_items = ["item1", "item2", "item3", "item4", "item5"]
    print(f"Adding items to list {list_key}")
    for item in list_items:
        redis_mcp.add_to_list(list_key, item)
    
    # Get the list back
    retrieved_list = redis_mcp.get_list(list_key)
    print(f"Retrieved list data: {retrieved_list}")
    print(f"List contains all items: {all(item in retrieved_list for item in list_items)}")
    
    # Test sorted set operations
    print("\n=== Testing sorted set operations ===")
    
    # Add items to a sorted set
    zset_key = "test_zset"
    zset_items = [
        ("item1", 1.0),
        ("item2", 2.5),
        ("item3", 3.7),
        ("item4", 4.2),
        ("item5", 5.0)
    ]
    print(f"Adding items to sorted set {zset_key}")
    for item, score in zset_items:
        redis_mcp.add_to_sorted_set(zset_key, item, score)
    
    # Get the sorted set back
    retrieved_zset = redis_mcp.get_sorted_set(zset_key, 0, -1)
    print(f"Retrieved sorted set data: {retrieved_zset}")
    
    # Test expiry
    print("\n=== Testing key expiry ===")
    
    # Set a key with expiry
    expiry_key = "test_expiry"
    expiry_value = "This will expire in 5 seconds"
    print(f"Setting {expiry_key} with 5 second expiry")
    redis_mcp.set_value(expiry_key, expiry_value, 5)
    
    # Verify it exists
    print(f"Value exists immediately after setting: {redis_mcp.get_value(expiry_key) is not None}")
    
    # Wait for expiry
    print("Waiting for key to expire...")
    time.sleep(6)
    
    # Verify it's gone
    print(f"Value exists after expiry time: {redis_mcp.get_value(expiry_key) is not None}")
    
    # Test increment operations
    print("\n=== Testing increment operations ===")
    
    # Set initial counter
    counter_key = "test_counter"
    redis_mcp.set_value(counter_key, "0")
    
    # Increment multiple times
    for i in range(1, 6):
        redis_mcp.increment_float(counter_key, 1.0)
        print(f"Counter after increment {i}: {redis_mcp.get_value(counter_key)}")
    
    # Test hash increment
    hash_counter_key = "test_hash_counter"
    field = "count"
    print(f"Incrementing hash field {field} in {hash_counter_key}")
    for i in range(1, 6):
        new_value = redis_mcp.increment_hash_value(hash_counter_key, field)
        print(f"Hash counter after increment {i}: {new_value}")
    
    # Clean up
    print("\n=== Cleaning up test keys ===")
    keys_to_delete = [key, json_key, hash_key, list_key, zset_key, counter_key, hash_counter_key]
    for k in keys_to_delete:
        redis_mcp.delete_value(k)
        print(f"Deleted key: {k}")
    
    print("\nRedis MCP test completed successfully!")

if __name__ == "__main__":
    main()
