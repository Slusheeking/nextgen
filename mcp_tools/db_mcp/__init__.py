"""
Database MCP Tools Package

This package contains Model Context Protocol (MCP) servers for accessing
various database systems. These servers provide a unified interface
for models to interact with different database technologies.

Available MCP Servers:
- RedisMCP: Access to Redis for state management, caching, and pub/sub messaging.
  Integrated with NetdataLogger for monitoring and metrics collection.
  Connects to the local Redis server for centralized Redis access.
"""

# Version
__version__ = "0.1.0"

# Import the RedisMCP class for easier access
from .redis_mcp import RedisMCP

__all__ = ["__version__", "RedisMCP"]

# Function to lazily import database MCP classes
def get_db_mcp(class_name):
    """
    Get a database MCP class lazily to avoid circular imports.
    
    Args:
        class_name (str): Name of the database MCP class
        
    Returns:
        class: The requested MCP class
        
    Raises:
        ImportError: If the class doesn't exist
    """
    class_map = {
        "RedisMCP": ".redis_mcp",
    }
    
    if class_name not in class_map:
        raise ImportError(f"Unknown database MCP class: {class_name}")
        
    module_name = class_map[class_name]
    import importlib
    module = importlib.import_module(module_name, package="mcp_tools.db_mcp")
    return getattr(module, class_name)
