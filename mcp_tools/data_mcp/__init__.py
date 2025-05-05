"""
Data MCP Tools Package

This package contains Model Context Protocol (MCP) servers for accessing
various financial data sources. These servers provide a unified interface
for models to access data from different providers dynamically.

All MCP servers in this package use the NetdataLogger system for monitoring,
logging, and metrics collection, providing consistent performance tracking
and unified logging across all data sources.

Available MCP Servers:
- PolygonRestMCP: Access to Polygon.io REST API for market data
- PolygonWsMCP: Access to Polygon.io WebSocket API for real-time data
- PolygonNewsMCP: Access to Polygon.io news and press releases
- YahooFinanceMCP: Access to Yahoo Finance market data
- YahooNewsMCP: Access to Yahoo Finance news and articles
- RedditMCP: Access to Reddit data and sentiment analysis
- UnusualWhalesMCP: Access to Unusual Whales options flow data
"""

# First import the base class to prevent circular dependencies
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP

# Version
__version__ = "0.1.0"

# Don't eagerly import all classes to avoid circular dependencies
# Import specific modules only when needed:
# from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
# from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWsMCP

# Export only the base class
__all__ = [
    "BaseDataMCP",
    "__version__",
]

# Function to lazily import classes
def get_mcp_class(class_name):
    """
    Lazily import and return an MCP class to avoid circular dependencies.
    
    Args:
        class_name (str): Name of the MCP class to import
        
    Returns:
        class: The imported class
    
    Raises:
        ImportError: If the class doesn't exist
    """
    class_map = {
        "PolygonRestMCP": ".polygon_rest_mcp",
        "PolygonWsMCP": ".polygon_ws_mcp",
        "PolygonNewsMCP": ".polygon_news_mcp",
        "YahooFinanceMCP": ".yahoo_finance_mcp",
        "YahooNewsMCP": ".yahoo_news_mcp",
        "RedditMCP": ".reddit_mcp",
        "UnusualWhalesMCP": ".unusual_whales_mcp",
    }
    
    if class_name not in class_map:
        raise ImportError(f"Unknown MCP class: {class_name}")
    
    module_name = class_map[class_name]
    import importlib
    module = importlib.import_module(module_name, package="mcp_tools.data_mcp")
    return getattr(module, class_name)
