"""
MCP Tools for FinGPT Trading System

This package contains Model Context Protocol (MCP) server implementations
for various components of the FinGPT trading system.
"""

__version__ = "0.1.0"

# First import the base classes to avoid circular imports
from mcp_tools.base_mcp_server import BaseMCPServer

# Then import the derived classes
# Note: Import specific modules only when needed to avoid circular imports
# If you need these classes, import them directly:
# from mcp_tools.data_mcp import BaseDataMCP, PolygonRestMCP
# from mcp_tools.alpaca_mcp import AlpacaMCP

__all__ = ["BaseMCPServer", "__version__"]
