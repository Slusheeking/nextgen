"""
MCP Tools for FinGPT Trading System

This package contains Model Context Protocol (MCP) server implementations
for various components of the FinGPT trading system.
"""

__version__ = '0.1.0'

from mcp_tools.base_mcp_server import BaseMCPServer
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP

__all__ = [
    'BaseMCPServer',
    'BaseDataMCP', 
    'PolygonRestMCP',
    'AlpacaMCP',
    '__version__'
]