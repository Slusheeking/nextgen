"""
Data MCP Tools Package

This package contains Model Context Protocol (MCP) servers for accessing
various financial data sources. These servers provide a unified interface
for models to access data from different providers dynamically.

Available MCP Servers:
- PolygonMCP: Access to Polygon.io REST and WebSocket APIs
- YahooFinanceMCP: Access to Yahoo Finance data
- YahooNewsMCP: Access to Yahoo Finance news and articles
- RedditMCP: Access to Reddit data and sentiment
- UnusualWhalesMCP: Access to Unusual Whales options flow data
"""

from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWsMCP
from mcp_tools.data_mcp.polygon_news_mcp import PolygonNewsMCP
from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP
from mcp_tools.data_mcp.yahoo_news_mcp import YahooNewsMCP
from mcp_tools.data_mcp.reddit_mcp import RedditMCP
from mcp_tools.data_mcp.unusual_whales_mcp import UnusualWhalesMCP

# Version
__version__ = '0.1.0'

# Export classes
__all__ = [
    'BaseDataMCP',
    'PolygonRestMCP',
    'PolygonWsMCP',
    'PolygonNewsMCP',
    'YahooFinanceMCP',
    'YahooNewsMCP',
    'RedditMCP',
    'UnusualWhalesMCP',
]
