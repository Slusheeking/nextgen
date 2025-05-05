"""
MCP Tools for FinGPT Trading System

This package contains Model Context Protocol (MCP) server implementations
for various components of the FinGPT trading system.

Available MCP modules:
- base_mcp_server: Base class for all MCP servers
- data_mcp: Data providers (Polygon, Yahoo Finance, Reddit, etc.)
- analysis_mcp: Market analysis tools (technical indicators, risk metrics, etc.)
- db_mcp: Database connectors (Redis, etc.)
- alpaca_mcp: Alpaca trading integration
"""

__version__ = "0.1.0"

# First import the base classes to avoid circular imports
from mcp_tools.base_mcp_server import BaseMCPServer

# Then import the derived classes
# Note: Import specific modules only when needed to avoid circular imports
# If you need these classes, import them directly:
# from mcp_tools.data_mcp import BaseDataMCP, PolygonRestMCP, PolygonWsMCP, PolygonNewsMCP
# from mcp_tools.analysis_mcp.get_analysis_mcp import get_analysis_mcp
# from mcp_tools.db_mcp.get_db_mcp import get_db_mcp
# from mcp_tools.alpaca_mcp import get_alpaca_mcp

from . import document_analysis_mcp
from . import financial_data_mcp
from . import financial_text_mcp
from . import risk_analysis_mcp
from . import time_series_mcp
from . import trading_mcp
from . import vector_store_mcp

__all__ = [
    "BaseMCPServer",
    "__version__",
    "document_analysis_mcp",
    "financial_data_mcp",
    "financial_text_mcp",
    "risk_analysis_mcp",
    "time_series_mcp",
    "trading_mcp",
    "vector_store_mcp",
]
