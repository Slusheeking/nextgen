"""
Financial Data MCP Tools Package

This package contains Model Context Protocol (MCP) servers for accessing
financial data analysis capabilities. These servers provide unified interfaces
for models to perform financial calculations, technical analysis, and data 
processing.

All MCP servers in this package use the NetdataLogger system for monitoring,
logging, and metrics collection, providing consistent performance tracking
and unified logging across all financial data services.

Available MCP Servers:
- FinancialDataMCP: Financial data analysis and technical indicators
"""

# Import the main class
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP

# Version
__version__ = "0.1.0"

# Export classes
__all__ = [
    "FinancialDataMCP",
    "__version__",
]
