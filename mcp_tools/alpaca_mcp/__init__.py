"""
Alpaca MCP Servers

This package contains MCP servers for Alpaca integration.
"""

# Use lazy imports to avoid circular dependencies
# Import AlpacaMCP when needed with:
# from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP

__version__ = "0.1.0"

__all__ = ["__version__"]

# Function to lazily load AlpacaMCP
def get_alpaca_mcp():
    """Get the AlpacaMCP class lazily to avoid circular imports."""
    from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP
    return AlpacaMCP
