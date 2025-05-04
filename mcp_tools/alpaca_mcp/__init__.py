"""
Alpaca MCP Servers

This package contains MCP servers for Alpaca integration.
Alpaca provides commission-free trading API for stocks and cryptocurrencies.

Available MCP Servers:
- AlpacaMCP: Access to Alpaca trading API for order execution, account management,
  and market data retrieval.
"""

# Use lazy imports to avoid circular dependencies
# Import AlpacaMCP when needed with:
# from mcp_tools.alpaca_mcp.alpaca_mcp import AlpacaMCP

__version__ = "0.1.0"

__all__ = ["__version__"]

# Function to lazily load AlpacaMCP
def get_alpaca_mcp(class_name="AlpacaMCP"):
    """
    Get the AlpacaMCP class lazily to avoid circular imports.
    
    Args:
        class_name (str): Name of the Alpaca MCP class to import
            (default: "AlpacaMCP", currently the only available class)
            
    Returns:
        class: The requested MCP class
        
    Raises:
        ImportError: If the class doesn't exist
    """
    class_map = {
        "AlpacaMCP": ".alpaca_mcp",
    }
    
    if class_name not in class_map:
        raise ImportError(f"Unknown Alpaca MCP class: {class_name}")
        
    module_name = class_map[class_name]
    import importlib
    module = importlib.import_module(module_name, package="mcp_tools.alpaca_mcp")
    return getattr(module, class_name)
