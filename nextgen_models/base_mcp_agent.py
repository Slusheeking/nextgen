import asyncio
import importlib
# Import redis directly
try:
    import redis
except ImportError:
    redis = None
from typing import Dict, Any, List, Optional

# Import all necessary MCP clients
from mcp_tools.financial_text_mcp.financial_text_mcp import FinancialTextMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
from mcp_tools.db_mcp.redis_mcp import RedisMCP
# Add imports for other MCP clients as needed based on other models

class BaseMCPAgent:
    """
    Base class for AutoGen agents that provides centralized access to MCP tools.

    This class initializes and manages instances of various MCP clients and
    provides generic methods for listing tools, calling tools, and reading resources
    that route calls to the appropriate client.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BaseMCPAgent with MCP client configurations.

        Args:
            config: Configuration dictionary containing configs for various MCP clients.
                    Expected keys: 'financial_text_config', 'financial_data_config',
                    'redis_config', etc.
        """
        self.config = config or {}
        self.mcp_clients: Dict[str, Any] = {}

        # Initialize MCP clients based on provided configuration
        if "financial_text_config" in self.config:
            self.mcp_clients["financial_text"] = FinancialTextMCP(self.config["financial_text_config"])
        if "financial_data_config" in self.config:
            self.mcp_clients["financial_data"] = FinancialDataMCP(self.config["financial_data_config"])
        if "redis_config" in self.config:
             self.mcp_clients["redis"] = RedisMCP(self.config["redis_config"])
        # Add initialization for other MCP clients here as needed

    def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        List all available tools on a specified MCP server.

        Args:
            server_name: The name of the MCP server (e.g., 'financial_text', 'financial_data', 'redis').

        Returns:
            A list of dictionaries, where each dictionary describes a tool.
            Returns an error dictionary if the server is not found or listing fails.
        """
        mcp_client = self.mcp_clients.get(server_name)
        if not mcp_client:
            return [{"error": f"MCP server '{server_name}' not found."}]

        try:
            # Assuming all MCP clients have a list_tools method
            return mcp_client.list_tools()
        except Exception as e:
            return [{"error": f"Error listing tools for server '{server_name}': {e}"}]

    def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific tool on a specified MCP server.

        Args:
            server_name: The name of the MCP server.
            tool_name: The name of the tool to call.
            arguments: A dictionary of arguments for the tool.

        Returns:
            The result of the tool execution, or an error dictionary if the server
            or tool is not found, or the call fails.
        """
        mcp_client = self.mcp_clients.get(server_name)
        if not mcp_client:
            return {"error": f"MCP server '{server_name}' not found."}

        try:
            # Assuming all MCP clients have a call_tool method
            return mcp_client.call_tool(tool_name, arguments)
        except Exception as e:
            return {"error": f"Error calling tool '{tool_name}' on server '{server_name}': {e}"}

    def read_resource(self, server_name: str, uri: str) -> Any:
        """
        Read a resource from a specified MCP server.

        Args:
            server_name: The name of the MCP server.
            uri: The URI of the resource to read.

        Returns:
            The content of the resource, or an error dictionary if the server
            or resource is not found, or reading fails.
        """
        mcp_client = self.mcp_clients.get(server_name)
        if not mcp_client:
            return {"error": f"MCP server '{server_name}' not found."}

        try:
            # Assuming all MCP clients have a read_resource method
            # Note: The document mentions read_resource, but the existing MCP clients
            # seem to use call_tool for data retrieval (e.g., RedisMCP.call_tool('get_json')).
            # This method might need adjustment based on actual MCP client implementations.
            # For now, providing a placeholder based on the document's description.
            # A more robust implementation would check for a specific read_resource method
            # or route based on URI scheme if clients support it.
             if hasattr(mcp_client, 'read_resource'):
                 return mcp_client.read_resource(uri)
             else:
                 return {"error": f"MCP server '{server_name}' does not support read_resource."}
        except Exception as e:
            return {"error": f"Error reading resource '{uri}' from server '{server_name}': {e}"}

    def shutdown(self):
        """
        Perform graceful shutdown of all initialized MCP clients.
        """
        for server_name, client in self.mcp_clients.items():
            if hasattr(client, 'shutdown'):
                try:
                    client.shutdown()
                    print(f"Shutdown {server_name} MCP client successfully.")
                except Exception as e:
                    print(f"Error shutting down {server_name} MCP client: {e}")
