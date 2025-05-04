"""
Base MCP Server Implementation

This module provides the base class for MCP (Model Context Protocol) servers
in the FinGPT trading system.
"""

import logging
import os
from typing import Dict, List, Any, Optional, Callable

# Import dotenv for environment variables
from dotenv import load_dotenv
from monitoring.system_monitor import MonitoringManager


class BaseMCPServer:
    """
    Base class for MCP servers.

    All specialized MCP servers should inherit from this class and implement
    their specific functionality.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base MCP server.

        Args:
            name: Name of the MCP server
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.logger = self._setup_logging()
        self.tools = {}
        self.resources = {}
        
        # Initialize monitoring
        self.monitor = MonitoringManager(
            service_name=f"{self.name}-mcp-server"
        )
        self.monitor.log_info(
            f"{self.name} MCP server initialized",
            component="base_mcp_server",
            action="initialization"
        )
        
        # Ensure environment variables are loaded
        load_dotenv()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the MCP server."""
        logger = logging.getLogger(f"mcp_tools.{self.name}")
        logger.setLevel(logging.INFO)

        # Add console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Register a tool with the MCP server.

        Args:
            func: Function implementing the tool
            name: Optional name for the tool (defaults to function name)
            description: Optional description of the tool
        """
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        self.tools[tool_name] = {
            "func": func,
            "description": tool_description,
        }

        self.logger.info(f"Registered tool: {tool_name}")
        self.monitor.log_info(
            f"Registered tool: {tool_name}",
            component="base_mcp_server",
            action="register_tool",
            tool_name=tool_name
        )

    def register_resource(
        self, name: str, resource: Any, description: Optional[str] = None
    ) -> None:
        """
        Register a resource with the MCP server.

        Args:
            name: Name of the resource
            resource: The resource object
            description: Optional description of the resource
        """
        self.resources[name] = {
            "resource": resource,
            "description": description or f"Resource: {name}",
        }

        self.logger.info(f"Registered resource: {name}")
        self.monitor.log_info(
            f"Registered resource: {name}",
            component="base_mcp_server",
            action="register_resource",
            resource_name=name
        )

    def list_tools(self) -> List[Dict[str, str]]:
        """
        List all available tools on this MCP server.

        Returns:
            List of tool information dictionaries
        """
        return [
            {"name": name, "description": info["description"]}
            for name, info in self.tools.items()
        ]

    def list_resources(self) -> List[Dict[str, str]]:
        """
        List all available resources on this MCP server.

        Returns:
            List of resource information dictionaries
        """
        return [
            {"name": name, "description": info["description"]}
            for name, info in self.resources.items()
        ]

    def call_tool(self, tool_name: str, args: Dict[str, Any] = None) -> Any:
        """
        Call a tool by name with the given arguments.

        Args:
            tool_name: Name of the tool to call
            args: Arguments to pass to the tool

        Returns:
            Result of the tool execution

        Raises:
            ValueError: If the tool doesn't exist
        """
        if tool_name not in self.tools:
            error_msg = f"Tool not found: {tool_name}"
            self.monitor.log_error(
                error_msg,
                component="base_mcp_server",
                action="call_tool_error",
                tool_name=tool_name
            )
            raise ValueError(error_msg)

        tool = self.tools[tool_name]["func"]
        args = args or {}

        self.logger.info(f"Calling tool: {tool_name} with args: {args}")
        self.monitor.log_info(
            f"Calling tool: {tool_name}",
            component="base_mcp_server",
            action="call_tool",
            tool_name=tool_name,
            args=str(args)
        )
        
        try:
            result = tool(**args)
            return result
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {e}"
            self.logger.error(error_msg)
            self.monitor.log_error(
                error_msg,
                component="base_mcp_server",
                action="call_tool_exception",
                tool_name=tool_name,
                error=str(e)
            )
            return {"error": str(e), "tool": tool_name}

    def access_resource(self, resource_name: str) -> Any:
        """
        Access a resource by name.

        Args:
            resource_name: Name of the resource to access

        Returns:
            The resource object

        Raises:
            ValueError: If the resource doesn't exist
        """
        if resource_name not in self.resources:
            error_msg = f"Resource not found: {resource_name}"
            self.monitor.log_error(
                error_msg,
                component="base_mcp_server",
                action="access_resource_error",
                resource_name=resource_name
            )
            raise ValueError(error_msg)

        self.logger.info(f"Accessing resource: {resource_name}")
        self.monitor.log_info(
            f"Accessing resource: {resource_name}",
            component="base_mcp_server",
            action="access_resource",
            resource_name=resource_name
        )
        return self.resources[resource_name]["resource"]
        
    def get_api_key(self, service_name: str, default: str = "") -> str:
        """
        Get API key for a specific service.
        
        Args:
            service_name: Service name (e.g., 'polygon', 'alpaca', 'openai')
            default: Default value if API key is not found
        
        Returns:
            API key value
        """
        # First check if there's a key in the config
        config_key = f"{service_name}_api_key"
        if config_key in self.config:
            return self.config[config_key]
        
        # Check environment variable using standard naming pattern
        env_var_names = [
            f"{service_name.upper()}_API_KEY",
            f"{service_name}_api_key"
        ]
        
        for env_var in env_var_names:
            api_key = os.environ.get(env_var)
            if api_key:
                return api_key
        
        # Log a warning if no API key found and no default provided
        if not default:
            warning_msg = f"API key for {service_name} not found in config or environment"
            self.logger.warning(warning_msg)
            self.monitor.log_error(
                warning_msg,
                component="base_mcp_server",
                action="api_key_missing",
                service_name=service_name
            )
            
        return default
        
    def get_env_value(self, key: str, default: Optional[str] = None) -> str:
        """
        Get environment variable value.
        
        Args:
            key: Environment variable key
            default: Default value if key is not found
        
        Returns:
            Environment variable value
        """
        # First check if there's a value in the config
        if key in self.config:
            return str(self.config[key])
        
        # Then check environment variable
        return os.environ.get(key, default)
