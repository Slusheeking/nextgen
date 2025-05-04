"""
Base MCP Server Implementation

This module provides the base class for MCP (Model Context Protocol) servers
in the FinGPT trading system.
"""

import logging
from typing import Dict, List, Any, Optional, Callable, Union

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
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the MCP server."""
        logger = logging.getLogger(f"mcp_tools.{self.name}")
        logger.setLevel(logging.INFO)
        
        # Add console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def register_tool(self, func: Callable, name: Optional[str] = None, 
                     description: Optional[str] = None) -> None:
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
        
    def register_resource(self, name: str, resource: Any, 
                         description: Optional[str] = None) -> None:
        """
        Register a resource with the MCP server.
        
        Args:
            name: Name of the resource
            resource: The resource object
            description: Optional description of the resource
        """
        self.resources[name] = {
            "resource": resource,
            "description": description or f"Resource: {name}"
        }
        
        self.logger.info(f"Registered resource: {name}")
    
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
            raise ValueError(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]["func"]
        args = args or {}
        
        self.logger.info(f"Calling tool: {tool_name} with args: {args}")
        try:
            result = tool(**args)
            return result
        except Exception as e:
            self.logger.error(f"Error calling tool {tool_name}: {e}")
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
            raise ValueError(f"Resource not found: {resource_name}")
        
        self.logger.info(f"Accessing resource: {resource_name}")
        return self.resources[resource_name]["resource"]