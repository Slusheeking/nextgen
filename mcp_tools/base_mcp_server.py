"""
Base MCP Server Implementation

This module provides the base class for MCP (Model Context Protocol) servers
in the Nextgen trading system. It includes comprehensive monitoring, metrics
collection, and diagnostic capabilities.
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

# Direct imports instead of dynamic loading
from dotenv import load_dotenv

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from monitoring.system_metrics import SystemMetricsCollector
from monitoring.stock_charts import StockChartGenerator


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
        init_start_time = time.time()
        self.name = name
        self.config = config or {}
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.resources: Dict[str, Dict[str, Any]] = {}
        self.start_time = datetime.now()
        
        # Ensure environment variables are loaded
        load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
        
        # Initialize NetdataLogger for logging and metrics
        self.logger = NetdataLogger(component_name=f"{self.name}-mcp-server")
        self.logger.info(f"Initializing {self.name} MCP server")
        
        # Initialize system metrics collector
        self.metrics_collector = SystemMetricsCollector(self.logger)
        self.logger.info("System metrics collector initialized")
        
        # Initialize chart generator for financial data visualization
        self.chart_generator = StockChartGenerator()
        self.logger.info("Stock chart generator initialized")
        
        # Start collecting system metrics
        self.metrics_collector.start()
        
        # Initialize detailed performance counters
        self.request_count = 0
        self.error_count = 0
        self.active_connections = 0
        self.peak_connections = 0
        self.total_response_time = 0
        self.slow_requests_count = 0
        self.timeout_count = 0
        self.last_request_time = time.time()
        self.consecutive_errors = 0
        self.max_consecutive_errors = 0
        self.tool_execution_counts: Dict[str, int] = {}
        self.resource_access_counts: Dict[str, int] = {}
        self.api_key_access_counts: Dict[str, int] = {}
        
        # Configure health thresholds
        self.slow_request_threshold_ms = self.config.get("slow_request_threshold_ms", 500)
        self.health_check_interval = self.config.get("health_check_interval", 60)  # seconds
        self.error_rate_threshold = self.config.get("error_rate_threshold", 0.1)  # 10%
        
        # Start health check background thread if enabled
        if self.config.get("enable_health_check", True):
            self._start_health_check_thread()
        
        # Set up initial gauge metrics
        self.logger.gauge("mcp.tools_count", 0)
        self.logger.gauge("mcp.resources_count", 0)
        self.logger.gauge("mcp.active_connections", 0)
        self.logger.gauge("mcp.queue_length", 0)
        self.logger.gauge("mcp.error_rate", 0)
        self.logger.gauge("mcp.avg_response_time_ms", 0)
        
        # Record initialization time
        init_duration = (time.time() - init_start_time) * 1000  # milliseconds
        self.logger.timing("mcp.initialization_time_ms", init_duration)
        
        # Log initialization complete
        self.logger.info(f"{self.name} MCP server initialized", 
                        init_time_ms=init_duration,
                        health_check_enabled=self.config.get("enable_health_check", True))

    def shutdown(self):
        """
        Shutdown the MCP server and stop metrics collection.
        """
        self.logger.info(f"Shutting down {self.name} MCP server")
        self.metrics_collector.stop()

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

        self.logger.info(f"Registered tool: {tool_name}", tool_name=tool_name)
        
        # Track metrics
        self.logger.gauge("mcp.tools_count", len(self.tools))

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

        self.logger.info(f"Registered resource: {name}", resource_name=name)
        
        # Track metrics
        self.logger.gauge("mcp.resources_count", len(self.resources))

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

    def call_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None) -> Any:
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
        # Increment active connections and request count
        self.active_connections += 1
        self.request_count += 1
        self.logger.counter("mcp.request_count")
        self.logger.gauge("mcp.active_connections", self.active_connections)
        self.logger.gauge("mcp.queue_length", self.active_connections) # Use active connections as proxy for queue length
        
        try:
            if tool_name not in self.tools:
                error_msg = f"Tool not found: {tool_name}"
                self.logger.error(error_msg, tool_name=tool_name)
                
                # Track error metrics
                self.error_count += 1
                self.logger.counter("mcp.error_count")
                self.logger.gauge("mcp.error_rate", (self.error_count / self.request_count) * 100)
                
                raise ValueError(error_msg)

            tool = self.tools[tool_name]["func"]
            args = args or {}

            self.logger.info(f"Calling tool: {tool_name}", tool_name=tool_name, args=str(args))
            
            # Measure response time
            start_time = time.time()
            
            try:
                result = tool(**args)
                
                # Calculate and record response time
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                self.logger.timing("mcp.response_time_ms", response_time)
                
                return result
            except Exception as e:
                error_msg = f"Error calling tool {tool_name}: {e}"
                self.logger.error(error_msg, tool_name=tool_name, error=str(e))
                
                # Track error metrics
                self.error_count += 1
                self.logger.counter("mcp.error_count")
                self.logger.gauge("mcp.error_rate", (self.error_count / self.request_count) * 100)
                
                # Calculate and record response time even for errors
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                self.logger.timing("mcp.error_response_time_ms", response_time)
                
                return {"error": str(e), "tool": tool_name}
        finally:
            # Decrement active connections
            self.active_connections -= 1
            self.logger.gauge("mcp.active_connections", self.active_connections)
            self.logger.gauge("mcp.queue_length", self.active_connections) # Update queue length metric

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
        # Increment active connections and request count
        self.active_connections += 1
        self.request_count += 1
        self.logger.counter("mcp.request_count")
        self.logger.gauge("mcp.active_connections", self.active_connections)
        self.logger.gauge("mcp.queue_length", self.active_connections) # Use active connections as proxy for queue length
        
        try:
            if resource_name not in self.resources:
                error_msg = f"Resource not found: {resource_name}"
                self.logger.error(error_msg, resource_name=resource_name)
                
                # Track error metrics
                self.error_count += 1
                self.logger.counter("mcp.error_count")
                
                raise ValueError(error_msg)

            # Measure access time
            start_time = time.time()
            
            self.logger.info(f"Accessing resource: {resource_name}", resource_name=resource_name)
            
            # Track resource access metrics
            self.logger.counter("mcp.resource_access_count")
            
            resource = self.resources[resource_name]["resource"]
            
            # Calculate and record access time
            access_time = (time.time() - start_time) * 1000  # Convert to ms
            self.logger.timing("mcp.resource_access_time_ms", access_time)
            
            return resource
        finally:
            # Decrement active connections
            self.active_connections -= 1
            self.logger.gauge("mcp.active_connections", self.active_connections)
            self.logger.gauge("mcp.queue_length", self.active_connections) # Update queue length metric
        
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
            self.logger.warning(warning_msg, service_name=service_name)
            
            # Track missing API key metric
            self.logger.counter("mcp.missing_api_key_count")
            
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
        
    def _start_health_check_thread(self):
        """
        Start a background thread to periodically check the health of the MCP server.
        
        This method is called during initialization if health_check is enabled in the config.
        The health check thread monitors system metrics, response times, error rates, etc.
        """
        self.logger.info("Stub health check thread - no actual thread started (for testing)")
        
        # In a production environment, this would start a real background thread
        # For example:
        # import threading
        # self.health_check_thread = threading.Thread(target=self._health_check_worker, daemon=True)
        # self.health_check_thread.start()
