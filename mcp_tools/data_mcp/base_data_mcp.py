"""
Base Data MCP Server

This module provides the base class for all data-focused MCP (Model Context Protocol) 
servers. It establishes common functionality for data fetching, caching, error handling,
and endpoint selection.
"""

import logging
import time
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import pandas as pd

from mcp_tools.base_mcp_server import BaseMCPServer

class BaseDataMCP(BaseMCPServer):
    """
    Base class for all data MCP servers.
    
    This abstract class provides common functionality for data source MCP servers,
    including dynamic endpoint selection, caching, error handling, and metric tracking.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base data MCP server.
        
        Args:
            name: Name of the MCP server
            config: Optional configuration dictionary
        """
        super().__init__(name=name, config=config)
        
        # Set up data cache with configurable TTL
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes default
        self.cache = {}
        self.cache_timestamps = {}
        
        # Set up endpoint definitions - to be populated by child classes
        self.endpoints = {}
        
        # Set up default error handling behavior
        self.error_retry_limit = self.config.get("error_retry_limit", 3)
        self.error_retry_delay = self.config.get("error_retry_delay", 1.0)
        
        # Initialize endpoint stats
        self.endpoint_stats = {}
        
        # Register common tools
        self._register_common_tools()
    
    def _register_common_tools(self) -> None:
        """Register common tools available in all data MCP servers."""
        self.register_tool(self.fetch_data)
        self.register_tool(self.list_available_endpoints)
        self.register_tool(self.get_endpoint_details)
        self.register_tool(self.clear_cache)
        self.register_tool(self.get_data_source_status)
    
    @abstractmethod
    def _initialize_client(self) -> Any:
        """
        Initialize the data source client.
        
        This method must be implemented by child classes to initialize
        their specific data source clients.
        
        Returns:
            Initialized data source client
        """
        pass
    
    @abstractmethod
    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for this data source.
        
        This method must be implemented by child classes to define
        the available endpoints and their configurations.
        
        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        pass
    
    def fetch_data(
        self, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Fetch data from the specified endpoint with given parameters.
        
        Args:
            endpoint: Name of the endpoint to fetch data from
            params: Optional parameters for the endpoint
            use_cache: Override default cache behavior for this request
            
        Returns:
            Fetched data or error information
        """
        params = params or {}
        if endpoint not in self.endpoints:
            error_msg = f"Unknown endpoint: {endpoint}"
            self.logger.error(error_msg)
            return {"error": error_msg, "available_endpoints": list(self.endpoints.keys())}
        
        # Determine if we should use cache
        should_use_cache = self.cache_enabled if use_cache is None else use_cache
        
        # Generate cache key
        cache_key = self._generate_cache_key(endpoint, params)
        
        # Check cache if enabled
        if should_use_cache and cache_key in self.cache:
            # Check if cache is still valid
            if time.time() - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl:
                self.logger.info(f"Returning cached data for {endpoint}")
                # Track cache hit in stats
                self._update_endpoint_stats(endpoint, "cache_hit")
                return self.cache[cache_key]
        
        # Track request in stats
        self._update_endpoint_stats(endpoint, "request")
        start_time = time.time()
        
        # Get endpoint configuration
        endpoint_config = self.endpoints[endpoint]
        
        # Try to fetch data with retry logic
        attempt = 0
        while attempt < self.error_retry_limit:
            try:
                # Execute the data fetch based on endpoint type
                result = self._execute_endpoint_fetch(endpoint_config, params)
                
                # Calculate fetch time
                elapsed = time.time() - start_time
                
                # Track successful request in stats
                self._update_endpoint_stats(endpoint, "success", elapsed)
                
                # Cache the result if caching is enabled
                if should_use_cache:
                    self.cache[cache_key] = result
                    self.cache_timestamps[cache_key] = time.time()
                
                self.logger.info(f"Successfully fetched data from {endpoint} in {elapsed:.2f}s")
                return result
                
            except Exception as e:
                attempt += 1
                self.logger.warning(f"Attempt {attempt}/{self.error_retry_limit} failed for {endpoint}: {e}")
                
                if attempt >= self.error_retry_limit:
                    # Track failed request in stats
                    self._update_endpoint_stats(endpoint, "error")
                    
                    error_msg = f"Error fetching data from {endpoint} after {attempt} attempts: {e}"
                    self.logger.error(error_msg)
                    return {"error": error_msg, "endpoint": endpoint}
                
                # Exponential backoff for retries
                time.sleep(self.error_retry_delay * (2 ** (attempt - 1)))
    
    def _execute_endpoint_fetch(
        self, 
        endpoint_config: Dict[str, Any], 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the fetch operation based on endpoint configuration.
        
        This method must be overridden by child classes to execute the actual
        data fetch operation for their specific data sources.
        
        Args:
            endpoint_config: Configuration for the endpoint
            params: Parameters for the request
            
        Returns:
            Fetched data
            
        Raises:
            NotImplementedError: If the child class does not implement this method
        """
        raise NotImplementedError("Child classes must implement _execute_endpoint_fetch")
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            endpoint: Endpoint name
            params: Request parameters
            
        Returns:
            Cache key string
        """
        # Sort params to ensure consistent keys regardless of parameter order
        params_str = json.dumps(params, sort_keys=True) if params else "{}"
        return f"{endpoint}:{params_str}"
    
    def _update_endpoint_stats(
        self, 
        endpoint: str, 
        event: str, 
        elapsed: Optional[float] = None
    ) -> None:
        """
        Update statistics for an endpoint.
        
        Args:
            endpoint: Name of the endpoint
            event: Type of event ("request", "success", "error", "cache_hit")
            elapsed: Optional elapsed time in seconds for the request
        """
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                "requests": 0,
                "successes": 0,
                "errors": 0,
                "cache_hits": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        stats = self.endpoint_stats[endpoint]
        
        if event == "request":
            stats["requests"] += 1
        elif event == "success":
            stats["successes"] += 1
            if elapsed is not None:
                stats["total_time"] += elapsed
                stats["avg_time"] = stats["total_time"] / stats["successes"]
        elif event == "error":
            stats["errors"] += 1
        elif event == "cache_hit":
            stats["cache_hits"] += 1
    
    def list_available_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all available endpoints for this data source.
        
        Returns:
            List of endpoint information dictionaries
        """
        return [
            {
                "name": name,
                "description": config.get("description", ""),
                "required_params": config.get("required_params", []),
                "optional_params": config.get("optional_params", []),
                "category": config.get("category", "general")
            }
            for name, config in self.endpoints.items()
        ]
    
    def get_endpoint_details(self, endpoint: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific endpoint.
        
        Args:
            endpoint: Name of the endpoint
            
        Returns:
            Endpoint details
        """
        if endpoint not in self.endpoints:
            return {"error": f"Unknown endpoint: {endpoint}"}
        
        config = self.endpoints[endpoint]
        stats = self.endpoint_stats.get(endpoint, {})
        
        return {
            "name": endpoint,
            "description": config.get("description", ""),
            "required_params": config.get("required_params", []),
            "optional_params": config.get("optional_params", []),
            "category": config.get("category", "general"),
            "method": config.get("method", "GET"),
            "path": config.get("path", ""),
            "stats": {
                "requests": stats.get("requests", 0),
                "successes": stats.get("successes", 0),
                "errors": stats.get("errors", 0),
                "cache_hits": stats.get("cache_hits", 0),
                "avg_time": stats.get("avg_time", 0)
            }
        }
    
    def clear_cache(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the data cache for this data source.
        
        Args:
            endpoint: Optional endpoint name to clear cache for that endpoint only
            
        Returns:
            Status information
        """
        if endpoint:
            # Clear cache for specific endpoint
            if endpoint not in self.endpoints:
                return {"error": f"Unknown endpoint: {endpoint}"}
            
            to_remove = []
            for key in self.cache.keys():
                if key.startswith(f"{endpoint}:"):
                    to_remove.append(key)
            
            for key in to_remove:
                del self.cache[key]
                if key in self.cache_timestamps:
                    del self.cache_timestamps[key]
            
            self.logger.info(f"Cleared cache for endpoint: {endpoint}")
            return {"status": "success", "message": f"Cleared cache for endpoint: {endpoint}", "items_removed": len(to_remove)}
        else:
            # Clear entire cache
            cache_size = len(self.cache)
            self.cache = {}
            self.cache_timestamps = {}
            self.logger.info(f"Cleared entire cache ({cache_size} items)")
            return {"status": "success", "message": "Cleared entire cache", "items_removed": cache_size}
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """
        Get the status and health of the data source.
        
        Returns:
            Status information
        """
        # Calculate overall success rate
        total_requests = sum(s.get("requests", 0) for s in self.endpoint_stats.values())
        total_successes = sum(s.get("successes", 0) for s in self.endpoint_stats.values())
        total_errors = sum(s.get("errors", 0) for s in self.endpoint_stats.values())
        total_cache_hits = sum(s.get("cache_hits", 0) for s in self.endpoint_stats.values())
        
        success_rate = (total_successes / total_requests) * 100 if total_requests > 0 else 0
        cache_hit_rate = (total_cache_hits / (total_requests + total_cache_hits)) * 100 if (total_requests + total_cache_hits) > 0 else 0
        
        # Calculate average response time across all endpoints
        total_time = sum(s.get("total_time", 0) for s in self.endpoint_stats.values())
        if total_successes > 0:
            avg_time = total_time / total_successes
        else:
            avg_time = 0
        
        return {
            "name": self.name,
            "status": "healthy" if success_rate >= 90 else "degraded" if success_rate >= 70 else "unhealthy",
            "endpoints": len(self.endpoints),
            "total_requests": total_requests,
            "total_successes": total_successes,
            "total_errors": total_errors,
            "total_cache_hits": total_cache_hits,
            "success_rate": f"{success_rate:.1f}%",
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "avg_response_time": f"{avg_time:.3f}s",
            "cache_enabled": self.cache_enabled,
            "cache_ttl": self.cache_ttl,
            "cache_size": len(self.cache)
        }
    
    def convert_to_dataframe(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Convert API response data to pandas DataFrame for easier analysis.
        
        Args:
            data: API response data
            
        Returns:
            Pandas DataFrame
        """
        try:
            if isinstance(data, dict):
                # Check if there's a specific results field
                if "results" in data and isinstance(data["results"], list):
                    return pd.DataFrame(data["results"])
                elif "data" in data and isinstance(data["data"], list):
                    return pd.DataFrame(data["data"])
                else:
                    # Try to convert the whole dict to a dataframe
                    return pd.DataFrame([data])
            elif isinstance(data, list):
                return pd.DataFrame(data)
            else:
                self.logger.error(f"Cannot convert data of type {type(data)} to DataFrame")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error converting to DataFrame: {e}")
            return pd.DataFrame()