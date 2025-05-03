"""
Polygon.io REST API integration for financial market data.

A production-ready client for accessing Polygon.io REST API endpoints to fetch
financial market data for algorithmic trading systems.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Union

import requests
from dotenv import load_dotenv
from loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

# Load environment variables
load_dotenv()

# Setup traditional logging as fallback
logger = logging.getLogger(__name__)

class PolygonRestClient:
    """
    Production client for the Polygon.io REST API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.polygon.io"):
        """
        Initialize the Polygon.io REST API client.
        
        Args:
            api_key: API key for Polygon.io (defaults to POLYGON_API_KEY environment variable)
            base_url: Base URL for Polygon.io API (defaults to https://api.polygon.io)
        """
        # Initialize observability tools
        self.loki = LokiManager(service_name="data-polygon-rest")
        self.prom = PrometheusManager(service_name="data-polygon-rest")
        
        # Create metrics
        self.request_counter = self.prom.create_counter(
            "polygon_api_requests_total", 
            "Total count of Polygon.io REST API requests",
            ["method", "endpoint", "status"]
        )
        
        self.request_latency = self.prom.create_histogram(
            "polygon_api_request_duration_seconds",
            "Polygon.io REST API request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Initialize client
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            self.loki.warning("No Polygon API key provided - API calls will fail")
            logger.warning("No Polygon API key provided - API calls will fail")
            
        self.base_url = base_url
        
        # Define the optimal endpoints for different data retrieval purposes
        self.optimal_endpoints = {
            "price_history": {
                "path": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}",
                "method": "GET",
                "description": "Daily bars"
            },
            "market_cap": {
                "path": "/v3/reference/tickers/{ticker}",
                "method": "GET",
                "description": "Company info"
            },
            "volume": {
                "path": "/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}",
                "method": "GET",
                "description": "Daily bars"
            },
            "snapshot": {
                "path": "/v2/snapshot/ticker/{ticker}",
                "method": "GET",
                "description": "Snapshot for a single ticker"
            },
            "tickers_list": {
                "path": "/v3/reference/tickers",
                "method": "GET",
                "description": "List/search all tickers"
            }
        }
        
        self.loki.info("PolygonRestClient initialized", component="polygon_rest")
        logger.info("PolygonRestClient initialized")

    def fetch_optimal_data(self, purpose: str, **params) -> Dict[str, Any]:
        """
        Fetch data using one of the predefined optimal endpoints.
        
        Args:
            purpose: The purpose of the data retrieval 
                    ("price_history", "market_cap", "volume", "snapshot", "tickers_list")
            **params: Parameters for the API call
            
        Returns:
            API response data as dictionary
        """
        if purpose not in self.optimal_endpoints:
            error_msg = f"Unknown endpoint purpose: {purpose}"
            self.loki.error(error_msg, component="polygon_rest", purpose=purpose)
            logger.error(error_msg)
            return {"error": error_msg}
            
        endpoint = self.optimal_endpoints.get(purpose)
        return self.fetch_data(endpoint_details=endpoint, **params)

    def fetch_data(
        self, 
        endpoint_details: Optional[Dict[str, Any]] = None,
        description_keyword: Optional[str] = None, 
        **params
    ) -> Dict[str, Any]:
        """
        Fetch data from Polygon.io API.

        Args:
            endpoint_details: The endpoint definition (with 'path', 'method', etc.)
            description_keyword: Keyword to match in endpoint description (alternative to endpoint_details)
            **params: Parameters to fill in the endpoint path and query

        Returns:
            API response data or error information
        """
        # Get endpoint details if only keyword provided
        if endpoint_details is None and description_keyword is not None:
            for purpose, endpoint in self.optimal_endpoints.items():
                if description_keyword.lower() in endpoint.get("description", "").lower():
                    endpoint_details = endpoint
                    break
                    
            if not endpoint_details:
                error_msg = f"No endpoint found for keyword: {description_keyword}"
                self.loki.error(error_msg, component="polygon_rest", keyword=description_keyword)
                logger.error(error_msg)
                return {"error": error_msg}
                
        elif endpoint_details is None:
            error_msg = "No endpoint details or description keyword provided"
            self.loki.error(error_msg, component="polygon_rest")
            logger.error(error_msg)
            return {"error": error_msg}

        # Extract path and method from endpoint details
        path = endpoint_details.get("path") or endpoint_details.get("endpoint")
        method = endpoint_details.get("method", "GET").upper()
        
        if not path:
            error_msg = "No path specified in endpoint details"
            self.loki.error(error_msg, component="polygon_rest")
            logger.error(error_msg)
            return {"error": error_msg}

        # Format URL with path parameters
        try:
            url_path = path.format(**params)
        except KeyError as e:
            error_msg = f"Missing required parameter: {e}"
            self.loki.error(error_msg, component="polygon_rest", params=str(params))
            logger.error(error_msg)
            return {"error": error_msg}

        # Construct full URL
        url = f"{self.base_url}{url_path}" if url_path.startswith("/") else f"{self.base_url}/{url_path}"
        
        # Filter query parameters
        query_params = {k: v for k, v in params.items() if "{" + k + "}" not in path}
        query_params["apiKey"] = self.api_key

        logger.info(f"API request: {method} {url}")
        self.loki.info(f"API request: {method} {url}", 
                     component="polygon_rest", 
                     method=method,
                     endpoint=url_path)

        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(url, params=query_params, timeout=15)
            elif method == "POST":
                response = requests.post(url, json=query_params, timeout=15)
            else:
                error_msg = f"Unsupported HTTP method: {method}"
                self.loki.error(error_msg, component="polygon_rest", method=method)
                logger.error(error_msg)
                return {"error": error_msg}

            elapsed = time.time() - start_time
            
            # Record metrics
            endpoint_path = path.split('?')[0] if '?' in path else path
            self.prom.observe_histogram(
                "polygon_api_request_duration_seconds", 
                elapsed, 
                method=method, 
                endpoint=endpoint_path
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Log success and metrics
                log_msg = f"API request successful ({elapsed:.2f}s)"
                logger.info(log_msg)
                self.loki.info(log_msg, 
                             component="polygon_rest", 
                             method=method, 
                             endpoint=endpoint_path,
                             duration=f"{elapsed:.2f}")
                
                # Count successful request
                self.prom.increment_counter(
                    "polygon_api_requests_total", 
                    1, 
                    method=method, 
                    endpoint=endpoint_path, 
                    status="success"
                )
                
                return data
            else:
                # Log error and metrics
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                self.loki.error(error_msg, 
                              component="polygon_rest", 
                              method=method, 
                              endpoint=endpoint_path,
                              status_code=str(response.status_code))
                
                # Count failed request
                self.prom.increment_counter(
                    "polygon_api_requests_total", 
                    1, 
                    method=method, 
                    endpoint=endpoint_path, 
                    status="error"
                )
                
                return {
                    "error": f"HTTP {response.status_code}",
                    "details": response.text
                }
                
        except requests.RequestException as e:
            error_msg = f"Request failed: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="polygon_rest", 
                          method=method, 
                          endpoint=path,
                          error_type="request_exception")
            
            # Count failed request
            self.prom.increment_counter(
                "polygon_api_requests_total", 
                1, 
                method=method, 
                endpoint=path, 
                status="exception"
            )
            
            return {"error": str(e)}
        
        except ValueError as e:
            error_msg = f"JSON parsing failed: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="polygon_rest", 
                          method=method, 
                          endpoint=path,
                          error_type="json_parse")
            
            # Count failed request
            self.prom.increment_counter(
                "polygon_api_requests_total", 
                1, 
                method=method, 
                endpoint=path, 
                status="parse_error"
            )
            
            return {"error": "Failed to parse response"}
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="polygon_rest", 
                          method=method, 
                          endpoint=path,
                          error_type="unexpected")
            
            # Count failed request
            self.prom.increment_counter(
                "polygon_api_requests_total", 
                1, 
                method=method, 
                endpoint=path, 
                status="error"
            )
            
            return {"error": str(e)}
