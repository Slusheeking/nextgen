"""
Unusual Whales API integration for options flow and unusual activity.

A production-ready client for accessing Unusual Whales API to track options flow,
unusual options activity, and market sentiment for algorithmic trading systems.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import requests
from dotenv import load_dotenv
from loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

# Load environment variables
load_dotenv()

# Setup traditional logging as fallback
logger = logging.getLogger(__name__)

class UnusualWhalesClient:
    """
    Production client for the Unusual Whales API focused on options flow and unusual activity.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.unusualwhales.com"):
        """
        Initialize the Unusual Whales API client.
        
        Args:
            api_key: API key for Unusual Whales (defaults to UNUSUAL_WHALES_API_KEY environment variable)
            base_url: Base URL for Unusual Whales API (defaults to https://api.unusualwhales.com)
        """
        # Initialize observability tools
        self.loki = LokiManager(service_name="data-unusual-whales")
        self.prom = PrometheusManager(service_name="data-unusual-whales")
        
        # Create metrics
        self.request_counter = self.prom.create_counter(
            "unusual_whales_api_requests_total", 
            "Total count of Unusual Whales API requests",
            ["endpoint", "purpose", "status"]
        )
        
        self.request_latency = self.prom.create_histogram(
            "unusual_whales_api_request_duration_seconds",
            "Unusual Whales API request duration in seconds",
            ["endpoint", "purpose"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.data_points_gauge = self.prom.create_gauge(
            "unusual_whales_data_points",
            "Number of data points returned from Unusual Whales API",
            ["endpoint", "purpose"]
        )
        
        # Initialize API client
        self.api_key = api_key or os.environ.get("UNUSUAL_WHALES_API_KEY")
        if not self.api_key:
            log_msg = "No Unusual Whales API key provided - API calls will fail"
            logger.warning(log_msg)
            self.loki.warning(log_msg, component="unusual_whales")
            
        # Ensure base URL doesn't have trailing slashes
        self.base_url = base_url.rstrip("/") if base_url else "https://api.unusualwhales.com"
        
        # Define the optimal endpoints for different data retrieval purposes
        self.optimal_endpoints = {
            "options_flow": {
                "path": "/api/stock/{ticker}/flow-recent",
                "method": "GET",
                "description": "Latest options flow"
            },
            "unusual_activity": {
                "path": "/api/stock/{ticker}/flow-alerts",
                "method": "GET",
                "description": "Latest flow alerts"
            },
            "dark_pool": {
                "path": "/api/darkpool/{ticker}",
                "method": "GET",
                "description": "Darkpool trades"
            },
            "flow_alerts": {
                "path": "/api/option-trades/flow-alerts",
                "method": "GET",
                "description": "Latest global flow alerts"
            }
        }
        
        logger.info("UnusualWhalesClient initialized")
        self.loki.info("UnusualWhalesClient initialized", component="unusual_whales")
        
        # Verify API connectivity
        self._verify_api_connectivity()

    def _verify_api_connectivity(self):
        """Verify that the API key and base URL are correct by making a simple test request."""
        try:
            # Try a simple API request to check connectivity
            test_url = f"{self.base_url}/api/option-trades/flow-alerts"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            logger.info(f"Verifying API connectivity with {test_url}")
            self.loki.info(f"Verifying API connectivity with {test_url}", 
                         component="unusual_whales", 
                         endpoint="connectivity_check")
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                log_msg = f"API connectivity verified: {self.base_url}"
                logger.info(log_msg)
                self.loki.info(log_msg, component="unusual_whales", status="success")
                
                # Track successful connectivity
                self.prom.increment_counter(
                    "unusual_whales_api_requests_total",
                    1,
                    endpoint="connectivity_check",
                    purpose="initialization",
                    status="success"
                )
            else:
                error_msg = f"API connectivity failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                logger.error(f"Please check the API base URL ({self.base_url}) and API key")
                
                self.loki.error(error_msg, 
                              component="unusual_whales", 
                              status_code=str(response.status_code),
                              endpoint="connectivity_check")
                
                # Track failed connectivity
                self.prom.increment_counter(
                    "unusual_whales_api_requests_total",
                    1,
                    endpoint="connectivity_check",
                    purpose="initialization",
                    status="error"
                )
        except Exception as e:
            error_msg = f"API connectivity check failed: {e}"
            logger.error(error_msg)
            logger.error(f"Please verify the API base URL ({self.base_url}) is accessible")
            
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          error_type="connectivity",
                          endpoint="connectivity_check")
            
            # Track failed connectivity
            self.prom.increment_counter(
                "unusual_whales_api_requests_total",
                1,
                endpoint="connectivity_check",
                purpose="initialization",
                status="exception"
            )

    def get_options_flow(self, ticker: str) -> Dict[str, Any]:
        """
        Get the latest options flow data for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Options flow data
        """
        return self.fetch_data(purpose="options_flow", ticker=ticker)
    
    def get_unusual_activity(self, ticker: str) -> Dict[str, Any]:
        """
        Get the latest unusual options activity for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Unusual activity data
        """
        return self.fetch_data(purpose="unusual_activity", ticker=ticker)
    
    def get_dark_pool(self, ticker: str) -> Dict[str, Any]:
        """
        Get the latest dark pool trades for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dark pool trading data
        """
        return self.fetch_data(purpose="dark_pool", ticker=ticker)
    
    def get_global_flow_alerts(self) -> Dict[str, Any]:
        """
        Get the latest global flow alerts across all tickers.
        
        Returns:
            Global flow alerts data
        """
        return self.fetch_data(purpose="flow_alerts")

    def fetch_data(self, purpose: str, **params) -> Dict[str, Any]:
        """
        Fetch data from Unusual Whales API using a predefined purpose.
        
        Args:
            purpose: The purpose of the data retrieval 
                    ("options_flow", "unusual_activity", "dark_pool", "flow_alerts")
            **params: Parameters for the API call (e.g., ticker)
            
        Returns:
            API response data
        """
        if purpose not in self.optimal_endpoints:
            error_msg = f"Unknown endpoint purpose: {purpose}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          purpose=purpose)
            return {"error": error_msg}
            
        endpoint = self.optimal_endpoints.get(purpose)
        if not endpoint:
            error_msg = f"No endpoint found for purpose: {purpose}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          purpose=purpose)
            return {"error": error_msg}
        
        path = endpoint.get("path")
        method = endpoint.get("method", "GET").upper()
        
        if not path:
            error_msg = "No path specified in endpoint"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          purpose=purpose)
            return {"error": error_msg}

        # Format the path with parameters (e.g., replace {ticker} with actual ticker)
        try:
            url_path = path.format(**params)
        except KeyError as e:
            error_msg = f"Missing required parameter: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          purpose=purpose,
                          params=str(params))
            return {"error": error_msg}

        # Construct full URL
        url = f"{self.base_url}{url_path}"
        
        # Prepare headers with authentication
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Prepare query parameters - exclude parameters already used in URL path
        query_params = {k: v for k, v in params.items() if "{" + k + "}" not in path}

        logger.info(f"Requesting {method} {url}")
        self.loki.info(f"Requesting {method} {url}", 
                     component="unusual_whales", 
                     purpose=purpose,
                     endpoint=url_path)
        
        try:
            start_time = time.time()
            
            if method == "GET":
                response = requests.get(url, headers=headers, params=query_params, timeout=15)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=query_params, timeout=15)
            else:
                error_msg = f"Unsupported HTTP method: {method}"
                logger.error(error_msg)
                self.loki.error(error_msg, 
                              component="unusual_whales", 
                              purpose=purpose,
                              method=method)
                return {"error": error_msg}

            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "unusual_whales_api_request_duration_seconds",
                elapsed,
                endpoint=path,
                purpose=purpose
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Count successful request
                self.prom.increment_counter(
                    "unusual_whales_api_requests_total",
                    1,
                    endpoint=path,
                    purpose=purpose,
                    status="success"
                )
                
                # Set gauge for data points if we have results
                results = data.get("results", [])
                if results:
                    data_points = len(results)
                    self.prom.set_gauge(
                        "unusual_whales_data_points",
                        data_points,
                        endpoint=path,
                        purpose=purpose
                    )
                
                log_msg = f"API request successful ({elapsed:.2f}s)"
                logger.info(log_msg)
                self.loki.info(log_msg, 
                             component="unusual_whales", 
                             purpose=purpose,
                             endpoint=path,
                             duration=f"{elapsed:.2f}",
                             data_points=len(results) if results else 0)
                
                return data
            else:
                error_msg = f"API request failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                self.loki.error(error_msg, 
                              component="unusual_whales", 
                              purpose=purpose,
                              endpoint=path,
                              status_code=str(response.status_code))
                
                # Count failed request
                self.prom.increment_counter(
                    "unusual_whales_api_requests_total",
                    1,
                    endpoint=path,
                    purpose=purpose,
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
                          component="unusual_whales", 
                          purpose=purpose,
                          endpoint=path,
                          error_type="request_exception")
            
            # Count failed request
            self.prom.increment_counter(
                "unusual_whales_api_requests_total",
                1,
                endpoint=path,
                purpose=purpose,
                status="exception"
            )
            
            return {"error": str(e)}
            
        except ValueError as e:
            error_msg = f"JSON parsing failed: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          purpose=purpose,
                          endpoint=path,
                          error_type="json_parse")
            
            # Count failed request
            self.prom.increment_counter(
                "unusual_whales_api_requests_total",
                1,
                endpoint=path,
                purpose=purpose,
                status="parse_error"
            )
            
            return {"error": "Failed to parse response"}
            
        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg, 
                          component="unusual_whales", 
                          purpose=purpose,
                          endpoint=path,
                          error_type="unexpected")
            
            # Count failed request
            self.prom.increment_counter(
                "unusual_whales_api_requests_total",
                1,
                endpoint=path,
                purpose=purpose,
                status="error"
            )
            
            return {"error": str(e)}
    
    def search_flow_by_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for options flow based on specific criteria.
        
        Args:
            criteria: Dictionary with search criteria such as:
                     - sentiment: str ("bullish", "bearish", "neutral")
                     - min_premium: float (minimum premium in dollars)
                     - contract_type: str ("call", "put")
                     - min_volume: int (minimum volume)
                     - expiration_range: tuple (start_date, end_date)
            
        Returns:
            Options flow data matching the criteria
        """
        # This would typically use a specific search endpoint
        # For now we'll use the flow_alerts endpoint and filter client-side
        logger.info(f"Searching flow with criteria: {criteria}")
        
        # Get global flow alerts
        flow_data = self.get_global_flow_alerts()
        
        # Check if there was an error
        if "error" in flow_data:
            return flow_data
            
        # Extract the results
        results = flow_data.get("results", [])
        if not results:
            return {"results": []}
            
        # Filter based on criteria
        filtered_results = []
        for item in results:
            matches = True
            
            # Check sentiment
            if "sentiment" in criteria and item.get("sentiment") != criteria["sentiment"]:
                matches = False
                
            # Check premium
            if "min_premium" in criteria and (not item.get("premium") or 
                                             float(item.get("premium", 0)) < criteria["min_premium"]):
                matches = False
                
            # Check contract type
            if "contract_type" in criteria and item.get("contract_type") != criteria["contract_type"]:
                matches = False
                
            # Check volume
            if "min_volume" in criteria and (not item.get("volume") or 
                                           int(item.get("volume", 0)) < criteria["min_volume"]):
                matches = False
                
            # Check expiration
            if "expiration_range" in criteria and item.get("expiration"):
                exp_range = criteria["expiration_range"]
                exp_date = item.get("expiration")
                if exp_date < exp_range[0] or exp_date > exp_range[1]:
                    matches = False
            
            if matches:
                filtered_results.append(item)
                
        return {"results": filtered_results}
