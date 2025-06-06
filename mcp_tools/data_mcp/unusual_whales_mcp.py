"""
Unusual Whales MCP Server

This module implements a Model Context Protocol (MCP) server for the Unusual Whales
API, providing access to options flow and unusual activity data.
"""

import os
import time

from dotenv import load_dotenv
load_dotenv(dotenv_path='/home/ubuntu/nextgen/.env')
from typing import Dict, Any, Optional

# Direct imports with graceful error handling

# Try to import required dependencies
try:
    import requests
except ImportError:
    requests = None

try:
    import dotenv
except ImportError:
    dotenv = None

# Import monitoring components
from monitoring.netdata_logger import NetdataLogger
from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP

# Load environment variables


class UnusualWhalesMCP(BaseDataMCP):
    """
    MCP server for Unusual Whales API.

    This server provides access to options flow, unusual activity, and dark pool
    data from Unusual Whales.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Unusual Whales MCP server.

        Args:
            config: Optional configuration dictionary. May contain:
                - api_key: Unusual Whales API key (overrides environment variable)
                - base_url: Base URL for API
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 300)
        """
        super().__init__(name="unusual_whales_mcp", config=config)

        # Initialize monitoring/logger
        self.logger = NetdataLogger(component_name="unusual-whales-mcp")
        self.logger.info("UnusualWhalesMCP initialized")

        # Initialize client configuration
        self.whales_client = self._initialize_client()

        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()

        # Register specific tools
        self._register_specific_tools()

        self.logger.info(
            f"UnusualWhalesMCP initialized with {len(self.endpoints)} endpoints"
        )
        # Removed self.monitor.log_info for endpoints

    def _initialize_client(self) -> Dict[str, Any]:
        """
        Initialize Unusual Whales client configuration.

        Returns:
            Client configuration or None if initialization fails
        """
        try:
            # Prioritize environment variable for API key
            api_key = os.environ.get("UNUSUAL_WHALES_API_KEY") or self.config.get("api_key")
            base_url = self.config.get("base_url", "https://api.unusualwhales.com").rstrip("/")

            if not api_key:
                self.logger.error("No Unusual Whales API key provided - API calls will fail")
                return None

            if api_key == "${UNUSUAL_WHALES_API_KEY}":
                self.logger.error("Unusual Whales API key is still set to the placeholder value. Please set the actual API key.")
                return None

            # Mask the API key for logging
            masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
            self.logger.info(f"Loaded Unusual Whales API key: {masked_key}")

            # Verify API connectivity
            connectivity_result = self._verify_api_connectivity(api_key, base_url)
            if not isinstance(connectivity_result, dict) or not connectivity_result.get('success'):
                error_msg = connectivity_result.get('message') if isinstance(connectivity_result, dict) else str(connectivity_result)
                self.logger.error(f"API connectivity verification failed: {error_msg}")
                return {"error": error_msg}

            self.logger.info("Unusual Whales client initialized successfully")
            return {"api_key": api_key, "base_url": base_url}

        except Exception as e:
            self.logger.error(f"Failed to initialize Unusual Whales client: {str(e)}")
            self.logger.counter("error_count", 1)
            return {"error": str(e)}

    def _verify_api_connectivity(self, api_key: str, base_url: str) -> Dict[str, Any]:
        """
        Verify API connectivity by making a simple test request.

        Args:
            api_key: API key
            base_url: Base URL

        Returns:
            Dictionary with success status and message
        """
        if not api_key:
            self.logger.error("No API key provided for Unusual Whales")
            return {"success": False, "message": "No API key provided"}

        try:
            # Try a simple API request to check connectivity
            test_url = f"{base_url}/api/option-trades/flow-alerts"
            headers = {"Authorization": f"Bearer {api_key}"}

            self.logger.info(f"Verifying API connectivity with {test_url}")
            response = requests.get(test_url, headers=headers, timeout=10)

            if response.status_code == 200:
                self.logger.info(f"API connectivity verified: {base_url}")
                return {"success": True, "message": "API connectivity verified"}
            else:
                error_msg = f"API connectivity failed with status {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                self.logger.error(
                    f"Please check the API base URL ({base_url}) and API key"
                )
                return {"success": False, "message": error_msg}

        except requests.RequestException as e:
            error_msg = f"API connectivity check failed: {e}"
            self.logger.error(error_msg)
            self.logger.counter("error_count", 1)
            self.logger.error(
                f"Please verify the API base URL ({base_url}) is accessible"
            )
            return {"success": False, "message": error_msg}

    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Unusual Whales API.

        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "options_flow": {
                "path": "/api/stock/{ticker}/flow-recent",
                "method": "GET",
                "description": "Get the latest options flow data for a specific ticker",
                "category": "options_flow",
                "required_params": ["ticker"],
                "optional_params": ["limit"],
                "default_values": {"limit": "50"},
                "handler": self._handle_options_flow,
            },
            "unusual_activity": {
                "path": "/api/stock/{ticker}/flow-alerts",
                "method": "GET",
                "description": "Get the latest unusual options activity for a specific ticker",
                "category": "options_flow",
                "required_params": ["ticker"],
                "optional_params": ["limit"],
                "default_values": {"limit": "50"},
                "handler": self._handle_unusual_activity,
            },
            "dark_pool": {
                "path": "/api/darkpool/{ticker}",
                "method": "GET",
                "description": "Get the latest dark pool trades for a specific ticker",
                "category": "dark_pool",
                "required_params": ["ticker"],
                "optional_params": ["limit"],
                "default_values": {"limit": "50"},
                "handler": self._handle_dark_pool,
            },
            "global_flow_alerts": {
                "path": "/api/option-trades/flow-alerts",
                "method": "GET",
                "description": "Get the latest global flow alerts across all tickers",
                "category": "options_flow",
                "required_params": [],
                "optional_params": ["limit"],
                "default_values": {"limit": "50"},
                "handler": self._handle_global_flow_alerts,
            },
            "flow_by_criteria": {
                "description": "Search for options flow based on specific criteria",
                "category": "options_flow",
                "required_params": ["criteria"],
                "optional_params": ["limit"],
                "default_values": {"limit": "50"},
                "handler": self._handle_flow_by_criteria,
            },
        }

    def _register_specific_tools(self):
        """Register tools specific to Unusual Whales API."""
        self.register_tool(self.get_options_flow)
        self.register_tool(self.get_unusual_activity)
        self.register_tool(self.get_dark_pool)
        self.register_tool(self.get_global_flow_alerts)
        self.register_tool(self.search_flow_by_criteria)

    def _execute_endpoint_fetch(
        self, endpoint_config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the fetch operation based on endpoint configuration.

        Args:
            endpoint_config: Configuration for the endpoint
            params: Parameters for the request

        Returns:
            Fetched data

        Raises:
            Exception: If the fetch fails
        """
        # Merge provided params with default values
        merged_params = {**endpoint_config.get("default_values", {}), **params}

        # Get the handler function for this endpoint
        handler = endpoint_config.get("handler")
        if handler and callable(handler):
            return handler(merged_params)

        # Default handling if no specific handler
        path = endpoint_config.get("path")
        method = endpoint_config.get("method", "GET")

        if not path:
            raise ValueError("No path specified in endpoint configuration")

        # Format the path with path parameters
        try:
            formatted_path = path.format(**merged_params)
        except KeyError as e:
            raise ValueError(f"Missing required parameter: {e}")

        # Extract query parameters (those not used in path formatting)
        query_params = {
            k: v for k, v in merged_params.items() if "{" + k + "}" not in path
        }

        # Execute the API request
        return self._execute_api_request(formatted_path, method, query_params)

    def _execute_api_request(
        self, path: str, method: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute an API request to Unusual Whales.

        Args:
            path: API path
            method: HTTP method
            params: Query parameters

        Returns:
            API response data

        Raises:
            Exception: If the request fails
        """
        if not self.whales_client:
            return {"error": "Unusual Whales client not initialized"}

        api_key = self.whales_client.get("api_key")
        base_url = self.whales_client.get("base_url")

        if not api_key:
            return {"error": "No API key provided"}

        # Construct full URL
        url = f"{base_url}{path}"

        # Check if the API key is the literal string "${UNUSUAL_WHALES_API_KEY}"
        if api_key == "${UNUSUAL_WHALES_API_KEY}":
            self.logger.error("API key is not properly set. It's still the placeholder value.")
            return {"error": "API key is not properly configured"}

        # Log the raw API key (for debugging purposes only, remove in production)
        self.logger.debug(f"Raw API key: {api_key}")

        # Prepare headers with authentication
        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"FinGPT/NextGen-{os.environ.get('VERSION', '1.0.0')}",
            "X-Request-Source": "fingpt-mcp-unusual-whales",
        }

        # Log the API key being used (masked for security)
        masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
        self.logger.info(f"Using API key: {masked_key}")
        
        # Log the full Authorization header (masked) for debugging
        masked_auth_header = f"Bearer {masked_key}"
        self.logger.debug(f"Authorization header: {masked_auth_header}")

        # Log the actual Authorization header being sent (for debugging purposes only, remove in production)
        self.logger.debug(f"Actual Authorization header: {headers['Authorization']}")

        # Execute request with retry logic
        retry_count = 0
        max_retries = 3
        retry_delay = 1.0  # seconds

        while retry_count <= max_retries:
            try:
                start_time = time.time()
                if method == "GET":
                    response = requests.get(
                        url, params=params, headers=headers, timeout=15
                    )
                elif method == "POST":
                    response = requests.post(
                        url, json=params, headers=headers, timeout=15
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                elapsed = (time.time() - start_time) * 1000  # ms
                self.logger.counter("external_api_call_count", 1)
                self.logger.timing("data_fetch_time_ms", elapsed)
                self.logger.gauge("response_status_code", response.status_code)
                if response.content:
                    self.logger.gauge("data_volume_bytes", len(response.content))

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limit exceeded
                    self.logger.counter("rate_limit_count", 1)
                    retry_count += 1
                    if retry_count <= max_retries:
                        sleep_time = retry_delay * (
                            2 ** (retry_count - 1)
                        )  # Exponential backoff
                        self.logger.warning(
                            f"Rate limit exceeded, retrying in {sleep_time}s (attempt {retry_count}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        continue
                elif response.status_code == 401:  # Unauthorized
                    error_msg = f"API request failed due to authentication error: {response.status_code} - {response.text}"
                    self.logger.error(error_msg)
                    return {"error": error_msg}

                # Handle other error codes
                error_msg = (
                    f"API request failed: {response.status_code} - {response.text}"
                )
                self.logger.error(error_msg)
                return {"error": error_msg}

            except requests.RequestException as e:
                retry_count += 1
                if retry_count <= max_retries:
                    sleep_time = retry_delay * (2 ** (retry_count - 1))
                    self.logger.warning(
                        f"Request failed, retrying in {sleep_time}s (attempt {retry_count}/{max_retries}): {e}"
                    )
                    time.sleep(sleep_time)
                else:
                    error_msg = f"Request failed after {max_retries} retries: {e}"
                    self.logger.error(error_msg)
                    self.logger.counter("error_count", 1)
                    return {"error": error_msg}

    # Handler methods for specific endpoints

    def _handle_options_flow(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle options flow endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        path = f"/api/stock/{ticker}/flow-recent"
        return self._execute_api_request(path, "GET", params)

    def _handle_unusual_activity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle unusual activity endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        path = f"/api/stock/{ticker}/flow-alerts"
        return self._execute_api_request(path, "GET", params)

    def _handle_dark_pool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dark pool endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")

        path = f"/api/darkpool/{ticker}"
        return self._execute_api_request(path, "GET", params)

    def _handle_global_flow_alerts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle global flow alerts endpoint."""
        path = "/api/option-trades/flow-alerts"
        return self._execute_api_request(path, "GET", params)

    def _handle_flow_by_criteria(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle flow by criteria endpoint.

        This endpoint allows searching for options flow based on specific criteria
        such as sentiment, premium, contract type, etc.
        """
        criteria = params.get("criteria")
        if not criteria:
            raise ValueError("Missing required parameter: criteria")

        # Get global flow alerts
        flow_data = self._handle_global_flow_alerts(params)

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
            if (
                "sentiment" in criteria
                and item.get("sentiment") != criteria["sentiment"]
            ):
                matches = False

            # Check premium
            if "min_premium" in criteria and (
                not item.get("premium")
                or float(item.get("premium", 0)) < criteria["min_premium"]
            ):
                matches = False

            # Check contract type
            if (
                "contract_type" in criteria
                and item.get("contract_type") != criteria["contract_type"]
            ):
                matches = False

            # Check volume
            if "min_volume" in criteria and (
                not item.get("volume")
                or int(item.get("volume", 0)) < criteria["min_volume"]
            ):
                matches = False

            # Check expiration
            if "expiration_range" in criteria and item.get("expiration"):
                exp_range = criteria["expiration_range"]
                exp_date = item.get("expiration")
                if exp_date < exp_range[0] or exp_date > exp_range[1]:
                    matches = False

            if matches:
                filtered_results.append(item)

        return {"results": filtered_results, "count": len(filtered_results)}

    # Public API methods for models to use directly

    def get_options_flow(self, ticker: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get the latest options flow data for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of results to return

        Returns:
            Options flow data
        """
        return self.fetch_data("options_flow", {"ticker": ticker, "limit": str(limit)})

    def get_unusual_activity(self, ticker: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get the latest unusual options activity for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of results to return

        Returns:
            Unusual activity data
        """
        return self.fetch_data(
            "unusual_activity", {"ticker": ticker, "limit": str(limit)}
        )

    def get_dark_pool(self, ticker: str, limit: int = 50) -> Dict[str, Any]:
        """
        Get the latest dark pool trades for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of results to return

        Returns:
            Dark pool trading data
        """
        return self.fetch_data("dark_pool", {"ticker": ticker, "limit": str(limit)})

    def get_global_flow_alerts(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get the latest global flow alerts across all tickers.

        Args:
            limit: Maximum number of results to return

        Returns:
            Global flow alerts data
        """
        return self.fetch_data("global_flow_alerts", {"limit": str(limit)})

    def search_flow_by_criteria(
        self, criteria: Dict[str, Any], limit: int = 50
    ) -> Dict[str, Any]:
        """
        Search for options flow based on specific criteria.

        Args:
            criteria: Dictionary with search criteria such as:
                    - sentiment: str ("bullish", "bearish", "neutral")
                    - min_premium: float (minimum premium in dollars)
                    - contract_type: str ("call", "put")
                    - min_volume: int (minimum volume)
                    - expiration_range: tuple (start_date, end_date)
            limit: Maximum number of results to return

        Returns:
            Options flow data matching the criteria
        """
        return self.fetch_data(
            "flow_by_criteria", {"criteria": criteria, "limit": str(limit)}
        )

    @property
    def api_key(self) -> str:
        """Get the API key from the client configuration."""
        if not self.whales_client:
            return ""
        return self.whales_client.get("api_key", "")

    @property
    def base_url(self) -> str:
        """Get the base URL from the client configuration."""
        if not self.whales_client:
            return "https://api.unusualwhales.com"
        return self.whales_client.get("base_url", "https://api.unusualwhales.com")
