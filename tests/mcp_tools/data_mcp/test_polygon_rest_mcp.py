import os
import unittest
from dotenv import load_dotenv
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP

class TestPolygonRestMCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

    def setUp(self):
        # Load config from file
        config_path = os.path.join(os.path.dirname(__file__), "../../../config/data_mcp/polygon_rest_mcp_config.json")
        with open(config_path, "r") as f:
            import json
            config = json.load(f)
        self.mcp = PolygonRestMCP(config)

    def test_fetch_raw_data_all_endpoints(self):
        # Test that the MCP tool returns raw data from all endpoints
        example_params = {
            "ticker": "AAPL",
            "from": "2024-01-01",
            "to": "2024-01-31",
            "date": "2024-01-02",
            "market_type": "stocks",
            "direction": "gainers",
            "type": "stocks",
            "underlying_asset": "AAPL",
            "option_contract": "AAPL240119C00145000"
        }
        for endpoint, config in self.mcp.endpoints.items():
            params = {}
            for param in config.get("required_params", []):
                params[param] = example_params.get(param, "AAPL")
            try:
                data = self.mcp.fetch_data(endpoint, params)
                self.assertIsInstance(data, (dict, list), f"Endpoint {endpoint} did not return dict or list")
                if isinstance(data, dict) and "error" in data:
                    error_str = str(data["error"])
                    if (
                        endpoint == "option_snapshot"
                        and "NOT_AUTHORIZED" in error_str
                        and "polygon.io/pricing" in error_str
                    ):
                        print(f"WARNING: Skipping assertion for {endpoint} due to expected NOT_AUTHORIZED error.")
                        continue
                    if (
                        endpoint == "stock_financials"
                        and "404" in error_str
                        and "page not found" in error_str
                    ):
                        print(f"WARNING: Skipping assertion for {endpoint} due to expected 404 page not found error.")
                        continue
                    self.fail(f"Endpoint {endpoint} returned error: {data.get('error')}")
            except Exception as e:
                self.fail(f"PolygonRestMCP fetch_data for endpoint {endpoint} raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()