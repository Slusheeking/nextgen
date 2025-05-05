import os
import unittest
from dotenv import load_dotenv
from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWsMCP

class TestPolygonWsMCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

    def setUp(self):
        # Load config from file
        config_path = os.path.join(os.path.dirname(__file__), "../../../config/data_mcp/polygon_ws_mcp_config.json")
        with open(config_path, "r") as f:
            import json
            config = json.load(f)
        self.mcp = PolygonWsMCP(config)

    def test_subscribe_to_stream(self):
        # Test that the MCP tool can subscribe to a stream and returns a valid status
        try:
            result = self.mcp.subscribe_to_stream("trades", ["AAPL"])
            self.assertIsInstance(result, dict)
            self.assertIn("subscription_id", result)
            self.assertIn("status", result)
        except Exception as e:
            self.fail(f"PolygonWsMCP subscribe_to_stream raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()