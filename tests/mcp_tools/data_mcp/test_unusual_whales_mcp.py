import os
import unittest
from dotenv import load_dotenv
from mcp_tools.data_mcp.unusual_whales_mcp import UnusualWhalesMCP

class TestUnusualWhalesMCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

    def setUp(self):
        # Load config from file
        config_path = os.path.join(os.path.dirname(__file__), "../../../config/data_mcp/unusual_whales_mcp_config.json")
        with open(config_path, "r") as f:
            import json
            config = json.load(f)
        self.mcp = UnusualWhalesMCP(config)

    def test_fetch_options_flow(self):
        # This test will fetch live options flow data from Unusual Whales API
        data = self.mcp.fetch_data("options_flow", {"symbol": "AAPL", "limit": 5})
        self.assertTrue(isinstance(data, dict) or isinstance(data, list))

if __name__ == "__main__":
    unittest.main()