import os
import unittest
from dotenv import load_dotenv
from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP

class TestYahooFinanceMCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

    def setUp(self):
        # Load config from file
        config_path = os.path.join(os.path.dirname(__file__), "../../../config/data_mcp/yahoo_finance_mcp_config.json")
        with open(config_path, "r") as f:
            import json
            config = json.load(f)
        self.mcp = YahooFinanceMCP(config)

    def test_fetch_quote(self):
        # This test will fetch live quote data for AAPL from Yahoo Finance API
        data = self.mcp.fetch_data("quote", {"symbol": "AAPL"})
        self.assertTrue(isinstance(data, dict) or isinstance(data, list))

if __name__ == "__main__":
    unittest.main()