import os
import unittest
from dotenv import load_dotenv
from mcp_tools.data_mcp.reddit_mcp import RedditMCP

class TestRedditMCP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables from .env file
        load_dotenv()

    def setUp(self):
        # Load config from file
        config_path = os.path.join(os.path.dirname(__file__), "../../../config/data_mcp/reddit_mcp_config.json")
        with open(config_path, "r") as f:
            import json
            config = json.load(f)
        self.mcp = RedditMCP(config)

    def test_fetch_subreddit_posts(self):
        # This test will fetch live data from Reddit's API
        data = self.mcp.fetch_data("recent_posts", {"subreddit": "python", "limit": 5})
        self.assertTrue(isinstance(data, dict))
        # Check for expected keys
        self.assertIn("posts", data)

if __name__ == "__main__":
    unittest.main()