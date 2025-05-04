#!/usr/bin/env python3
"""
Standalone test for Polygon.io API without depending on the data package.
"""

import os
import json
import logging
import time
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("polygon_direct_test")

# Load environment variables
load_dotenv()

class SimplePolygonClient:
    """
    Simple client for the Polygon.io REST API.
    """
    
    def __init__(self, api_key=None, base_url="https://api.polygon.io"):
        """
        Initialize the Polygon.io REST API client.
        
        Args:
            api_key: API key for Polygon.io (defaults to POLYGON_API_KEY environment variable)
            base_url: Base URL for Polygon.io API
        """
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            logger.warning("No Polygon API key provided - API calls will fail")
        
        self.base_url = base_url
        logger.info("SimplePolygonClient initialized")

    def get_ticker_details(self, ticker):
        """
        Get details for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        url = f"{self.base_url}{endpoint}"
        
        params = {"apiKey": self.api_key}
        logger.info(f"Fetching ticker details for {ticker}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved ticker details for {ticker}")
                return data
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {"error": f"HTTP {response.status_code}", "details": response.text}
                
        except Exception as e:
            logger.error(f"Error fetching ticker details: {e}")
            return {"error": str(e)}

    def get_price_history(self, ticker, multiplier=1, timespan="day", from_date=None, to_date=None):
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            multiplier: The size of the timespan multiplier (e.g., 1, 2, etc.)
            timespan: The timespan unit (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
        
        Returns:
            Dictionary with historical price data
        """
        # Use last week if dates not provided
        if not from_date or not to_date:
            today = datetime.now()
            week_ago = today - timedelta(days=7)
            from_date = from_date or week_ago.strftime("%Y-%m-%d")
            to_date = to_date or today.strftime("%Y-%m-%d")
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        url = f"{self.base_url}{endpoint}"
        
        params = {"apiKey": self.api_key}
        logger.info(f"Fetching price history for {ticker} from {from_date} to {to_date}")
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Successfully retrieved price history for {ticker}")
                return data
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return {"error": f"HTTP {response.status_code}", "details": response.text}
                
        except Exception as e:
            logger.error(f"Error fetching price history: {e}")
            return {"error": str(e)}

def test_polygon_api():
    """Test the Polygon.io REST API with real API key."""
    
    # Initialize client
    logger.info("Initializing SimplePolygonClient...")
    polygon_client = SimplePolygonClient()
    
    # Test ticker details
    ticker = "AAPL"
    logger.info(f"Fetching company info for {ticker}...")
    ticker_details = polygon_client.get_ticker_details(ticker)
    
    if "error" in ticker_details:
        logger.error(f"Error fetching ticker details: {ticker_details['error']}")
    else:
        logger.info(f"Successfully retrieved info for {ticker}")
        company_name = ticker_details.get("results", {}).get("name", "Unknown")
        market_cap = ticker_details.get("results", {}).get("market_cap", "Unknown")
        logger.info(f"Company: {company_name}")
        logger.info(f"Market Cap: {market_cap}")
    
    # Test historical price data
    logger.info(f"Fetching historical price data for {ticker}...")
    
    # Calculate date range for past week
    today = datetime.now()
    week_ago = today - timedelta(days=7)
    
    # Format dates for Polygon API (YYYY-MM-DD)
    from_date = week_ago.strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    
    price_data = polygon_client.get_price_history(
        ticker=ticker,
        from_date=from_date,
        to_date=to_date
    )
    
    if "error" in price_data:
        logger.error(f"Error fetching price data: {price_data['error']}")
    else:
        results = price_data.get("results", [])
        num_results = len(results)
        logger.info(f"Successfully retrieved {num_results} days of price data")
        
        if num_results > 0:
            latest = results[-1]
            logger.info(f"Latest close price: ${latest.get('c', 'Unknown')}")
            logger.info(f"Volume: {latest.get('v', 'Unknown')}")

if __name__ == "__main__":
    logger.info("Starting Polygon API direct test...")
    test_polygon_api()
    logger.info("Polygon API test completed.")