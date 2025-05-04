#!/usr/bin/env python3
"""
Simple Polygon.io MCP Server implementation.

This module creates a Model Context Protocol (MCP) server that provides
tools for accessing Polygon.io market data.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any, Optional

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("polygon_mcp_server")

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

class PolygonMCPServer:
    """
    MCP server for Polygon.io data.
    """
    
    def __init__(self):
        """Initialize the Polygon MCP server."""
        self.polygon_client = SimplePolygonClient()
        logger.info("Polygon MCP Server initialized")
        
    def get_stock_price(self, ticker: str):
        """
        MCP Tool: Get current stock price and basic info.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Stock price and basic information
        """
        logger.info(f"MCP Tool 'get_stock_price' called with ticker={ticker}")
        
        # Get ticker details
        ticker_details = self.polygon_client.get_ticker_details(ticker)
        
        # Get latest price data (last week)
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        from_date = week_ago.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        price_data = self.polygon_client.get_price_history(
            ticker=ticker,
            from_date=from_date,
            to_date=to_date
        )
        
        # Extract relevant information
        result = {
            "ticker": ticker,
            "company_name": ticker_details.get("results", {}).get("name", "Unknown"),
            "price_data": {}
        }
        
        # Add price data if available
        price_results = price_data.get("results", [])
        if price_results:
            latest = price_results[-1]
            result["price_data"] = {
                "latest_close": latest.get("c"),
                "latest_open": latest.get("o"),
                "latest_high": latest.get("h"),
                "latest_low": latest.get("l"),
                "latest_volume": latest.get("v"),
                "date": today.strftime("%Y-%m-%d")
            }
        
        return result
    
    def get_company_info(self, ticker: str):
        """
        MCP Tool: Get company fundamental information.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Company fundamental information
        """
        logger.info(f"MCP Tool 'get_company_info' called with ticker={ticker}")
        
        # Get ticker details
        ticker_details = self.polygon_client.get_ticker_details(ticker)
        
        # Extract relevant information
        results = ticker_details.get("results", {})
        
        result = {
            "ticker": ticker,
            "company_name": results.get("name", "Unknown"),
            "description": results.get("description", ""),
            "market_cap": results.get("market_cap"),
            "primary_exchange": results.get("primary_exchange", ""),
            "homepage_url": results.get("homepage_url", ""),
            "sector": results.get("sic_description", ""),
            "currency_name": results.get("currency_name", "USD")
        }
        
        return result
    
    def get_historical_data(self, ticker: str, period: str = "1mo", interval: str = "1d"):
        """
        MCP Tool: Get historical price data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period ("1d", "1w", "1mo", "3mo", "6mo", "1y")
            interval: Data interval ("1d", "1h", "1m")
            
        Returns:
            Historical price data
        """
        logger.info(f"MCP Tool 'get_historical_data' called with ticker={ticker}, period={period}, interval={interval}")
        
        # Map period to days
        period_mapping = {
            "1d": 1,
            "1w": 7,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
        }
        days = period_mapping.get(period, 30)
        
        # Map interval to Polygon timespan
        interval_mapping = {
            "1m": "minute",
            "1h": "hour",
            "1d": "day"
        }
        timespan = interval_mapping.get(interval, "day")
        
        # Calculate date range
        today = datetime.now()
        start_date = today - timedelta(days=days)
        
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        # Get price data
        price_data = self.polygon_client.get_price_history(
            ticker=ticker,
            timespan=timespan,
            from_date=from_date,
            to_date=to_date
        )
        
        # Convert to a cleaner format
        results = price_data.get("results", [])
        formatted_data = [
            {
                "date": datetime.fromtimestamp(item["t"]/1000).strftime("%Y-%m-%d"),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v")
            }
            for item in results
        ]
        
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "data": formatted_data
        }

def print_mcp_server_info():
    """Print information about the MCP server tools."""
    print("\nPolygon MCP Server Tools:")
    print("------------------------")
    print("1. get_stock_price(ticker)")
    print("   - Get current stock price and basic information")
    print("   - Example: get_stock_price(ticker='AAPL')")
    print("\n2. get_company_info(ticker)")
    print("   - Get company fundamental information")
    print("   - Example: get_company_info(ticker='MSFT')")
    print("\n3. get_historical_data(ticker, period, interval)")
    print("   - Get historical price data")
    print("   - Example: get_historical_data(ticker='TSLA', period='1mo', interval='1d')")
    print("\nUsage Example:")
    print("  server = PolygonMCPServer()")
    print("  apple_price = server.get_stock_price('AAPL')")
    print("  print(apple_price)")

if __name__ == "__main__":
    # Initialize server
    print("Initializing Polygon MCP Server...")
    server = PolygonMCPServer()
    
    # Print server info
    print_mcp_server_info()
    
    # Test with a sample query
    print("\nRunning test query for AAPL...")
    result = server.get_stock_price("AAPL")
    print(json.dumps(result, indent=2))