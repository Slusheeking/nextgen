#!/usr/bin/env python3
"""
Test script for Polygon.io API integration.
"""

import os
import json
import logging
import sys
from datetime import datetime, timedelta

# Import directly from the module file to avoid __init__.py import issues
sys.path.append('.')  # Add current directory to path
from data.polygon_rest import PolygonRestClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("polygon_test")

def test_polygon_api():
    """Test the Polygon.io REST API with real API key."""
    
    logger.info("Initializing Polygon REST client...")
    polygon_client = PolygonRestClient()
    
    # Test ticker details
    ticker = "AAPL"
    logger.info(f"Fetching company info for {ticker}...")
    ticker_details = polygon_client.fetch_optimal_data("market_cap", ticker=ticker)
    
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
    
    price_data = polygon_client.fetch_optimal_data(
        "price_history",
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=from_date,
        to=to_date
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
    
    # Test market snapshot
    logger.info(f"Fetching market snapshot for {ticker}...")
    snapshot = polygon_client.fetch_optimal_data("snapshot", ticker=ticker)
    
    if "error" in snapshot:
        logger.error(f"Error fetching snapshot: {snapshot['error']}")
    else:
        logger.info(f"Successfully retrieved market snapshot for {ticker}")
        last_trade = snapshot.get("last", {}).get("price", "Unknown")
        logger.info(f"Last trade price: ${last_trade}")

if __name__ == "__main__":
    logger.info("Starting Polygon API test...")
    test_polygon_api()
    logger.info("Polygon API test completed.")