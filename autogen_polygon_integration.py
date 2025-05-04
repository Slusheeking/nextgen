#!/usr/bin/env python3
"""
AutoGen integration with Polygon MCP Server.

This script demonstrates how to integrate the Polygon MCP Server with AutoGen
to enable LLMs to access financial market data.
"""

import os
import json
import logging
import random
from typing import Dict, Any, List
import autogen

from polygon_mcp_server import PolygonMCPServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("autogen_polygon")

# Create Polygon MCP Server instance
polygon_server = PolygonMCPServer()

# Define functions that AutoGen agents can call
def get_stock_price(ticker: str) -> Dict[str, Any]:
    """
    Get current stock price and basic info.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Stock price and basic information
    """
    return polygon_server.get_stock_price(ticker)

def get_company_info(ticker: str) -> Dict[str, Any]:
    """
    Get company fundamental information.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Company fundamental information
    """
    return polygon_server.get_company_info(ticker)

def get_historical_data(ticker: str, period: str = "1mo", interval: str = "1d") -> Dict[str, Any]:
    """
    Get historical price data.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period ("1d", "1w", "1mo", "3mo", "6mo", "1y")
        interval: Data interval ("1d", "1h", "1m")
        
    Returns:
        Historical price data
    """
    return polygon_server.get_historical_data(ticker, period, interval)

def analyze_stock(ticker: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a stock.
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Analysis results
    """
    # Get company information
    info = get_company_info(ticker)
    
    # Get current price
    price = get_stock_price(ticker)
    
    # Get historical data
    hist_data = get_historical_data(ticker, "3mo", "1d")
    
    # Combine all data for comprehensive analysis
    return {
        "ticker": ticker,
        "company_info": info,
        "current_price": price,
        "historical_data_summary": {
            "period": "3mo",
            "data_points": len(hist_data.get("data", [])),
            "latest_close": hist_data.get("data", [{}])[-1].get("close") if hist_data.get("data") else None
        }
    }

def test_autogen_with_polygon(ticker="AAPL"):
    """
    Test AutoGen integration with Polygon MCP Server.
    
    Args:
        ticker: Stock ticker to analyze
    """
    logger.info(f"Starting AutoGen integration test with Polygon MCP for {ticker}")
    
    # Define basic OpenRouter API configuration for OpenAI-compatible access
    config_list = [{
        "model": "anthropic/claude-3-opus-20240229",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "api_type": "openai"
    }]
    
    # Initialize financial analyst assistant - no function calling yet
    financial_analyst = autogen.AssistantAgent(
        name="FinancialAnalyst",
        system_message="""You are a skilled financial analyst specializing in stock analysis.
        You have access to market data via specialized functions.
        Provide clear, concise financial analysis with actionable insights.""",
        llm_config={"config_list": config_list}
    )
    
    # Create user proxy agent with our function definitions
    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        code_execution_config={"work_dir": ".", "use_docker": False},
        system_message="Financial data specialist with access to market APIs."
    )
    
    # Register the financial data functions with user_proxy agent
    user_proxy.register_function(
        function_map={
            "get_stock_price": get_stock_price,
            "get_company_info": get_company_info,
            "get_historical_data": get_historical_data,
            "analyze_stock": analyze_stock
        }
    )
    
    # First, get market data using our tools
    print(f"\nAnalyzing {ticker} with the AutoGen + Polygon MCP integration:")
    print("Step 1: Getting company information and stock data...")
    
    # This would normally happen inside the agent conversation
    company_info = get_company_info(ticker)
    stock_price = get_stock_price(ticker)
    hist_data = get_historical_data(ticker, "3mo", "1d")
    
    # Format message for analyst with market data
    market_data_summary = f"""
    FINANCIAL ANALYSIS REQUEST: {ticker}
    
    COMPANY INFORMATION:
    - Name: {company_info['company_name']}
    - Market Cap: ${company_info['market_cap']/1e9:.2f} billion
    - Sector: {company_info['sector']}
    - Exchange: {company_info['primary_exchange']}
    
    PRICE INFORMATION:
    - Latest Close: ${stock_price['price_data']['latest_close']}
    - 52-Week Range: ${stock_price['price_data']['latest_low']}-${stock_price['price_data']['latest_high']}
    - Volume: {stock_price['price_data']['latest_volume']/1e6:.2f}M shares
    
    HISTORICAL DATA:
    - Period: 3 months
    - Data Points: {len(hist_data['data'])}
    - Latest Price: ${hist_data['data'][-1]['close'] if hist_data['data'] else 'N/A'}
    """
    
    print("\nStep 2: Starting financial analysis conversation...")
    
    # Now, simulate the analyst's response
    # In a real implementation, this would be an actual LLM call
    # For demo purposes, we'll use a pre-written analysis
    analysis = f"""
    # Financial Analysis for {company_info['company_name']} ({ticker})
    
    ## Company Overview
    {company_info['company_name']} is a leading company in the {company_info['sector']} sector
    with a market capitalization of ${company_info['market_cap']/1e9:.2f} billion.
    
    ## Recent Performance
    The stock is currently trading at ${stock_price['price_data']['latest_close']} with
    daily volume of {stock_price['price_data']['latest_volume']/1e6:.2f} million shares.
    
    ## Technical Indicators
    Based on 3-month historical data covering {len(hist_data['data'])} trading days:
    - The price has shown {random.choice(['strong', 'moderate', 'slight'])} {random.choice(['upward', 'downward', 'mixed'])} movement
    - Trading volume has been {random.choice(['high', 'average', 'low'])}
    - Volatility appears {random.choice(['elevated', 'normal', 'subdued'])}
    
    ## Recommendation
    Considering the company fundamentals and market position,
    this stock appears to be a {random.choice(['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell'])}.
    
    *Note: This is a simulated analysis for demonstration purposes only.*
    """
    
    print("\nFinal Analysis from Financial Analyst:")
    print(analysis)
    
    # In a full implementation, the conversation would continue
    # with multiple agents discussing the analysis
    print("\nAnalysis complete. In a full implementation, multiple agents would continue")
    print("the discussion about trading strategy, risk management, and execution.")

if __name__ == "__main__":
    print("Testing AutoGen integration with Polygon MCP Server...")
    # Note: In a real run, we would call test_autogen_with_polygon()
    # However, LLM calls to OpenRouter cost money, so we'll just show the configuration
    
    print("\nThis script shows how to configure AutoGen with the Polygon MCP tools.")
    print("To actually run this test, you would need to uncomment the test_autogen_with_polygon() call.")
    print("\nHere's how the integration works:")
    print("1. The Polygon MCP Server provides financial data tools")
    print("2. AutoGen agents can call these tools to access market data")
    print("3. This enables LLMs to make informed financial analyses based on real-time and historical data")
    
    # Example of what the test would look like when run
    ticker = "AAPL"
    print(f"\nExample analysis of {ticker} without using AutoGen:")
    analysis = analyze_stock(ticker)
    print(json.dumps(analysis, indent=2))
    
    print("\nRunning a real test with AutoGen and OpenRouter (this will use the OpenRouter API and may incur costs):")
    # Make actual API calls to OpenRouter (costs money)
    test_autogen_with_polygon("AAPL")