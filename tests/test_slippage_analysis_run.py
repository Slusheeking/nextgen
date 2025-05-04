#!/usr/bin/env python3
"""
Test script for the Slippage Analysis MCP module.

This script demonstrates the usage of the SlippageAnalysisMCP class by:
1. Creating sample trade and market data
2. Initializing the SlippageAnalysisMCP server
3. Running various slippage and market impact analysis methods
4. Displaying the results
"""

import numpy as np
import matplotlib.pyplot as plt
from mcp_tools.analysis_mcp.slippage_analysis_mcp import SlippageAnalysisMCP

def main():
    """Run tests on the SlippageAnalysisMCP module with sample data."""
    print("Testing Slippage Analysis MCP module...")
    
    # Initialize the SlippageAnalysisMCP server
    config = {
        "use_gpu": False,  # Force CPU usage for testing
    }
    slippage_analysis = SlippageAnalysisMCP(config)
    
    # Generate sample price data
    np.random.seed(42)  # For reproducibility
    
    # Sample price series (100 data points)
    base_prices = 100 + np.cumsum(np.random.normal(0, 1, 100)) * 0.1
    
    # Create pre and post trade price series
    pre_trade_prices = base_prices[:50]
    post_trade_prices = base_prices[50:] + 0.2  # Slight price increase after trade
    
    # Sample bid-ask data
    pre_trade_bid_ask = {
        "bid_price": 99.5,
        "ask_price": 100.5,
        "bid_size": 1000,
        "ask_size": 800
    }
    
    post_trade_bid_ask = {
        "bid_price": 99.4,
        "ask_price": 100.6,
        "bid_size": 900,
        "ask_size": 700
    }
    
    # Test 1: Calculate slippage
    print("\n1. Testing slippage calculation...")
    
    # Buy side slippage
    buy_slippage = slippage_analysis._handle_calculate_slippage({
        "expected_price": 100.0,
        "executed_price": 100.5,
        "side": "buy",
        "quantity": 100
    })
    
    print("Buy Side Slippage:")
    print(f"Expected Price: {buy_slippage['expected_price']}")
    print(f"Executed Price: {buy_slippage['executed_price']}")
    print(f"Slippage Amount: {buy_slippage['slippage_amount']}")
    print(f"Slippage Percent: {buy_slippage['slippage_percent']}%")
    print(f"Cost Impact: ${buy_slippage['cost_impact']}")
    
    # Sell side slippage
    sell_slippage = slippage_analysis._handle_calculate_slippage({
        "expected_price": 100.0,
        "executed_price": 99.5,
        "side": "sell",
        "quantity": 100
    })
    
    print("\nSell Side Slippage:")
    print(f"Expected Price: {sell_slippage['expected_price']}")
    print(f"Executed Price: {sell_slippage['executed_price']}")
    print(f"Slippage Amount: {sell_slippage['slippage_amount']}")
    print(f"Slippage Percent: {sell_slippage['slippage_percent']}%")
    print(f"Cost Impact: ${sell_slippage['cost_impact']}")
    
    # Test 2: Analyze market impact
    print("\n2. Testing market impact analysis...")
    market_impact = slippage_analysis._handle_analyze_market_impact({
        "pre_trade_prices": pre_trade_prices.tolist(),
        "post_trade_prices": post_trade_prices.tolist(),
        "trade_size": 1000,
        "average_volume": 10000,
        "time_window": 5
    })
    
    print(f"Price Change: {market_impact['price_change']}")
    print(f"Price Change Percent: {market_impact['price_change_percent']}%")
    print(f"Volatility Change: {market_impact['volatility_change']}")
    print(f"Volatility Change Percent: {market_impact['volatility_change_percent']}%")
    print(f"Relative Trade Size: {market_impact['relative_trade_size']}")
    print(f"Estimated Impact: {market_impact['estimated_impact']}")
    print(f"Computed On: {market_impact['computed_on']}")
    
    # Test 3: Evaluate execution timing
    print("\n3. Testing execution timing evaluation...")
    # Create price series with a known optimal point
    price_series = np.array([100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 98, 99, 100])
    execution_index = 6  # Execute at price 100 (middle of the series)
    
    timing_result = slippage_analysis._handle_evaluate_execution_timing({
        "execution_price": price_series[execution_index],
        "price_series": price_series.tolist(),
        "execution_index": execution_index,
        "window_size": 5
    })
    
    print(f"Execution Price: {timing_result['execution_price']}")
    print(f"Min Price in Window: {timing_result['min_price']}")
    print(f"Max Price in Window: {timing_result['max_price']}")
    print(f"Mean Price in Window: {timing_result['mean_price']}")
    print(f"Percentile: {timing_result['percentile']}%")
    print(f"Timing Score: {timing_result['timing_score']}")
    print(f"Computed On: {timing_result['computed_on']}")
    
    # Test 4: Analyze liquidity impact
    print("\n4. Testing liquidity impact analysis...")
    liquidity_impact = slippage_analysis._handle_analyze_liquidity_impact({
        "pre_trade_bid_ask": pre_trade_bid_ask,
        "post_trade_bid_ask": post_trade_bid_ask,
        "side": "buy",
        "time_window": 5
    })
    
    print(f"Pre-Trade Spread: {liquidity_impact['pre_spread']}")
    print(f"Post-Trade Spread: {liquidity_impact['post_spread']}")
    print(f"Spread Change: {liquidity_impact['spread_change']}")
    print(f"Spread Change Percent: {liquidity_impact['spread_change_percent']}%")
    print(f"Size Change: {liquidity_impact['size_change']}")
    print(f"Size Change Percent: {liquidity_impact['size_change_percent']}%")
    print(f"Liquidity Score: {liquidity_impact['liquidity_score']}")
    
    # Optional: Plot the results
    try:
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Slippage Comparison
        plt.subplot(2, 2, 1)
        slippage_data = [buy_slippage['slippage_amount'], sell_slippage['slippage_amount']]
        plt.bar(['Buy', 'Sell'], slippage_data, color=['red', 'green'])
        plt.title('Slippage Comparison')
        plt.ylabel('Slippage Amount')
        
        # Plot 2: Market Impact - Price Series
        plt.subplot(2, 2, 2)
        plt.plot(range(len(pre_trade_prices)), pre_trade_prices, label='Pre-Trade')
        plt.plot(range(len(pre_trade_prices), len(pre_trade_prices) + len(post_trade_prices)), 
                 post_trade_prices, label='Post-Trade')
        plt.axvline(x=len(pre_trade_prices), color='red', linestyle='--', label='Trade Point')
        plt.title('Market Impact on Price')
        plt.legend()
        
        # Plot 3: Execution Timing
        plt.subplot(2, 2, 3)
        plt.plot(price_series, label='Price Series')
        plt.scatter(execution_index, price_series[execution_index], color='red', s=100, label='Execution Point')
        plt.axhline(y=timing_result['min_price'], color='green', linestyle='--', label='Min Price')
        plt.axhline(y=timing_result['max_price'], color='red', linestyle='--', label='Max Price')
        plt.axhline(y=timing_result['mean_price'], color='blue', linestyle='--', label='Mean Price')
        plt.title('Execution Timing Analysis')
        plt.legend()
        
        # Plot 4: Liquidity Impact
        plt.subplot(2, 2, 4)
        
        # Create a bar chart for bid-ask spreads
        labels = ['Pre-Trade', 'Post-Trade']
        bid_values = [pre_trade_bid_ask['bid_price'], post_trade_bid_ask['bid_price']]
        ask_values = [pre_trade_bid_ask['ask_price'], post_trade_bid_ask['ask_price']]
        
        x = np.arange(len(labels))
        width = 0.35
        
        plt.bar(x - width/2, bid_values, width, label='Bid')
        plt.bar(x + width/2, ask_values, width, label='Ask')
        
        plt.title('Liquidity Impact - Bid/Ask Prices')
        plt.xticks(x, labels)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('slippage_analysis_results.png')
        print("\nPlot saved as 'slippage_analysis_results.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
