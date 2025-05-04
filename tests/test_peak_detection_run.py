#!/usr/bin/env python3
"""
Test script for the Peak Detection MCP module.

This script demonstrates the usage of the PeakDetectionMCP class by:
1. Creating sample price and volume data
2. Initializing the PeakDetectionMCP server
3. Running various detection methods
4. Displaying the results
"""

import numpy as np
import matplotlib.pyplot as plt
from mcp_tools.analysis_mcp.peak_detection_mcp import PeakDetectionMCP

def main():
    """Run tests on the PeakDetectionMCP module with sample data."""
    print("Testing Peak Detection MCP module...")
    
    # Initialize the PeakDetectionMCP server
    config = {
        "use_gpu": False,  # Force CPU usage for testing
        "min_data_size_for_gpu": 1000,
    }
    peak_detection = PeakDetectionMCP(config)
    
    # Generate sample price data with peaks and valleys
    # Create a sine wave with some noise
    x = np.linspace(0, 4*np.pi, 200)
    sample_prices = 100 + 10 * np.sin(x) + np.random.normal(0, 1, 200)
    
    # Generate sample volume data
    sample_volumes = 1000 + 500 * np.abs(np.sin(x)) + np.random.normal(0, 100, 200)
    
    # Generate uptrend data with a breakout
    uptrend_prices = np.array([100 + i*0.1 + np.random.normal(0, 0.2) for i in range(100)])
    # Add a breakout at the end
    uptrend_prices = np.append(uptrend_prices, [uptrend_prices[-1] + 2 + np.random.normal(0, 0.1) for _ in range(10)])
    
    # Generate consolidation data
    consolidation_prices = np.array([100 + np.random.normal(0, 1) for _ in range(50)])
    
    # Convert numpy arrays to lists for the API
    sample_prices_list = sample_prices.tolist()
    sample_volumes_list = sample_volumes.tolist()
    uptrend_prices_list = uptrend_prices.tolist()
    consolidation_prices_list = consolidation_prices.tolist()
    
    # Test 1: Detect peaks and valleys
    print("\n1. Testing peak and valley detection...")
    peaks_result = peak_detection._handle_detect_peaks(
        {
            "prices": sample_prices_list,
            "window_size": 5,
            "prominence": 0.1  # Lower prominence to detect more peaks
        }
    )
    
    print(f"Detected {peaks_result['count_peaks']} peaks and {peaks_result['count_valleys']} valleys")
    print("First peak:", peaks_result['peaks'][0] if peaks_result['peaks'] else "None")
    print("First valley:", peaks_result['valleys'][0] if peaks_result['valleys'] else "None")
    
    # Test 2: Detect support and resistance levels
    print("\n2. Testing support and resistance detection...")
    sr_result = peak_detection._handle_detect_support_resistance(
        {
            "prices": sample_prices_list,
            "volumes": sample_volumes_list,
            "num_levels": 3
        }
    )
    
    print(f"Support levels: {sr_result['support_levels']}")
    print(f"Resistance levels: {sr_result['resistance_levels']}")
    print(f"Current price: {sr_result['current_price']}")
    
    # Test 3: Detect breakout patterns
    print("\n3. Testing breakout pattern detection...")
    breakout_result = peak_detection._handle_detect_breakout(
        {
            "prices": uptrend_prices_list,
            "volumes": sample_volumes_list[:len(uptrend_prices_list)],
            "lookback_period": 20
        }
    )
    
    if breakout_result.get('breakout_detected', False):
        print(f"Breakout detected!")
        print(f"Direction: {breakout_result['direction']}")
        print(f"Price: {breakout_result['price']}")
        print(f"Breakout level: {breakout_result['breakout_level']}")
        print(f"Breakout percentage: {breakout_result['breakout_percentage']:.2f}%")
        print(f"Volume confirmation: {breakout_result['volume_confirmation']}")
    else:
        print("No breakout detected")
        print(f"Current price: {breakout_result['price']}")
        print(f"Recent high: {breakout_result['recent_high']}")
        print(f"Recent low: {breakout_result['recent_low']}")
    
    # Test 4: Detect consolidation patterns
    print("\n4. Testing consolidation pattern detection...")
    consolidation_result = peak_detection._handle_detect_consolidation(
        {
            "prices": consolidation_prices_list,
            "window_size": 10
        }
    )
    
    print(f"Consolidation detected: {consolidation_result['consolidation_detected']}")
    print(f"Recent volatility: {consolidation_result['recent_volatility']:.6f}")
    print(f"Previous volatility: {consolidation_result['previous_volatility']:.6f}")
    print(f"Volatility reduction: {consolidation_result['volatility_reduction']}")
    print(f"Price range: {consolidation_result['price_range']}")
    
    # Optional: Plot the data with detected peaks and valleys
    try:
        plt.figure(figsize=(12, 8))
        
        # Plot 1: Price data with peaks and valleys
        plt.subplot(2, 2, 1)
        plt.plot(sample_prices, label='Price')
        
        # Plot peaks
        peak_indices = [p['index'] for p in peaks_result['peaks']]
        peak_values = [p['price'] for p in peaks_result['peaks']]
        plt.scatter(peak_indices, peak_values, color='green', marker='^', label='Peaks')
        
        # Plot valleys
        valley_indices = [v['index'] for v in peaks_result['valleys']]
        valley_values = [sample_prices[v['index']] for v in peaks_result['valleys']]
        plt.scatter(valley_indices, valley_values, color='red', marker='v', label='Valleys')
        
        # Plot support and resistance levels
        for level in sr_result['support_levels']:
            plt.axhline(y=level, color='blue', linestyle='--', alpha=0.5)
        
        for level in sr_result['resistance_levels']:
            plt.axhline(y=level, color='red', linestyle='--', alpha=0.5)
        
        plt.title('Price Data with Peaks, Valleys, Support and Resistance')
        plt.legend()
        
        # Plot 2: Uptrend data with breakout
        plt.subplot(2, 2, 2)
        plt.plot(uptrend_prices, label='Price')
        
        if breakout_result.get('breakout_detected', False):
            # Mark the breakout point
            plt.axvline(x=len(uptrend_prices)-10, color='red', linestyle='--', label='Breakout Point')
            plt.axhline(y=breakout_result['breakout_level'], color='green', linestyle='--', label='Breakout Level')
        
        plt.title('Uptrend Data with Breakout')
        plt.legend()
        
        # Plot 3: Consolidation data
        plt.subplot(2, 2, 3)
        plt.plot(consolidation_prices, label='Price')
        
        # Plot the min and max of the price range
        plt.axhline(y=consolidation_result['price_range']['min'], color='blue', linestyle='--', label='Min Price')
        plt.axhline(y=consolidation_result['price_range']['max'], color='red', linestyle='--', label='Max Price')
        
        plt.title('Consolidation Data')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('peak_detection_results.png')
        print("\nPlot saved as 'peak_detection_results.png'")
    except Exception as e:
        print(f"Could not generate plot: {e}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
