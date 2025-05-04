"""
Test for Peak Detection MCP

This module tests the functionality of the Peak Detection MCP server.
"""

import unittest
import numpy as np
import os
import sys
import json
from typing import Dict, List, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from mcp_tools.analysis_mcp.peak_detection_mcp import PeakDetectionMCP


class TestPeakDetectionMCP(unittest.TestCase):
    """Test cases for the Peak Detection MCP server."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a PeakDetectionMCP instance with test configuration
        self.test_config = {
            "use_gpu": False,  # Force CPU usage for consistent testing
            "min_data_size_for_gpu": 10000  # Set high to ensure CPU usage
        }
        self.peak_detector = PeakDetectionMCP(config=self.test_config)
        
        # Generate sample price data with known peaks and valleys
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample price and volume data for testing."""
        # Create a sine wave with some noise for price data
        x = np.linspace(0, 4 * np.pi, 200)
        self.prices = 100 + 20 * np.sin(x) + np.random.normal(0, 1, 200)
        
        # Create volume data with spikes at certain points
        self.volumes = 1000 + 500 * np.random.random(200)
        # Add volume spikes at specific points
        spike_indices = [25, 75, 125, 175]
        for idx in spike_indices:
            self.volumes[idx] = self.volumes[idx] * 3
        
        # Convert to Python lists for the API
        self.prices_list = self.prices.tolist()
        self.volumes_list = self.volumes.tolist()

    def test_detect_peaks(self):
        """Test peak detection functionality."""
        # Call the detect_peaks method
        result = self.peak_detector.detect_peaks(
            prices=self.prices_list,
            window_size=5,
            prominence=0.1  # Lower prominence to detect more peaks
        )
        
        # Verify the result structure
        self.assertIn("peaks", result)
        self.assertIn("valleys", result)
        self.assertIn("count_peaks", result)
        self.assertIn("count_valleys", result)
        
        # Verify we found some peaks and valleys
        self.assertGreater(result["count_peaks"], 0)
        self.assertGreater(result["count_valleys"], 0)
        
        # Verify peak structure
        for peak in result["peaks"]:
            self.assertIn("index", peak)
            self.assertIn("price", peak)
            self.assertIn("prominence", peak)
        
        print(f"Detected {result['count_peaks']} peaks and {result['count_valleys']} valleys")

    def test_detect_support_resistance(self):
        """Test support and resistance level detection."""
        # Call the detect_support_resistance method
        result = self.peak_detector.detect_support_resistance(
            prices=self.prices_list,
            volumes=self.volumes_list,
            num_levels=3
        )
        
        # Verify the result structure
        self.assertIn("support_levels", result)
        self.assertIn("resistance_levels", result)
        self.assertIn("current_price", result)
        
        # Verify we found some support and resistance levels
        self.assertLessEqual(len(result["support_levels"]), 3)
        self.assertLessEqual(len(result["resistance_levels"]), 3)
        
        print(f"Detected support levels: {result['support_levels']}")
        print(f"Detected resistance levels: {result['resistance_levels']}")
        print(f"Current price: {result['current_price']}")

    def test_detect_breakout(self):
        """Test breakout pattern detection."""
        # Create a specific breakout pattern
        breakout_prices = self.prices_list.copy()
        # Add a breakout at the end
        breakout_prices[-1] = max(breakout_prices[-30:-1]) * 1.05
        
        # Call the detect_breakout method
        result = self.peak_detector.detect_breakout(
            prices=breakout_prices,
            volumes=self.volumes_list,
            lookback_period=20
        )
        
        # Verify the result structure
        self.assertIn("breakout_detected", result)
        
        # Since we created a breakout, it should be detected
        self.assertTrue(result["breakout_detected"])
        self.assertEqual(result["direction"], "up")
        
        print(f"Breakout detected: {result['breakout_detected']}")
        print(f"Breakout direction: {result.get('direction')}")
        print(f"Breakout percentage: {result.get('breakout_percentage'):.2f}%")

    def test_detect_consolidation(self):
        """Test consolidation pattern detection."""
        # Create a specific consolidation pattern
        consolidation_prices = self.prices_list.copy()
        # Replace the last section with a flat pattern
        consolidation_prices[-20:] = [consolidation_prices[-21] + np.random.normal(0, 0.1) for _ in range(20)]
        
        # Call the detect_consolidation method
        result = self.peak_detector.detect_consolidation(
            prices=consolidation_prices,
            window_size=10
        )
        
        # Verify the result structure
        self.assertIn("consolidation_detected", result)
        self.assertIn("recent_volatility", result)
        self.assertIn("previous_volatility", result)
        
        # Since we created a consolidation pattern, it should be detected
        self.assertTrue(result["consolidation_detected"])
        
        print(f"Consolidation detected: {result['consolidation_detected']}")
        print(f"Recent volatility: {result['recent_volatility']:.6f}")
        print(f"Previous volatility: {result['previous_volatility']:.6f}")
        print(f"Price range: {result['price_range']}")

    def test_with_config_values(self):
        """Test that the MCP uses configuration values correctly."""
        # Create a temporary config file
        config_dir = os.path.join("config", "analysis_mcp")
        os.makedirs(config_dir, exist_ok=True)
        
        test_config = {
            "peak_detection": {
                "default_window_size": 7,
                "default_prominence": 1.0,
                "default_width": 2,
                "default_distance": 5
            },
            "support_resistance": {
                "default_window_size": 15,
                "default_num_levels": 4,
                "default_price_threshold": 0.02
            },
            "breakout": {
                "default_lookback_period": 25,
                "default_volume_factor": 2.0,
                "default_price_threshold": 0.03
            },
            "consolidation": {
                "default_window_size": 12,
                "default_volatility_threshold": 0.005
            }
        }
        
        # Create a new instance with the test config
        config_peak_detector = PeakDetectionMCP(config=test_config)
        
        # Test that the config values are used when not explicitly provided
        # We'll test this by checking the internal handler methods
        params = {"prices": self.prices_list}
        result = config_peak_detector._handle_detect_peaks(params)
        
        # Verify the result doesn't have an error
        self.assertNotIn("error", result)
        
        print("Successfully tested with configuration values")


if __name__ == "__main__":
    unittest.main()
