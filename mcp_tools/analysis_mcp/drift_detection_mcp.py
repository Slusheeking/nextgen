"""
Drift Detection MCP Server

This module implements a Model Context Protocol (MCP) server for detecting
price drift, trend changes, and momentum shifts in financial data.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from mcp_tools.base_mcp_server import BaseMCPServer

class DriftDetectionMCP(BaseMCPServer):
    """
    MCP server for price drift and trend detection.
    
    This server provides tools for identifying price drift from moving averages,
    detecting trend changes, and analyzing momentum shifts.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Drift Detection MCP server.
        
        Args:
            config: Optional configuration dictionary
        """
        super().__init__(name="drift_detection_mcp", config=config)
        
        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()
        
        # Register specific tools
        self._register_specific_tools()
        
        self.logger.info("DriftDetectionMCP initialized")
    
    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints.
        
        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "detect_ma_drift": {
                "description": "Detect drift from moving averages",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["short_window", "long_window", "drift_threshold"],
                "handler": self._handle_detect_ma_drift
            },
            "detect_trend_change": {
                "description": "Detect trend changes",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["window_size", "change_threshold"],
                "handler": self._handle_detect_trend_change
            },
            "analyze_momentum": {
                "description": "Analyze price momentum",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["window_size", "momentum_threshold"],
                "handler": self._handle_analyze_momentum
            },
            "detect_volatility_shift": {
                "description": "Detect shifts in volatility",
                "category": "trend_analysis",
                "required_params": ["prices"],
                "optional_params": ["window_size", "shift_threshold"],
                "handler": self._handle_detect_volatility_shift
            }
        }
    
    def _register_specific_tools(self):
        """Register tools specific to Drift Detection MCP."""
        self.register_tool(self.detect_ma_drift)
        self.register_tool(self.detect_trend_change)
        self.register_tool(self.analyze_momentum)
        self.register_tool(self.detect_volatility_shift)
    
    # Handler methods for specific endpoints
    
    def _handle_detect_ma_drift(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_ma_drift endpoint."""
        prices = params.get("prices", [])
        short_window = params.get("short_window", 5)
        long_window = params.get("long_window", 20)
        drift_threshold = params.get("drift_threshold", 0.02)
        
        if not prices or len(prices) < long_window:
            return {"error": "Insufficient price data"}
        
        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            
            # Calculate moving averages
            short_ma = self._calculate_moving_average(prices, short_window)
            long_ma = self._calculate_moving_average(prices, long_window)
            
            # Calculate drift
            drift_result = self._calculate_ma_drift(
                prices, short_ma, long_ma, drift_threshold
            )
            
            return drift_result
        except Exception as e:
            self.logger.error(f"Error detecting MA drift: {e}")
            return {"error": f"Failed to detect MA drift: {str(e)}"}
    
    def _handle_detect_trend_change(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_trend_change endpoint."""
        prices = params.get("prices", [])
        window_size = params.get("window_size", 10)
        change_threshold = params.get("change_threshold", 0.03)
        
        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}
        
        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            
            # Detect trend change
            trend_result = self._detect_trend_change(
                prices, window_size, change_threshold
            )
            
            return trend_result
        except Exception as e:
            self.logger.error(f"Error detecting trend change: {e}")
            return {"error": f"Failed to detect trend change: {str(e)}"}
    
    def _handle_analyze_momentum(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analyze_momentum endpoint."""
        prices = params.get("prices", [])
        window_size = params.get("window_size", 14)
        momentum_threshold = params.get("momentum_threshold", 0.1)
        
        if not prices or len(prices) < window_size:
            return {"error": "Insufficient price data"}
        
        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            
            # Analyze momentum
            momentum_result = self._analyze_momentum(
                prices, window_size, momentum_threshold
            )
            
            return momentum_result
        except Exception as e:
            self.logger.error(f"Error analyzing momentum: {e}")
            return {"error": f"Failed to analyze momentum: {str(e)}"}
    
    def _handle_detect_volatility_shift(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle detect_volatility_shift endpoint."""
        prices = params.get("prices", [])
        window_size = params.get("window_size", 10)
        shift_threshold = params.get("shift_threshold", 0.5)
        
        if not prices or len(prices) < window_size * 2:
            return {"error": "Insufficient price data"}
        
        try:
            # Convert to numpy array if needed
            if not isinstance(prices, np.ndarray):
                prices = np.array(prices)
            
            # Detect volatility shift
            volatility_result = self._detect_volatility_shift(
                prices, window_size, shift_threshold
            )
            
            return volatility_result
        except Exception as e:
            self.logger.error(f"Error detecting volatility shift: {e}")
            return {"error": f"Failed to detect volatility shift: {str(e)}"}
    
    # Core analysis methods
    
    def _calculate_moving_average(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate simple moving average.
        
        Args:
            prices: Array of price data
            window: Window size for moving average
            
        Returns:
            Array of moving averages
        """
        if len(prices) < window:
            return np.array([])
        
        # Calculate moving average
        ma = np.zeros_like(prices)
        for i in range(len(prices)):
            if i < window - 1:
                ma[i] = np.nan
            else:
                ma[i] = np.mean(prices[i-window+1:i+1])
        
        return ma
    
    def _calculate_ma_drift(self, prices: np.ndarray, short_ma: np.ndarray, 
                          long_ma: np.ndarray, drift_threshold: float) -> Dict[str, Any]:
        """
        Calculate drift from moving averages.
        
        Args:
            prices: Array of price data
            short_ma: Short-term moving average
            long_ma: Long-term moving average
            drift_threshold: Threshold for significant drift
            
        Returns:
            Drift analysis results
        """
        # Get current values
        current_price = prices[-1]
        current_short_ma = short_ma[-1]
        current_long_ma = long_ma[-1]
        
        # Calculate drifts
        drift_from_short = (current_price - current_short_ma) / current_short_ma
        drift_from_long = (current_price - current_long_ma) / current_long_ma
        ma_spread = (current_short_ma - current_long_ma) / current_long_ma
        
        # Determine drift direction and significance
        if abs(drift_from_short) > drift_threshold:
            short_drift_significant = True
            short_drift_direction = "up" if drift_from_short > 0 else "down"
        else:
            short_drift_significant = False
            short_drift_direction = "neutral"
            
        if abs(drift_from_long) > drift_threshold:
            long_drift_significant = True
            long_drift_direction = "up" if drift_from_long > 0 else "down"
        else:
            long_drift_significant = False
            long_drift_direction = "neutral"
        
        # Determine trend based on MA relationship
        if current_short_ma > current_long_ma:
            trend = "uptrend"
        elif current_short_ma < current_long_ma:
            trend = "downtrend"
        else:
            trend = "neutral"
        
        # Check for crossover
        crossover = False
        crossover_direction = "none"
        
        if len(short_ma) > 1 and len(long_ma) > 1:
            prev_short_ma = short_ma[-2]
            prev_long_ma = long_ma[-2]
            
            if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
                crossover = True
                crossover_direction = "bullish"
            elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
                crossover = True
                crossover_direction = "bearish"
        
        return {
            "current_price": float(current_price),
            "short_ma": float(current_short_ma),
            "long_ma": float(current_long_ma),
            "drift_from_short_ma": float(drift_from_short),
            "drift_from_long_ma": float(drift_from_long),
            "ma_spread": float(ma_spread),
            "short_drift_significant": short_drift_significant,
            "short_drift_direction": short_drift_direction,
            "long_drift_significant": long_drift_significant,
            "long_drift_direction": long_drift_direction,
            "trend": trend,
            "crossover": crossover,
            "crossover_direction": crossover_direction
        }
    
    def _detect_trend_change(self, prices: np.ndarray, window_size: int = 10,
                           change_threshold: float = 0.03) -> Dict[str, Any]:
        """
        Detect trend changes.
        
        Args:
            prices: Array of price data
            window_size: Window size for trend analysis
            change_threshold: Threshold for significant change
            
        Returns:
            Trend change detection results
        """
        if len(prices) < window_size * 2:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate linear regression for recent window
        recent_window = prices[-window_size:]
        recent_x = np.arange(window_size)
        recent_slope, recent_intercept = np.polyfit(recent_x, recent_window, 1)
        
        # Calculate linear regression for previous window
        prev_window = prices[-window_size*2:-window_size]
        prev_x = np.arange(window_size)
        prev_slope, prev_intercept = np.polyfit(prev_x, prev_window, 1)
        
        # Calculate slope change
        slope_change = recent_slope - prev_slope
        slope_change_percent = slope_change / abs(prev_slope) if prev_slope != 0 else float('inf')
        
        # Determine trend directions
        if recent_slope > 0:
            recent_trend = "uptrend"
        elif recent_slope < 0:
            recent_trend = "downtrend"
        else:
            recent_trend = "neutral"
            
        if prev_slope > 0:
            prev_trend = "uptrend"
        elif prev_slope < 0:
            prev_trend = "downtrend"
        else:
            prev_trend = "neutral"
        
        # Determine if trend change is significant
        significant_change = abs(slope_change_percent) > change_threshold
        
        # Determine trend change type
        if recent_trend != prev_trend:
            trend_change_type = f"{prev_trend}_to_{recent_trend}"
        elif significant_change:
            if slope_change > 0:
                trend_change_type = "acceleration"
            else:
                trend_change_type = "deceleration"
        else:
            trend_change_type = "continuation"
        
        return {
            "recent_trend": recent_trend,
            "previous_trend": prev_trend,
            "recent_slope": float(recent_slope),
            "previous_slope": float(prev_slope),
            "slope_change": float(slope_change),
            "slope_change_percent": float(slope_change_percent),
            "significant_change": significant_change,
            "trend_change_type": trend_change_type,
            "window_size": window_size
        }
    
    def _analyze_momentum(self, prices: np.ndarray, window_size: int = 14,
                        momentum_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Analyze price momentum.
        
        Args:
            prices: Array of price data
            window_size: Window size for momentum calculation
            momentum_threshold: Threshold for significant momentum
            
        Returns:
            Momentum analysis results
        """
        if len(prices) < window_size:
            return {"error": "Insufficient data for momentum analysis"}
        
        # Calculate rate of change (ROC)
        roc = (prices[-1] / prices[-window_size] - 1) * 100
        
        # Calculate RSI
        returns = np.diff(prices)
        gains = np.copy(returns)
        losses = np.copy(returns)
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        if len(gains) >= window_size:
            avg_gain = np.mean(gains[-window_size:])
            avg_loss = np.mean(losses[-window_size:])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Default if not enough data
        
        # Calculate momentum strength
        momentum_strength = abs(roc)
        
        # Determine momentum direction
        if roc > 0:
            momentum_direction = "positive"
        elif roc < 0:
            momentum_direction = "negative"
        else:
            momentum_direction = "neutral"
        
        # Determine if momentum is significant
        significant_momentum = momentum_strength > momentum_threshold
        
        # Determine momentum state based on RSI
        if rsi > 70:
            momentum_state = "overbought"
        elif rsi < 30:
            momentum_state = "oversold"
        else:
            momentum_state = "neutral"
        
        return {
            "rate_of_change": float(roc),
            "rsi": float(rsi),
            "momentum_strength": float(momentum_strength),
            "momentum_direction": momentum_direction,
            "significant_momentum": significant_momentum,
            "momentum_state": momentum_state,
            "window_size": window_size
        }
    
    def _detect_volatility_shift(self, prices: np.ndarray, window_size: int = 10,
                               shift_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Detect shifts in volatility.
        
        Args:
            prices: Array of price data
            window_size: Window size for volatility calculation
            shift_threshold: Threshold for significant volatility shift
            
        Returns:
            Volatility shift detection results
        """
        if len(prices) < window_size * 2:
            return {"error": "Insufficient data for volatility analysis"}
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate recent volatility
        recent_returns = returns[-window_size:]
        recent_volatility = np.std(recent_returns) * np.sqrt(252)  # Annualized
        
        # Calculate previous volatility
        prev_returns = returns[-window_size*2:-window_size]
        prev_volatility = np.std(prev_returns) * np.sqrt(252)  # Annualized
        
        # Calculate volatility shift
        volatility_shift = recent_volatility - prev_volatility
        volatility_shift_percent = volatility_shift / prev_volatility if prev_volatility > 0 else float('inf')
        
        # Determine if shift is significant
        significant_shift = abs(volatility_shift_percent) > shift_threshold
        
        # Determine shift direction
        if volatility_shift > 0:
            shift_direction = "increasing"
        elif volatility_shift < 0:
            shift_direction = "decreasing"
        else:
            shift_direction = "stable"
        
        # Determine volatility state
        if recent_volatility > 0.3:  # 30% annualized volatility is high
            volatility_state = "high"
        elif recent_volatility < 0.1:  # 10% annualized volatility is low
            volatility_state = "low"
        else:
            volatility_state = "normal"
        
        return {
            "recent_volatility": float(recent_volatility),
            "previous_volatility": float(prev_volatility),
            "volatility_shift": float(volatility_shift),
            "volatility_shift_percent": float(volatility_shift_percent),
            "significant_shift": significant_shift,
            "shift_direction": shift_direction,
            "volatility_state": volatility_state,
            "window_size": window_size
        }
    
    # Public API methods
    
    def detect_ma_drift(self, prices: List[float], short_window: int = 5,
                       long_window: int = 20) -> Dict[str, Any]:
        """
        Detect drift from moving averages.
        
        Args:
            prices: List of price data points
            short_window: Window size for short-term MA
            long_window: Window size for long-term MA
            
        Returns:
            Dictionary with drift analysis results
        """
        params = {
            "prices": prices,
            "short_window": short_window,
            "long_window": long_window
        }
        return self.call_endpoint("detect_ma_drift", params)
    
    def detect_trend_change(self, prices: List[float], 
                           window_size: int = 10) -> Dict[str, Any]:
        """
        Detect trend changes.
        
        Args:
            prices: List of price data points
            window_size: Window size for trend analysis
            
        Returns:
            Dictionary with trend change detection results
        """
        params = {
            "prices": prices,
            "window_size": window_size
        }
        return self.call_endpoint("detect_trend_change", params)
    
    def analyze_momentum(self, prices: List[float], 
                        window_size: int = 14) -> Dict[str, Any]:
        """
        Analyze price momentum.
        
        Args:
            prices: List of price data points
            window_size: Window size for momentum calculation
            
        Returns:
            Dictionary with momentum analysis results
        """
        params = {
            "prices": prices,
            "window_size": window_size
        }
        return self.call_endpoint("analyze_momentum", params)
    
    def detect_volatility_shift(self, prices: List[float], 
                              window_size: int = 10) -> Dict[str, Any]:
        """
        Detect shifts in volatility.
        
        Args:
            prices: List of price data points
            window_size: Window size for volatility calculation
            
        Returns:
            Dictionary with volatility shift detection results
        """
        params = {
            "prices": prices,
            "window_size": window_size
        }
        return self.call_endpoint("detect_volatility_shift", params)