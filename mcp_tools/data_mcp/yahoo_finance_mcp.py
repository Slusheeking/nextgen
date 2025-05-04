"""
Yahoo Finance MCP Server

This module implements a Model Context Protocol (MCP) server for the Yahoo Finance
API, providing access to stock data and company information.
"""

import os
import logging
import time
import json
import requests
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from mcp_tools.data_mcp.base_data_mcp import BaseDataMCP

class YahooFinanceMCP(BaseDataMCP):
    """
    MCP server for Yahoo Finance API.
    
    This server provides access to Yahoo Finance data including stock prices,
    company information, financial statements, and news.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Yahoo Finance MCP server.
        
        Args:
            config: Optional configuration dictionary. May contain:
                - cache_enabled: Whether to enable caching (default: True)
                - cache_ttl: Time-to-live for cached data in seconds (default: 300)
                - default_period: Default time period for historical data
                - default_interval: Default interval for historical data
        """
        super().__init__(name="yahoo_finance_mcp", config=config)
        
        # Initialize client configuration
        self.finance_client = self._initialize_client()
        
        # Initialize endpoint definitions
        self.endpoints = self._initialize_endpoints()
        
        # Register specific tools
        self._register_specific_tools()
        
        self.logger.info("YahooFinanceMCP initialized with %d endpoints", len(self.endpoints))
    
    def _initialize_client(self) -> Dict[str, Any]:
        """
        Initialize client configuration.
        
        Yahoo Finance API doesn't require an API key as we're using yfinance.
        
        Returns:
            Client configuration or None if initialization fails
        """
        try:
            # Setup default configuration
            config = {
                "default_period": self.config.get("default_period", "1y"),
                "default_interval": self.config.get("default_interval", "1d")
            }
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Yahoo Finance client: {e}")
            return None
    
    def _initialize_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize available endpoints for Yahoo Finance.
        
        Returns:
            Dictionary mapping endpoint names to their configurations
        """
        return {
            "stock_data": {
                "description": "Get historical stock price data",
                "category": "market_data",
                "required_params": ["ticker"],
                "optional_params": ["period", "interval", "start", "end"],
                "default_values": {
                    "period": self.finance_client.get("default_period", "1y"),
                    "interval": self.finance_client.get("default_interval", "1d")
                },
                "handler": self._handle_stock_data
            },
            "company_info": {
                "description": "Get company information and fundamentals",
                "category": "fundamentals",
                "required_params": ["ticker"],
                "optional_params": [],
                "handler": self._handle_company_info
            },
            "financial_statements": {
                "description": "Get company financial statements",
                "category": "fundamentals",
                "required_params": ["ticker"],
                "optional_params": ["statement_type", "period_type"],
                "default_values": {
                    "statement_type": "income",  # income, balance, cash
                    "period_type": "annual"      # annual or quarterly
                },
                "handler": self._handle_financial_statements
            },
            "earnings": {
                "description": "Get company earnings information",
                "category": "fundamentals",
                "required_params": ["ticker"],
                "optional_params": [],
                "handler": self._handle_earnings
            },
            "recommendations": {
                "description": "Get analyst recommendations for a stock",
                "category": "analysis",
                "required_params": ["ticker"],
                "optional_params": [],
                "handler": self._handle_recommendations
            },
            "holders": {
                "description": "Get major holders of a stock",
                "category": "fundamentals",
                "required_params": ["ticker"],
                "optional_params": ["holder_type"],
                "default_values": {
                    "holder_type": "institutional"  # institutional or mutualfund
                },
                "handler": self._handle_holders
            },
            "options": {
                "description": "Get options chain for a ticker",
                "category": "options",
                "required_params": ["ticker"],
                "optional_params": ["expiration", "option_type"],
                "default_values": {
                    "option_type": "both"  # call, put, or both
                },
                "handler": self._handle_options
            },
            "market_status": {
                "description": "Check if market is currently open",
                "category": "market_info",
                "required_params": [],
                "optional_params": [],
                "handler": self._handle_market_status
            }
        }
    
    def _register_specific_tools(self):
        """Register tools specific to Yahoo Finance API."""
        self.register_tool(self.get_stock_price)
        self.register_tool(self.get_company_info)
        self.register_tool(self.get_historical_data)
        self.register_tool(self.get_financial_statements)
        self.register_tool(self.get_market_movers)
    
    def _execute_endpoint_fetch(self, endpoint_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the fetch operation based on endpoint configuration.
        
        Args:
            endpoint_config: Configuration for the endpoint
            params: Parameters for the request
            
        Returns:
            Fetched data
            
        Raises:
            Exception: If the fetch fails
        """
        # Merge provided params with default values
        merged_params = {**endpoint_config.get("default_values", {}), **params}
        
        # Get the handler function for this endpoint
        handler = endpoint_config.get("handler")
        if handler and callable(handler):
            return handler(merged_params)
        else:
            raise ValueError(f"No handler defined for endpoint")
    
    # Handler methods for specific endpoints
    
    def _handle_stock_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle stock data endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        period = params.get("period")
        interval = params.get("interval")
        start_date = params.get("start")
        end_date = params.get("end")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # If start/end dates are provided, use them instead of period
            if start_date and end_date:
                hist_data = ticker_obj.history(start=start_date, end=end_date, interval=interval)
            else:
                hist_data = ticker_obj.history(period=period, interval=interval)
            
            # Convert DataFrame to dictionary
            if hist_data.empty:
                return {"error": f"No data found for {ticker}"}
                
            data_dict = hist_data.reset_index().to_dict(orient="records")
            
            # Process dates to make them JSON serializable
            for item in data_dict:
                if "Date" in item and hasattr(item["Date"], "isoformat"):
                    item["Date"] = item["Date"].isoformat()
            
            return {
                "ticker": ticker,
                "period": period if not (start_date and end_date) else f"{start_date} to {end_date}",
                "interval": interval,
                "data_points": len(data_dict),
                "data": data_dict
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {ticker}: {e}")
            return {"error": f"Failed to fetch stock data: {str(e)}"}
    
    def _handle_company_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle company info endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Clean up the info dict to ensure all values are JSON serializable
            clean_info = {}
            for k, v in info.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    clean_info[k] = v
                else:
                    clean_info[k] = str(v)
            
            return {
                "ticker": ticker,
                "company_info": clean_info
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching company info for {ticker}: {e}")
            return {"error": f"Failed to fetch company info: {str(e)}"}
    
    def _handle_financial_statements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle financial statements endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        statement_type = params.get("statement_type", "income")
        period_type = params.get("period_type", "annual")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Get the appropriate financial statement
            if statement_type == "income":
                if period_type == "annual":
                    financials = ticker_obj.income_stmt
                else:
                    financials = ticker_obj.quarterly_income_stmt
            elif statement_type == "balance":
                if period_type == "annual":
                    financials = ticker_obj.balance_sheet
                else:
                    financials = ticker_obj.quarterly_balance_sheet
            elif statement_type == "cash":
                if period_type == "annual":
                    financials = ticker_obj.cashflow
                else:
                    financials = ticker_obj.quarterly_cashflow
            else:
                return {"error": f"Invalid statement type: {statement_type}"}
            
            # Convert to dict (if it exists)
            if financials is None or financials.empty:
                return {"error": f"No {statement_type} statements found for {ticker}"}
                
            # Process the DataFrame to make it JSON serializable
            financials_dict = {}
            for col in financials.columns:
                col_name = col.strftime('%Y-%m-%d') if hasattr(col, 'strftime') else str(col)
                financials_dict[col_name] = {}
                for idx in financials.index:
                    financials_dict[col_name][idx] = financials.loc[idx, col]
            
            return {
                "ticker": ticker,
                "statement_type": statement_type,
                "period_type": period_type,
                "data": financials_dict
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching financial statements for {ticker}: {e}")
            return {"error": f"Failed to fetch financial statements: {str(e)}"}
    
    def _handle_earnings(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle earnings endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        try:
            ticker_obj = yf.Ticker(ticker)
            earnings = ticker_obj.earnings
            earnings_dates = ticker_obj.earnings_dates
            
            result = {"ticker": ticker}
            
            # Process earnings history
            if earnings is not None and not earnings.empty:
                result["earnings_history"] = earnings.reset_index().to_dict(orient="records")
            
            # Process upcoming earnings dates
            if earnings_dates is not None and not earnings_dates.empty:
                earnings_dates_dict = earnings_dates.reset_index().to_dict(orient="records")
                
                # Process dates to make them JSON serializable
                for item in earnings_dates_dict:
                    if "Earnings Date" in item and hasattr(item["Earnings Date"], "isoformat"):
                        item["Earnings Date"] = item["Earnings Date"].isoformat()
                        
                result["earnings_dates"] = earnings_dates_dict
            
            # Get next earnings date if available
            next_date = ticker_obj.calendar
            if next_date is not None:
                result["next_earnings"] = {
                    "earnings_date": next_date.get("Earnings Date", "Unknown"),
                    "earnings_avg": next_date.get("EPS Estimate", "Unknown"),
                    "revenue_avg": next_date.get("Revenue Estimate", "Unknown")
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching earnings for {ticker}: {e}")
            return {"error": f"Failed to fetch earnings: {str(e)}"}
    
    def _handle_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle recommendations endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        try:
            ticker_obj = yf.Ticker(ticker)
            recommendations = ticker_obj.recommendations
            
            if recommendations is None or recommendations.empty:
                return {"error": f"No recommendations found for {ticker}"}
                
            # Convert DataFrame to dict and make it JSON serializable
            recommendations_dict = recommendations.reset_index().to_dict(orient="records")
            
            # Process dates to make them JSON serializable
            for item in recommendations_dict:
                if "Date" in item and hasattr(item["Date"], "isoformat"):
                    item["Date"] = item["Date"].isoformat()
            
            return {
                "ticker": ticker,
                "recommendations": recommendations_dict
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching recommendations for {ticker}: {e}")
            return {"error": f"Failed to fetch recommendations: {str(e)}"}
    
    def _handle_holders(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle holders endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        holder_type = params.get("holder_type", "institutional")
        
        try:
            ticker_obj = yf.Ticker(ticker)
            
            result = {"ticker": ticker}
            
            if holder_type == "institutional":
                holders = ticker_obj.institutional_holders
                if holders is not None and not holders.empty:
                    # Convert DataFrames to dicts
                    result["institutional_holders"] = holders.reset_index().to_dict(orient="records")
            elif holder_type == "mutualfund":
                holders = ticker_obj.mutualfund_holders
                if holders is not None and not holders.empty:
                    # Convert DataFrames to dicts
                    result["mutualfund_holders"] = holders.reset_index().to_dict(orient="records")
            elif holder_type == "all":
                institutional = ticker_obj.institutional_holders
                mutualfund = ticker_obj.mutualfund_holders
                
                if institutional is not None and not institutional.empty:
                    result["institutional_holders"] = institutional.reset_index().to_dict(orient="records")
                
                if mutualfund is not None and not mutualfund.empty:
                    result["mutualfund_holders"] = mutualfund.reset_index().to_dict(orient="records")
            else:
                return {"error": f"Invalid holder type: {holder_type}"}
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching holders for {ticker}: {e}")
            return {"error": f"Failed to fetch holders: {str(e)}"}
    
    
    def _handle_options(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle options endpoint."""
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("Missing required parameter: ticker")
            
        option_type = params.get("option_type", "both")
        expiration = params.get("expiration", None)  # None means earliest expiration
        
        try:
            ticker_obj = yf.Ticker(ticker)
            
            # Get available expirations
            expirations = ticker_obj.options
            
            if not expirations:
                return {"error": f"No options data found for {ticker}"}
                
            # If no expiration specified, use the earliest one
            if expiration is None:
                expiration = expirations[0]
            elif expiration not in expirations:
                return {"error": f"Invalid expiration date. Available dates: {expirations}"}
            
            # Get options data
            options = ticker_obj.option_chain(expiration)
            
            result = {
                "ticker": ticker,
                "expiration": expiration,
                "available_expirations": expirations,
                "options": {}
            }
            
            # Process options data based on type
            if option_type in ["both", "call"]:
                calls = options.calls
                if not calls.empty:
                    result["options"]["calls"] = calls.reset_index().to_dict(orient="records")
            
            if option_type in ["both", "put"]:
                puts = options.puts
                if not puts.empty:
                    result["options"]["puts"] = puts.reset_index().to_dict(orient="records")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching options for {ticker}: {e}")
            return {"error": f"Failed to fetch options: {str(e)}"}
    
    def _handle_market_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle market status endpoint."""
        try:
            # This is a simplified approximation of market hours
            now = datetime.now()
            
            # Check if current day is weekend
            if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                return {"is_market_open": False, "reason": "Weekend"}
            
            # Check if current time is within market hours (9:30 AM to 4:00 PM Eastern Time)
            # Note: This is a simplification and doesn't account for holidays
            eastern_time = datetime.utcnow() - timedelta(hours=4)  # Approximate ET
            market_open = eastern_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = eastern_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if eastern_time < market_open:
                return {"is_market_open": False, "reason": "Before market hours"}
            elif eastern_time > market_close:
                return {"is_market_open": False, "reason": "After market hours"}
            else:
                return {"is_market_open": True}
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return {"error": f"Failed to check market status: {str(e)}"}
    
    # Public API methods for models to use directly
    
    def get_stock_price(self, ticker: str, include_after_hours: bool = False) -> Dict[str, Any]:
        """
        Get current stock price and basic information.
        
        Args:
            ticker: Stock ticker symbol
            include_after_hours: Whether to include after-hours price (if available)
            
        Returns:
            Dictionary with stock price information
        """
        # Use the stock data endpoint with minimal period to get latest price
        data = self.fetch_data("stock_data", {
            "ticker": ticker,
            "period": "1d",
            "interval": "1m"
        })
        
        # Extract the most recent data point
        if "data" in data and data["data"]:
            latest = data["data"][-1]
            
            # Get company info for additional details
            info = self.fetch_data("company_info", {"ticker": ticker})
            company_info = info.get("company_info", {})
            
            result = {
                "ticker": ticker,
                "price": latest.get("Close"),
                "change": latest.get("Close") - latest.get("Open") if "Close" in latest and "Open" in latest else None,
                "change_percent": ((latest.get("Close") / latest.get("Open") - 1) * 100) if "Close" in latest and "Open" in latest and latest.get("Open") > 0 else None,
                "volume": latest.get("Volume"),
                "company_name": company_info.get("shortName", ""),
                "market_cap": company_info.get("marketCap"),
                "pe_ratio": company_info.get("trailingPE"),
                "dividend_yield": company_info.get("dividendYield")
            }
            
            return result
        
        return {"error": "No price data found"}
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information and fundamentals.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        return self.fetch_data("company_info", {"ticker": ticker})
    
    def get_historical_data(
        self, 
        ticker: str, 
        period: str = "1mo", 
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Get historical price data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max")
            interval: Data interval ("1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            Historical price data
        """
        return self.fetch_data("stock_data", {
            "ticker": ticker,
            "period": period,
            "interval": interval
        })
    
    def get_financial_statements(
        self, 
        ticker: str, 
        statement_type: str = "income", 
        period_type: str = "annual"
    ) -> Dict[str, Any]:
        """
        Get company financial statements.
        
        Args:
            ticker: Stock ticker symbol
            statement_type: Type of statement ("income", "balance", or "cash")
            period_type: Period type ("annual" or "quarterly")
            
        Returns:
            Financial statement data
        """
        return self.fetch_data("financial_statements", {
            "ticker": ticker,
            "statement_type": statement_type,
            "period_type": period_type
        })
    
    
    def get_market_movers(self, category: str = "gainers", limit: int = 10) -> Dict[str, Any]:
        """
        Get market movers (gainers, losers, active).
        
        Args:
            category: Type of movers ("gainers", "losers", "active")
            limit: Maximum number of stocks to return
            
        Returns:
            List of market movers
        """
        # Yahoo Finance API doesn't directly expose market movers through yfinance
        # We would need to implement a web scraping solution or use another data source
        return {
            "error": "Market movers functionality not available through direct API",
            "message": "Consider using Polygon.io for market movers data"
        }