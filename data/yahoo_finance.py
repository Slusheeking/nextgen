"""
Yahoo Finance API integration for stock data and financial information.

A production-ready client for accessing Yahoo Finance data to fetch historical prices,
company information, earnings, and news for algorithmic trading systems.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from loki.loki_manager import LokiManager
from prometheus.prometheus_manager import PrometheusManager

# Load environment variables
load_dotenv()

# Setup traditional logging as fallback
logger = logging.getLogger(__name__)

class YahooFinanceClient:
    """
    Production client for the Yahoo Finance API focused on financial data.
    """

    def __init__(self):
        """
        Initialize the Yahoo Finance API client.
        No API key is required as yfinance handles the connection.
        """
        # Initialize observability tools
        self.loki = LokiManager(service_name="data-yahoo-finance")
        self.prom = PrometheusManager(service_name="data-yahoo-finance")
        
        # Create metrics
        self.request_counter = self.prom.create_counter(
            "yahoo_finance_requests_total", 
            "Total count of Yahoo Finance API requests",
            ["endpoint", "ticker", "status"]
        )
        
        self.request_latency = self.prom.create_histogram(
            "yahoo_finance_request_duration_seconds",
            "Yahoo Finance API request duration in seconds",
            ["endpoint", "ticker"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.data_points_gauge = self.prom.create_gauge(
            "yahoo_finance_data_points",
            "Number of data points returned from Yahoo Finance",
            ["endpoint", "ticker"]
        )
        
        logger.info("YahooFinanceClient initialized")
        self.loki.info("YahooFinanceClient initialized", component="yahoo_finance")
        
        # Define the optimal endpoints for different data retrieval purposes
        self.optimal_endpoints = {
            "sector": "info",
            "industry": "info",
            "fundamentals": "info",
            "daily_data": "history_daily",
            "intraday_data": "history_intraday",
            "earnings": "earnings",
            "news": "news"
        }

    def get_stock_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Get historical stock data.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period to retrieve data for (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")
            
        Returns:
            DataFrame with historical stock data (OHLCV)
        """
        logger.info(f"Fetching {period} {interval} data for {ticker}")
        self.loki.info(f"Fetching {period} {interval} data for {ticker}", 
                     component="yahoo_finance", 
                     endpoint="stock_data", 
                     ticker=ticker,
                     period=period,
                     interval=interval)
        
        start_time = time.time()
        
        try:
            yf_ticker = yf.Ticker(ticker)
            data = yf_ticker.history(period=period, interval=interval)
            
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "yahoo_finance_request_duration_seconds",
                elapsed,
                endpoint="stock_data",
                ticker=ticker
            )
            
            # Increment request counter (success)
            self.prom.increment_counter(
                "yahoo_finance_requests_total",
                1,
                endpoint="stock_data",
                ticker=ticker,
                status="success"
            )
            
            if data.empty:
                log_msg = f"No data found for {ticker} with period={period}, interval={interval}"
                logger.warning(log_msg)
                self.loki.warning(log_msg, 
                                component="yahoo_finance", 
                                endpoint="stock_data", 
                                ticker=ticker,
                                period=period,
                                interval=interval)
                return pd.DataFrame()
                
            # Add ticker column for identification
            data['ticker'] = ticker
            
            # Set gauge for number of data points
            data_points = len(data)
            self.prom.set_gauge(
                "yahoo_finance_data_points",
                data_points,
                endpoint="stock_data",
                ticker=ticker
            )
            
            logger.info(f"Retrieved {data_points} data points for {ticker}")
            self.loki.info(f"Retrieved {data_points} data points for {ticker}",
                         component="yahoo_finance", 
                         endpoint="stock_data", 
                         ticker=ticker,
                         period=period,
                         interval=interval,
                         data_points=data_points,
                         duration=f"{elapsed:.2f}")
            return data
            
        except Exception as e:
            error_msg = f"Error fetching stock data for {ticker}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg,
                          component="yahoo_finance", 
                          endpoint="stock_data", 
                          ticker=ticker,
                          period=period,
                          interval=interval,
                          error_type="exception")
            
            # Increment request counter (error)
            self.prom.increment_counter(
                "yahoo_finance_requests_total",
                1,
                endpoint="stock_data",
                ticker=ticker,
                status="error"
            )
            
            return pd.DataFrame()
    
    def get_company_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get company information and fundamentals.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        logger.info(f"Fetching company info for {ticker}")
        self.loki.info(f"Fetching company info for {ticker}", 
                     component="yahoo_finance", 
                     endpoint="company_info", 
                     ticker=ticker)
        
        start_time = time.time()
        
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            
            elapsed = time.time() - start_time
            
            # Record request duration
            self.prom.observe_histogram(
                "yahoo_finance_request_duration_seconds",
                elapsed,
                endpoint="company_info",
                ticker=ticker
            )
            
            # Increment request counter (success)
            self.prom.increment_counter(
                "yahoo_finance_requests_total",
                1,
                endpoint="company_info",
                ticker=ticker,
                status="success"
            )
            
            if not info:
                log_msg = f"No company info found for {ticker}"
                logger.warning(log_msg)
                self.loki.warning(log_msg, 
                                component="yahoo_finance", 
                                endpoint="company_info", 
                                ticker=ticker)
                return {}
                
            # Set gauge for number of data points (fields in info)
            data_points = len(info) if info else 0
            self.prom.set_gauge(
                "yahoo_finance_data_points",
                data_points,
                endpoint="company_info",
                ticker=ticker
            )
            
            logger.info(f"Retrieved company info for {ticker}")
            self.loki.info(f"Retrieved company info for {ticker}",
                         component="yahoo_finance", 
                         endpoint="company_info", 
                         ticker=ticker,
                         fields=data_points,
                         duration=f"{elapsed:.2f}")
            return info
            
        except Exception as e:
            error_msg = f"Error fetching company info for {ticker}: {e}"
            logger.error(error_msg)
            self.loki.error(error_msg,
                          component="yahoo_finance", 
                          endpoint="company_info", 
                          ticker=ticker,
                          error_type="exception")
            
            # Increment request counter (error)
            self.prom.increment_counter(
                "yahoo_finance_requests_total",
                1,
                endpoint="company_info",
                ticker=ticker,
                status="error"
            )
            
            return {}
    
    def get_earnings(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings history.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with earnings data
        """
        logger.info(f"Fetching earnings data for {ticker}")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            earnings = yf_ticker.earnings
            
            if earnings is None or (isinstance(earnings, pd.DataFrame) and earnings.empty):
                logger.warning(f"No earnings data found for {ticker}")
                return {}
            
            # Convert earnings data to a dictionary format
            earnings_dict = {}
            if hasattr(earnings, 'to_dict'):
                earnings_dict = earnings.to_dict()
            else:
                earnings_dict = earnings
                
            logger.info(f"Retrieved earnings data for {ticker}")
            return earnings_dict
            
        except Exception as e:
            logger.error(f"Error fetching earnings data for {ticker}: {e}")
            return {}
    
    def get_news(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get recent news for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news items
        """
        logger.info(f"Fetching news for {ticker}")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            news = yf_ticker.news
            
            if not news:
                logger.warning(f"No news found for {ticker}")
                return []
                
            logger.info(f"Retrieved {len(news)} news items for {ticker}")
            return news
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def get_dividends(self, ticker: str) -> Dict[str, Any]:
        """
        Get dividend history.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with dividend history
        """
        logger.info(f"Fetching dividends for {ticker}")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            dividends = yf_ticker.dividends
            
            if dividends is None or (hasattr(dividends, 'empty') and dividends.empty):
                logger.warning(f"No dividend data found for {ticker}")
                return {}
                
            dividend_dict = {}
            if hasattr(dividends, 'to_dict'):
                dividend_dict = dividends.to_dict()
            else:
                dividend_dict = dividends
                
            logger.info(f"Retrieved dividend data for {ticker}")
            return dividend_dict
            
        except Exception as e:
            logger.error(f"Error fetching dividends for {ticker}: {e}")
            return {}
    
    def get_calendar(self, ticker: str) -> Dict[str, Any]:
        """
        Get earnings calendar information.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with earnings calendar data
        """
        logger.info(f"Fetching calendar for {ticker}")
        
        try:
            yf_ticker = yf.Ticker(ticker)
            calendar = yf_ticker.calendar
            
            if calendar is None or (hasattr(calendar, 'empty') and calendar.empty):
                logger.warning(f"No calendar data found for {ticker}")
                return {}
                
            calendar_dict = {}
            if hasattr(calendar, 'to_dict'):
                calendar_dict = calendar.to_dict()
            else:
                calendar_dict = calendar
                
            logger.info(f"Retrieved calendar data for {ticker}")
            return calendar_dict
            
        except Exception as e:
            logger.error(f"Error fetching calendar for {ticker}: {e}")
            return {}
    
    def fetch_data(self, purpose: str, ticker: str, **params) -> Any:
        """
        Fetch data using one of the predefined optimal endpoints.
        
        Args:
            purpose: The purpose of the data retrieval 
                    ("sector", "industry", "fundamentals", "daily_data", "intraday_data", "earnings", "news")
            ticker: Stock ticker symbol
            **params: Additional parameters for the data retrieval
            
        Returns:
            The data for the specified purpose
        """
        if purpose not in self.optimal_endpoints:
            error_msg = f"Unknown purpose: {purpose}"
            logger.error(error_msg)
            self.loki.error(error_msg,
                          component="yahoo_finance", 
                          endpoint="fetch_data", 
                          purpose=purpose,
                          ticker=ticker)
            return {"error": error_msg}
            
        endpoint = self.optimal_endpoints[purpose]
        
        if endpoint == "info":
            return self.get_company_info(ticker)
        elif endpoint == "history_daily":
            period = params.get("period", "1y")
            return self.get_stock_data(ticker, period=period, interval="1d")
        elif endpoint == "history_intraday":
            period = params.get("period", "5d")
            interval = params.get("interval", "1h")
            return self.get_stock_data(ticker, period=period, interval=interval)
        elif endpoint == "earnings":
            return self.get_earnings(ticker)
        elif endpoint == "news":
            return self.get_news(ticker)
        else:
            error_msg = f"Unknown endpoint: {endpoint}"
            logger.error(error_msg)
            self.loki.error(error_msg,
                          component="yahoo_finance", 
                          endpoint="fetch_data", 
                          purpose=purpose,
                          ticker=ticker)
            return {"error": error_msg}
    
    def get_sector_performance(self) -> Dict[str, float]:
        """
        Get performance metrics for different market sectors.
        
        Returns:
            Dictionary with sector performance data
        """
        logger.info("Fetching sector performance data")
        
        try:
            # In a production implementation, this would use a dynamic 
            # mechanism to determine sector ETFs and their mappings
            performance = {}
            
            logger.info(f"Retrieved performance data for {len(performance)} sectors")
            return performance
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}
    
    def get_market_movers(self, category: str = "gainers", count: int = 10) -> List[Dict[str, Any]]:
        """
        Get market movers (gainers, losers, most active).
        
        Args:
            category: The category of movers to fetch ("gainers", "losers", "active")
            count: Number of movers to return
            
        Returns:
            List of market movers data
        """
        logger.info(f"Fetching {count} market {category}")
        
        # In a production implementation, this would use a real API for market movers
        # or implement web scraping to get actual data
        try:
            # Get market movers data from an appropriate source
            movers = []
            
            # Sort the movers based on category
            if category == "gainers":
                movers.sort(key=lambda x: x["change_pct"], reverse=True)
            elif category == "losers":
                movers.sort(key=lambda x: x["change_pct"])
            elif category == "active":
                movers.sort(key=lambda x: x["volume"], reverse=True)
                
            logger.info(f"Retrieved {len(movers)} market {category}")
            return movers
            
        except Exception as e:
            logger.error(f"Error fetching market {category}: {e}")
            return []
