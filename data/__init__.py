"""
FinGPT Data Package

Provides data connectors and preprocessing utilities for financial data.
"""

# Import main data components
from data.data_preprocessor import DataPreprocessor
from data.polygon_rest import PolygonRestClient
from data.polygon_ws import PolygonWebSocketClient
from data.reddit import RedditClient
from data.unusual_whales import UnusualWhalesClient
from data.yahoo_finance import YahooFinanceClient
