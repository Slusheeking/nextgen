#!/usr/bin/env python3
"""
Unified MCP Server Launcher

This script instantiates all MCP servers in the project and keeps them running.
Adapt as needed if new MCPs are added.
"""

import time

from mcp_tools.data_mcp.polygon_news_mcp import PolygonNewsMCP
from mcp_tools.data_mcp.polygon_rest_mcp import PolygonRestMCP
from mcp_tools.data_mcp.polygon_ws_mcp import PolygonWSMCP
from mcp_tools.data_mcp.reddit_mcp import RedditMCP
from mcp_tools.data_mcp.unusual_whales_mcp import UnusualWhalesMCP
from mcp_tools.data_mcp.yahoo_finance_mcp import YahooFinanceMCP
from mcp_tools.data_mcp.yahoo_news_mcp import YahooNewsMCP

from mcp_tools.db_mcp.redis_mcp import RedisMCP

from mcp_tools.document_analysis_mcp.document_analysis_mcp import DocumentAnalysisMCP
from mcp_tools.financial_data_mcp.financial_data_mcp import FinancialDataMCP
# FinancialTextMCP removed - functionality consolidated into FinancialDataMCP
from mcp_tools.risk_analysis_mcp.risk_analysis_mcp import RiskAnalysisMCP
from mcp_tools.time_series_mcp.time_series_mcp import TimeSeriesMCP
from mcp_tools.trading_mcp.trading_mcp import TradingMCP
from mcp_tools.vector_store_mcp.vector_store_mcp import VectorStoreMCP

def main():
    print("Starting all MCP servers...")
    servers = [
        PolygonNewsMCP(),
        PolygonRestMCP(),
        PolygonWSMCP(),
        RedditMCP(),
        UnusualWhalesMCP(),
        YahooFinanceMCP(),
        YahooNewsMCP(),
        RedisMCP(),
        DocumentAnalysisMCP(),
        FinancialDataMCP(),
        # FinancialTextMCP removed - functionality consolidated into FinancialDataMCP
        RiskAnalysisMCP(),
        TimeSeriesMCP(),
        TradingMCP(),
        VectorStoreMCP(),
    ]
    print(f"{len(servers)} MCP servers instantiated and running.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Unified MCP server launcher stopped.")

if __name__ == "__main__":
    main()